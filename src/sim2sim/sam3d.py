"""SAM 3D Objects (one RGB image + one object mask -> textured-vertex mesh and camera-frame layout) as a GPU service.

The researcher runs the service (scripts/sam3d-service.slurm, a non-H100/H200 GPU) and the strict controller relays an
agent's `sam3d` action to it through a private spool directory (owned by the researcher, mode 0700) on a filesystem that
both jobs see. The agent container never sees this module, the spool, the service or any credential; only the image,
mask and seed the agent sent cross the boundary, and only the mesh and layout return. Spool layout:
  service.json         heartbeat and pinned model identity, rewritten every HEARTBEAT seconds
  requests/<id>.json   written atomically by the controller; the service claims it by renaming into work/
  responses/<id>.json  written atomically by the service, consumed and removed by the controller
A missing, stale or silent service stops the run (ServiceFailure); there is no CPU or degraded path.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import struct
import sys
import threading
import time
import uuid

from .runtime import RuntimeFailure, SandboxFailure

MAX_PNG_BYTES = 4 * 1024 * 1024
MAX_SIDE = 2048
MAX_FACES = 40000
MAX_MESH_BYTES = 4 * 1024 * 1024
HEARTBEAT = 5.
STALE = 120.  # generous: NFS attribute caching can delay a fresh heartbeat by tens of seconds across nodes
FRAME = ("Mesh vertices are in the camera frame of the input image: x right, y up, -z forward (the observation calibration "
         "convention), in the model's monocular metric estimate, so scale and depth are approximate. layout gives the "
         "object-to-camera similarity in the model's own PyTorch3D camera frame (x left, y up, z forward), row-vector "
         "convention: camera = (local * scale) @ R(rotation_wxyz) + translation.")


class ServiceFailure(SandboxFailure):
    """The SAM 3D service is unavailable or broke its contract: the run stops loudly."""


def png_size(data):
    """(width, height, bit_depth, color_type) from a PNG header; no decoding happens on the host."""
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise RuntimeFailure("sam3d inputs must be PNG files")
    return struct.unpack(">IIBB", data[16:26])


def validate_request(request):
    """The agent's sam3d request as bounded PNG bytes and a seed; raises RuntimeFailure (an agent-visible error)."""
    if set(request) - {"op", "image_png_b64", "mask_png_b64", "seed"}:
        raise RuntimeFailure("sam3d accepts only image_png_b64, mask_png_b64 and seed")
    decoded = {}
    for key in ("image_png_b64", "mask_png_b64"):
        value = request.get(key)
        if not isinstance(value, str) or len(value) > MAX_PNG_BYTES * 4 // 3 + 4:
            raise RuntimeFailure("sam3d %s must be a base64 PNG of at most %d bytes" % (key, MAX_PNG_BYTES))
        try:
            decoded[key] = base64.b64decode(value, validate=True)
        except ValueError as exc:
            raise RuntimeFailure("sam3d %s is not valid base64" % key) from exc
    (width, height, depth, color), mask = png_size(decoded["image_png_b64"]), png_size(decoded["mask_png_b64"])
    if not (0 < width <= MAX_SIDE and 0 < height <= MAX_SIDE) or depth != 8 or color not in (2, 6):
        raise RuntimeFailure("sam3d image must be an 8-bit RGB or RGBA PNG of at most %dx%d pixels" % (MAX_SIDE, MAX_SIDE))
    if mask[:2] != (width, height) or mask[3] != 0 or mask[2] not in (1, 8):
        raise RuntimeFailure("sam3d mask must be a 1- or 8-bit grayscale PNG with the image's size (nonzero = object)")
    seed = request.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 31:
        raise RuntimeFailure("sam3d seed must be an integer in [0, 2^31)")
    return {"image_png_b64": request["image_png_b64"], "mask_png_b64": request["mask_png_b64"], "seed": seed}


def _write_json(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")))
    os.replace(temporary, path)


def _private_spool(spool):
    spool = Path(spool)
    info = spool.stat()
    if not spool.is_dir() or spool.is_symlink() or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ServiceFailure("SAM 3D spool must be a private (0700) directory owned by the researcher: %s" % spool)
    return spool


class Relay:
    """Controller side: forwards one validated agent request at a time and waits for the service's reply."""
    def __init__(self, spool, timeout=900.):
        self.spool, self.timeout = _private_spool(spool), float(timeout)
        self.service = self.status()

    def status(self):
        try:
            service = json.loads((self.spool / "service.json").read_text())
        except (OSError, ValueError) as exc:
            raise ServiceFailure("SAM 3D service is not running (start scripts/sam3d-service.slurm)") from exc
        age = time.time() - float(service["heartbeat_unix"])
        if not -STALE < age < STALE:  # clock skew between nodes is far below STALE
            raise ServiceFailure("SAM 3D service heartbeat is stale (%.0f s)" % age)
        return service

    def call(self, request, deadline, check):
        """(public result, private audit record). RuntimeFailure: bad input or a failed inference; ServiceFailure: no service."""
        payload = validate_request(request)
        identifier = uuid.uuid4().hex
        pending, answer = self.spool / "requests" / (identifier + ".json"), self.spool / "responses" / (identifier + ".json")
        started = time.monotonic()
        _write_json(pending, {"id": identifier, **payload})
        limit, polled = min(deadline, started + self.timeout), started
        try:
            while answer.name not in os.listdir(answer.parent):  # a directory read revalidates NFS caches across nodes
                check()
                now = time.monotonic()
                if now >= limit:
                    raise ServiceFailure("SAM 3D service did not answer within %.0f s" % (now - started))
                if now - polled >= HEARTBEAT:
                    self.status()
                    polled = now
                time.sleep(.5)
        finally:
            pending.unlink(missing_ok=True)  # never leave an unclaimed request behind for a later run
        raw = answer.read_bytes()
        answer.unlink()
        try:
            reply = json.loads(raw)
            assert reply["id"] == identifier and isinstance(reply["ok"], bool)
            if reply["ok"]:
                mesh = base64.b64decode(reply["mesh_ply_b64"], validate=True)
                assert mesh.startswith(b"ply\nformat binary_little_endian 1.0\n") and len(mesh) <= MAX_MESH_BYTES
                assert 0 < reply["faces"] <= MAX_FACES and len(reply["layout"]["rotation_wxyz"]) == 4
            else:
                assert isinstance(reply["error"], str)
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            raise ServiceFailure("SAM 3D service returned a malformed reply") from exc
        record = {"kind": "sam3d_call", "request_id": identifier, "ok": reply["ok"], "seed": payload["seed"],
                  "relay_seconds": time.monotonic() - started, "inference_seconds": reply.get("seconds"),
                  "service": {key: self.service[key] for key in ("host", "job_id", "model")},
                  "mesh_sha256": hashlib.sha256(mesh).hexdigest() if reply["ok"] else None}
        if not reply["ok"]:
            return None, {**record, "error": reply["error"][:2000]}
        public = {key: reply[key] for key in ("mesh_ply_b64", "vertices", "faces", "layout")}
        return {**public, "seed": payload["seed"], "frame": FRAME, "model": self.service["model"]["name"]}, record


def encode_ply(vertices, faces, colors):
    """Binary little-endian PLY with float32 xyz, uint8 rgb vertices and triangle faces."""
    import numpy as np
    vertices, faces, colors = np.asarray(vertices, "<f4"), np.asarray(faces, "<i4"), np.asarray(colors, "u1")
    assert vertices.ndim == 2 and vertices.shape[1] == 3 and colors.shape == vertices.shape and np.isfinite(vertices).all()
    assert faces.ndim == 2 and faces.shape[1] == 3 and 0 < len(faces) and faces.min() >= 0 and faces.max() < len(vertices)
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nelement face %d\n"
              "property list uchar int vertex_indices\nend_header\n" % (len(vertices), len(faces))).encode()
    vertex = np.empty(len(vertices), [("p", "<f4", 3), ("c", "u1", 3)])
    vertex["p"], vertex["c"] = vertices, colors
    face = np.empty(len(faces), [("n", "u1"), ("i", "<i4", 3)])
    face["n"], face["i"] = 3, faces
    return header + vertex.tobytes() + face.tobytes()


def handle(model, request):
    """One service reply. Input or inference errors become an agent-visible error; nothing else leaves the service."""
    import io
    import numpy as np
    from PIL import Image
    started = time.monotonic()
    try:
        image = Image.open(io.BytesIO(base64.b64decode(request["image_png_b64"], validate=True)))
        mask = Image.open(io.BytesIO(base64.b64decode(request["mask_png_b64"], validate=True)))
        assert image.size == mask.size and max(image.size) <= MAX_SIDE, "image and mask must have one size"
        rgb, mask = np.asarray(image.convert("RGB")), np.asarray(mask.convert("L")) > 0
        assert mask.sum() >= 16, "the mask must cover at least 16 pixels"
        vertices, faces, colors, layout = model(rgb, mask, int(request["seed"]))
        mesh = encode_ply(vertices, faces, colors)
        assert len(faces) <= MAX_FACES and len(mesh) <= MAX_MESH_BYTES, "mesh exceeds the output bound"
    except Exception as exc:  # the request is reported as failed; a broken GPU process dies with its heartbeat instead
        print("sam3d request %s failed: %r" % (request["id"], exc), file=sys.stderr, flush=True)
        return {"id": request["id"], "ok": False, "error": "SAM 3D failed: %s" % (exc if isinstance(exc, AssertionError) else type(exc).__name__)}
    return {"id": request["id"], "ok": True, "mesh_ply_b64": base64.b64encode(mesh).decode(), "vertices": len(vertices),
            "faces": len(faces), "layout": layout, "seconds": round(time.monotonic() - started, 3)}


def serve(spool, model, identity, stop, poll=.2):
    """Process requests oldest first until stop is set; the heartbeat runs on its own thread."""
    spool = _private_spool(spool)
    for name in ("requests", "work", "responses"):
        (spool / name).mkdir(mode=0o700, exist_ok=True)
    status = {"host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"), "pid": os.getpid(), "model": identity}

    def beat():
        while not stop.is_set():
            _write_json(spool / "service.json", {**status, "heartbeat_unix": time.time()})
            stop.wait(HEARTBEAT)
        (spool / "service.json").unlink(missing_ok=True)
    heart = threading.Thread(target=beat, daemon=True)
    heart.start()
    try:
        while not stop.is_set():
            pending = sorted((spool / "requests").glob("*.json"), key=lambda path: path.stat().st_mtime)
            if not pending:
                stop.wait(poll)
                continue
            claimed = spool / "work" / pending[0].name
            try:
                os.rename(pending[0], claimed)
            except FileNotFoundError:  # withdrawn by the controller after a timeout
                continue
            request = json.loads(claimed.read_text())
            _write_json(spool / "responses" / claimed.name, handle(model, request))
            claimed.unlink()
            for old in (spool / "responses").glob("*.json"):  # replies to runs that stopped waiting
                if time.time() - old.stat().st_mtime > 3600:
                    old.unlink(missing_ok=True)
    finally:
        stop.set()
        heart.join()


class Sam3dObjects:
    """The pinned SAM 3D Objects pipeline on one CUDA device (imported only inside the service environment)."""
    def __init__(self, repo, checkpoints):
        import torch
        assert torch.cuda.is_available(), "SAM 3D needs a CUDA GPU"
        torch.zeros(1, device="cuda")  # torch owns the primary context before kaolin's warp initializes (exclusive-process GPUs)
        sys.path.insert(0, str(Path(repo) / "notebook"))
        from inference import Inference
        self.torch, self.inference = torch, Inference(str(Path(checkpoints) / "pipeline.yaml"), compile=False)

    def __call__(self, rgb, mask, seed):
        import numpy as np
        import open3d as o3d
        from pytorch3d.transforms import quaternion_to_matrix
        from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
        out = self.inference(rgb, mask, seed=seed)
        glb = out["glb"]
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(glb.vertices, float)), o3d.utility.Vector3iVector(np.asarray(glb.faces)))
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(glb.visual.vertex_colors)[:, :3] / 255.)
        if len(mesh.triangles) > MAX_FACES:
            mesh = mesh.simplify_quadric_decimation(MAX_FACES)
        mesh.remove_unreferenced_vertices()
        # to_glb rotated the decoder's z-up local frame to y-up (v @ A); undo it, then apply the predicted layout.
        local = np.asarray(mesh.vertices) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T
        scale, rotation, translation = (out[key].float() for key in ("scale", "rotation", "translation"))
        transform = compose_transform(scale=scale, rotation=quaternion_to_matrix(rotation), translation=translation)
        camera = transform.transform_points(self.torch.as_tensor(local, dtype=self.torch.float32, device=scale.device)[None])[0].cpu().numpy()
        layout = {"rotation_wxyz": rotation[0].tolist(), "translation": translation[0].tolist(), "scale": scale[0].tolist()}
        colors = np.clip(np.asarray(mesh.vertex_colors) * 255 + .5, 0, 255)
        return camera * [-1, 1, -1], np.asarray(mesh.triangles), colors, layout  # PyTorch3D camera -> x right, y up, -z forward


def verify_lock(lock_path, root):
    """sha256 of every pinned file (gated checkpoints and cached auxiliary weights) against the lock; returns its digest."""
    lock = json.loads(Path(lock_path).read_text())
    for relative, digest in lock["files_sha256"].items():
        hasher = hashlib.sha256()
        with open(Path(root) / relative, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                hasher.update(block)
        assert hasher.hexdigest() == digest, "pinned SAM 3D file changed: " + relative
    return lock, hashlib.sha256(Path(lock_path).read_bytes()).hexdigest()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Serve SAM 3D Objects to strict sim2sim controllers through a private spool.")
    parser.add_argument("--spool", required=True)
    parser.add_argument("--root", required=True, help="SAM3D_ROOT holding sam-3d-objects/, the cache and the lock")
    args = parser.parse_args(argv)
    root = Path(args.root)
    lock, lock_sha256 = verify_lock(root / "sam3d-lock.json", root)
    assert (root / "sam-3d-objects/.git/HEAD").read_text().strip() == lock["code_commit"], "SAM 3D code is not at the pinned commit"
    model = Sam3dObjects(root / "sam-3d-objects", root / lock["checkpoints"])
    identity = {"name": "facebook/sam-3d-objects@" + lock["revision"][:12], "revision": lock["revision"], "code_commit": lock["code_commit"],
                "lock_sha256": lock_sha256, "device": model.torch.cuda.get_device_name(0)}
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())  # finish the current request, then withdraw the heartbeat
    print("sam3d service ready:", json.dumps(identity), flush=True)
    serve(args.spool, model, identity, stop)


if __name__ == "__main__":
    main()
