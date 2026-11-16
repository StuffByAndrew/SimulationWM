#!/usr/bin/env bash
#SBATCH --job-name=sam3d-smoke
#SBATCH --partition=kempner_gpu_priority
#SBATCH --account=kempner_qianqian_lab
#SBATCH --qos=kempner_qianqian_priority
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=sam3d-smoke-%j.log
# SAM 3D service environment check on one A100 (never H100/H200; the pinned cu121 stack has no Blackwell sm_120 kernels).
#   sbatch scripts/sam3d-smoke.slurm env    CUDA visibility, compiled extensions and their CUDA kernels (no weights)
#   sbatch scripts/sam3d-smoke.slurm full   also loads the gated checkpoints (network on: the pipeline fetches its pinned
#                                           auxiliary weights into the cache), runs one real inference through the spool
#                                           relay, and writes SAM3D_ROOT/sam3d-lock.json (sha256 of every weight file)
set -euo pipefail
mode="${1:?usage: sam3d-smoke.slurm env|full}"
[[ "$mode" == env || "$mode" == full ]]
repo_root="${SIM2SIM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
root="${SAM3D_ROOT:-/n/netscratch/qianqian_lab/Lab/aejxie/sam3d}"
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
[[ "$gpu" != *H100* && "$gpu" != *H200* ]] || { echo "refusing H100/H200 GPU: $gpu" >&2; exit 2; }
export PYTHONPATH="$repo_root/src" CONDA_PREFIX="$root/env" CUDA_HOME="$root/env" PATH="$root/env/bin:$PATH"
export HF_HUB_CACHE="$root/cache/hf-hub" TORCH_HOME="$root/cache/torch" XDG_CACHE_HOME="$root/cache/xdg" HF_HUB_DISABLE_TELEMETRY=1
export TORCH_EXTENSIONS_DIR="$root/cache/torch-extensions" PYTHONUNBUFFERED=1 LIDRA_SKIP_INIT=true
printf 'node=%s job=%s gpu=%s mode=%s compute_mode=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$gpu" "$mode" "$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader)"
"$root/env/bin/python" - "$mode" "$root" <<'EOF'
import importlib, sys, time
import torch
assert torch.cuda.is_available(), "CUDA is not visible"
device = torch.cuda.get_device_properties(0)
print("torch", torch.__version__, "cuda", torch.version.cuda, device.name, "sm_%d%d" % (device.major, device.minor), "%.1f GiB" % (device.total_memory / 2 ** 30))
assert device.total_memory >= 32 * 10 ** 9, "SAM 3D Objects needs at least 32 GB of GPU memory"
torch.zeros(1, device="cuda")  # torch owns the primary context before kaolin's warp initializes (exclusive-process GPUs)
for name in ("pytorch3d._C", "kaolin._C", "gsplat", "flash_attn", "spconv.pytorch", "xformers.ops", "moge", "open3d", "sam3d_objects"):
    module = importlib.import_module(name)
    print("import", name, getattr(module, "__version__", ""))
from pytorch3d.ops import knn_points
from flash_attn import flash_attn_func
import xformers.ops
points = torch.rand(1, 4096, 3, device="cuda")
assert knn_points(points, points, K=1).dists.abs().max().item() < 1e-6
q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.float16)
reference = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, q, q))).transpose(1, 2)
assert (flash_attn_func(q, q, q) - reference).abs().max().item() < 1e-2 and (xformers.ops.memory_efficient_attention(q, q, q) - reference).abs().max().item() < 1e-2
print("CUDA kernels: pytorch3d knn, flash-attn and xformers attention match torch sdpa")
sys.path.insert(0, sys.argv[2] + "/sam-3d-objects/notebook")
from inference import Inference  # the pinned pipeline's import chain (hydra patch, kaolin, gradio, pytorch3d)
print("notebook inference module imports")
if sys.argv[1] == "env":
    sys.exit(0)
import base64, hashlib, io, json, os, shutil, threading
from pathlib import Path
import numpy as np
from PIL import Image
from sim2sim.sam3d import Relay, Sam3dObjects, serve
root, job = Path(sys.argv[2]), os.environ["SLURM_JOB_ID"]
repo, lock_path = root / "sam-3d-objects", root / "sam3d-lock.json"
assert not lock_path.exists(), "sam3d-lock.json exists; delete it deliberately to re-pin"
clock = time.monotonic()
model = Sam3dObjects(repo, root / "checkpoints")
print("pipeline loaded in %.1f s" % (time.monotonic() - clock))


def encode(image):
    data = io.BytesIO()
    image.save(data, "PNG")
    return base64.b64encode(data.getvalue()).decode()
example = repo / "notebook/images/shutterstock_stylish_kidsroom_1640806567"
rgb = Image.open(example / "image.png").convert("RGB")
mask = np.asarray(Image.open(example / "14.png")) > 0  # as notebook/inference.py load_mask: alpha if multichannel
mask = Image.fromarray((mask[..., -1] if mask.ndim == 3 else mask).astype(np.uint8) * 255)
size = tuple(round(v * 1024 / max(rgb.size)) for v in rgb.size)  # agent-sized input: the full photo exceeds MAX_PNG_BYTES
rgb, mask = rgb.resize(size, Image.LANCZOS), mask.resize(size, Image.NEAREST)
spool, stop = root / ("smoke-spool-" + job), threading.Event()
spool.mkdir(mode=0o700)
thread = threading.Thread(target=serve, args=(spool, model, {"name": "smoke"}, stop))
thread.start()
time.sleep(1)
clock = time.monotonic()
try:
    result, record = Relay(spool).call({"image_png_b64": encode(rgb), "mask_png_b64": encode(mask), "seed": 42}, time.monotonic() + 900, lambda: None)
finally:  # a failed call must not leave the service thread holding the GPU job open
    stop.set()
    thread.join()
assert result is not None, record
print("inference through the relay in %.1f s: %d vertices, %d faces, layout %s" % (time.monotonic() - clock, result["vertices"], result["faces"], json.dumps(result["layout"])))
mesh = base64.b64decode(result["mesh_ply_b64"])
vertices = np.frombuffer(mesh.split(b"end_header\n", 1)[1][:result["vertices"] * 15], [("p", "<f4", 3), ("c", "u1", 3)])["p"]
print("camera-frame vertex z range (in front of the camera means negative): %.3f..%.3f" % (vertices[:, 2].min(), vertices[:, 2].max()))
(root / ("smoke-%s.ply" % job)).write_bytes(mesh)
assert not list(spool.rglob("*.json"))
shutil.rmtree(spool)
files = {}
for path in sorted([*(root / "checkpoints").rglob("*"), *(root / "cache/hf-hub").rglob("*"), *(root / "cache/torch").rglob("*")]):
    if path.is_file() and not path.is_symlink() and ".lock" not in path.suffixes and "__pycache__" not in path.parts:
        hasher = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                hasher.update(block)
        files[path.relative_to(root).as_posix()] = hasher.hexdigest()
pins = json.loads((Path(os.environ["PYTHONPATH"]).parent / "experiments/sam3d/checkpoint-pins.json").read_text())
assert (repo / ".git/HEAD").read_text().strip() == pins["code_commit"]
lock_path.write_text(json.dumps({"revision": pins["revision"], "code_commit": pins["code_commit"], "checkpoints": "checkpoints",
                                 "files_sha256": files}, indent=1, sort_keys=True))
print("wrote", lock_path, "with", len(files), "pinned files")
EOF
