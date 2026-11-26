"""Dependency-free synchronous client for the sandbox's JSON-lines protocol."""
import base64
import io
import json
from pathlib import Path
import socket
import sys
import zipfile


class Client:
    def __init__(self, reader=None, writer=None):
        self.reader = reader or sys.stdin
        self.writer = writer or sys.stdout
        self.task = self._read()

    def _read(self):
        line = self.reader.readline()
        if not line:
            raise EOFError("Host closed the protocol")
        return json.loads(line)

    def request(self, op, **kwargs):
        self.writer.write(json.dumps({"op": op, **kwargs}, separators=(",", ":")) + "\n")
        self.writer.flush()
        response = self._read()
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Host operation failed"))
        return response

    def observe(self, ticks=0, *, world="reference"):
        return self.request("observe", ticks=ticks, world=world)["observation"]

    def arm_target(self, joint_targets, gripper_width_m=None, ticks=20, *, world="reference"):
        """gripper_width_m only for an arm with the Panda gripper (robot panda)."""
        gripper = {} if gripper_width_m is None else {"gripper_width_m": gripper_width_m}
        return self.request("arm_target", joint_targets=list(joint_targets), **gripper, ticks=ticks, world=world)["observation"]

    def hand_target(self, actuator_targets, ticks=20, *, world="reference"):
        return self.request("hand_target", actuator_targets=list(actuator_targets), ticks=ticks, world=world)["observation"]

    # fields: grasp_index=0 or 1 when the task offers two point grasps (public_tool_config point_grasps = 2).
    def grasp(self, pixel, *, world="reference", **fields):
        return self.request("grasp", pixel=list(pixel), world=world, **fields)["observation"]

    # rotation: [rx, ry, rz] world rotation vector (radians) about the grasp point, only with public_tool_config grasp = "weld".
    def move(self, displacement, ticks=20, *, world="reference", rotation=None, **fields):
        turn = {} if rotation is None else {"rotation": list(rotation)}
        return self.request("move", displacement=list(displacement), ticks=ticks, world=world, **turn, **fields)["observation"]

    def release(self, ticks=0, *, world="reference", **fields):
        return self.request("release", ticks=ticks, world=world, **fields)["observation"]

    def reset(self, *, world="reference"):
        return self.request("reset", world=world)["observation"]

    def inference(self, messages, tools=None):
        return self.request("inference", messages=messages, **({"tools": tools} if tools else {}))["inference"]

    def sam3d(self, image_png, mask_png, seed=0):
        """SAM 3D Objects on one image and object mask (PNG bytes): mesh_ply_b64, vertices, faces, layout, frame."""
        encode = lambda data: base64.b64encode(data).decode()
        return self.request("sam3d", image_png_b64=encode(image_png), mask_png_b64=encode(mask_png), seed=seed)["sam3d"]

    def _project(self, project_dir):
        root = Path(project_dir).resolve()
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    raise ValueError("Project must not contain symlinks")
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())
        maximum = int(self.task["submission"]["max_bytes"])
        if len(data.getvalue()) > maximum:
            raise ValueError("Project exceeds compressed submission limit")
        return base64.b64encode(data.getvalue()).decode()

    def load_reconstruction(self, project_dir, *, entrypoint="scene.xml", camera="fixed"):
        return self.request("load_reconstruction", world="reconstruction", project_b64=self._project(project_dir),
                            entrypoint=entrypoint, camera=camera)

    def submit(self, project_dir):
        return self.request("submit", project_b64=self._project(project_dir))

    def progress(self, text, *, origin="agent"):
        return self.request("progress", text=text, origin=origin)

    def finish(self):
        return self.request("finish")


class Channel:
    """Agent code's controller channel (task code_channel): the sandbox harness relays physical actions while a bash command
    runs. request() returns the controller's reply (base64 images included) and raises on an unsuccessful one."""
    def __init__(self, path="/tmp/sim2sim-controller.sock"):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.connect(path)
        self.stream = self.socket.makefile("rwb")

    def request(self, op, **kwargs):
        self.stream.write(json.dumps({"op": op, **kwargs}, separators=(",", ":")).encode() + b"\n")
        self.stream.flush()
        line = self.stream.readline()
        if not line:
            raise EOFError("Controller channel closed")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Host operation failed"))
        return response

    def close(self):
        self.stream.close()
        self.socket.close()
