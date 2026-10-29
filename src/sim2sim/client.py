"""Dependency-free synchronous client for the sandbox's JSON-lines protocol."""
import base64
import io
import json
from pathlib import Path
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

    def observe(self, ticks=0):
        return self.request("observe", ticks=ticks)["observation"]

    def grasp(self, pixel):
        return self.request("grasp", pixel=list(pixel))["observation"]

    def move(self, displacement, ticks=20):
        return self.request("move", displacement=list(displacement), ticks=ticks)["observation"]

    def release(self, ticks=0):
        return self.request("release", ticks=ticks)["observation"]

    def reset(self):
        return self.request("reset")["observation"]

    def inference(self, messages):
        return self.request("inference", messages=messages)["inference"]

    def submit(self, project_dir):
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
        return self.request("submit", project_b64=base64.b64encode(data.getvalue()).decode())

    def finish(self):
        return self.request("finish")
