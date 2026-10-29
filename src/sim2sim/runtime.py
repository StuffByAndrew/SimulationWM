"""Rootless Podman boundary and host-owned agent protocol."""
from __future__ import annotations

import base64
import io
import hashlib
import importlib.metadata
import platform
import json
import os
import queue
from pathlib import Path
import selectors
import subprocess
import time
import threading
import tomllib
import urllib.request
import urllib.parse
import uuid
import zipfile


class RuntimeFailure(RuntimeError):
    pass


class SandboxFailure(RuntimeFailure):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeFailure("Inference redirects are forbidden")


def check_rootless():
    try:
        p = subprocess.run(["podman", "info", "--format=json"],
                           capture_output=True, text=True, timeout=20, check=True)
        info = json.loads(p.stdout)
        host = info["host"]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeFailure("A working rootless Podman installation is required") from exc
    if host.get("security", {}).get("rootless") is not True:
        raise RuntimeFailure("Refusing non-rootless Podman")
    if host.get("cgroupVersion") != "v2" or not {"cpu", "memory", "pids"}.issubset(set(host.get("cgroupControllers") or [])):
        raise RuntimeFailure("Rootless cgroup v2 with delegated cpu, memory and pids controllers is required")


def container_command(image, command, limits, name):
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise RuntimeFailure("container command must be a nonempty string array")
    return ["podman", "run", "--rm", "--pull=never", "--name", name,
            "--network=none", "--read-only", "--log-driver=none", "--http-proxy=false", "--image-volume=ignore", "--entrypoint=", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--user=65532:65532", "--pid=private", "--ipc=private", "--uts=private",
            "--cgroups=enabled", "--pids-limit=" + str(int(limits.get("pids", 64))),
            "--memory=" + str(limits.get("memory", "1g")), "--memory-swap=" + str(limits.get("memory", "1g")),
            "--cpus=" + str(float(limits.get("cpus", 1))), "--ulimit=nofile=256:256",
            "--ulimit=core=0:0", "--tmpfs=/workspace:rw,noexec,nosuid,nodev,size=" + str(limits.get("workspace", "256m")) + ",mode=1777",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777", "--workdir=/workspace",
            "--env=HOME=/workspace", "--env=PYTHONDONTWRITEBYTECODE=1", "--env=OPENBLAS_NUM_THREADS=1", "--env=OMP_NUM_THREADS=1", "--env=LP_NUM_THREADS=1", "-i", image, *command]


def stop_container(proc, name):
    if proc.poll() is None:
        proc.kill()
    try:
        subprocess.run(["podman", "rm", "--force", "--time=0", name], capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


class BoundedEvents:
    def __init__(self, path, limit=8 * 1024 * 1024):
        self.handle = path.open("w")
        self.limit = limit
        self.written = 0

    def write(self, value):
        size = len(value.encode())
        if self.written + size > self.limit:
            raise RuntimeFailure("Event log byte budget exhausted")
        self.handle.write(value)
        self.written += size

    def flush(self):
        self.handle.flush()

    def close(self):
        self.handle.close()


class BoundedStderr:
    def __init__(self, stream, path, limit=1024 * 1024):
        self.stream = stream
        self.path = path
        self.limit = limit
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self):
        remaining = self.limit
        with self.path.open("wb") as handle:
            while True:
                data = self.stream.read(65536)
                if not data:
                    break
                if remaining:
                    saved = data[:remaining]
                    handle.write(saved)
                    remaining -= len(saved)

    def close(self):
        self.thread.join(timeout=2)


class Transport:
    def __init__(self, proc, max_bytes):
        self.proc = proc
        self.max_bytes = max_bytes
        self.buffer = bytearray()
        os.set_blocking(proc.stdout.fileno(), False)
        os.set_blocking(proc.stdin.fileno(), False)

    def _wait(self, stream, event, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeFailure("Active wall-time budget exhausted")
        with selectors.DefaultSelector() as selector:
            selector.register(stream, event)
            if not selector.select(remaining):
                raise RuntimeFailure("Active wall-time budget exhausted")

    def send(self, payload, deadline):
        data = memoryview(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
        while data:
            self._wait(self.proc.stdin, selectors.EVENT_WRITE, deadline)
            try:
                n = os.write(self.proc.stdin.fileno(), data)
                data = data[n:]
            except BlockingIOError:
                continue
            except BrokenPipeError as exc:
                raise RuntimeFailure("Agent closed stdin") from exc

    def receive(self, deadline):
        while b"\n" not in self.buffer:
            self._wait(self.proc.stdout, selectors.EVENT_READ, deadline)
            try:
                chunk = os.read(self.proc.stdout.fileno(), min(65536, self.max_bytes + 1 - len(self.buffer)))
            except BlockingIOError:
                continue
            if not chunk:
                raise RuntimeFailure("Agent exited before completing protocol")
            self.buffer.extend(chunk)
            if len(self.buffer) > self.max_bytes:
                raise RuntimeFailure("Agent protocol message exceeds byte limit")
        line, _, rest = self.buffer.partition(b"\n")
        self.buffer = bytearray(rest)
        try:
            value = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            raise RuntimeFailure("Agent emitted invalid JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeFailure("Agent message must be a JSON object")
        return value


def decode_submission(value, max_bytes):
    if not isinstance(value, str) or len(value) > ((max_bytes + 2) // 3) * 4:
        raise RuntimeFailure("Submission exceeds compressed byte limit")
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise RuntimeFailure("Invalid submission base64") from exc
    if len(data) > max_bytes:
        raise RuntimeFailure("Submission exceeds compressed byte limit")
    return data


class InferenceBroker:
    """Agent supplies messages only; destination, model and key remain host controlled."""
    def __init__(self, config):
        self.config = config
        self.calls = 0
        self.total_tokens = 0
        self.budget_lock = threading.Lock()
        endpoint = config.get("endpoint", "")
        parsed = urllib.parse.urlsplit(endpoint)
        if endpoint and (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment):
            raise RuntimeFailure("Inference endpoint must be a fixed HTTPS URL")

    def call(self, request, timeout):
        # A daemon owns blocking HTTPS I/O; the agent's deadline never waits on it.
        with self.budget_lock:
            if self.calls >= int(self.config.get("max_calls", 20)):
                raise RuntimeFailure("Inference call budget exhausted")
            ceiling = self.config.get("max_total_tokens")
            if ceiling is not None and self.total_tokens >= int(ceiling):
                raise RuntimeFailure("Inference reported-token budget exhausted")
            self.calls += 1
        allowance = max(0, min(timeout, float(self.config.get("timeout_seconds", 60))))
        result = queue.Queue(maxsize=1)
        def invoke():
            try:
                result.put((True, self._call(request, allowance)))
            except Exception as exc:
                result.put((False, exc))
        thread = threading.Thread(target=invoke, daemon=True)
        thread.start()
        try:
            ok, value = result.get(timeout=allowance)
        except queue.Empty as exc:
            raise RuntimeFailure("Inference wall-time budget exhausted") from exc
        if not ok:
            raise value
        return value

    def _call(self, request, timeout):
        cfg = self.config
        if not cfg.get("endpoint"):
            raise RuntimeFailure("Inference is disabled")
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise RuntimeFailure("Inference messages must be a nonempty array")
        clean = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant"):
                raise RuntimeFailure("Invalid inference message role")
            content = message.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                        parts.append({"type": "text", "text": part["text"]})
                    elif isinstance(part, dict) and part.get("type") == "image_url" and isinstance(part.get("image_url"), dict):
                        url = part["image_url"].get("url")
                        if not isinstance(url, str) or not url.startswith(("data:image/png;base64,", "data:image/jpeg;base64,")):
                            raise RuntimeFailure("Inference images must be inline PNG/JPEG data URLs")
                        raw_image = decode_submission(url.split(",", 1)[1], int(cfg.get("max_request_bytes", 4 * 1024 * 1024)))
                        if not (raw_image.startswith(b"\x89PNG\r\n\x1a\n") or raw_image.startswith(b"\xff\xd8\xff")):
                            raise RuntimeFailure("Invalid inline image signature")
                        parts.append({"type": "image_url", "image_url": {"url": url}})
                    else:
                        raise RuntimeFailure("Invalid inference content part")
                content = parts
            elif not isinstance(content, str):
                raise RuntimeFailure("Invalid inference message content")
            clean.append({"role": message["role"], "content": content})
        body = json.dumps({"model": cfg["model"], "messages": clean,
                           "max_tokens": int(cfg.get("max_output_tokens", 1024))}).encode()
        if len(body) > int(cfg.get("max_request_bytes", 4 * 1024 * 1024)):
            raise RuntimeFailure("Inference request exceeds byte limit")
        key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"), "")
        if not key:
            raise RuntimeFailure("Host inference credential is missing")
        req = urllib.request.Request(cfg["endpoint"], data=body,
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        opener = urllib.request.build_opener(NoRedirect())
        inference_deadline = time.monotonic() + max(0, min(timeout, float(cfg.get("timeout_seconds", 60))))
        with opener.open(req, timeout=max(0.001, inference_deadline - time.monotonic())) as response:
            limit = int(cfg.get("max_response_bytes", 1048576))
            chunks = []
            length = 0
            while length <= limit:
                if time.monotonic() >= inference_deadline:
                    raise RuntimeFailure("Inference wall-time budget exhausted")
                chunk = response.read1(min(65536, limit + 1 - length))
                if not chunk:
                    break
                chunks.append(chunk)
                length += len(chunk)
            raw = b"".join(chunks)
        if len(raw) > limit:
            raise RuntimeFailure("Inference response exceeds byte limit")
        result = json.loads(raw)
        # Expose only generated content and token accounting, never headers or request metadata.
        usage = result.get("usage", {})
        total = usage.get("total_tokens") if isinstance(usage, dict) else None
        with self.budget_lock:
            if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
                self.total_tokens += total
            elif cfg.get("max_total_tokens") is not None:
                self.total_tokens = int(cfg["max_total_tokens"])
                raise RuntimeFailure("Provider omitted token usage; further inference is disabled")
            cumulative = self.total_tokens
        return {"content": result["choices"][0]["message"]["content"], "usage": usage, "cumulative_total_tokens": cumulative}


def zip_directory(directory, max_bytes=64 * 1024 * 1024):
    root = Path(directory).resolve()
    buf = io.BytesIO()
    total = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise RuntimeFailure("Bundle directory may not contain symlinks")
            if path.is_file():
                total += path.stat().st_size
                if total > max_bytes:
                    raise RuntimeFailure("Bundle exceeds byte limit")
                archive.write(path, path.relative_to(root).as_posix())
    return buf.getvalue()


def extract_bundle(data, directory, max_bytes=128 * 1024 * 1024):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if len(infos) > 4096 or sum(i.file_size for i in infos) > max_bytes:
            raise RuntimeFailure("Bundle exceeds extracted limits")
        for info in infos:
            target = root / info.filename
            if not info.filename or "\\" in info.filename or Path(info.filename).is_absolute() or ".." in Path(info.filename).parts or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeFailure("Unsafe bundle member")
            if not target.resolve().is_relative_to(root):
                raise RuntimeFailure("Unsafe bundle destination")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))


def evaluate_in_container(submission, evaluation, sandbox, output, source_bundle=None):
    output.mkdir(parents=True, exist_ok=False)
    source = source_bundle if source_bundle is not None else zip_directory(evaluation["source_dir"])
    cfg = dict(evaluation)
    cfg.pop("source_dir", None)
    cfg.pop("output_dir", None)
    payload = {"submission_b64": base64.b64encode(submission).decode(),
               "source_b64": base64.b64encode(source).decode(), "config": cfg}
    name = "sim2sim-eval-" + uuid.uuid4().hex
    command = container_command(sandbox.get("evaluator_image", "localhost/sim2sim:latest"), ["python", "-m", "sim2sim.worker"], sandbox, name)
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = BoundedStderr(proc.stderr, output / "evaluator.stderr.log")
    try:
        transport = Transport(proc, 180 * 1024 * 1024)
        deadline = time.monotonic() + float(evaluation.get("timeout_seconds", 180))
        transport.send(payload, deadline)
        result = transport.receive(deadline)
        if "error" in result:
            raise RuntimeFailure("Evaluation failed: " + str(result["error"]))
        artifacts = decode_submission(result["artifacts_b64"], 128 * 1024 * 1024)
        extract_bundle(artifacts, output / "evaluation")
        report = result["report"]
        (output / "evaluation.json").write_text(json.dumps(report, indent=2))
        return report
    finally:
        stop_container(proc, name)
        err.close()


class VideoRecorder:
    """Separate reset episodes, sampled against simulation time at the encoded FPS."""
    def __init__(self, output, sim, fps):
        if not 1 <= fps <= 120:
            raise RuntimeFailure("video_fps must be between 1 and 120")
        import imageio.v2 as imageio
        self.imageio = imageio
        self.output, self.sim, self.fps = output, sim, fps
        self.segments = []
        self.writer = None
        self.start_segment()

    def _frame(self):
        from PIL import Image
        import numpy as np
        observation = self.sim.observe(full=True)
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(observation["rgb"]))))

    def start_segment(self):
        self.segment = {"file": "interaction-%03d.mp4" % len(self.segments),
                        "start_total_ticks": self.sim.total_ticks, "start_time": float(self.sim._data.time)}
        self.writer = self.imageio.get_writer(str(self.output / self.segment["file"]), fps=self.fps)
        self.writer.append_data(self._frame())
        self.frames = 1
        self.last_time = float(self.sim._data.time)
        self.next_time = self.last_time + 1 / self.fps

    def tick(self):
        now = float(self.sim._data.time)
        if self.writer is not None and now + 1e-10 >= self.next_time:
            frame = self._frame()
            while now + 1e-10 >= self.next_time:
                self.writer.append_data(frame)
                self.frames += 1
                self.next_time += 1 / self.fps
            self.last_time = now

    def finish_segment(self):
        if self.writer is None:
            return
        now = float(self.sim._data.time)
        if now > self.last_time + 1e-10:
            self.writer.append_data(self._frame())
            self.frames += 1
        self.writer.close()
        self.writer = None
        self.segment.update(end_total_ticks=self.sim.total_ticks, end_time=now, frames=self.frames, fps=self.fps)
        self.segments.append(self.segment)
        (self.output / "video-index.json").write_text(json.dumps(self.segments, indent=2))

    def close(self):
        self.finish_segment()


def run(config):
    from .simulation import Simulation

    check_rootless()
    sandbox = config["sandbox"]
    limits = config.get("budget", {})
    max_ticks = int(limits.get("physics_ticks", 3000))
    seconds = float(limits.get("active_seconds", 300))
    if max_ticks < 0 or seconds <= 0:
        raise RuntimeFailure("Budgets must be positive")
    output = Path(config.get("output_dir", "runs/latest")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    broker = InferenceBroker(config.get("inference", {}))
    source_root = Path(config.get("evaluation", {}).get("source_dir", Path(config["source"]["scene"]).resolve().parent)).resolve()
    source_bundle = zip_directory(source_root)
    (output / "source-snapshot.zip").write_bytes(source_bundle)
    name = "sim2sim-agent-" + uuid.uuid4().hex
    scene = config["source"]["scene"]
    sim_config = dict(config.get("simulation", {}))
    sim_config.setdefault("default_ticks", 20)
    sim_config.setdefault("rgb", True)
    sim_config.setdefault("depth", False)
    sim_config.setdefault("calibration", True)
    source_manifest = Path(scene).resolve().parent / config["source"].get("manifest", "manifest.toml")
    if source_manifest.is_file():
        with source_manifest.open("rb") as handle:
            private_manifest = tomllib.load(handle)
        for key in ("camera", "graspable_bodies"):
            if key in private_manifest:
                sim_config.setdefault(key, private_manifest[key])
    sim = Simulation(scene, sim_config)
    events = BoundedEvents(output / "events.jsonl", int(limits.get("event_log_bytes", 8 * 1024 * 1024)))
    video = VideoRecorder(output, sim, int(config.get("video_fps", 30))) if config.get("record_video", True) else None
    def recording_callback(simulation):
        if time.monotonic() >= deadline:
            raise RuntimeFailure("Active wall-time budget exhausted")
        if video:
            video.tick()
    sim._callback = recording_callback
    stderr = None
    proc = None
    started = time.monotonic()
    deadline = started + seconds
    submissions = 0
    termination = "failed"
    reason = "initialization"
    manifest = json.loads(json.dumps(config))
    manifest["source"]["scene"] = str(Path(scene).resolve())
    manifest["resolved_output_dir"] = str(output)
    manifest["source_bundle_sha256"] = hashlib.sha256(source_bundle).hexdigest()
    manifest["versions"] = {"python": platform.python_version()}
    for package in ("mujoco", "numpy"):
        try:
            manifest["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            manifest["versions"][package] = "unavailable"
    manifest["image_ids"] = {}
    for label, image in (("agent", sandbox["image"]), ("evaluator", sandbox.get("evaluator_image", "localhost/sim2sim:latest"))):
        try:
            inspected = subprocess.run(["podman", "image", "inspect", image, "--format", "{{.Id}}"], capture_output=True, text=True, timeout=10, check=True)
            manifest["image_ids"][label] = str(inspected.stdout).strip()
        except (OSError, subprocess.SubprocessError):
            manifest["image_ids"][label] = "unavailable"
    if manifest.get("inference", {}).get("endpoint"):
        manifest["inference"]["endpoint"] = urllib.parse.urlunsplit(urllib.parse.urlsplit(manifest["inference"]["endpoint"])._replace(query=""))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    started = time.monotonic()
    deadline = started + seconds
    try:
        proc = subprocess.Popen(container_command(sandbox["image"], sandbox.get("command", ["python", "/opt/agent/scripted_agent.py"]), sandbox, name),
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stderr = BoundedStderr(proc.stderr, output / "agent.stderr.log")
        transport = Transport(proc, int(limits.get("max_message_bytes", 12 * 1024 * 1024)))
        transport.send({"type": "task", "protocol": 1, "task": config.get("task", "Reconstruct the observed scene as a MuJoCo XML project."),
                        "actions": ["observe", "grasp", "move", "release", "reset", "submit", "inference", "finish"],
                        "action_schema": {
                            "observe": {"ticks": "optional nonnegative integer; advance then render"},
                            "grasp": {"pixel": "[u,v] pixels from top-left; pin first visible movable surface point"},
                            "move": {"displacement": "[dx,dy,dz] meters in world coordinates; translate the pinned point, allowing physical rotation", "ticks": "positive integer; default " + str(sim_config["default_ticks"])},
                            "release": {"ticks": "optional nonnegative integer; advance after unpinning"},
                            "reset": "restore initial scene and release; never refunds ticks or time",
                            "submit": {"project_b64": "base64 ZIP with scene.xml and relative assets; scores remain private"},
                            "inference": {"messages": "chat messages with text or inline PNG/JPEG data URLs; fixed host endpoint"},
                            "finish": "end the run"},
                        "observation_contract": {"rgb": "base64 RGB8 PNG when enabled", "depth": "base64 uint16 PNG millimeters when enabled", "calibration": "intrinsics plus world_from_camera; camera looks down -Z", "modalities": {k: sim_config[k] for k in ("rgb", "depth", "calibration")}},
                        "budgets": {"active_seconds": seconds, "physics_ticks": max_ticks, "submissions": int(limits.get("submissions", 3))},
                        "submission": {"format": "base64 ZIP", "entrypoint": "scene.xml", "max_bytes": int(limits.get("submission_bytes", 4 * 1024 * 1024))},
                        "observation": sim.observe()}, deadline)
        while True:
            request = transport.receive(deadline)
            op = request.get("op")
            event = {"active_elapsed": seconds - max(0, deadline - time.monotonic()), "op": op, "total_ticks": sim.total_ticks}
            # Image observations and submitted project bytes stay out of the compact event stream.
            event["request"] = {k: v for k, v in request.items() if k in ("ticks", "pixel", "displacement") and len(json.dumps(v)) < 256}
            response = {}
            try:
                if op in ("observe", "grasp", "move", "release", "reset"):
                    ticks = request.get("ticks", config.get("simulation", {}).get("default_ticks", 20) if op == "move" else 0)
                    if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 0 or ticks > max_ticks - sim.total_ticks:
                        raise RuntimeFailure("Cumulative physics tick budget exceeded")
                    if op in ("reset", "grasp") and ticks:
                        raise RuntimeFailure("This action does not accept ticks")
                    action = dict(request)
                    if op in ("move", "release", "observe"):
                        action["ticks"] = ticks
                    if op == "reset" and video:
                        video.finish_segment()
                    observation = sim.action(action)
                    if op == "reset" and video:
                        video.start_segment()
                    response = {"ok": True, "observation": observation, "remaining_ticks": max_ticks - sim.total_ticks}
                elif op == "inference":
                    response = {"ok": True, "inference": broker.call(request, deadline - time.monotonic())}
                elif op == "submit":
                    if submissions >= int(limits.get("submissions", 3)):
                        raise RuntimeFailure("Submission budget exhausted")
                    submission = decode_submission(request.get("project_b64"), int(limits.get("submission_bytes", 4 * 1024 * 1024)))
                    submissions += 1
                    (output / ("submission-%03d.zip" % submissions)).write_bytes(submission)
                    version_output = output / ("version-%03d" % submissions)
                    report = None
                    paused = False
                    try:
                        evaluation = dict(config["evaluation"])
                        evaluation.setdefault("source_dir", str(Path(scene).resolve().parent))
                        evaluation.setdefault("source_entrypoint", Path(scene).resolve().relative_to(source_root).as_posix())
                        evaluation.setdefault("source_config", dict(sim_config))
                        for key in ("camera", "timestep", "grasp_solref", "grasp_solimp"):
                            if key in sim_config:
                                evaluation.setdefault(key, sim_config[key])
                        subprocess.run(["podman", "pause", name], capture_output=True, timeout=10, check=True)
                        paused = True
                        pause = time.monotonic()
                        try:
                            report = evaluate_in_container(submission, evaluation, sandbox, version_output, source_bundle)
                        finally:
                            deadline += time.monotonic() - pause
                            try:
                                subprocess.run(["podman", "unpause", name], capture_output=True, timeout=10, check=True)
                            except Exception as exc:
                                raise SandboxFailure("Cannot resume paused agent container") from exc
                    except Exception as exc:
                        report = {"status": "failed", "scores": None, "error_type": type(exc).__name__,
                                  "error": str(exc) if isinstance(exc, RuntimeFailure) else "Evaluation worker failed"}
                        version_output.mkdir(parents=True, exist_ok=True)
                        (version_output / "evaluation.json").write_text(json.dumps(report, indent=2))
                        if not paused or isinstance(exc, SandboxFailure):
                            raise SandboxFailure("Cannot safely pause/resume agent for private evaluation") from exc
                    response = {"ok": True, "submission": submissions, "status": report.get("status", "failed")}
                    event["evaluation"] = report
                elif op == "finish":
                    transport.send({"ok": True, "status": "finished"}, deadline)
                    termination = "finished"
                    reason = "agent_finish"
                    event["status"] = "finished"
                    events.write(json.dumps(event) + "\n")
                    break
                else:
                    raise RuntimeFailure("Unknown operation")
            except SandboxFailure:
                raise
            except Exception as exc:
                # Never expose provider exceptions, host paths, credentials, or evaluator internals.
                response = {"ok": False, "error": str(exc) if isinstance(exc, RuntimeFailure) and op != "submit" else "Operation failed"}
                event["error_type"] = type(exc).__name__
            event["ok"] = response.get("ok")
            events.write(json.dumps(event) + "\n")
            events.flush()
            transport.send(response, deadline)
        return {"output_dir": str(output), "submissions": submissions, "total_ticks": sim.total_ticks}
    except BaseException as exc:
        reason = str(exc) if isinstance(exc, RuntimeFailure) else type(exc).__name__
        raise
    finally:
        (output / "termination.json").write_text(json.dumps({"status": termination, "reason": reason,
            "remaining_active_seconds": max(0, deadline - time.monotonic()),
            "total_ticks": sim.total_ticks, "remaining_ticks": max_ticks - sim.total_ticks,
            "submissions": submissions}, indent=2))
        if proc is not None:
            stop_container(proc, name)
        if video:
            video.close()
        sim.close()
        events.close()
        if stderr is not None:
            stderr.close()
