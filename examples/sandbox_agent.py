"""Tool-using reconstruction agent for the no-network OCI sandbox.

Runs inside the container; the model is reached only through the host inference broker, which fixes the
endpoint, model and credential. The system prompt is identical across experimental conditions: only the
controller-provided task text, action contract and observations differ."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import threading
import time

from sim2sim.client import Client

SYSTEM = """You are an autonomous research agent working inside an isolated Linux sandbox. Your job is to reconstruct an unknown simulated scene as a MuJoCo MJCF project, using only the observations and interactions offered by the controller. The task statement, action contract and first observation follow in the first user message.

Environment
- You act only through function tools. `bash` runs shell commands in /workspace. python3 with mujoco, numpy, scipy and matplotlib is installed; there is no network. Files in /workspace and /tmp cannot be executed directly, so run scripts as `python3 file.py` or `bash file.sh`. Use code freely: analyze observations quantitatively, author your project files, and simulate or render your own candidates offscreen before loading them.
- Controller tools act on two independent worlds: "reference" (the scene to reconstruct) and "reconstruction" (your most recently loaded candidate). The same actions are available in both, so you can run identical experiments in each and compare the results directly.
- Any image a controller tool returns is also saved under /workspace/observations/ for analysis with code; only the most recent files are kept. `view_image` shows you any PNG or JPEG file, including renders you produce yourself.
- Only recent turns and images stay in your context; older history is replaced by a summary. Keep a running notes file at /workspace/NOTES.md (hypotheses, experiments, measurements, and what is still unverified) and re-read it whenever you need it.

How your work is evaluated
- Only submitted reconstructions are evaluated, by hidden tests. They probe geometry and appearance from viewpoints other than the one you observe, and dynamic behavior under interactions you have not tried. You never see scores.
- First-pass reconstructions are almost always wrong in respects that were not directly observed or tested. Treat every property of your model as a hypothesis until a direct comparison experiment in both worlds supports it.

How to work
- Every tool result reports the remaining wall time; the session ends automatically at the deadline.
- Work in cycles: form hypotheses, design experiments that discriminate between them, run the same actions in both worlds, compare quantitatively, update your project, and resubmit whenever the reconstruction improves. A submission is a checkpoint, not an ending: keep running new discriminating experiments after every submission.
- `finish` is irreversible. Call it when you believe nothing about your reconstruction differs from the reference world: every property of your model has been checked by a direct comparison experiment in both worlds, and no experiment you can still design would reveal a discrepancy. Work that was not submitted is lost, so keep your best current reconstruction submitted.
- In each turn, write one or two sentences of public status describing what you are testing and why, alongside your tool calls. Use only the provided tools and your own sandbox."""

COMPACT = """Your context is nearly full, so older turns will now be removed. Write a complete, self-contained summary of the work so far for your future self: the current reconstruction and where its files are, every experiment you ran with its measured outcome in both worlds, what each result established, which aspects are still unverified, which submissions you made, and your plan. Reply with text only; do not call tools in this reply. Also update /workspace/NOTES.md in your next turn."""

GUARD = """finish was not executed yet (%s of wall time remain). Before ending, list every aspect of your reconstruction that you have NOT validated by a direct comparison experiment between the reference and reconstruction worlds. If any remain, design and run those experiments and resubmit when the reconstruction improves. If none remain, call finish again in a later turn and it will be executed."""

# Tasks whose submission is a predictor program (task["submission"]["kind"] == "predictor", e.g. the real-time die level) get
# their own fixed texts; every scene-reconstruction condition keeps the texts above unchanged.
SYSTEM_PREDICTOR = """You are an autonomous research agent working inside an isolated Linux sandbox. Your job is to study an object in a simulated world through the observations and interactions offered by the controller, build a model that predicts its behavior, and submit that model as a predictor program. The task statement, action contract and first observation follow in the first user message.

Environment
- You act only through function tools. `bash` runs shell commands in /workspace. python3 with mujoco, numpy, scipy and matplotlib is installed; there is no network. Files in /workspace and /tmp cannot be executed directly, so run scripts as `python3 file.py` or `bash file.sh`. Use code freely: analyze observations quantitatively, simulate your own models offscreen, and write and test your predictor.
- The world runs continuously in real time, also while you think and while your code runs. Controller tools act on it directly; when the task offers a controller channel, programs you start with bash can act on the world and read its camera frames with low latency.
- Any image a controller tool returns is also saved under /workspace/observations/ for analysis with code; only the most recent files are kept. `view_image` shows you any PNG or JPEG file, including renders you produce yourself.
- Only recent turns and images stay in your context; older history is replaced by a summary. Keep a running notes file at /workspace/NOTES.md (hypotheses, experiments, measurements, and what is still unverified) and re-read it whenever you need it.

How your work is evaluated
- Only submitted predictors are evaluated, by hidden tests that run them on new throws as the task describes. You never see scores.
- First versions of a predictor are almost always wrong in respects that were not directly tested. Treat every assumption in your predictor as a hypothesis until experiments in the world support it.

How to work
- Every tool result reports the remaining wall time; the session ends automatically at the deadline.
- Work in cycles: form hypotheses, design experiments that discriminate between them, compare your predictor's answers with what actually happens, update it, and resubmit whenever it improves. A submission is a checkpoint, not an ending: keep testing after every submission.
- `finish` is irreversible. Call it when you believe no experiment you can still design would improve your predictor. Work that was not submitted is lost, so keep your best current predictor submitted.
- In each turn, write one or two sentences of public status describing what you are testing and why, alongside your tool calls. Use only the provided tools and your own sandbox."""

COMPACT_PREDICTOR = """Your context is nearly full, so older turns will now be removed. Write a complete, self-contained summary of the work so far for your future self: the current predictor and where its files are, every experiment you ran with its measured outcome, what each result established, which aspects are still unverified, which submissions you made, and your plan. Reply with text only; do not call tools in this reply. Also update /workspace/NOTES.md in your next turn."""

GUARD_PREDICTOR = """finish was not executed yet (%s of wall time remain). Before ending, list every assumption in your predictor that you have NOT checked against what actually happens in the world. If any remain, design and run those experiments and resubmit when the predictor improves. If none remain, call finish again in a later turn and it will be executed."""
PROMPTS = {"scene": (SYSTEM, COMPACT, GUARD), "predictor": (SYSTEM_PREDICTOR, COMPACT_PREDICTOR, GUARD_PREDICTOR)}

WORLD = {"type": "string", "enum": ["reference", "reconstruction"]}
STRING = {"type": "string"}
INTEGERS = lambda n: {"type": "array", "items": {"type": "integer"}, "minItems": n, "maxItems": n}
NUMBERS = lambda n: {"type": "array", "items": {"type": "number"}, "minItems": n, "maxItems": n}
ACTIONS = {
    "observe": ({"world": WORLD, "ticks": {"type": "integer", "minimum": 0}}, ["world"]),
    "grasp": ({"world": WORLD, "pixel": INTEGERS(2)}, ["world", "pixel"]),
    "move": ({"world": WORLD, "displacement": NUMBERS(3), "ticks": {"type": "integer", "minimum": 1}}, ["world", "displacement"]),
    "release": ({"world": WORLD, "ticks": {"type": "integer", "minimum": 0}}, ["world"]),
    "arm_target": ({"world": WORLD, "joint_targets": NUMBERS(7), "gripper_width_m": {"type": "number"},
                    "ticks": {"type": "integer", "minimum": 1}}, ["world", "joint_targets", "gripper_width_m"]),
    "hand_target": ({"world": WORLD, "actuator_targets": NUMBERS(20), "ticks": {"type": "integer", "minimum": 1}}, ["world", "actuator_targets"]),
    "camera": ({"world": WORLD, "pos": NUMBERS(3), "lookat": NUMBERS(3)}, ["world", "pos", "lookat"]),
    "reset": ({"world": WORLD}, ["world"]),
    "load_reconstruction": ({"project_dir": STRING, "entrypoint": STRING, "camera": STRING}, []),
    "submit": ({"project_dir": STRING}, []),
    "sam3d": ({"image": STRING, "mask": STRING, "seed": {"type": "integer", "minimum": 0}}, ["image", "mask"]),
    "sequence": ({"world": WORLD, "steps_file": STRING, "images": {"type": "string", "enum": ["last", "none"]}}, ["world", "steps_file"]),
    "frames": ({"since": {"type": "integer"}, "limit": {"type": "integer", "minimum": 1, "maximum": 100}}, []),
    "throw": ({"protocol": STRING}, ["protocol"]),
    "finish": ({}, []),
}
SEQUENCE_OPS = ("observe", "grasp", "move", "release", "arm_target", "hand_target", "camera", "reset", "throw")  # physical actions a sequence may contain
SEQUENCE_STEPS = 2000
LOCAL = {
    "bash": ("Run a bash command in /workspace and return its exit status and combined stdout/stderr (long output is truncated in the middle). "
             "Each command runs in its own process group, which is terminated when the command returns or times out (default 300 s).",
             {"command": STRING, "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 1800}}, ["command"]),
    "view_image": ("Show a PNG or JPEG file from the sandbox, for example your own offscreen render.", {"path": STRING}, ["path"]),
}
IMAGE_KEYS = ("rgb", "depth", "wrist_rgb", "wrist_depth")
FORBIDDEN_PATHS = ("/opt/sim2sim/environments", "/opt/sim2sim/experiments", "/opt/sim2sim/runs", "/opt/sim2sim/.git", "/opt/sim2sim/configs")
CREDENTIALS = ("INFERENCE_API_KEY", "OPENAI_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AZURE_OPENAI_API_KEY")


def text(value):
    return {"type": "text", "text": value}


def clock(seconds):
    seconds = int(max(0, seconds))
    return "%d:%02d:%02d" % (seconds // 3600, seconds // 60 % 60, seconds % 60)


def project_digest(root):
    files = sorted(path for path in Path(root).rglob("*") if path.is_file()) if Path(root).is_dir() else []
    return hashlib.sha256(b"".join(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() for path in files)).hexdigest() if files else None


def check_boundary():
    if any(Path(path).exists() for path in FORBIDDEN_PATHS):
        raise RuntimeError("Agent image contains a forbidden researcher directory")
    if any(os.environ.get(name) for name in CREDENTIALS):
        raise RuntimeError("A host credential was exposed to the agent")


def tool_schemas(task):
    tools = [{"name": name, "description": description, "parameters": {"type": "object", "properties": properties,
              "required": required, "additionalProperties": False}} for name, (description, properties, required) in LOCAL.items()]
    for op in task["actions"]:
        if op in ("inference", "progress"):
            continue
        properties, required = ACTIONS[op]
        if "realtime" in task:  # a real-time task has only the reference world
            properties, required = {k: v for k, v in properties.items() if k != "world"}, [k for k in required if k != "world"]
        contract = task["action_schema"][op]
        notes = {"load_reconstruction": " Packages project_dir (default /workspace/project) as a ZIP and loads it into the reconstruction world.",
                 "submit": " Packages project_dir (default /workspace/project) as a ZIP and submits it for hidden evaluation.",
                 "finish": " Irreversibly ends the session.",
                 "sam3d": " image and mask are PNG file paths in the sandbox; the mesh is saved as a PLY file under /workspace/sam3d/.",
                 "sequence": " Write steps_file with your own code, then analyze the saved per-step outputs with code.",
                 "frames": " Images are saved as files (paths in the reply), not shown; use view_image or your code."}.get(op, "")
        tools.append({"name": op, "description": "Controller action. Contract: %s.%s" % (json.dumps(contract), notes),
                      "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}})
    return tools


class Agent:
    def __init__(self, client, workspace="/workspace", compact_tokens=100000, keep_turns=8, max_images=12,
                 reserve_seconds=360, output_chars=16000, keep_image_files=400, now=time.time):
        self.client, self.task, self.now = client, client.task, now
        assert self.task["inference"]["enabled"], "The host inference broker is not configured"
        assert self.task.get("deadline_unix"), "The sandbox agent requires a hard wall-time deadline"
        budgets = self.task["budgets"]
        self.budget = float(budgets["wall_seconds"])
        self.active_deadline = now() + float(budgets["active_seconds"]) if budgets.get("active_seconds") else float("inf")
        self.deadline = min(float(self.task["deadline_unix"]), self.active_deadline)
        self.workspace = Path(workspace)
        self.observations = self.workspace / "observations"
        self.observations.mkdir(parents=True, exist_ok=True)
        self.compact_tokens, self.keep_turns, self.max_images = compact_tokens, keep_turns, max_images
        self.reserve, self.output_chars, self.keep_image_files = reserve_seconds, output_chars, keep_image_files
        self.tools = tool_schemas(self.task)
        self.system, self.compact_text, self.guard = PROMPTS[self.task["submission"].get("kind", "scene")]
        self.schemas = {tool["name"]: tool["parameters"] for tool in self.tools}
        self.turns, self.summary, self.saved, self.images = [], None, [], 0
        self.turn, self.submissions, self.intercepted, self.channel_requests = 0, 0, None, 0
        self.project, self.submitted = self.workspace / "project", None
        public = {key: value for key, value in self.task.items() if key not in ("type", "protocol", "observation")}
        self.task_message = {"role": "user", "content": [text("Controller task and contract:\n" + json.dumps(public, indent=1))]
                             + self.render({"observation": self.task["observation"]}, "reference")}

    def left(self):
        return self.deadline - self.now()

    def stamp(self):
        return "[remaining wall time %s; the session ends automatically when %s remain]" % (clock(self.left()), clock(self.reserve))

    def note(self, message, origin="adapter"):
        self.client.progress(message[:1000], origin=origin)

    def messages(self):
        head = [{"role": "system", "content": self.system}, self.task_message]
        if self.summary is not None:
            head.append({"role": "user", "content": "Summary of earlier work (older turns were removed from context):\n" + self.summary})
        return head + [message for turn in self.turns for message in turn]

    def infer(self, messages):
        for attempt in range(4):
            try:
                reply = self.client.inference(messages, tools=self.tools)
            except RuntimeError as exc:
                wait = 15 * 2 ** attempt
                if attempt == 3 or self.left() - wait <= self.reserve:
                    raise
                self.note("Model call failed (%s); retrying in %d s." % (str(exc)[:300], wait))
                time.sleep(wait)
                continue
            # The host re-anchors the wall deadline on every reply (host-side holds do not count as run time).
            self.deadline = min(float(reply.pop("deadline_unix")), self.active_deadline)
            return reply

    def run(self):
        self.note("Sandbox agent started with %s of wall time." % clock(self.left()))
        while self.left() > self.reserve:
            reply = self.infer(self.messages())
            if reply.get("budget_stop"):
                return self.budget_stop(reply["reason"])
            self.turn += 1
            content = reply["content"].strip()
            if content:
                self.note(content, "model")
            turn = [{"role": "assistant", "content": reply["content"], "tool_calls": reply["tool_calls"]}] if content or reply["tool_calls"] else []
            if not reply["tool_calls"]:
                turn.append({"role": "user", "content": "No tool was called. Continue the work using the tools. " + self.stamp()})
            for call in reply["tool_calls"]:
                parts, done = self.execute(call)
                if done:
                    return
                turn.append({"role": "tool", "tool_call_id": call["id"], "content": parts + [text(self.stamp())]})
            self.turns.append(turn)
            self.prune_images()
            usage = reply.get("usage")
            if usage and usage["promptTokens"] + usage["cacheReadTokens"] + usage["cacheWriteTokens"] > self.compact_tokens:
                self.compact()
        else:
            self.note("Ending the session automatically with %s remaining." % clock(self.left()))
        self.client.finish()

    def compact(self):
        reply = self.infer(self.messages() + [{"role": "user", "content": self.compact_text + " " + self.stamp()}])
        if reply.get("budget_stop"):
            return
        self.summary = reply["content"].strip() or "(No summary was produced; consult /workspace/NOTES.md.)"
        self.turns = self.turns[-self.keep_turns:]
        self.note("Context compacted; kept the latest %d turns and a model-written summary." % len(self.turns))

    def budget_stop(self, reason):
        self.note("Inference budget exhausted (%s); submitting any unsubmitted project changes, then finishing." % reason)
        digest = project_digest(self.project)
        if digest is not None and digest != self.submitted:
            try:
                self.client.submit(self.project)
            except (RuntimeError, ValueError, OSError) as exc:
                self.note("Final submission was rejected: %s" % str(exc)[:300])
        self.client.finish()

    def prune_images(self):
        images = [(turn, message, index) for turn in self.turns for message in turn if isinstance(message["content"], list)
                  for index, part in enumerate(message["content"]) if part["type"] == "image_url"]
        if len(images) > self.max_images:
            for _, message, index in images[:len(images) - self.max_images // 2]:
                message["content"][index] = text("[older image removed from context; use view_image on its saved file if needed]")

    def execute(self, call):
        name, schema = call["name"], self.schemas[call["name"]]
        try:
            args = json.loads(call["arguments"] or "{}")
        except ValueError:
            return [text("ERROR: arguments must be a JSON object")], False
        if not isinstance(args, dict) or set(args) - set(schema["properties"]) or set(schema["required"]) - set(args):
            return [text("ERROR: arguments must match the tool schema %s" % json.dumps(schema))], False
        try:
            if name == "bash":
                return [text(self.bash(args["command"], int(args.get("timeout_seconds", 300))))], False
            if name == "view_image":
                return self.view_image(args["path"]), False
            if name == "finish":
                return self.finish()
            if name == "sam3d":
                return [text(self.sam3d(args))], False
            if name == "sequence":
                return self.sequence(args), False
            if name in ("load_reconstruction", "submit"):
                project = Path(args.get("project_dir", self.workspace / "project"))
                self.project = project
                if name == "submit":
                    digest = project_digest(project)
                    response = self.client.submit(project)
                    self.submissions, self.submitted = self.submissions + 1, digest
                else:
                    response = self.client.load_reconstruction(project, entrypoint=args.get("entrypoint", "scene.xml"), camera=args.get("camera", "fixed"))
            else:
                response = self.client.request(name, **args)
            return self.render(response, args.get("world", "reconstruction" if name == "load_reconstruction" else "reference")), False
        except (RuntimeError, ValueError, OSError) as exc:
            return [text("ERROR: %s" % exc)], False

    def finish(self):
        left = self.left()
        if self.submissions == 0:
            return [text("finish was not executed: no reconstruction has been submitted, and only submitted reconstructions are evaluated.")], False
        if left > max(900, .1 * self.budget) and self.intercepted in (None, self.turn):
            self.intercepted = self.turn
            self.note("Intercepted an early finish request with %s remaining." % clock(left))
            return [text(self.guard % clock(left))], False
        self.note("Model ended the session with %s remaining." % clock(left))
        self.client.finish()
        return [], True

    def save_images(self, observation, world):
        """Move every image of observation into a file; return [(key, path)]."""
        saved = []
        for key in IMAGE_KEYS:
            if isinstance(observation.get(key), str):
                self.images += 1
                path = self.observations / ("%05d-%s-%s.png" % (self.images, world, key))
                path.write_bytes(base64.b64decode(observation.pop(key), validate=True))
                self.saved.append(path)
                saved.append((key, path))
        for path in self.saved[:-self.keep_image_files]:
            path.unlink()
        self.saved = self.saved[-self.keep_image_files:]
        return saved

    def render(self, response, world):
        response = dict(response)
        observation = dict(response.pop("observation", None) or {})
        saved = self.save_images(observation, world)
        parts = [part for key, path in saved if key.endswith("rgb") for part in
                 (text("%s image (%s):" % (key, path)), {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()}})]
        if "frames" in response:  # real-time frame batches: files only, never inline images
            response["frames"] = [dict(frame, saved_images=[str(path) for _, path in self.save_images(frame, world)]) for frame in map(dict, response["frames"])]
        if observation:
            response["observation"] = observation
        if saved:
            response["saved_images"] = [str(path) for _, path in saved]
        return [text(json.dumps(response, separators=(",", ":")))] + parts

    def relay(self, server, stop):
        """Controller channel (task code_channel): serve the bash command's programs one connection at a time, forwarding each
        JSON request line that names an offered physical action to the controller and writing back its reply line."""
        allowed = set(self.task["code_channel"]["ops"])
        while not stop.is_set():
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            with connection, connection.makefile("rwb") as stream:
                for line in stream:
                    try:
                        request = json.loads(line)
                        assert isinstance(request, dict) and request.get("op") in allowed, "op must be one of %s" % sorted(allowed)
                        reply = self.client.request(request.pop("op"), **request)
                    except (AssertionError, ValueError, RuntimeError, TypeError) as exc:
                        reply = {"ok": False, "error": str(exc)}
                    stream.write(json.dumps(reply, separators=(",", ":")).encode() + b"\n")
                    stream.flush()
                    self.channel_requests += 1

    def bash(self, command, timeout):
        timeout = max(1, min(timeout, 1800, int(self.left() - self.reserve)))
        stop, relay = threading.Event(), None
        if "code_channel" in self.task:
            path = Path(self.task["code_channel"]["path"])
            path.unlink(missing_ok=True)
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(path))
            server.listen(1)
            server.settimeout(.2)
            relay = threading.Thread(target=self.relay, args=(server, stop), daemon=True)
            relay.start()
        try:
            return self.run_bash(command, timeout)
        finally:
            if relay is not None:
                stop.set()
                relay.join()
                server.close()
                path.unlink()

    def run_bash(self, command, timeout):
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(["bash", "-c", command], cwd=self.workspace, stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       env={**os.environ, "HOME": str(self.workspace), "MPLBACKEND": "Agg"})
            try:
                status = "exit code %d" % process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                status = "killed after the %d s timeout" % timeout
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            size = output.seek(0, os.SEEK_END)
            half = self.output_chars // 2
            if size <= self.output_chars:
                output.seek(0)
                body = output.read().decode(errors="replace")
            else:
                output.seek(0)
                head = output.read(half).decode(errors="replace")
                output.seek(size - half)
                body = head + "\n[... %d bytes omitted ...]\n" % (size - 2 * half) + output.read().decode(errors="replace")
        return "%s\n%s" % (status, body)

    def sam3d(self, args):
        result = self.client.sam3d(Path(args["image"]).read_bytes(), Path(args["mask"]).read_bytes(), args.get("seed", 0))
        folder = self.workspace / "sam3d"
        folder.mkdir(exist_ok=True)
        path = folder / ("%03d-mesh.ply" % (len(list(folder.glob("*-mesh.ply"))) + 1))
        path.write_bytes(base64.b64decode(result.pop("mesh_ply_b64"), validate=True))
        return json.dumps({"mesh_ply": str(path), **result}, separators=(",", ":"))

    def sequence(self, args):
        """Run an agent-written JSON list of physical actions back to back in one world as ordinary controller requests."""
        steps = json.loads(Path(args["steps_file"]).read_text())
        allowed = [op for op in SEQUENCE_OPS if op in self.task["actions"]]
        if not (isinstance(steps, list) and 1 <= len(steps) <= SEQUENCE_STEPS
                and all(isinstance(step, dict) and step.get("op") in allowed and "world" not in step for step in steps)):
            raise ValueError("steps_file must hold a JSON array of 1 to %d action objects with op in %s and no world field" % (SEQUENCE_STEPS, allowed))
        folder = self.workspace / "sequences"
        folder.mkdir(exist_ok=True)
        path = folder / ("%04d.json" % (len(list(folder.glob("*.json"))) + 1))
        records, response, error = [], None, None
        for index, step in enumerate(steps):
            try:
                response = self.client.request(step["op"], world=args["world"], **{k: v for k, v in step.items() if k != "op"})
            except RuntimeError as exc:
                error = "step %d (%s) was rejected: %s" % (index, step["op"], exc)
                break
            observation = {k: v for k, v in response["observation"].items() if k not in IMAGE_KEYS and not k.endswith(("_format", "calibration"))}
            records.append({"step": index, "op": step["op"], "observation": observation})
        path.write_text(json.dumps(records, separators=(",", ":")))
        summary = text("Ran %d of %d steps in the %s world; each step's non-image observation fields are saved in %s.%s"
                       % (len(records), len(steps), args["world"], path, " Stopped: " + error if error else ""))
        if response is None or args.get("images", "last") == "none":
            return [summary]
        return [summary] + self.render(response, args["world"])

    def view_image(self, path):
        data = Path(path).read_bytes()
        kind = "png" if data.startswith(b"\x89PNG\r\n\x1a\n") else "jpeg" if data.startswith(b"\xff\xd8\xff") else None
        if kind is None or len(data) > 2 * 1024 * 1024:
            raise ValueError("view_image supports PNG or JPEG files up to 2 MiB")
        return [text("Image %s:" % path), {"type": "image_url", "image_url": {"url": "data:image/%s;base64,%s" % (kind, base64.b64encode(data).decode())}}]


def main(argv=None, client=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compact-tokens", type=int, default=100000, help="compact history when prompt tokens exceed this")
    parser.add_argument("--keep-turns", type=int, default=8, help="recent turns kept verbatim after compaction")
    parser.add_argument("--max-images", type=int, default=12, help="images kept in context before older ones are dropped")
    parser.add_argument("--reserve-seconds", type=int, default=360, help="end the session when this much wall time remains")
    parser.add_argument("--output-chars", type=int, default=16000, help="bash output returned to the model")
    parser.add_argument("--keep-image-files", type=int, default=400, help="saved observation images retained in /workspace")
    parser.add_argument("--workspace", default="/workspace")
    args = parser.parse_args(argv)
    assert args.compact_tokens > 0 and args.keep_turns > 0 and args.max_images > 1 and args.reserve_seconds > 0 and args.keep_image_files > 0
    check_boundary()
    Agent(client or Client(), workspace=args.workspace, compact_tokens=args.compact_tokens, keep_turns=args.keep_turns,
          max_images=args.max_images, reserve_seconds=args.reserve_seconds, output_chars=args.output_chars,
          keep_image_files=args.keep_image_files).run()


if __name__ == "__main__":
    main()
