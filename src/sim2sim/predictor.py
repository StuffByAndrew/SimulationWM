"""Isolated runner for agent-submitted predictor programs (strict OCI only).

A predictor metric (runtime.PREDICTOR_METRICS) scores a program instead of a scene. For each evaluated submission the host
starts one container, `python -m sim2sim.predictor MODE`: no network, read-only root, no mounts, never the source scene,
manifest, controller or credentials. The host sends the agent's ZIP first, as {"op": "load", "project_b64", "entrypoint",
"call_seconds"}; the worker unpacks it with the submission checks and replies {"ok": true} or {"ok": false, "error"}.
Logs go to the metric's predictor directory (RUN/predictor-NNN/): predictor.stderr.log and the mode's own log.

call mode (`calls`; evaluator image, [sandbox.worker] bounds): the entrypoint (e.g. predictor.py) is imported under the
alarm, then the metric's `collect(call)` calls its functions one at a time:
  {"op": "call", "function": NAME, "kwargs": {...}}  ->  {"ok": true, "value": V} | {"ok": false, "error": ...}
Each load or call has call_seconds of wall time (an alarm agent code cannot catch with `except Exception`) and the whole
session the metric's timeout; a value must be JSON (finite numbers) of at most MAX_VALUE_BYTES. Anything the program
prints goes to the stderr log. A failed call is that call's error; a timed-out, crashed or oversized worker fails every
later call. The call log is predictor-calls.json.

stream mode (`stream`; agent image and agent bounds, because answer latency is scored and the program gets exactly the
resources it was developed with): after the load reply the worker execs `python3 ENTRYPOINT` in the unpacked directory,
whose stdin/stdout are JSON lines in real time. The host sends the metric's public start message and waits for
{"type":"ready"}; per trial it sends the trial header at release, frame k at k / frame_rate seconds after release (only
frames before the deadline), then {"type":"trial_end"} at the deadline, and {"type":"end"} after the last trial. The first
{"type":"prediction","trial":i,...} line for trial i counts: "on_time" if received by the deadline and accepted by the
metric's `validate`, "malformed" if not accepted (or unparsable), "late" if received after the deadline within the gap
before the next trial, "missing" if none. Every output line is logged with its receipt time (predictor-output.jsonl).
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
from typing import Callable
import uuid

ROOT = Path("/workspace/predictor")
MAX_LINE = 16 * 1024 * 1024     # one protocol line (call mode) or program output line (stream mode)
MAX_VALUE_BYTES = 1024 * 1024
MAX_LOG = 16 << 20              # researcher log of the program's output lines per evaluation (stream mode)
CALL_SECONDS = 60
CONTRACT = {"runner": "isolated container: your ZIP only, no network, no controller, no source scene; python3 with numpy, scipy and mujoco",
            "calls": "each function is called with keyword arguments only and must return JSON-compatible finite numbers, "
                     f"lists or objects of at most {MAX_VALUE_BYTES} bytes, within {CALL_SECONDS} s per call; printed output is discarded"}


class CallTimeout(BaseException):
    """Raised by the worker alarm; a BaseException so `except Exception` in agent code cannot swallow it."""


def _alarm(signum, frame):
    raise CallTimeout("call exceeded its time limit")


def _reply(stream, message):
    line = json.dumps(message, separators=(",", ":"), allow_nan=False).encode()
    if len(line) > MAX_VALUE_BYTES + 64:
        raise ValueError(f"returned value exceeds {MAX_VALUE_BYTES} bytes")
    stream.write(line + b"\n")
    stream.flush()


def main(mode, root=ROOT):
    """Worker entrypoint; runs only inside the predictor container."""
    from .submission import decode_submission, extract_submission
    assert mode in ("call", "stream"), mode
    protocol = os.fdopen(os.dup(1), "wb")
    if mode == "call":
        os.dup2(2, 1)  # the program's prints (fd 1 and sys.stdout) go to the stderr log
        sys.stdout = sys.stderr
    signal.signal(signal.SIGALRM, _alarm)
    module, seconds = None, CALL_SECONDS
    while True:
        raw = sys.stdin.buffer.readline(MAX_LINE + 1)
        if not raw:
            return
        try:
            message = json.loads(raw)
            if message["op"] == "load":
                seconds = int(message.get("call_seconds", CALL_SECONDS))
            signal.alarm(seconds)
            try:
                if message["op"] == "load":
                    assert module is None, "one load per worker"
                    entry = extract_submission(decode_submission(message["project_b64"]), root, message["entrypoint"])
                    if mode == "stream":
                        signal.alarm(0)
                        _reply(protocol, {"ok": True})
                        os.chdir(root)
                        os.execv(sys.executable, [sys.executable, str(entry.relative_to(root))])
                    sys.path.insert(0, str(root))
                    spec = importlib.util.spec_from_file_location("predictor", entry)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    reply = {"ok": True}
                else:
                    assert message["op"] == "call" and module is not None, "load the predictor before calling it"
                    function = getattr(module, message["function"], None)
                    if not callable(function):
                        raise ValueError(f"the predictor defines no function {message['function']}")
                    reply = {"ok": True, "value": function(**message["kwargs"])}
            finally:
                signal.alarm(0)
            _reply(protocol, reply)
        except (Exception, CallTimeout) as exc:
            _reply(protocol, {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]})


@dataclass
class Job:
    """One predictor submission as the runtime hands it to its metric's evaluate_submission(job) (runtime.PREDICTOR_METRICS)."""
    data: bytes                 # the submitted ZIP
    evaluation: dict            # the run's [evaluation] table with the resolved source settings
    output: Path                # RUN/
    version_output: Path        # RUN/version-NNN/ (created by the evaluator; must not exist yet)
    logs: Path                  # RUN/predictor-NNN/: container stderr and the mode's log
    scene: Path                 # trusted source scene
    sim_config: dict
    manifest: dict              # private source manifest
    wall_elapsed: float
    sandbox: dict
    supervisor: object
    evaluate_scene: Callable    # evaluate_scene(evaluation): the private evaluator container's report

    def container(self, mode):
        return container(mode, self.sandbox, self.logs, self.supervisor)


@contextmanager
def container(mode, sandbox, logs, supervisor):
    """The isolated predictor container of one evaluated submission (see the module docstring for each mode's bounds)."""
    from .runtime import launch_container, stop_container, worker_limits
    logs.mkdir(parents=True, exist_ok=True)
    image, limits = (sandbox["image"], sandbox) if mode == "stream" else (sandbox["evaluator_image"], worker_limits(sandbox))
    name = "sim2sim-predictor-" + uuid.uuid4().hex
    proc = launch_container(image, ["python", "-m", "sim2sim.predictor", mode], limits, name, logs / "predictor.stderr.log", supervisor)
    try:
        yield proc
    finally:
        stop_container(proc, name)
        supervisor.unregister(proc)
        proc.stderr_log.close()


def _load(submission, entrypoint, call_seconds=CALL_SECONDS):
    return {"op": "load", "project_b64": base64.b64encode(submission).decode(), "entrypoint": entrypoint, "call_seconds": call_seconds}


def calls(proc, submission, entrypoint, collect, deadline, log_path, call_seconds=CALL_SECONDS):
    """Call mode, host side: load the program, let `collect(call)` make the metric's calls before the monotonic deadline,
    write the call log (arguments, values, errors, seconds) to log_path and return collect's result."""
    from .runtime import Transport, RuntimeFailure
    transport, log, broken = Transport(proc, MAX_LINE), [], None

    def exchange(message):
        nonlocal broken
        if broken is None:
            try:
                transport.send(message, deadline)
                return transport.receive(deadline)
            except RuntimeFailure as exc:  # timeout, crash or oversized reply: the worker is no longer trusted
                broken = f"predictor worker failed: {exc}"
        return {"ok": False, "error": broken}

    def call(function, **kwargs):
        started = time.monotonic()
        reply = exchange({"op": "call", "function": function, "kwargs": kwargs})
        log.append({"function": function, "kwargs": kwargs, **reply, "seconds": round(time.monotonic() - started, 3)})
        return reply

    try:
        loaded = exchange(_load(submission, entrypoint, call_seconds))
        log.append({"function": "<load>", **loaded})
        return collect(call) if loaded["ok"] else collect(lambda function, **kwargs: {"ok": False, "error": loaded["error"]})
    finally:
        Path(log_path).write_text(json.dumps(log, indent=1))


def stream(proc, submission, entrypoint, start, trials, frame_rate, deadline_s, validate, log_path, gap_s=.25, setup_s=120.):
    """Stream mode, host side: run the program over trials ((header, frames) pairs) in real time; return (ready, results),
    one {"trial", "status", "latency_s", "message"} per trial (message: the counted prediction line, or None)."""
    lines, outbox = queue.Queue(), queue.Queue()
    log, logged = Path(log_path).open("w"), [0]

    def reader():
        for raw in iter(lambda: proc.stdout.readline(MAX_LINE + 1), b""):
            lines.put((time.monotonic(), raw))
        lines.put((time.monotonic(), None))

    def writer():
        try:
            for payload in iter(outbox.get, None):
                proc.stdin.write(payload)
                proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass

    def send(message):
        outbox.put(json.dumps(message, separators=(",", ":"), allow_nan=False).encode() + b"\n")

    def record(stamp, raw, origin):
        entry = json.dumps({"t": round(stamp - origin, 6), "line": raw.decode(errors="replace")[:65536]}) + "\n"
        if logged[0] + len(entry) <= MAX_LOG:
            log.write(entry)
            logged[0] += len(entry)

    def until(predicate, deadline, origin):
        """Consume output lines until predicate(message) or deadline; True if found."""
        while True:
            try:
                stamp, raw = lines.get(timeout=max(0, deadline - time.monotonic()))
            except queue.Empty:
                return False
            if raw is None:
                lines.put((stamp, None))
                return False
            record(stamp, raw, origin)
            try:
                if predicate(json.loads(raw)):
                    return True
            except ValueError:
                pass

    for target in (reader, writer):
        threading.Thread(target=target, daemon=True).start()
    began = time.monotonic()
    results = []
    try:
        send(_load(submission, entrypoint))
        ready = until(lambda m: m == {"ok": True}, began + setup_s, began)
        if ready:
            send(start)
            ready = until(lambda m: isinstance(m, dict) and m.get("type") == "ready", time.monotonic() + setup_s, began)
        for header, frames in trials:
            result = {"trial": header["trial"], "status": "missing", "latency_s": None, "message": None}
            results.append(result)
            if not ready:
                continue
            release = time.monotonic()
            send(header)
            for k, frame in enumerate(frames):
                delay = release + k / frame_rate - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                send({"type": "frame", "trial": header["trial"], "index": k, **frame})
            time.sleep(max(0, release + deadline_s - time.monotonic()))
            send({"type": "trial_end", "trial": header["trial"]})
            time.sleep(gap_s)
            while not lines.empty():
                stamp, raw = lines.get()
                if raw is None:
                    lines.put((stamp, None))
                    break
                record(stamp, raw, began)
                if result["status"] != "missing" or stamp < release:
                    continue
                try:
                    message = json.loads(raw)
                except ValueError:
                    message = None
                if isinstance(message, dict) and message.get("type") == "prediction" and message.get("trial") != header["trial"]:
                    continue  # an answer for another trial: logged, never counted
                prediction = isinstance(message, dict) and message.get("type") == "prediction"
                result.update(latency_s=stamp - release, message=message if prediction else None,
                              status="late" if stamp > release + deadline_s else "on_time" if prediction and validate(message) else "malformed")
        send({"type": "end"})
        outbox.put(None)
        time.sleep(.5)
        return ready, results
    finally:
        outbox.put(None)
        log.close()


if __name__ == "__main__":
    main(sys.argv[1], *map(Path, sys.argv[2:]))  # optional root: local tests and checks only
