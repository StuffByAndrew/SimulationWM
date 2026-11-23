"""Isolated runner for agent-submitted predictor programs.

A predictor submission is the agent's ZIP project whose entrypoint (e.g. ``predictor.py``) defines plain Python functions.
The host starts one worker container per evaluated submission (``python -m sim2sim.predictor`` in the evaluator image,
``[sandbox.worker]`` limits, no network, read-only root, no mounts, never the source scene or manifest), sends the ZIP,
then calls named functions with JSON keyword arguments one at a time and receives JSON values. The metric that needs the
predictions decides which calls to make (``collect_predictions(table, call)`` in its module) and scores the answers in
the private evaluator. Nothing a predictor returns or prints reaches the agent.

Wire (one JSON object per line on the worker's stdin/stdout):
  {"op": "load", "project_b64": ZIP, "entrypoint": "predictor.py", "call_seconds": S}  ->  {"ok": true} | {"ok": false, "error": ...}
  {"op": "call", "function": NAME, "kwargs": {...}}                 ->  {"ok": true, "value": V} | {"ok": false, "error": ...}
Each load or call is limited to call_seconds (default CALL_SECONDS) of wall time inside the worker and the whole session to the
host's predictor timeout; a returned value must be JSON (finite numbers) of at most MAX_VALUE_BYTES serialized. Anything
the program prints goes to the worker's stderr log, never into the protocol. A failed call is recorded as that call's
error and the session continues; a timed-out or crashed worker fails every later call.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid

MAX_LINE = 16 * 1024 * 1024
MAX_VALUE_BYTES = 1024 * 1024
CALL_SECONDS = 60
CONTRACT = {"runner": "isolated container: your ZIP only, no network, no controller, no source scene; python3 with numpy, scipy and mujoco",
            "calls": "each function is called with keyword arguments only and must return JSON-compatible finite numbers, "
                     f"lists or objects of at most {MAX_VALUE_BYTES} bytes, within {CALL_SECONDS} s per call; printed output is discarded"}


class CallTimeout(BaseException):
    """Raised by the worker alarm; a BaseException so `except Exception` in agent code cannot swallow it."""


def _alarm(signum, frame):
    raise CallTimeout("call exceeded its time limit")


def main(*, root=Path("/workspace/predictor")):
    """Worker entrypoint; runs only inside the predictor container."""
    from .evaluation import extract_submission
    from .runtime import decode_submission
    protocol = os.fdopen(os.dup(1), "wb")
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
                seconds = int(message["call_seconds"])
            signal.alarm(seconds)
            try:
                if message["op"] == "load":
                    assert module is None, "one load per worker"
                    entry = extract_submission(decode_submission(message["project_b64"], 8 * 1024 * 1024), root, message["entrypoint"])
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
            line = json.dumps(reply, separators=(",", ":"), allow_nan=False).encode()
            if len(line) > MAX_VALUE_BYTES + 64:
                raise ValueError(f"returned value exceeds {MAX_VALUE_BYTES} bytes")
        except (Exception, CallTimeout) as exc:
            line = json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}).encode()
        protocol.write(line + b"\n")
        protocol.flush()


def run_predictor(submission, entrypoint, sandbox, output, supervisor, collect, timeout, call_seconds=CALL_SECONDS):
    """Host side: one isolated worker for this submission; `collect(call)` makes the metric's calls. Returns its result and
    writes the call log (calls, values, errors, seconds) to output/predictor-calls.json."""
    from .runtime import Transport, launch_container, stop_container, worker_limits, RuntimeFailure
    output.mkdir(parents=True, exist_ok=True)
    name = "sim2sim-predictor-" + uuid.uuid4().hex
    proc = launch_container(sandbox["evaluator_image"], ["python", "-m", "sim2sim.predictor"], worker_limits(sandbox), name,
                            output / "predictor.stderr.log", supervisor)
    deadline = min(time.monotonic() + timeout, supervisor.deadline)
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
        loaded = exchange({"op": "load", "project_b64": base64.b64encode(submission).decode(), "entrypoint": entrypoint, "call_seconds": call_seconds})
        log.append({"function": "<load>", **loaded})
        result = collect(call) if loaded["ok"] else collect(lambda function, **kwargs: {"ok": False, "error": loaded["error"]})
    finally:
        stop_container(proc, name)
        supervisor.unregister(proc)
        proc.stderr_log.close()
    (output / "predictor-calls.json").write_text(json.dumps(log, indent=1))
    return result


if __name__ == "__main__":
    main()
