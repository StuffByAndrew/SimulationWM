"""Explicit approved development bridge. No OS filesystem/network isolation claim."""
from __future__ import annotations
import base64
from contextlib import contextmanager
import copy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .runtime import (BoundedStderr, CandidateSession, RuntimeFailure, Transport, WallClock,
                      _run, decode_submission, extract_bundle)

MODE = "development_shared_filesystem"


class GroupProcess(subprocess.Popen):
    def kill(self):
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def native_process(role, workspace, config, supervisor):
    supervisor.check()
    workspace.mkdir(parents=True, exist_ok=False)
    limits = dict(config.get("development_worker", {}))
    limits["cpu_seconds"] = max(1, math.ceil(supervisor.remaining()))
    command = [sys.executable, "-m", "sim2sim.dev_worker", role, "--workspace", str(workspace),
               "--limits", json.dumps(limits, separators=(",", ":"))]
    # No inherited API/cloud credentials; this remains a shared filesystem process.
    environment = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(workspace),
                   "PYTHONPATH": str(Path(__file__).resolve().parents[1]), "PYTHONDONTWRITEBYTECODE": "1",
                   "MUJOCO_GL": config.get("development_worker", {}).get("render_backend", "egl"),
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "LP_NUM_THREADS": "1",
                   "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", ""), "SLURM_JOB_PARTITION": os.environ.get("SLURM_JOB_PARTITION", "")}
    return GroupProcess(command, cwd=workspace, env=environment, start_new_session=True,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def stop_native(proc):
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


class NativeCandidate(CandidateSession):
    def __init__(self, config, output, audit, supervisor):
        self.name = None  # WallClock uses killpg via GroupProcess, never Podman.
        self.audit, self.token, self.action_id, self.last = audit, None, None, None
        self.closed = False
        self.supervisor = supervisor
        workspace = Path(config["output_dir"])/"workers"/("candidate-"+uuid.uuid4().hex)
        self.proc = native_process("candidate", workspace, config, supervisor)
        self.stderr = BoundedStderr(self.proc.stderr, output)
        self.transport = Transport(self.proc, 48*1024*1024,
            on_receive=lambda raw, complete: self._wire("response", raw, complete),
            on_send=lambda raw, complete: self._wire("request", raw, complete))

    def close(self):
        if self.closed:
            return
        self.closed = True
        stop_native(self.proc)
        self.supervisor.unregister(self.proc)
        self.stderr.close()
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            stream.close()


class DevelopmentBackend:
    def __init__(self, config, public_dir, supervisor):
        self.config, self.public_dir, self.supervisor = config, Path(public_dir), supervisor
        self.transport = None

    def preflight(self):
        if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") not in ("test", "sapphire"):
            raise RuntimeFailure("Development physics workers require an active Slurm test or sapphire allocation")
        if self.config.get("inference", {}).get("endpoint"):
            raise RuntimeFailure("Development bridge does not use a host inference API")
        wall = self.config.get("budget", {}).get("wall_seconds")
        if not isinstance(wall, (int, float)) or isinstance(wall, bool) or not math.isfinite(wall) or not 0 < wall <= 21600:
            raise RuntimeFailure("Development bridge requires a finite wall_seconds cap of at most 21600")

    def agent_transport(self, maximum, on_receive, on_send):
        from .dev_transport import MailboxTransport
        self.transport = MailboxTransport(self.public_dir, maximum, on_receive=on_receive, on_send=on_send,
                                          stop_event=self.supervisor.expired)
        self.supervisor.add_cleanup(lambda: self.transport.close({"status": "stopped", "reason": "controller_stopped"}))
        return self.transport

    def candidate(self, output, audit):
        return NativeCandidate(self.config, output, audit, self.supervisor)

    @contextmanager
    def paused(self, names):
        # The external model is not OS-pausable. Request serialization and the
        # independent hard wall cap remain authoritative in this development mode.
        yield

    def evaluate(self, submission, evaluation, sandbox, output, source_bundle, supervisor):
        supervisor.check()
        output.mkdir(parents=True, exist_ok=False)
        workspace = Path(self.config["output_dir"])/"workers"/("evaluation-"+uuid.uuid4().hex)
        proc = native_process("evaluation", workspace, self.config, supervisor)
        err = BoundedStderr(proc.stderr, output/"evaluator.stderr.log")
        try:
            supervisor.register(proc)
            cfg = dict(evaluation)
            cfg.pop("source_dir", None)
            cfg.pop("output_dir", None)
            payload = {"submission_b64": base64.b64encode(submission).decode(),
                       "source_b64": base64.b64encode(source_bundle).decode(), "config": cfg}
            transport = Transport(proc, 180*1024*1024)
            deadline = min(supervisor.deadline, time.monotonic()+float(evaluation.get("timeout_seconds", 180)))
            transport.send(payload, deadline)
            result = transport.receive(deadline)
            if "error" in result:
                raise RuntimeFailure("Development evaluation worker failed")
            extract_bundle(decode_submission(result["artifacts_b64"], 128*1024*1024), output/"evaluation")
            report = result["report"]
            (output/"evaluation.json").write_text(json.dumps(report, indent=2))
            return report
        finally:
            stop_native(proc)
            supervisor.unregister(proc)
            err.close()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                stream.close()

    def close_agent(self):
        if self.transport:
            self.transport.close({"status": "stopped", "reason": "controller_stopped"})


def run_development(config, public_dir, *, prepared_output=False, model="gpt-6-astra"):
    if model != "gpt-6-astra":
        raise RuntimeFailure("Development pilot requires exact gpt-6-astra; no substitution")
    config = copy.deepcopy(config)
    config.update(execution_mode=MODE, agent_model=model)
    supervisor = WallClock(config.get("budget", {}).get("wall_seconds"))
    backend = DevelopmentBackend(config, public_dir, supervisor)
    try:
        return _run(config, supervisor, prepared_output=prepared_output, backend=backend)
    except BaseException as exc:
        if supervisor.expired.is_set() or supervisor.remaining() <= 0:
            raise RuntimeFailure("Hard wall-time budget exhausted") from exc
        raise
    finally:
        backend.close_agent()
        supervisor.close()
