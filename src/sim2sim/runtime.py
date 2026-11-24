"""Rootless Podman boundary and host-owned agent protocol."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from contextlib import contextmanager, nullcontext
import io
import hashlib
import importlib
import importlib.metadata
import platform
import multiprocessing
import math
import json
import os
import re
from pathlib import Path
import secrets
import selectors
import stat
import subprocess
import tarfile
import time
import threading
import tomllib
import urllib.error
import urllib.request
import urllib.parse
import uuid
import zipfile

from .robots import GRIPPER, HAND_ACTUATORS, ROBOTS, joints


class RuntimeFailure(RuntimeError):
    pass


class SandboxFailure(RuntimeFailure):
    pass


class WallClock:
    """A hard child-work deadline, independent of the active-time accounting."""
    def __init__(self, seconds=None):
        self.started = time.monotonic()
        if seconds is not None and (not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0):
            raise RuntimeFailure("wall_seconds must be positive")
        self.deadline = self.started + float(seconds) if seconds is not None else float("inf")
        self.expired = threading.Event()
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.processes = {}
        self.callbacks = []
        self.finished_at = None
        self.held, self.hold_started = 0., None
        if seconds is not None:
            threading.Thread(target=self._watch, daemon=True).start()

    def remaining(self):
        now = self.finished_at if self.finished_at is not None else self.hold_started or time.monotonic()
        return max(0., self.deadline - now)

    def held_seconds(self):
        return self.held + (time.monotonic() - self.hold_started if self.hold_started is not None else 0.)

    @contextmanager
    def hold(self):
        """An orchestrator hold is outside the run: its duration extends the deadline and is excluded from elapsed time."""
        assert self.hold_started is None, "holds do not nest"
        self.hold_started = time.monotonic()
        try:
            yield
        finally:
            duration = time.monotonic() - self.hold_started
            self.deadline += duration
            self.held += duration
            self.hold_started = None

    def credit(self, seconds):
        """Give back time already spent that belongs outside the run (a provider-failed inference attempt), as if held."""
        assert seconds >= 0 and self.hold_started is None
        self.deadline += seconds
        self.held += seconds

    def check(self):
        if self.expired.is_set() or self.remaining() <= 0:
            raise RuntimeFailure("Hard wall-time budget exhausted")

    def register(self, proc, name=None):
        with self.lock:
            expired = self.closed.is_set() or self.expired.is_set() or self.remaining() <= 0
            if not expired:
                self.processes[id(proc)] = (proc, name)
        if expired:
            self._stop(proc, name)
            raise RuntimeFailure("Hard wall-time budget exhausted")

    def unregister(self, proc):
        with self.lock:
            self.processes.pop(id(proc), None)

    def add_cleanup(self, callback):
        with self.lock:
            self.callbacks.append(callback)

    @staticmethod
    def _stop(proc, name):
        if name:
            def kill_container():
                try:
                    subprocess.run(["podman", "kill", "--signal=KILL", name], capture_output=True, timeout=10)
                except (OSError, subprocess.SubprocessError):
                    pass
            threading.Thread(target=kill_container, daemon=True).start()
        try:
            if proc.poll() is None:
                proc.kill()
        except (OSError, ProcessLookupError):
            pass

    def _stop_all(self):
        with self.lock:
            processes, callbacks = list(self.processes.values()), list(self.callbacks)
        for proc, name in processes:
            self._stop(proc, name)
        for callback in callbacks:
            try:
                callback()
            except Exception:
                pass

    def _watch(self):
        while not self.closed.wait(min(max(self.remaining(), .01), 60.)):
            if self.hold_started is None and self.remaining() <= 0:
                self.expired.set()
                self._stop_all()
                return

    def close(self):
        if self.finished_at is None:
            self.finished_at = time.monotonic()
        self.closed.set()
        self._stop_all()


def _inference_process(config, request, timeout, connection):
    """Trusted host-only network process; no agent code or hidden state is passed."""
    try:
        # This worker only performs the HTTP exchange; budget and cost-pool accounting stay in the parent broker.
        connection.send((True, InferenceBroker({k: v for k, v in config.items() if k not in POOL_SETTINGS})._call(request, timeout), False))
    except Exception as exc:
        message = str(exc) if isinstance(exc, RuntimeFailure) else "Inference provider request failed (%s)" % type(exc).__name__
        # Transport errors (timeouts, resets, truncated bodies) and ProviderUnavailable are outages; other failures are not.
        connection.send((False, message, isinstance(exc, ProviderUnavailable) or not isinstance(exc, RuntimeFailure)))
    finally:
        connection.close()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeFailure("Inference redirects are forbidden")


BACKENDS = ("cgroup", "rlimit", "slurm_step")
CONTAINER_ENV = {"HOME": "/workspace", "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "LP_NUM_THREADS": "1"}
PUBLIC_ROBOT = "/opt/public/robot"
PUBLIC_PRIOR = "/opt/public/prior"
PUBLIC_TOOLS = "/opt/public/tools/"
TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
TOOL_IMAGE_LABEL = "sim2sim.tool-runtime"
# Optional ablatable agent tools ([tools] name = true): fixed neutral task text and pinned file sha256s. A disabled
# tool is absent from the container; an enabled one is mounted read-only at PUBLIC_TOOLS + name. A service tool (pins
# None) mounts nothing: it is a controller action the host relays to a researcher-run service (see sam3d.py).
AGENT_TOOLS = {
    "segmentation": ("Optional tool: a promptable image segmentation model (MobileSAM, CPU) is installed read-only at "
                     "/opt/public/tools/segmentation. Run `python3 /opt/public/tools/segmentation/segment.py --help` for usage. "
                     "Given an image and point or box prompts, or automatic mode, it writes masks as PNG and NPZ files and "
                     "prints a JSON summary.",
                     {"vendor/mobile_sam.pt": "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"}),
    "planning": ("Optional tool: Panda arm planning helpers are installed read-only at /opt/public/tools/planning. Run "
                 "`python3 /opt/public/tools/planning/plan.py --help` for usage. On your own scene.xml they provide "
                 "damped-least-squares inverse kinematics for hand or fingertip targets, forward kinematics with contact and "
                 "clearance reports, joint-space interpolation, a collision-checked straight-line or RRT-connect planner, "
                 "straight-line Cartesian moves for approach, retreat and lifting, and antipodal parallel-jaw grasp "
                 "candidates for a body with pregrasp and approach motions. Each returns joint target sequences for arm_target.",
                 {}),
    "sam3d": ("Optional tool: the controller action sam3d runs SAM 3D Objects, a single-image 3D object reconstruction model. "
              "Give it one RGB PNG image and a grayscale PNG mask of one object in it (same size, nonzero = object). It "
              "returns a triangle mesh with vertex colors of the whole object, including sides the image does not show, "
              "placed in that image's camera frame, with the predicted pose and scale. Shape, scale and depth are learned "
              "estimates from one image and can be wrong. Each call takes about a minute or longer.",
              None),
    "sequence": ("Optional tool: the sequence action runs a list of physical actions that your code writes to a JSON file, back to back "
                 "in one world, and saves every step's non-image observation fields to a file for analysis.",
                 None),
    # Shadow hand only. Runtime adds public_tool_config.json: the exact public settings the reconstruction world receives.
    "local_controller": ("Optional tool: a local copy of the controller is installed read-only at /opt/public/tools/local_controller. Run "
                         "`python3 /opt/public/tools/local_controller/localsim.py --help` for usage. It runs your own project in your "
                         "sandbox with the same simulation code and settings as the reconstruction world, so the same actions from a "
                         "fresh load give the same observations, and it gives the position of every tactile sensor cell for any hand "
                         "joint positions.",
                         {}),
    "channel": ("Optional tool: while a bash command runs, programs it starts can send controller actions and read camera frames "
                "with low latency through the controller channel described under code_channel.",
                None),
}
CHANNEL_SOCKET = "/tmp/sim2sim-controller.sock"  # served by the sandbox harness only while a bash command runs


def podman_env():
    """The explicit environment of every srun/podman launch; host credentials never reach a container step."""
    return {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "XDG_RUNTIME_DIR", "CONTAINERS_STORAGE_CONF", "DBUS_SESSION_BUS_ADDRESS") if k in os.environ}


def memory_bytes(limits):
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([kmgt]?)(?:i?b)?", str(limits.get("memory", "1g")).lower())
    if not match:
        raise RuntimeFailure("Unsupported memory size for resource verification")
    return int(float(match.group(1)) * 1024 ** ("kmgt".find(match.group(2)) + 1 if match.group(2) else 0))


def worker_limits(sandbox):
    """Evaluator and candidate-worker bounds: optional [sandbox.worker] keys override the agent bounds."""
    return {**sandbox, **sandbox.get("worker", {})}


def check_rootless(process_backend="cgroup", supervisor=None):
    if process_backend not in BACKENDS:
        raise RuntimeFailure("process_limit_backend must be cgroup, rlimit or slurm_step")
    try:
        p = subprocess.run(["podman", "info", "--format=json"], env=podman_env(),
                           capture_output=True, text=True, timeout=min(20, supervisor.remaining()) if supervisor else 20, check=True)
        info = json.loads(p.stdout)
        host = info["host"]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeFailure("A working rootless Podman installation is required") from exc
    if host.get("security", {}).get("rootless") is not True:
        raise RuntimeFailure("Refusing non-rootless Podman")
    # slurm_step enforcement comes from Slurm's step cgroups and is verified per container from the host.
    required = {"cgroup": {"cpu", "memory", "pids"}, "rlimit": {"cpu", "memory"}, "slurm_step": set()}[process_backend]
    if host.get("cgroupVersion") != "v2" or not required.issubset(set(host.get("cgroupControllers") or [])):
        raise RuntimeFailure("Rootless cgroup v2 with delegated " + ", ".join(sorted(required)) + " controllers is required")
    if process_backend == "slurm_step" and not (os.environ.get("SLURM_JOB_ID") and os.environ.get("XDG_RUNTIME_DIR")):
        raise SandboxFailure("slurm_step requires a Slurm allocation and the job-private Podman environment")


def container_command(image, command, limits, name, mounts=()):
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise RuntimeFailure("container command must be a nonempty string array")
    backend = limits.get("process_limit_backend", "cgroup")
    if backend not in BACKENDS:
        raise RuntimeFailure("process_limit_backend must be cgroup, rlimit or slurm_step")
    pids = int(limits.get("pids", 64))
    if pids < 1:
        raise RuntimeFailure("pids limit must be positive")
    if backend == "slurm_step":
        cpus = limits.get("cpus", 1)
        if not isinstance(cpus, int) or isinstance(cpus, bool) or cpus < 1 or memory_bytes(limits) % 2 ** 20:
            raise RuntimeFailure("slurm_step needs integer cpus and whole-MiB memory")
        # One exact step per container: Slurm's step cgroup holds memory.max (swap 0) and a dedicated cpuset.
        # Slurm drops XDG_RUNTIME_DIR, so env -i restores the job-private podman environment and nothing else.
        prefix = ["srun", "--jobid=" + os.environ["SLURM_JOB_ID"], "--exact", "--nodes=1", "--ntasks=1", "--cpus-per-task=" + str(cpus),
                  "--mem=" + str(memory_bytes(limits) // 2 ** 20) + "M", "--quiet", "--job-name=" + name,
                  "/usr/bin/env", "-i", *(k + "=" + v for k, v in podman_env().items())]
        cgroups = ["--cgroups=disabled"]
    else:
        prefix = []
        cgroups = ["--cgroups=enabled", "--pids-limit=" + str(pids if backend == "cgroup" else -1),
                   "--memory=" + str(limits.get("memory", "1g")), "--memory-swap=" + str(limits.get("memory", "1g")), "--cpus=" + str(float(limits.get("cpus", 1)))]
    return [*prefix, "podman", "run", "--rm", "--pull=never", "--name", name,
            "--network=none", "--read-only", "--log-driver=none", "--http-proxy=false", "--image-volume=ignore", "--entrypoint=", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--user=65532:65532", "--cgroupns=private", "--pid=private", "--ipc=private", "--uts=private", *cgroups,
            "--ulimit=nofile=256:256",
            "--ulimit=core=0:0", "--tmpfs=/workspace:rw,noexec,nosuid,nodev,size=" + str(limits.get("workspace", "256m")) + ",mode=1777",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777", "--workdir=/workspace",
            *("--env=" + k + "=" + v for k, v in CONTAINER_ENV.items()),
            *("--mount=type=volume,source=" + volume + ",destination=" + target + ",ro=true" for volume, target in mounts), "-i", image,
            # RLIMIT_NPROC is set by trusted prlimit as the sandbox UID: crun would apply --ulimit before its own
            # clone, which counts every process of the invoking host user and fails spuriously on shared nodes.
            "/usr/bin/prlimit", "--nproc=%d:%d" % (pids, pids), "--", *command]


def launch_container(image, command, limits, name, stderr_log, supervisor=None, mounts=()):
    """Start a container, draining stderr to a bounded log; a slurm_step container is verified from the host before any input."""
    proc = subprocess.Popen(container_command(image, command, limits, name, mounts), env=podman_env(),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.stderr_log = BoundedStderr(proc.stderr, stderr_log)
    try:
        if supervisor:
            supervisor.register(proc, name)
        deadline = min(time.monotonic() + 120, supervisor.deadline if supervisor else float("inf"))
        proc.verified = verify_host_limits(name, limits, deadline, proc) if limits.get("process_limit_backend", "cgroup") == "slurm_step" else None
    except BaseException:
        stop_container(proc, name)
        if supervisor:
            supervisor.unregister(proc)
        proc.stderr_log.close()
        raise
    return proc


def stop_container(proc, name):
    if proc.poll() is None:
        proc.kill()
    try:
        subprocess.run(["podman", "rm", "--force", "--time=0", name], capture_output=True, timeout=15, env=podman_env())
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def container_pid(name, deadline, proc=None):
    while True:
        if proc is not None and proc.poll() is not None:
            raise SandboxFailure("Container step exited with code %d before host-side verification" % proc.returncode)
        inspected = subprocess.run(["podman", "inspect", "--format", "{{.State.Pid}}", name], capture_output=True, text=True, timeout=10, env=podman_env())
        pid = int(inspected.stdout.strip() or 0) if inspected.returncode == 0 else 0
        if pid > 0:
            return pid
        if time.monotonic() > deadline:
            raise SandboxFailure("Container is not running for host-side verification")
        time.sleep(.1)


def container_step(pid):
    """Slurm step and leaf cgroup of a container host PID: a numbered step of this job, not the controller's own."""
    job = os.environ["SLURM_JOB_ID"]
    relative = Path("/proc/%d/cgroup" % pid).read_text().strip().split("::", 1)[1]
    match = re.fullmatch(r"(.*/job_" + job + r"/step_(\d+))/user/task_\d+", relative)
    if not match or match.group(1) + "/" in Path("/proc/self/cgroup").read_text():
        raise SandboxFailure("Container is not confined to its own Slurm step: " + relative)
    return job + "." + match.group(2), Path("/sys/fs/cgroup" + relative)


# Container name -> (Slurm step, leaf cgroup), recorded once at launch verification. Pause/resume must not query podman:
# a container frozen mid-start can hold its podman lock, and inspect then blocks (wave-1 job 57188794 resume failure).
SLURM_STEPS = {}


def verify_host_limits(name, limits, deadline, proc=None):
    """Trusted host-side proof; a container cgroup namespace cannot see the enforcing ancestor step cgroup."""
    pid = container_pid(name, deadline, proc)
    step, leaf = container_step(pid)
    SLURM_STEPS[name] = step, leaf
    levels = [leaf, *leaf.parents]
    levels = levels[:[p.name for p in levels].index("job_" + os.environ["SLURM_JOB_ID"]) + 1]
    def finite(key):
        values = [int(v) for v in ((p / key).read_text().strip() for p in levels) if v != "max"]
        return min(values) if values else None
    cpus = sum(int(b or a) - int(a) + 1 for a, _, b in (part.partition("-") for part in (leaf / "cpuset.cpus.effective").read_text().strip().split(",")))
    status = dict(line.split(":", 1) for line in Path("/proc/%d/status" % pid).read_text().splitlines())
    uid_map = [tuple(map(int, line.split())) for line in Path("/proc/%d/uid_map" % pid).read_text().splitlines()]
    host_uid = [outside + 65532 - inside for inside, outside, count in uid_map if inside <= 65532 < inside + count]
    pids = int(limits.get("pids", 64))
    while True:  # the container init applies its own process bound with prlimit before exec
        nproc = [line.split()[2:4] for line in Path("/proc/%d/limits" % pid).read_text().splitlines() if line.startswith("Max processes")][0]
        if nproc[1] != "unlimited" and int(nproc[1]) <= pids or time.monotonic() > deadline:
            break
        time.sleep(.05)
    pause = int((Path(os.environ["XDG_RUNTIME_DIR"]) / "libpod/tmp/pause.pid").read_text())
    observed = {"step": step, "cgroup": str(leaf), "memory_max": finite("memory.max"), "memory_swap_max": finite("memory.swap.max"),
                "cpuset_cpus": cpus, "uids": sorted(set(status["Uid"].split())), "expected_host_uid": host_uid, "nproc": nproc,
                **{key: status[key].strip() for key in ("CapEff", "CapPrm", "CapBnd", "NoNewPrivs")},
                "podman_userns": os.readlink("/proc/%d/ns/user" % pid) == os.readlink("/proc/%d/ns/user" % pause)}
    valid = observed["memory_max"] is not None and 0 < observed["memory_max"] <= memory_bytes(limits) and observed["memory_swap_max"] == 0
    valid = valid and 1 <= cpus <= limits.get("cpus", 1) and len(host_uid) == 1 and observed["uids"] == [str(host_uid[0])] and host_uid[0] != os.getuid()
    valid = valid and nproc[0] == nproc[1] != "unlimited" and 0 < int(nproc[1]) <= pids and observed["NoNewPrivs"] == "1" and observed["podman_userns"]
    if not valid or any(observed[key] != "0000000000000000" for key in ("CapEff", "CapPrm", "CapBnd")):
        raise SandboxFailure("Host-side slurm_step verification failed: " + json.dumps(observed))
    return observed


def _set_paused(name, paused, backend):
    if backend != "slurm_step":
        subprocess.run(["podman", "pause" if paused else "unpause", name], capture_output=True, timeout=10, check=True, env=podman_env())
        return
    # Podman cannot pause without its own cgroup; Slurm applies step STOP/CONT with the cgroup v2 freezer.
    step, leaf = SLURM_STEPS[name]
    # scancel is a slurmctld RPC that can stall under controller load (wave-2 job 57226610: one 10 s CONT timeout ended a
    # run). Signalling is idempotent, so resend until the step freezer itself reports the requested state, else fail.
    deadline = time.monotonic() + 180
    while True:
        try:
            subprocess.run(["scancel", "--signal=" + ("STOP" if paused else "CONT"), step], capture_output=True, timeout=30, check=True, env=podman_env())
        except subprocess.TimeoutExpired:
            pass
        settle = time.monotonic() + 15
        while time.monotonic() < settle:
            if "frozen " + str(int(paused)) in (leaf / "cgroup.events").read_text().splitlines():
                return
            time.sleep(.05)
        if time.monotonic() > deadline:
            raise SandboxFailure("Slurm step %s freezer did not reach frozen=%d" % (step, paused))


@contextmanager
def paused_containers(names, backend="cgroup"):
    attempted = []
    try:
        for name in names:
            attempted.append(name)
            _set_paused(name, True, backend)
    except Exception as exc:
        for name in reversed(attempted):
            try:
                _set_paused(name, False, backend)
            except Exception:
                pass
        raise SandboxFailure("Cannot safely pause all containers for private evaluation: %r" % exc) from exc
    try:
        yield
    finally:
        failure = None
        for name in reversed(attempted):
            try:
                _set_paused(name, False, backend)
            except Exception as exc:
                failure = exc
        if failure is not None:
            raise SandboxFailure("Cannot safely resume paused containers: %r" % failure) from failure


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_effective_limits(observed, limits):
    maximum_memory = memory_bytes(limits)
    backend = limits.get("process_limit_backend", "cgroup")
    try:
        nproc_soft, nproc_hard = observed["nproc"]
        pids = int(limits.get("pids", 64))
        valid = 0 < nproc_soft <= nproc_hard <= pids
        valid = valid and observed["uid"] == 65532 and observed["cap_eff"] == "0000000000000000" and observed["no_new_privs"] == "1"
        if backend != "slurm_step":  # slurm_step memory/CPU bounds sit on ancestor step cgroups, verified host-side
            memory = int(observed["memory.max"])
            quota, period = observed["cpu.max"].split()
            valid = valid and 0 < memory <= maximum_memory and 0 < int(quota) / int(period) <= float(limits.get("cpus", 1)) + 1e-9
        if backend == "cgroup":
            valid = valid and 0 < int(observed["pids.max"]) <= pids
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        valid = False
    if not valid:
        raise RuntimeFailure("OCI resource verification failed: CPU/memory/process limits must be finite and enforced; observed memory.max=" + str(observed.get("memory.max")) + ", cpu.max=" + str(observed.get("cpu.max")))


def public_robot_files(robot_dir, robot="panda"):
    """Hashed public robot fixture: allowlisted files, IMPORT.md cut before its private scene notes."""
    robot_dir = Path(robot_dir)
    if not robot_dir.is_dir() or robot_dir.is_symlink():
        raise RuntimeFailure("Public robot fixture unavailable")
    files = {}
    for item in sorted(robot_dir.rglob("*")):
        relative = item.relative_to(robot_dir)
        if item.is_symlink() or not (item.is_file() or item.is_dir()) or (relative.as_posix() not in (ROBOTS[robot][0], "LICENSE", "IMPORT.md") and relative.parts[0] != "assets"):
            raise RuntimeFailure("Unexpected public robot fixture entry")
        if item.is_file():
            files[relative.as_posix()] = item.read_bytes()
    files["IMPORT.md"] = files["IMPORT.md"].decode().split("\n\nThe sibling scenes ", 1)[0].rstrip("\n").encode() + b"\n"
    if re.search(r"sibling|block|cube|proprio|lever|spring|boat|assembly|tetrahedral|probe|prism|spik|smooth", files["IMPORT.md"].decode(), re.I):
        raise RuntimeFailure("Public robot IMPORT.md still describes private scenes")
    return files


def public_prior_files(prior_dir):
    """Prepared continuation carry-over (see experiments.stage_continuation): regular files exactly as staged."""
    root = Path(prior_dir)
    if not root.is_dir() or root.is_symlink():
        raise RuntimeFailure("Public prior directory unavailable")
    files = {}
    for item in sorted(root.rglob("*")):
        if item.is_symlink() or not (item.is_file() or item.is_dir()):
            raise RuntimeFailure("Unexpected public prior entry")
        if item.is_file():
            files[item.relative_to(root).as_posix()] = item.read_bytes()
    return files


def enabled_tools(config):
    """Names of the optional agent tools a [tools] table enables, in AGENT_TOOLS order (all disabled by default)."""
    table = config.get("tools", {})
    if not isinstance(table, dict) or set(table) - set(AGENT_TOOLS) or any(not isinstance(value, bool) for value in table.values()):
        raise RuntimeFailure("[tools] accepts only boolean " + ", ".join(AGENT_TOOLS))
    if table.get("planning") and config.get("simulation", {}).get("robot") != "panda":
        raise RuntimeFailure('tools.planning requires simulation.robot = "panda"')
    robot = config.get("simulation", {}).get("robot")
    if table.get("local_controller") and (robot not in ROBOTS or "hand_target" not in ROBOTS[robot][2]):
        raise RuntimeFailure('tools.local_controller requires a hand robot (simulation.robot = "shadow_hand" or "panda_shadow")')
    return tuple(name for name in AGENT_TOOLS if table.get(name))


def public_tool_files(name):
    """Regular files of tools/<name> (bytecode caches excluded), with pinned third-party files verified."""
    root = TOOLS_DIR / name
    if not root.is_dir() or root.is_symlink():
        raise RuntimeFailure("Agent tool directory unavailable: " + name)
    files = {}
    for item in sorted(root.rglob("*")):
        relative = item.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if item.is_symlink() or not (item.is_file() or item.is_dir()):
            raise RuntimeFailure("Unexpected agent tool entry: " + name + "/" + relative.as_posix())
        if item.is_file():
            files[relative.as_posix()] = item.read_bytes()
    for relative, digest in AGENT_TOOLS[name][1].items():
        if hashlib.sha256(files.get(relative, b"")).hexdigest() != digest:
            raise RuntimeFailure("Pinned agent tool file missing or changed: %s/%s (scripts/fetch-%s.sh)" % (name, relative, name))
    return files


def stage_public_volume(files, supervisor):
    """Trusted read-only staging: a fresh, randomly named Podman volume imported from an in-memory tar.
    Entries are owned by container root and the mount is read-only; owner-write modes only let the
    sandbox user's plain `cp -r` copies stay editable."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for directory in sorted({str(parent) for relative in files for parent in Path(relative).parents if str(parent) != "."}):
            info = tarfile.TarInfo(directory)
            info.type, info.mode = tarfile.DIRTYPE, 0o755
            archive.addfile(info)
        for relative, data in sorted(files.items()):
            info = tarfile.TarInfo(relative)
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
    name = "sim2sim-public-" + uuid.uuid4().hex
    for arguments, payload in ((["create", name], None), (["import", name, "-"], buffer.getvalue())):
        subprocess.run(["podman", "volume", *arguments], input=payload, capture_output=True, timeout=min(60, supervisor.remaining()), check=True, env=podman_env())
    return name


def image_id(image, supervisor):
    try:
        return subprocess.run(["podman", "image", "inspect", image, "--format", "{{.Id}}"], capture_output=True, text=True,
                              timeout=min(10, supervisor.remaining()), check=True, env=podman_env()).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise SandboxFailure("Trusted image is unavailable: " + image) from exc


def check_limits(sandbox, supervisor=None):
    """Probe the researcher-trusted evaluator image with the agent bounds before any agent can execute."""
    code = """import json,os,resource
from pathlib import Path
root=Path('/sys/fs/cgroup')
relative=Path('/proc/self/cgroup').read_text().strip().split('::',1)[1].lstrip('/')
root=root/relative
result={name:(root/name).read_text().strip() if (root/name).exists() else None for name in ['memory.max','cpu.max','pids.max']}
status=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines())
result.update(nproc=resource.getrlimit(resource.RLIMIT_NPROC),uid=os.getuid(),cap_eff=status['CapEff'].strip(),no_new_privs=status['NoNewPrivs'].strip(),env=sorted(os.environ))
print(json.dumps(result),flush=True)
input()
"""
    name = "sim2sim-probe-" + uuid.uuid4().hex
    image = sandbox.get("evaluator_image", "localhost/sim2sim:latest")
    image_env = json.loads(subprocess.run(["podman", "image", "inspect", "--format", "{{json .Config.Env}}", image], capture_output=True,
                                          text=True, timeout=20, check=True, env=podman_env()).stdout)
    allowed_env = {item.split("=", 1)[0] for item in image_env} | set(CONTAINER_ENV) | {"container", "HOSTNAME", "TERM"}
    proc = launch_container(image, ["python", "-c", code], sandbox, name, Path(os.devnull), supervisor)
    try:
        result = Transport(proc, 4096).receive(min(time.monotonic() + 30, supervisor.deadline) if supervisor else time.monotonic() + 30)
        validate_effective_limits(result, sandbox)
        if not set(result["env"]) <= allowed_env:
            raise SandboxFailure("Container environment has unexpected variables: " + ", ".join(sorted(set(result["env"]) - allowed_env)))
        return {"container": result, "host": proc.verified}
    finally:
        stop_container(proc, name)
        if supervisor:
            supervisor.unregister(proc)
        proc.stderr_log.close()
        proc.stdin.close()
        proc.stdout.close()


class TimedAudit:
    """Attach run-local clocks to every host and worker audit event."""
    def __init__(self, audit, clock):
        self.audit, self.clock = audit, clock
    def record(self, event, reservation=None):
        return self.audit.record({**self.clock(), **event}, reservation=reservation)
    def __getattr__(self, name):
        return getattr(self.audit, name)


def validate_prepared_output(output):
    """Allow only wrapper metadata/source plus kernel-managed NFS silly-renames."""
    allowed = {"experiment.json", "status.json", "config-private.json", "source", "prior"}
    for path in Path(output).iterdir():
        nfs_temporary = re.fullmatch(r"\.nfs[0-9a-f]+", path.name) is not None
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            if nfs_temporary:
                continue
            raise RuntimeFailure("Prepared metadata disappeared during validation: " + repr(path.name))
        if nfs_temporary and stat.S_ISREG(mode):
            # NFS retains an unlinked/replaced open metadata file under this name.
            # It belongs to the kernel/client; never unlink or move it ourselves.
            continue
        if path.name not in allowed or stat.S_ISLNK(mode):
            raise RuntimeFailure("Prepared output directory contains runtime data or unexpected files: " + repr(path.name))
        if path.name in ("source", "prior"):  # private source copy; continuation carry-over (hash-checked at start)
            if not stat.S_ISDIR(mode):
                raise RuntimeFailure("Prepared %s must be a regular directory" % path.name)
            if any(item.is_symlink() or not (item.is_file() or item.is_dir()) for item in path.rglob("*")):
                raise RuntimeFailure("Prepared %s must contain regular files and directories only" % path.name)
        elif not stat.S_ISREG(mode):
            raise RuntimeFailure("Prepared metadata must be regular files")


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
    def __init__(self, proc, max_bytes, on_receive=None, on_send=None):
        self.proc = proc
        self.on_receive = on_receive
        self.on_send = on_send
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
        raw = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        if self.on_send:
            self.on_send(raw, True)
        data = memoryview(raw)
        while data:
            self._wait(self.proc.stdin, selectors.EVENT_WRITE, deadline)
            try:
                n = os.write(self.proc.stdin.fileno(), data)
                data = data[n:]
            except BlockingIOError:
                continue
            except BrokenPipeError as exc:
                raise RuntimeFailure("Agent closed stdin") from exc

    def audit_pending(self):
        if self.buffer and self.on_receive:
            self.on_receive(bytes(self.buffer), False)
            self.buffer.clear()

    def receive(self, deadline):
        while b"\n" not in self.buffer:
            try:
                self._wait(self.proc.stdout, selectors.EVENT_READ, deadline)
            except RuntimeFailure:
                if self.buffer and self.on_receive:
                    self.on_receive(bytes(self.buffer), False)
                    self.buffer.clear()
                raise
            try:
                chunk = os.read(self.proc.stdout.fileno(), min(65536, self.max_bytes + 1 - len(self.buffer)))
            except BlockingIOError:
                continue
            if not chunk:
                if self.buffer and self.on_receive:
                    self.on_receive(bytes(self.buffer), False)
                    self.buffer.clear()
                raise RuntimeFailure("Agent exited before completing protocol")
            self.buffer.extend(chunk)
            if len(self.buffer) > self.max_bytes:
                if self.on_receive:
                    self.on_receive(bytes(self.buffer), False)
                    self.buffer.clear()
                raise RuntimeFailure("Agent protocol message exceeds byte limit")
        line, _, rest = self.buffer.partition(b"\n")
        self.buffer = bytearray(rest)
        if self.on_receive:
            self.on_receive(bytes(line) + b"\n", True)
        try:
            value = json.loads(line)
            json.dumps(value, allow_nan=False)
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


TOOL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}\Z")
INFERENCE_SETTINGS = {"endpoint", "model", "provider", "metadata", "reasoning_effort", "reasoning_summary", "api_key_env", "max_calls",
                      "max_total_tokens", "max_cost_usd", "max_output_tokens", "max_request_bytes", "max_response_bytes", "timeout_seconds",
                      "cost_pool", "cost_pool_usd", "cost_ledger_dir", "call_reserve_usd", "budget_hold_seconds",
                      "checkpoint_wall_seconds", "api", "price_usd_per_mtok",
                      "provider_hold_seconds"}
# "responses": the aqinference gateway's normalized Responses route (reports cost). "gemini": Google's native
# generateContent, which reports no cost, so the host prices its token counts from the configured list prices.
INFERENCE_APIS = ("responses", "gemini")
GEMINI_PRICES = {"input", "cached_input", "output"}  # USD per million tokens; thinking tokens bill as output
# Spend-cap reasons a budget hold may resume from; other stops (call/token caps, missing accounting) always suspend.
HOLDABLE = ("Run inference cost budget exhausted", "Shared inference cost pool exhausted")
INFERENCE_ACCOUNTING = ("model", "finish_reason", "usage", "cost_usd", "calls", "remaining_calls", "cumulative_tokens",
                        "remaining_tokens", "cumulative_cost_usd", "remaining_cost_usd")
CONTEXT_USAGE = ("promptTokens", "cacheReadTokens", "cacheWriteTokens")  # context size, for harness compaction only
INTERACTIONS = ("grasp", "move", "release", "arm_target", "hand_target")  # every physical action; blocked in the reference world when reference_interaction=false
FIXTURE_OPS = ("reset_carriage", "release_carriage")  # positioning-fixture levels only (manifest [fixture]); reference world only
PHYSICAL = ("observe", "grasp", "move", "release", "reset", "arm_target", "hand_target", "camera") + FIXTURE_OPS
# Metrics that score a submitted predictor program (src/sim2sim/predictor.py) instead of a scene: metric -> module whose
# evaluate_submission(predictor.Job) runs the program in isolation and returns the evaluation report (strict_oci only).
PREDICTOR_METRICS = {"fixture": "fixture", "die": "die"}
POOL_SETTINGS = {"cost_pool", "cost_pool_usd", "cost_ledger_dir", "call_reserve_usd"}
POOL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
TOOL_TRANSPORT_NOTE = ("This API transport cannot carry native tool-call or tool-result items, so the host reports each genuine function "
                       "result in a following user message labelled with the function name, call id and arguments. Treat those labelled "
                       "results as the real tool outputs. Always invoke tools through the function-calling interface: text that describes "
                       "or imitates a call does not execute anything.")
USAGE_FIELDS = ("promptTokens", "completionTokens", "totalTokens", "cacheReadTokens", "cacheWriteTokens", "reasoningTokens")


def _count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


PROVIDER_BACKOFF = 15.  # seconds before the first host-side retry after a provider outage; doubles per failure, max 300


class ProviderUnavailable(RuntimeFailure):
    """The provider did not answer in time or was overloaded (timeout, HTTP 429/5xx, transport failure), not a model output."""


class BudgetStop(RuntimeFailure):
    """A run or shared-pool inference budget cannot safely fund another call; the agent should stop gracefully."""


def read_cost_pool(directory, attempts=6):
    """Host-only per-run accounting files of one shared cost pool (no prompts or content).

    Another run's atomic replace on NFS can briefly make its file unreadable (stale handle or a torn cached view), so
    each file is re-read a bounded number of times; a file that stays unreadable still raises and fails closed."""
    records = {}
    for path in sorted(Path(directory).glob("*.json")):
        for attempt in range(attempts):
            try:
                record = json.loads(path.read_text())
                break
            except (OSError, ValueError):
                if attempt == attempts - 1:
                    raise
                time.sleep(0.25 * (attempt + 1))
        if not (isinstance(record, dict) and all(isinstance(record.get(k), (int, float)) and not isinstance(record.get(k), bool) and record[k] >= 0
                                                 for k in ("cost_usd", "inflight_reserve_usd", "calls", "tokens"))):
            raise RuntimeFailure("Invalid cost ledger " + path.name)
        records[path.stem] = record
    return records


class InferenceBroker:
    """Agent supplies messages and function schemas only; destination, model, metadata and key stay host controlled.

    The fixed gateway Responses route accepts only system/user/assistant messages with text/image parts, so the
    broker forwards earlier tool calls and tool results as labelled text turns, explained by a fixed transport note."""
    def __init__(self, config, run_id=None):
        unknown = set(config) - INFERENCE_SETTINGS
        if unknown:
            raise RuntimeFailure("Unknown inference settings: " + ", ".join(sorted(unknown)))
        self.config = config
        self.calls = 0
        self.tokens = 0
        self.cost = 0.
        self.budget_lock = threading.Lock()
        self.cancelled = threading.Event()
        endpoint = config.get("endpoint", "")
        parsed = urllib.parse.urlsplit(endpoint)
        if endpoint and (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment):
            raise RuntimeFailure("Inference endpoint must be a fixed HTTPS URL")
        metadata = config.get("metadata", {})
        api = config.get("api", "responses")
        if api not in INFERENCE_APIS:
            raise RuntimeFailure("Unknown inference api")
        if endpoint and not (isinstance(config.get("model"), str) and isinstance(config.get("api_key_env"), str)):
            raise RuntimeFailure("Inference requires model and api_key_env")
        if endpoint and api == "responses" and not (isinstance(metadata, dict) and {"project", "feature"} <= set(metadata)
                                                    and all(isinstance(value, str) and value for value in metadata.values())):
            raise RuntimeFailure("Gateway inference requires string metadata project/feature")
        prices = config.get("price_usd_per_mtok")
        if endpoint and api == "gemini" and not (
                parsed.path == "/v1beta/models/%s:generateContent" % config["model"] and not parsed.query and "metadata" not in config
                and "provider" not in config and isinstance(prices, dict) and set(prices) == GEMINI_PRICES
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in prices.values())):
            raise RuntimeFailure("Gemini inference requires the model's own generateContent endpoint and positive input/cached_input/output "
                                 "list prices, and takes no gateway metadata or provider")
        if api == "responses" and prices is not None:
            raise RuntimeFailure("The gateway reports its own cost; price_usd_per_mtok is only for the gemini api")
        if config.get("reasoning_summary") not in (None, "auto", "concise", "detailed"):
            raise RuntimeFailure("Invalid reasoning_summary")
        self.stopped = None
        self.reserve = float(config.get("call_reserve_usd", 3.0))
        self.ledger = None
        if config.get("cost_pool") is not None:
            # One file per run, written only by this broker, so concurrent runs on other nodes need no shared lock.
            if not (isinstance(config["cost_pool"], str) and POOL_NAME.match(config["cost_pool"]) and isinstance(run_id, str) and POOL_NAME.match(run_id)
                    and float(config.get("cost_pool_usd", 0)) > 0 and self.reserve > 0):
                raise RuntimeFailure("cost_pool requires a valid pool name, run id, positive cost_pool_usd and call_reserve_usd")
            self.run_id = run_id
            self.ledger = Path(config.get("cost_ledger_dir", "experiments/cost-ledger")).resolve() / config["cost_pool"]
            self.ledger.mkdir(parents=True, exist_ok=True)
            if (self.ledger / (run_id + ".json")).exists():
                raise RuntimeFailure("A cost ledger already exists for this run id")
            self.write_ledger(0.)

    def write_ledger(self, inflight):
        if self.ledger is not None:
            record = {"run_id": self.run_id, "pool": self.config["cost_pool"], "cost_usd": self.cost, "inflight_reserve_usd": inflight,
                      "calls": self.calls, "tokens": self.tokens, "updated_at": datetime.now(timezone.utc).isoformat()}
            temporary = self.ledger / ("." + self.run_id + ".json.tmp")
            temporary.write_text(json.dumps(record) + "\n")
            os.replace(temporary, self.ledger / (self.run_id + ".json"))

    def stop_reason(self):
        """Soft threshold: stop while every cap still has room for two worst-case calls."""
        max_calls, max_tokens, max_cost = self.limits()
        if self.stopped:
            return self.stopped
        if self.calls >= max_calls:
            return "Inference call budget exhausted"
        if max_tokens is not None and self.tokens >= max_tokens:
            return "Inference reported-token budget exhausted"
        if max_cost is not None and max_cost - self.cost < 2 * self.reserve:
            return "Run inference cost budget exhausted"
        if self.ledger is not None:
            try:
                others = sum(r["cost_usd"] + r["inflight_reserve_usd"] for run, r in read_cost_pool(self.ledger).items() if run != self.run_id)
            except (OSError, ValueError, RuntimeFailure) as exc:
                return "Shared cost ledger unreadable (%s); failing closed" % type(exc).__name__
            if float(self.config["cost_pool_usd"]) - others - self.cost < 2 * self.reserve:
                return "Shared inference cost pool exhausted"
        return None

    def limits(self):
        cfg = self.config
        return (int(cfg.get("max_calls", 20)), None if cfg.get("max_total_tokens") is None else int(cfg["max_total_tokens"]),
                None if cfg.get("max_cost_usd") is None else float(cfg["max_cost_usd"]))

    def cancel(self):
        self.cancelled.set()
        with self.budget_lock:
            process = getattr(self, "process", None)
        if process is not None and process.is_alive():
            process.kill()

    def call(self, request, timeout):
        if self.cancelled.is_set():
            raise RuntimeFailure("Inference was cancelled")
        if not self.config.get("endpoint"):
            raise RuntimeFailure("Inference is disabled")
        with self.budget_lock:
            reason = self.stop_reason()
            if reason:
                raise BudgetStop(reason)
            self.calls += 1
            self.write_ledger(self.reserve)
        allowance = max(0., min(timeout, float(self.config.get("timeout_seconds", 60))))
        if allowance <= 0:
            raise RuntimeFailure("Inference wall-time budget exhausted")
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=_inference_process, args=(self.config, request, allowance, child), daemon=True)
        with self.budget_lock:
            self.process = process
        try:
            process.start()
            if self.cancelled.is_set():
                raise RuntimeFailure("Inference was cancelled")
            child.close()
            if not parent.poll(allowance):
                with self.budget_lock:
                    self.cost += self.reserve  # An abandoned request may still be billed; charge its worst case.
                # Only the provider's own per-call limit is an outage; the run's remaining time running out is not.
                raise (ProviderUnavailable if allowance >= float(self.config.get("timeout_seconds", 60)) else RuntimeFailure)(
                    "Inference wall-time budget exhausted")
            ok, value, outage = parent.recv()
            if not ok:
                raise (ProviderUnavailable if outage else RuntimeFailure)(value)
            return self.charge(value)
        finally:
            if process.is_alive():
                process.kill()
            if process.pid is not None:
                process.join(timeout=1)
            parent.close()
            child.close()
            with self.budget_lock:
                self.process = None
                self.write_ledger(0.)

    def charge(self, value):
        max_calls, max_tokens, max_cost = self.limits()
        with self.budget_lock:
            # Fail closed: an unaccountable completed call stops further inference and is withheld.
            if value["usage"] is None and max_tokens is not None:
                self.stopped = "Provider omitted token usage; further inference is disabled"
            if value["cost_usd"] is None and (max_cost is not None or self.ledger is not None):
                self.stopped = "Provider omitted cost; further inference is disabled"
                self.cost += self.reserve
            if self.stopped:
                raise RuntimeFailure(self.stopped)
            self.tokens += value["usage"]["accountedTokens"] if value["usage"] else 0
            self.cost += value["cost_usd"] or 0.
            value.update(calls=self.calls, remaining_calls=max(0, max_calls - self.calls), cumulative_tokens=self.tokens,
                         remaining_tokens=None if max_tokens is None else max(0, max_tokens - self.tokens),
                         cumulative_cost_usd=self.cost, remaining_cost_usd=None if max_cost is None else max(0., max_cost - self.cost))
        return value

    def _parts(self, content, images):
        if isinstance(content, str):
            content = [{"type": "text", "text": content}] if content else []
        if not isinstance(content, list):
            raise RuntimeFailure("Invalid inference message content")
        parts = []
        for part in content:
            if isinstance(part, dict) and set(part) == {"type", "text"} and part["type"] == "text" and isinstance(part["text"], str):
                parts.append({"type": "input_text", "text": part["text"]})
            elif images and isinstance(part, dict) and set(part) == {"type", "image_url"} and part["type"] == "image_url" \
                    and isinstance(part["image_url"], dict) and set(part["image_url"]) == {"url"}:
                url = part["image_url"]["url"]
                if not isinstance(url, str) or not url.startswith(("data:image/png;base64,", "data:image/jpeg;base64,")):
                    raise RuntimeFailure("Inference images must be inline PNG/JPEG data URLs")
                raw_image = decode_submission(url.split(",", 1)[1], int(self.config.get("max_request_bytes", 4 * 1024 * 1024)))
                if not (raw_image.startswith(b"\x89PNG\r\n\x1a\n") or raw_image.startswith(b"\xff\xd8\xff")):
                    raise RuntimeFailure("Invalid inline image signature")
                parts.append({"type": "input_image", "imageUrl": url})
            else:
                raise RuntimeFailure("Invalid inference content part")
        return parts

    def payload(self, request):
        cfg = self.config
        messages, tools = request.get("messages"), request.get("tools", [])
        if set(request) - {"op", "messages", "tools"}:
            raise RuntimeFailure("Inference requests contain only messages and tools")
        if not isinstance(messages, list) or not messages:
            raise RuntimeFailure("Inference messages must be a nonempty array")
        if not isinstance(tools, list) or len(tools) > 64:
            raise RuntimeFailure("Inference tools must be an array of at most 64 function schemas")
        functions = []
        for tool in tools:
            if not (isinstance(tool, dict) and {"name", "parameters"} <= set(tool) <= {"name", "description", "parameters"}
                    and isinstance(tool["name"], str) and TOOL_NAME.match(tool["name"]) and isinstance(tool.get("description", ""), str)
                    and isinstance(tool["parameters"], dict) and tool["parameters"].get("type") == "object"):
                raise RuntimeFailure("Invalid inference function schema")
            functions.append({"name": tool["name"], "description": tool.get("description", ""), "parameters": tool["parameters"]})
        names = {function["name"] for function in functions}
        if len(names) != len(functions):
            raise RuntimeFailure("Duplicate inference function name")
        inputs, issued = [], {}
        for message in messages:
            role = message.get("role") if isinstance(message, dict) else None
            extra = {"assistant": {"tool_calls"}, "tool": {"tool_call_id"}}.get(role, set())
            if role not in ("system", "user", "assistant", "tool") or not {"role", "content"} <= set(message) <= {"role", "content"} | extra:
                raise RuntimeFailure("Invalid inference message")
            parts = self._parts(message["content"], role in ("user", "tool"))
            if role == "assistant":
                calls = message.get("tool_calls", [])
                if not isinstance(calls, list):
                    raise RuntimeFailure("Invalid assistant tool calls")
                for call in calls:
                    if not (isinstance(call, dict) and set(call) == {"id", "name", "arguments"} and all(isinstance(call[k], str) for k in call)
                            and call["name"] in names and call["id"] not in issued):
                        raise RuntimeFailure("Invalid assistant tool call")
                    issued[call["id"]] = call
            elif role == "tool":
                if message.get("tool_call_id") not in issued:
                    raise RuntimeFailure("Tool result does not answer an earlier tool call")
                call = issued[message["tool_call_id"]]
                parts.insert(0, {"type": "input_text", "text": "[host] Result of your %s call (id %s, arguments %s):" % (call["name"], call["id"], call["arguments"])})
            if role == "assistant" and not parts:
                continue  # Calls are reported with their results, so a call-only turn carries no assistant text.
            if not parts:
                raise RuntimeFailure("Inference messages must not be empty")
            inputs.append({"role": "user" if role == "tool" else role, "content": parts})
        payload = {"model": cfg["model"], "input": inputs, "max_output_tokens": int(cfg.get("max_output_tokens", 4096)),
                   "store": False, "metadata": dict(cfg.get("metadata", {}))}
        if cfg.get("provider"):
            payload["provider"] = cfg["provider"]
        if functions:
            payload["tools"] = [{"type": "function", "function": function} for function in functions]
            payload["input"].insert(0, {"role": "system", "content": [{"type": "input_text", "text": TOOL_TRANSPORT_NOTE}]})
        reasoning = {key: cfg[name] for key, name in (("effort", "reasoning_effort"), ("summary", "reasoning_summary")) if cfg.get(name)}
        if reasoning:
            payload["reasoning"] = reasoning
        return payload, names

    def parse(self, result, names):
        cfg = self.config
        if not isinstance(result, dict) or result.get("model") != cfg["model"]:
            raise RuntimeFailure("Provider returned an unexpected model; no substitution is allowed")
        route = result.get("route")
        if not isinstance(route, dict) or route.get("fallbackUsed") is not False or route.get("model") != cfg["model"] \
                or (cfg.get("provider") and route.get("provider") != cfg["provider"]):
            raise RuntimeFailure("Provider route used a fallback or substitution; no substitution is allowed")
        content = result.get("content")
        if not isinstance(content, str):
            raise RuntimeFailure("Provider returned invalid public output text")
        calls = []
        for call in result.get("toolCalls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if not (isinstance(function, dict) and call.get("type") == "function" and isinstance(call.get("id"), str)
                    and function.get("name") in names and isinstance(function.get("arguments"), str)):
                raise RuntimeFailure("Provider returned an invalid tool call")
            calls.append({"id": call["id"], "name": function["name"], "arguments": function["arguments"]})
        usage = result.get("usage")
        usage = {key: usage[key] for key in USAGE_FIELDS} if isinstance(usage, dict) and all(_count(usage.get(key)) for key in USAGE_FIELDS) else None
        if usage:
            # Gateway totals exclude cached prompt tokens; account reads and writes as well.
            usage["accountedTokens"] = usage["totalTokens"] + usage["cacheReadTokens"] + usage["cacheWriteTokens"]
        cost = result.get("cost")
        cost = cost.get("total") if isinstance(cost, dict) and cost.get("currency") == "USD" else None
        cost = float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0 else None
        summary = result.get("reasoning") if cfg.get("reasoning_summary") else None
        return {"content": content, "tool_calls": calls, "finish_reason": result.get("finishReason") if isinstance(result.get("finishReason"), str) else None,
                "model": result["model"], "usage": usage, "cost_usd": cost, "summaries": [summary] if isinstance(summary, str) and summary.strip() else []}

    def gemini_payload(self, payload):
        """The same validated transcript as the gateway route (tool calls and results as labelled text turns), so both APIs see
        identical conversations; only the wire format differs."""
        cfg = self.config
        system, contents = [], []
        for item in payload["input"]:
            parts = [{"text": part["text"]} if part["type"] == "input_text" else
                     {"inlineData": {"mimeType": part["imageUrl"][5:].split(";", 1)[0], "data": part["imageUrl"].split(",", 1)[1]}}
                     for part in item["content"]]
            if item["role"] == "system":
                assert not contents, "system messages must precede the conversation"
                system += parts
            else:
                contents.append({"role": "model" if item["role"] == "assistant" else "user", "parts": parts})
        thinking = {"includeThoughts": bool(cfg.get("reasoning_summary"))}
        if cfg.get("reasoning_effort"):
            thinking["thinkingLevel"] = cfg["reasoning_effort"]
        body = {"contents": contents, "generationConfig": {"maxOutputTokens": payload["max_output_tokens"], "thinkingConfig": thinking}}
        if system:
            body["systemInstruction"] = {"parts": system}
        if "tools" in payload:
            body["tools"] = [{"functionDeclarations": [{"name": tool["function"]["name"], "description": tool["function"]["description"],
                                                        "parametersJsonSchema": tool["function"]["parameters"]} for tool in payload["tools"]]}]
        return body

    def parse_gemini(self, result, names):
        cfg = self.config
        if not isinstance(result, dict) or result.get("modelVersion") != cfg["model"]:
            raise RuntimeFailure("Provider returned an unexpected model; no substitution is allowed")
        candidates = result.get("candidates")
        if not (isinstance(candidates, list) and len(candidates) == 1 and isinstance(candidates[0], dict)):
            blocked = (result.get("promptFeedback") or {}).get("blockReason") if isinstance(result.get("promptFeedback"), dict) else None
            raise RuntimeFailure("Provider returned no candidate (%s)" % (blocked if isinstance(blocked, str) else "no reason given"))
        candidate = candidates[0]
        content = candidate.get("content", {})  # absent when the output budget ran out during thinking
        parts = content.get("parts", []) if isinstance(content, dict) else None
        if not isinstance(parts, list):
            raise RuntimeFailure("Provider returned invalid public output")
        text, summaries, calls = [], [], []
        for part in parts:
            call = part.get("functionCall") if isinstance(part, dict) else None
            if call is not None:
                if not (isinstance(call, dict) and call.get("name") in names and isinstance(call.get("args", {}), dict)):
                    raise RuntimeFailure("Provider returned an invalid tool call")
                # Gemini ids are optional; the broker issues unique ones because transcripts must not repeat a call id.
                calls.append({"id": "call_" + uuid.uuid4().hex[:16], "name": call["name"], "arguments": json.dumps(call.get("args", {}))})
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                (summaries if part.get("thought") is True else text).append(part["text"])
            elif not (isinstance(part, dict) and set(part) <= {"thoughtSignature"}):
                raise RuntimeFailure("Provider returned an unsupported output part")
        meta = result.get("usageMetadata")
        counts = {key: meta.get(key, 0) for key in ("promptTokenCount", "cachedContentTokenCount", "candidatesTokenCount", "thoughtsTokenCount")} \
            if isinstance(meta, dict) and _count(meta.get("promptTokenCount")) else None
        usage = cost = None
        if counts and all(_count(value) for value in counts.values()) and counts["cachedContentTokenCount"] <= counts["promptTokenCount"]:
            # Same fields as the gateway: promptTokens excludes cached reads; completion includes thinking (billed as output).
            fresh, cached = counts["promptTokenCount"] - counts["cachedContentTokenCount"], counts["cachedContentTokenCount"]
            output = counts["candidatesTokenCount"] + counts["thoughtsTokenCount"]
            usage = {"promptTokens": fresh, "completionTokens": output, "totalTokens": fresh + output, "cacheReadTokens": cached,
                     "cacheWriteTokens": 0, "reasoningTokens": counts["thoughtsTokenCount"], "accountedTokens": fresh + output + cached}
            prices = cfg["price_usd_per_mtok"]
            cost = (fresh * prices["input"] + cached * prices["cached_input"] + output * prices["output"]) / 1e6
        summary = "\n\n".join(s for s in summaries if s.strip()) if cfg.get("reasoning_summary") else ""
        return {"content": "".join(text), "tool_calls": calls, "finish_reason": candidate.get("finishReason") if isinstance(candidate.get("finishReason"), str) else None,
                "model": result["modelVersion"], "usage": usage, "cost_usd": cost, "summaries": [summary] if summary.strip() else []}

    def _call(self, request, timeout):
        cfg = self.config
        if not cfg.get("endpoint"):
            raise RuntimeFailure("Inference is disabled")
        gemini = cfg.get("api", "responses") == "gemini"
        payload, names = self.payload(request)
        body = json.dumps(self.gemini_payload(payload) if gemini else payload).encode()
        if len(body) > int(cfg.get("max_request_bytes", 4 * 1024 * 1024)):
            raise RuntimeFailure("Inference request exceeds byte limit")
        key = os.environ.get(cfg["api_key_env"], "")
        if not key:
            raise RuntimeFailure("Host inference credential is missing")
        req = urllib.request.Request(cfg["endpoint"], data=body, headers={"x-goog-api-key" if gemini else "Authorization": key if gemini else "Bearer " + key,
                                                                          "Content-Type": "application/json"})
        opener = urllib.request.build_opener(NoRedirect())
        inference_deadline = time.monotonic() + max(0, min(timeout, float(cfg.get("timeout_seconds", 60))))
        limit = int(cfg.get("max_response_bytes", 1048576))
        attempt = 0
        while True:
            try:
                response = opener.open(req, timeout=max(0.001, inference_deadline - time.monotonic()))
                break
            except urllib.error.HTTPError as exc:
                # Unbilled overload/rate errors are retried host-side within this call's deadline (the model sees nothing).
                wait = 5 * 2 ** attempt
                if exc.code in (429, 500, 503, 504) and time.monotonic() + wait + 60 < inference_deadline:
                    attempt += 1
                    time.sleep(wait)
                    continue
                try:
                    error = json.loads(exc.read(65536)).get("error")
                except (ValueError, AttributeError, OSError):
                    error = None
                error = error.get("message") if isinstance(error, dict) else error  # Google nests {code, message, status}
                error = error.replace(key, "") if isinstance(error, str) else "no gateway error text"
                raise (ProviderUnavailable if exc.code in (429, 500, 503, 504) else RuntimeFailure)(
                    "Inference provider HTTP %d: %s" % (exc.code, error[:500])) from None
        with response:
            chunks = []
            length = 0
            while length <= limit:
                if time.monotonic() >= inference_deadline:
                    raise ProviderUnavailable("Inference wall-time budget exhausted")
                chunk = response.read1(min(65536, limit + 1 - length))
                if not chunk:
                    break
                chunks.append(chunk)
                length += len(chunk)
            raw = b"".join(chunks)
        if len(raw) > limit:
            raise RuntimeFailure("Inference response exceeds byte limit")
        # Expose only generated content, tool calls, accounting and documented summaries; never headers or route internals.
        return (self.parse_gemini if gemini else self.parse)(json.loads(raw), names)


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


def evaluate_in_container(submission, evaluation, sandbox, output, source_bundle=None, supervisor=None):
    if supervisor:
        supervisor.check()
    output.mkdir(parents=True, exist_ok=False)
    source = source_bundle if source_bundle is not None else zip_directory(evaluation["source_dir"])
    cfg = dict(evaluation)
    cfg.pop("source_dir", None)
    cfg.pop("output_dir", None)
    payload = {"submission_b64": base64.b64encode(submission).decode(),
               "source_b64": base64.b64encode(source).decode(), "config": cfg}
    name = "sim2sim-eval-" + uuid.uuid4().hex
    proc = launch_container(sandbox.get("evaluator_image", "localhost/sim2sim:latest"), ["python", "-m", "sim2sim.worker"], worker_limits(sandbox), name,
                            output / "evaluator.stderr.log", supervisor)
    err = proc.stderr_log
    try:
        transport = Transport(proc, 180 * 1024 * 1024)
        deadline = time.monotonic() + float(evaluation.get("timeout_seconds", 180))
        if supervisor:
            deadline = min(deadline, supervisor.deadline)
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
        if supervisor:
            supervisor.unregister(proc)
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


class FrameVideoRecorder:
    """Encode already-recorded candidate PNGs; no candidate model touches the host."""
    def __init__(self, output, fps):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.writer = None
        self.segments = []
        self.episode = None

    def push(self, observation, total_ticks, episode, final=False):
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image
        now = float(observation.get("time", 0))
        if self.writer is None or episode != self.episode:
            self.finish()
            self.episode = episode
            self.segment = {"file": "interaction-%03d.mp4" % len(self.segments), "episode": episode,
                            "start_time": now, "start_total_ticks": total_ticks, "fps": self.fps}
            self.writer = imageio.get_writer(str(self.output / self.segment["file"]), fps=self.fps)
            self.next_time = now
            self.last_time = -1.0
            self.frames = 0
        self.pending_observation = observation
        self.pending_time = now
        if now + 1e-10 >= self.next_time:
            frame = np.asarray(Image.open(io.BytesIO(base64.b64decode(observation["rgb"]))))
            self.writer.append_data(frame)
            self.frames += 1
            self.last_time = now
            while self.next_time <= now + 1e-10:
                self.next_time += 1 / self.fps
        self.segment.update(end_time=now, end_total_ticks=total_ticks, frames=self.frames)

    def finish(self):
        if self.writer is not None:
            if self.pending_time > self.last_time + 1e-10:
                from PIL import Image
                import numpy as np
                frame = np.asarray(Image.open(io.BytesIO(base64.b64decode(self.pending_observation["rgb"]))))
                self.writer.append_data(frame)
                self.frames += 1
                self.segment["frames"] = self.frames
            self.writer.close()
            self.writer = None
            self.segments.append(self.segment)
            (self.output / "video-index.json").write_text(json.dumps(self.segments, indent=2))

    close = finish


class CandidateSession:
    """A persistent, source-free worker in the trusted simulation image."""
    def __init__(self, sandbox, output, audit):
        self.name = "sim2sim-reconstruction-" + uuid.uuid4().hex
        self.audit = audit
        self.token = None
        self.action_id = None
        self.last = None
        self.closed = False
        self.proc = launch_container(sandbox.get("evaluator_image", "localhost/sim2sim:latest"),
                                     ["python", "-m", "sim2sim.candidate_worker"], worker_limits(sandbox), self.name, output)
        self.stderr = self.proc.stderr_log
        self.transport = Transport(self.proc, 48 * 1024 * 1024,
                                   on_receive=lambda raw, complete: self._wire("response", raw, complete),
                                   on_send=lambda raw, complete: self._wire("request", raw, complete))

    def _wire(self, direction, raw, complete):
        event = {"kind": "worker_boundary", "direction": direction, "world": "reconstruction",
                 "action_id": self.action_id, "complete": complete,
                 "wire_b64": base64.b64encode(raw).decode()}
        try:
            payload = json.loads(raw)
            json.dumps(payload, allow_nan=False)
            event["payload"] = payload
        except (ValueError, UnicodeError):
            pass
        self.audit.record(event, reservation=self.token)

    def exchange(self, payload, deadline, token, action_id, on_tick=None):
        self.token, self.action_id = token, action_id
        try:
            self.transport.send(payload, deadline)
            while True:
                packet = self.transport.receive(deadline)
                if packet.get("type") == "tick":
                    self.last = packet
                    if on_tick is None:
                        raise RuntimeFailure("Unexpected candidate tick")
                    on_tick(packet)
                    # Persisted acknowledgements gate each subsequent physics tick.
                    self.transport.send({"continue": True}, deadline)
                elif packet.get("type") == "result":
                    self.last = packet
                    return packet
                else:
                    raise RuntimeFailure("Invalid candidate worker response")
        except BaseException:
            self.close()
            raise
        finally:
            self.token = None

    def close(self):
        if self.closed:
            return
        self.closed = True
        stop_container(self.proc, self.name)
        if getattr(self, "supervisor", None):
            self.supervisor.unregister(self.proc)
        self.stderr.close()
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            stream.close()


def run(config, *, prepared_output=False):
    supervisor = WallClock(config.get("budget", {}).get("wall_seconds"))
    try:
        return _run(config, supervisor, prepared_output=prepared_output)
    except BaseException as exc:
        if supervisor.expired.is_set() or supervisor.remaining() <= 0:
            raise RuntimeFailure("Hard wall-time budget exhausted") from exc
        raise
    finally:
        supervisor.close()


def _run(config, supervisor, *, prepared_output=False, backend=None):
    from .audit import AuditLog, AuditFailure
    from .candidate_worker import snapshot_simulation, full_observation, state_digest, public_observation

    run_started = supervisor.started
    started_at_unix = time.time()
    timing = {"active_start": None, "paused_total": 0., "pause_start": None}
    def clocks():
        now = time.monotonic()
        active = 0. if timing["active_start"] is None else now - timing["active_start"] - timing["paused_total"]
        if timing["pause_start"] is not None:
            active -= now - timing["pause_start"]
        return {"wall_elapsed": now - run_started - supervisor.held_seconds(), "active_elapsed": max(0., active)}
    sandbox = config["sandbox"]
    tools = enabled_tools(config)
    if tools and backend is not None:
        raise RuntimeFailure("Optional agent tools are strict_oci only")
    sam3d_relay = None
    if "sam3d" in tools:
        from .sam3d import Relay
        if not os.environ.get("SIM2SIM_SAM3D_SPOOL"):
            raise SandboxFailure("tools.sam3d needs SIM2SIM_SAM3D_SPOOL and a running scripts/sam3d-service.slurm")
        sam3d_relay = Relay(os.environ["SIM2SIM_SAM3D_SPOOL"])
    supervisor.check()
    if backend is None:
        check_rootless(sandbox.get("process_limit_backend", "cgroup"), supervisor)
    else:
        backend.preflight()
    supervisor.check()
    inference_config = config.get("inference", {})
    if inference_config.get("endpoint") and not os.environ.get(inference_config.get("api_key_env", "")):
        raise RuntimeFailure("Host inference credential is missing; agent was not launched")
    if backend is None:
        # Pin tags to immutable IDs: every container of this run uses the image that passed preflight.
        sandbox = {**sandbox, **{key: image_id(sandbox.get(key, "localhost/sim2sim:latest"), supervisor) for key in ("image", "evaluator_image")}}
        if "segmentation" in tools:
            labels = json.loads(subprocess.run(["podman", "image", "inspect", "--format", "{{json .Config.Labels}}", sandbox["image"]], capture_output=True,
                                               text=True, timeout=min(20, supervisor.remaining()), check=True, env=podman_env()).stdout) or {}
            if labels.get(TOOL_IMAGE_LABEL) != "torch-cpu":
                raise SandboxFailure("tools.segmentation needs the tools image (SIM2SIM_TRUSTED_IMAGE_RECORD=experiments/images/trusted-image-tools.json)")
        sandbox_verification = check_limits(sandbox, supervisor)
        supervisor.check()
    limits = config.get("budget", {})
    wall_only = limits.get("wall_only", False)
    if not isinstance(wall_only, bool):
        raise RuntimeFailure("wall_only must be boolean")
    if wall_only and not math.isfinite(supervisor.deadline):
        raise RuntimeFailure("wall_only requires a finite wall_seconds deadline")
    max_ticks = None if wall_only else int(limits.get("physics_ticks", 3000))
    max_submissions = None if wall_only else int(limits.get("submissions", 3))
    seconds = None if wall_only else float(limits.get("active_seconds", 300))
    if not wall_only and (max_ticks < 0 or not math.isfinite(seconds) or seconds <= 0):
        raise RuntimeFailure("Budgets must be positive")
    output = Path(config.get("output_dir", "runs/latest")).resolve()
    if output.exists() and prepared_output:
        validate_prepared_output(output)
    else:
        output.mkdir(parents=True, exist_ok=False)
    audit = TimedAudit(AuditLog(output, int(limits.get("audit_bytes", 512 * 1024 * 1024))), clocks)
    emergency = audit.reserve(256 * 1024)
    broker = InferenceBroker(config.get("inference", {}), run_id=output.name)
    supervisor.add_cleanup(broker.cancel)
    scene = config["source"]["scene"]
    source_root = Path(config.get("evaluation", {}).get("source_dir", Path(scene).resolve().parent)).resolve()
    source_bundle = zip_directory(source_root)
    (output / "source-snapshot.zip").write_bytes(source_bundle)
    sim_config = dict(config.get("simulation", {}))
    for key, default in (("default_ticks", 20), ("rgb", True), ("depth", False), ("calibration", True), ("wrench", False),
                         ("tactile", False), ("reference_interaction", True)):
        sim_config.setdefault(key, default)
    if not isinstance(sim_config["reference_interaction"], bool):
        raise RuntimeFailure("simulation.reference_interaction must be a boolean")
    if sim_config.get("movable_camera", False) not in (True, False):
        raise RuntimeFailure("simulation.movable_camera must be a boolean")
    if sim_config.get("effort", True) is not True and (sim_config["effort"] is not False or not sim_config.get("robot")):
        raise RuntimeFailure("simulation.effort is a boolean for a robot (false hides actuator_effort_nm: no force sensing)")
    source_manifest = Path(scene).resolve().parent / config["source"].get("manifest", "manifest.toml")
    private_manifest = {}
    if source_manifest.is_file():
        with source_manifest.open("rb") as handle:
            private_manifest = tomllib.load(handle)
        for key in ("camera", "graspable_bodies"):
            if key in private_manifest:
                sim_config.setdefault(key, private_manifest[key])
    fixture_table = private_manifest.get("fixture")
    if fixture_table:  # private seed of the fixture's reset perturbations (src/sim2sim/fixture.py); never public
        sim_config["fixture_seed"] = secrets.randbits(63)
    realtime = sim_config.get("realtime", False)
    metrics = list(config.get("evaluation", {}).get("metrics", []))
    predictor_metric = next((metric for metric in metrics if metric in PREDICTOR_METRICS), None)
    if predictor_metric and metrics != [predictor_metric]:
        raise RuntimeFailure(f"the {predictor_metric} metric scores a predictor program and combines with no other metric")
    if predictor_metric and not str(config["evaluation"].get("entrypoint", "")).endswith(".py"):
        raise RuntimeFailure(f"the {predictor_metric} metric needs [evaluation] entrypoint naming the predictor program (*.py)")
    die_metric = predictor_metric == "die"
    if realtime not in (True, False) or ("launcher" in private_manifest or die_metric) and not realtime or die_metric and "launcher" not in private_manifest:
        raise RuntimeFailure("simulation.realtime is a boolean; a manifest [launcher] needs it, and the die metric needs both")
    if realtime and not (wall_only and type(sim_config.get("frame_rate")) is int and sim_config["frame_rate"] > 0 and backend is None):
        raise RuntimeFailure("simulation.realtime needs wall_only, a positive integer frame_rate and strict_oci")
    supervisor.check()
    from .simulation import Simulation
    robot = sim_config.get("robot")
    sim = Simulation(scene, sim_config)
    live = None  # the real-time reference world (realtime.py), started below; it alone touches sim once started
    launcher = None
    if realtime:
        from .launcher import Launcher
        from .realtime import RealtimeWorld, WorldFailure
        launcher = Launcher(private_manifest["launcher"]) if "launcher" in private_manifest else None
    supervisor.check()
    public_config = {k: sim_config[k] for k in ("width", "height", "robot", "wrist_camera", "rgb", "depth", "calibration", "wrench", "tactile", "effort", "movable_camera", "realtime", "frame_rate", "default_ticks", "grasp_solref", "grasp_solimp") if k in sim_config}
    public_config.setdefault("width", getattr(sim, "width", 640))
    public_config.setdefault("height", getattr(sim, "height", 480))
    public_config["timestep"] = float(sim._model.opt.timestep) if hasattr(sim, "_model") else float(sim_config.get("timestep", .002))
    public_config.setdefault("grasp_solref", [.01, 1])
    public_config.setdefault("grasp_solimp", [.9, .95, .001, .5, 2])
    manifest = json.loads(json.dumps(config))
    manifest["execution_mode"] = "strict_oci" if backend is None else "development_shared_filesystem"
    if backend is not None:
        manifest["model_usage"] = {"backend": "external_spawned_agent", "requested_model": config.get("agent_model"), "tokens_available": False}
    manifest["source"]["scene"] = str(Path(scene).resolve())
    manifest.update(resolved_output_dir=str(output), source_bundle_sha256=hashlib.sha256(source_bundle).hexdigest(),
                    resolved_source_config=sim_config, public_tool_config=public_config,
                    versions={"python": platform.python_version()}, image_ids={}, started_at_unix=started_at_unix,
                    started_at_utc=datetime.fromtimestamp(started_at_unix, timezone.utc).isoformat())
    for package in ("mujoco", "numpy"):
        try:
            manifest["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            manifest["versions"][package] = "unavailable"
    if backend is None:
        manifest.update(image_ids={"agent": sandbox["image"], "evaluator": sandbox["evaluator_image"]}, sandbox_verification=sandbox_verification)
    if sam3d_relay:
        manifest["sam3d_service"] = {"spool": str(sam3d_relay.spool), **sam3d_relay.service}
    if manifest.get("inference", {}).get("endpoint"):
        manifest["inference"]["endpoint"] = urllib.parse.urlunsplit(urllib.parse.urlsplit(manifest["inference"]["endpoint"])._replace(query=""))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    audit.record({"kind": "run_start", "manifest": manifest, "source_snapshot": "source-snapshot.zip"})
    events = BoundedEvents(output / "events.jsonl", int(limits.get("event_log_bytes", 8 * 1024 * 1024)))
    progress = (output / "progress.jsonl").open("a", encoding="utf-8")
    progress_used = 0
    video = FrameVideoRecorder(output, int(config.get("video_fps", 30))) if config.get("record_video", True) else None
    candidate_video = FrameVideoRecorder(output / "reconstruction", int(config.get("video_fps", 30))) if config.get("record_video", True) else None
    charged = {"reference": 0, "reconstruction": 0}
    episodes = {"reference": 0, "reconstruction": -1}
    candidate = None
    candidate_version = None
    candidate_worker_ticks = 0
    fps = int(config.get("video_fps", 30))
    next_reference_frame = 1 / fps
    cache = {"reference": None, "reconstruction": None}
    context = {"action_id": 0, "world": "reference", "token": None, "incoming": None}
    proc = None
    volumes = []
    stderr = None
    transport = None
    name = "sim2sim-agent-" + uuid.uuid4().hex
    submissions = 0
    evaluated = 0
    candidate_changes = 0
    termination, reason = "failed", "initialization"
    suspended = None  # host budget reason once the orchestrator stops inference; the container never sees it
    timing["active_start"] = time.monotonic()
    deadline = float("inf") if wall_only else timing["active_start"] + seconds

    hold_seconds = inference_config.get("budget_hold_seconds")
    # Provider outages (timeouts, 429/5xx) are not the model's doing: with provider_hold_seconds set, each failed attempt's
    # time is given back and the backoff before the retry runs with containers paused and clocks stopped, up to this total.
    provider_hold = inference_config.get("provider_hold_seconds")
    provider_held = 0.
    # Orchestrator review point inside the visible wall budget (user: "4 h extendable to 6 h"): the agent sees the full
    # budget from the start; at the checkpoint the run holds exactly like a spend-cap hold, and is resumed or stopped.
    checkpoint = inference_config.get("checkpoint_wall_seconds")
    assert checkpoint is None or hold_seconds is not None, "checkpoint_wall_seconds requires budget_hold_seconds"
    holds = []

    def budget_hold(reason):
        """Freeze the run at a spend cap until the orchestrator decides (hold-decision.json in the run directory).

        The agent and candidate containers are paused and the wall and active clocks stop, so a resumed run is the same
        run it would have been with the raised caps from the start. Returns True to retry the call, False to suspend."""
        nonlocal deadline
        decision_path = output / "hold-decision.json"
        assert not decision_path.exists(), "a hold decision exists before its hold"
        (output / "hold.json").write_text(json.dumps({"reason": reason, "held_at": datetime.now(timezone.utc).isoformat(),
                                                      "cost_usd": broker.cost, "limits": broker.limits(), **clocks()}) + "\n")
        audit.record({"kind": "inference_hold", "reason": reason, "cost_usd": broker.cost}, reservation=context["token"])
        with frozen(), supervisor.hold():
            pause = timing["pause_start"] = time.monotonic()
            try:
                while not decision_path.exists() and time.monotonic() - pause < float(hold_seconds):
                    time.sleep(5)
                decision = json.loads(decision_path.read_text()) if decision_path.exists() else {"action": "stop", "reason": "hold timeout"}
            finally:
                duration = time.monotonic() - pause
                deadline += duration
                timing["paused_total"] += duration
                timing["pause_start"] = None
        index = len(holds) + 1
        for path in (decision_path, output / "hold.json"):
            if path.exists():
                path.rename(path.with_name("%s-%03d.json" % (path.stem, index)))
        assert decision["action"] in ("resume", "stop"), decision
        if decision["action"] == "resume":
            assert set(decision) == {"action", "max_cost_usd", "cost_pool_usd"}, decision
            assert decision["max_cost_usd"] > broker.cost and decision["cost_pool_usd"] > 0, decision
            broker.config = {**broker.config, "max_cost_usd": float(decision["max_cost_usd"]), "cost_pool_usd": float(decision["cost_pool_usd"])}
        holds.append({"reason": reason, "held_seconds": duration, "decision": decision})
        audit.record({"kind": "inference_hold_decision", **holds[-1]}, reservation=context["token"])
        return decision["action"] == "resume"

    def provider_pause(reason, lost, failures):
        """Credit a provider-failed attempt's time and wait out a backoff outside the run; the agent never sees the outage."""
        nonlocal deadline, provider_held
        supervisor.credit(lost)
        deadline += lost
        timing["paused_total"] += lost
        wait = min(PROVIDER_BACKOFF * 2 ** failures, 300.)
        with frozen(), supervisor.hold():
            pause = timing["pause_start"] = time.monotonic()
            try:
                time.sleep(wait)
            finally:
                duration = time.monotonic() - pause
                deadline += duration
                timing["paused_total"] += duration
                timing["pause_start"] = None
        provider_held += lost + duration
        audit.record({"kind": "provider_hold", "reason": reason, "failed_attempt_seconds": lost, "backoff_seconds": duration,
                      "consecutive_failures": failures + 1, "provider_held_total_seconds": provider_held}, reservation=context["token"])

    def progress_note(text, source, origin, model=None):
        nonlocal progress_used
        if not isinstance(text, str) or not text.strip():
            raise RuntimeFailure("Progress text must be nonempty")
        if source == "agent" and len(text) > 4096:
            raise RuntimeFailure("Progress text exceeds character limit")
        event = {**clocks(), "utc_time": datetime.now(timezone.utc).isoformat(), "kind": "progress",
                 "source": source, "origin": origin, "text": text, "model": model or config.get("agent_model") or config.get("inference", {}).get("model"),
                 "action_id": context["action_id"], "world": context["world"]}
        raw = json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
        size = len(raw.encode())
        if progress_used + size > int(limits.get("progress_bytes", 8 * 1024 * 1024)):
            raise RuntimeFailure("Progress log byte budget exhausted")
        audit.record(event, reservation=context["token"])
        progress.write(raw)
        progress.flush()
        os.fsync(progress.fileno())
        progress_used += size

    def canonical_world(request):
        value = request.get("world", "reference")
        if value == "real":
            value = "reference"
        if not isinstance(value, str) or value not in ("reference", "reconstruction"):
            raise RuntimeFailure("world must be reference or reconstruction")
        return value

    def image_hashes(observation):
        return {key + "_sha256": hashlib.sha256(base64.b64decode(observation[key])).hexdigest()
                for key in ("rgb", "depth", "wrist_rgb", "wrist_depth") if isinstance(observation.get(key), str)}

    def save_checkpoint(world, state, observation, kind="checkpoint", token=None):
        prior = cache.get(world) or {}
        cache[world] = {"state": state, "observation": observation if observation is not None else prior.get("observation")}
        audit.record({"kind": kind, "world": world, "action_id": context["action_id"],
                      "episode": episodes[world], "world_total_ticks": charged[world],
                      "shared_charged_ticks": sum(charged.values()),
                      "model_digest": manifest["source_bundle_sha256"] if world == "reference" else candidate_version,
                      "state": state,
                      "state_sha256": state_digest(state), "observation": observation,
                      **image_hashes(observation or {})}, reservation=token)
        recorder = candidate_video if world == "reconstruction" else video
        if recorder and observation is not None and "rgb" in observation:
            recorder.push(observation, charged[world], episodes[world], final=(kind != "tick"))

    def recording_callback(simulation):
        nonlocal next_reference_frame
        charged["reference"] = int(simulation.total_ticks)
        now = float(simulation._data.time) if hasattr(simulation, "_data") else simulation.total_ticks * public_config["timestep"]
        due = now + 1e-10 >= next_reference_frame
        expired = time.monotonic() >= min(deadline, supervisor.deadline)
        observation = full_observation(simulation) if due or expired else None
        if due:
            while next_reference_frame <= now + 1e-10:
                next_reference_frame += 1 / fps
        save_checkpoint("reference", snapshot_simulation(simulation), observation, "tick", context["token"])
        if expired:
            supervisor.check()
            raise RuntimeFailure("Active wall-time budget exhausted")
    if not realtime:  # the real-time world records each action's state and every frame it delivers, not every tick
        sim._callback = recording_callback

    @contextmanager
    def frozen():
        """Host-side work outside the run: the agent and candidate containers (and a real-time world) are paused."""
        to_pause = [name] + ([candidate.name] if candidate and not candidate.closed else [])
        with (paused_containers(to_pause, sandbox.get("process_limit_backend", "cgroup")) if backend is None else backend.paused(to_pause)), \
             (live.paused() if live else nullcontext()):
            yield

    def boundary(direction, raw, complete):
        if direction == "request":
            context["action_id"] += 1
        event = {"kind": "boundary", "direction": direction, "world": context["world"],
                 "action_id": context["action_id"], "complete": complete,
                 "wire_b64": base64.b64encode(raw).decode(), "world_ticks": dict(charged)}
        try:
            payload = json.loads(raw)
            json.dumps(payload, allow_nan=False)
            event["payload"] = payload
            if direction == "request" and isinstance(payload, dict):
                try:
                    context["world"] = canonical_world(payload)
                except RuntimeFailure:
                    context["world"] = "invalid"
                if payload.get("op") == "load_reconstruction":
                    context["world"] = "reconstruction"
                event["world"] = context["world"]
        except (ValueError, UnicodeError):
            pass
        token = context["incoming"] if direction == "request" else context["token"]
        audit.record(event, reservation=token)

    def reserve_for(request, world, ticks):
        state = cache.get(world, {}) or {}
        state_bytes = len(json.dumps(state.get("state", {})))
        pixels = public_config["width"] * public_config["height"]
        if request.get("op") == "load_reconstruction":
            size = 64 * 1024 * 1024
        elif request.get("op") == "inference":
            size = int(config.get("inference", {}).get("max_response_bytes", 1048576)) * 4 + state_bytes * 3 + 262144
        elif request.get("op") == "progress":
            size = 262144
        elif request.get("op") == "sam3d":
            size = 3 * 1024 * 1024 * 4 + 262144  # the base64 mesh is recorded in the reply wire and action_end
        elif request.get("op") == "submit":
            size = 4 * 1024 * 1024 + state_bytes * 3
        else:
            factor = 16 if world == "reconstruction" else 8
            frames = int(ticks * public_config["timestep"] * fps + 1) + 4
            size = (ticks + 4) * (state_bytes * 6 + 8192) + frames * (pixels * factor + 65536)
        size += len(json.dumps(request)) * 2 + 65536
        return audit.reserve(size)

    try:
        if realtime:
            live = RealtimeWorld(sim, Simulation(scene, sim_config), sim_config["frame_rate"], launcher)
        with audit.reserve(public_config["width"] * public_config["height"] * 10 + 1024 * 1024) as initial:
            if live:
                entry = live.latest()
                save_checkpoint("reference", entry["state"], live.render(entry), token=initial)
            else:
                save_checkpoint("reference", snapshot_simulation(sim), full_observation(sim), token=initial)
        supervisor.check()
        max_message = int(limits.get("max_message_bytes", 12 * 1024 * 1024))
        robot_files = public_robot_files(source_root / "robot", sim_config["robot"]) if sim_config.get("robot") else None
        if robot_files is not None:
            audit.record({"kind": "public_fixture", "path": PUBLIC_ROBOT if backend is None else "robot",
                          "files_sha256": {relative: hashlib.sha256(data).hexdigest() for relative, data in robot_files.items()}})
        prior = config.get("continuation")
        prior_files = public_prior_files(prior["public_dir"]) if prior else None
        if prior_files is not None:
            assert backend is None, "continuation is strict_oci only"
            hashes = {relative: hashlib.sha256(data).hexdigest() for relative, data in prior_files.items()}
            if hashes != prior["files_sha256"]:
                raise RuntimeFailure("Public prior files changed after preparation")
            audit.record({"kind": "public_fixture", "path": PUBLIC_PRIOR, "files_sha256": hashes})
        tool_files = {PUBLIC_TOOLS + name: public_tool_files(name) for name in tools if AGENT_TOOLS[name][1] is not None}
        if "local_controller" in tools:
            tool_files[PUBLIC_TOOLS + "local_controller"]["public_tool_config.json"] = json.dumps(public_config, sort_keys=True).encode()
        for target, files in tool_files.items():
            audit.record({"kind": "public_fixture", "path": target, "files_sha256": {relative: hashlib.sha256(data).hexdigest() for relative, data in files.items()}})
        if backend is None:
            for files, target in ((robot_files, PUBLIC_ROBOT), (prior_files, PUBLIC_PRIOR), *((files, target) for target, files in tool_files.items())):
                if files is not None:
                    volumes.append((stage_public_volume(files, supervisor), target))
            proc = launch_container(sandbox["image"], sandbox["command"], sandbox, name, output / "agent.stderr.log", supervisor,
                                    mounts=tuple(volumes))
            stderr = proc.stderr_log
            transport = Transport(proc, max_message, on_receive=lambda raw, complete: boundary("request", raw, complete),
                                  on_send=lambda raw, complete: boundary("response", raw, complete))
        else:
            transport = backend.agent_transport(max_message, on_receive=lambda raw, complete: boundary("request", raw, complete),
                                                on_send=lambda raw, complete: boundary("response", raw, complete))
        task = {"type": "task", "protocol": 1,
                "execution_mode": "strict_oci" if backend is None else "development_shared_filesystem",
                "agent_model": config.get("agent_model") or config.get("inference", {}).get("model"),
                "model_usage": {"tokens_available": backend is None},
                "deadline_unix": time.time()+supervisor.remaining() if supervisor.deadline != float("inf") else None,
                "task": config.get("task", "Reconstruct the observed scene as a MuJoCo XML project.") + "".join("\n" + AGENT_TOOLS[name][0] for name in tools),
                "actions": ["observe", "grasp", "move", "release", "reset", "load_reconstruction", "submit", "inference", "progress", "finish"],
                "worlds": {"default": "reference", "available": ["reference", "reconstruction"],
                           "reconstruction": "Load your own ZIP project independently of scoring; each world has independent state."},
                "action_schema": {"observe": {"ticks": "optional nonnegative integer"},
                    "grasp": {"pixel": "[u,v] from top left; pin first visible movable point"},
                    "move": {"displacement": "[dx,dy,dz] world meters; translate pinned point allowing physical rotation", "ticks": sim_config["default_ticks"]},
                    "release": {"ticks": "optional nonnegative integer"}, "reset": "Restore selected world; budgets never refunded",
                    "load_reconstruction": {"project_b64": "base64 ZIP", "entrypoint": "scene.xml", "camera": "own authored camera, default fixed"},
                    "submit": {"project_b64": "base64 ZIP; scores stay private"},
                    "inference": {"messages": "system/user/assistant(+tool_calls)/tool(+tool_call_id) messages; text or inline PNG/JPEG parts",
                                  "tools": "optional [{name, description, parameters: JSON object schema}]"},
                    "progress": {"text": "concise public progress note, at most4096characters", "origin": "model, adapter or agent; never hidden reasoning"}, "finish": "end run"},
                "public_tool_config": public_config, "seconds_per_tick": public_config["timestep"],
                "observation_contract": {"rgb": "base64 RGB8 PNG", "depth": "base64 uint16 PNG millimeters",
                    "calibration": "intrinsics plus world_from_camera, -Z forward",
                    "wrench": "optional world object-on-handle N/N m at grasp point; ideal point torque zero",
                    "modalities": {k: sim_config[k] for k in ("rgb", "depth", "calibration", "wrench")}},
                # Spend caps (calls/tokens/USD) stay host-enforced and are not shown to the model: a visible cap could
                # shape behaviour and differed across waves (w1 $58, w2 $44). Exhaustion arrives as budget_stop.
                "inference": {"model": config.get("inference", {}).get("model"), "enabled": bool(config.get("inference", {}).get("endpoint")),
                              "max_output_tokens": int(config.get("inference", {}).get("max_output_tokens", 4096)),
                              "max_request_bytes": int(config.get("inference", {}).get("max_request_bytes", 4*1024*1024))},
                "budgets": {"wall_only": wall_only, "active_seconds": seconds, "wall_seconds": config.get("budget", {}).get("wall_seconds"),
                            "remaining_wall_seconds": supervisor.remaining() if supervisor.deadline != float("inf") else None,
                            "physics_ticks": max_ticks, "physics_ticks_scope": "shared across both worlds; resets/loads never refund; uncertain worker failure conservatively charges requested ticks",
                            "submissions": max_submissions, "audit_bytes": audit.limit},
                "submission": {"format": "base64 ZIP", "entrypoint": config.get("evaluation", {}).get("entrypoint", "scene.xml"),
                               "max_bytes": int(limits.get("submission_bytes", 4 * 1024 * 1024))},
                "observation": public_observation(cache["reference"]["observation"], sim_config)}
        if sim_config.get("movable_camera"):
            task["actions"].insert(task["actions"].index("reset"), "camera")
            task["action_schema"]["camera"] = {"pos": "[x,y,z] world meters", "lookat": "[x,y,z] world meters; up is +z",
                                               "effect": "re-aims the selected world's observation camera and returns an observation; "
                                                         "no physics time passes; the pose persists across resets",
                                               "limits": "each coordinate within +-4 m; pos and lookat at least 0.05 m apart"}
        if "sequence" in tools:
            task["actions"].insert(task["actions"].index("reset"), "sequence")
            task["action_schema"]["sequence"] = {
                "world": "reference or reconstruction; every step runs in this world",
                "steps_file": "sandbox path of a JSON array of 1 to 2000 action objects, each {\"op\": observe, grasp, move, release, "
                              "arm_target, hand_target, camera or reset (as offered), plus that action's own fields, no world}",
                "images": "last (default): return the final step's images; none: return none",
                "execution": "the sandbox harness sends the steps in order as ordinary controller actions, each charged and recorded as "
                             "usual, and stops at the first rejected step",
                "outputs": "/workspace/sequences/NNNN.json: per step its index, op and all non-image observation fields"}
        if fixture_table:
            from .fixture import PREDICTOR_FUNCTIONS
            from .predictor import CONTRACT
            for op in FIXTURE_OPS:
                task["actions"].insert(task["actions"].index("reset"), op)
            task["action_schema"]["reset_carriage"] = (
                "reference world only: the fixture's cocking cycle. It retracts the plunger to the compression set on the compression "
                "dial, latches it with the carriage against it, sets the brake pad's spring preload from the pressure dial, and re-seats "
                "the pad, which settles for a fixed number of physics ticks (charged); the dials and the robot are untouched. Dial "
                "changes take effect at the next reset_carriage")
            task["action_schema"]["release_carriage"] = {"ticks": "optional nonnegative integer, physics ticks to advance after the release",
                                                         "effect": "reference world only: opens the latch of a cocked carriage (after reset_carriage)"}
            task["observation_contract"]["fixture"] = ("fixture instrument readings: compression_dial and pressure_dial positions in scale "
                                                      "divisions (0 to 10, as on the dial scales) and carriage_cocked")
            if predictor_metric == "fixture":
                task["submission"]["predictor"] = {**CONTRACT, "functions": PREDICTOR_FUNCTIONS}
            if "sequence" in task["action_schema"]:
                sequence = task["action_schema"]["sequence"]
                sequence["steps_file"] = sequence["steps_file"].replace("camera or reset", "camera, reset, reset_carriage or release_carriage")
                assert "release_carriage" in sequence["steps_file"]
        if sam3d_relay:
            task["actions"].insert(task["actions"].index("progress"), "sam3d")
            task["action_schema"]["sam3d"] = {"image_png_b64": "8-bit RGB(A) PNG, at most 2048x2048", "mask_png_b64": "grayscale PNG, same size, nonzero = object",
                                              "seed": "optional integer, default 0", "returns": "mesh_ply_b64 (binary PLY, vertex colors), vertices, faces, layout, frame"}
        if robot:
            fixture, robot_ops = ROBOTS[robot][0], ROBOTS[robot][2]
            task["actions"] = [op for op in task["actions"] if op not in ("grasp", "move", "release")]
            task["actions"][1:1] = robot_ops
            for op in ("grasp", "move", "release"):
                task["action_schema"].pop(op)
            for robot_op in robot_ops:
                task["action_schema"][robot_op] = {
                    "arm_target": {"joint_targets": "seven finite joint position targets in radians, Panda joints 1 through 7",
                                   **({"gripper_width_m": "finite parallel-finger opening width in meters, 0 through 0.08"} if robot in GRIPPER else {})},
                    "hand_target": {"actuator_targets": "20 finite position targets in this actuator order: " + ", ".join(HAND_ACTUATORS)
                                    + f"; each within that actuator's ctrlrange in robot/{fixture} (radians; a coupled *J0 target is "
                                      "the sum of its finger's J1 and J2 angles); interpolated linearly from the current targets over the ticks"},
                }[robot_op] | {"ticks": "positive integer; default " + str(sim_config["default_ticks"])}
            task["action_schema"]["load_reconstruction"]["robot"] = f"Include public robot/{fixture} using <include file=\"robot/{fixture}\"/>; preserve the supplied robot and copy robot/assets with it"
            task["public_robot_path"] = PUBLIC_ROBOT
            if backend is not None:
                robot_destination = Path(backend.public_dir) / "robot"
                robot_destination.mkdir(exist_ok=False)
                for relative, data in robot_files.items():
                    (robot_destination / relative).parent.mkdir(parents=True, exist_ok=True)
                    (robot_destination / relative).write_bytes(data)
                task["public_robot_path"] = str(robot_destination)
            if sim_config.get("wrist_camera"):
                task["observation_contract"]["wrist_camera"] = "wrist_rgb, wrist_depth and wrist_calibration follow enabled global modalities; camera moves with Panda hand, same dimensions and formats"
            task["observation_contract"].pop("wrench", None)
            effort = sim_config.get("effort", True)
            if robot == "panda":
                tactile = ("left/right finger: pad_normal_force_n 3x3 taxels over the fingertip pad (rows distal to proximal, columns "
                           "across the finger) and finger_normal_force_n over the whole finger, newtons; any contact on the finger counts")
                proprioception = ("joint_positions_rad[7], joint_velocities_rad_s[7], " + ("actuator_effort_nm[7], " if effort else "")
                                  + "finger_positions_m[2], finger_velocities_m_s[2], gripper_width_m, gripper_opening (0..1); robot sensors only")
            else:
                tactile = ("thumb, first, middle, ring, little: pad_normal_force_n 3x3 taxels over the palmar side (body y <= 1 mm) of that "
                           "finger's distal phalanx body (rh_thdistal, rh_ffdistal, ...), body x -9.5..9.5 mm in 3 columns (increasing x) by "
                           "body z 34..4 mm in 3 rows (distal first), and total_normal_force_n over the whole distal phalanx; palm: "
                           "pad_normal_force_n 4x3 taxels over the palmar face of rh_palm (body y <= -4 mm), body z 96..-12 mm in 4 rows "
                           "(distal first) by body x -42..44 mm in 3 columns (increasing x), and total_normal_force_n over the whole palm "
                           "body; contact normal force in newtons summed per taxel; any contact on the body counts")
                names = joints(robot)
                proprioception = (f"joint_positions_rad[{len(names)}] and joint_velocities_rad_s[{len(names)}] in joint order " + ", ".join(names)
                                  + ({("hand_target",): "; actuator_effort_nm[20] in hand_target actuator order",
                                      ("arm_target", "hand_target"): "; actuator_effort_nm[27]: Panda joints 1 through 7 (arm_target order), then "
                                                                     "the hand_target actuator order"}[robot_ops] if effort else "") + "; robot sensors only")
            if sim_config["tactile"]:
                task["observation_contract"]["tactile"] = tactile
                task["observation_contract"]["modalities"]["tactile"] = True
            task["observation_contract"]["proprioception"] = proprioception
        if realtime:
            task["actions"] = [op for op in task["actions"] if op != "load_reconstruction"]
            for op in ("frames",) + (("throw",) if launcher else ()):
                task["actions"].insert(task["actions"].index("reset"), op)
            task["worlds"] = {"default": "reference", "available": ["reference"]}
            task["action_schema"].pop("load_reconstruction")
            task["action_schema"].update(
                observe="returns the latest camera frame; no ticks: the world advances on its own",
                frames={"since": "frame index; returns kept frames with a larger index, oldest first (default -1: all kept)",
                        "limit": "1 to 100 frames, default 25", "wait": "waits up to 1 s for the next frame when none is newer"},
                reset="restores the initial state (robot home, grasp released)")
            if launcher:
                task["action_schema"]["throw"] = {"protocol": "one of " + ", ".join(launcher.protocols) + " (task launcher_protocols)",
                                                  "effect": "resets the world (robot home, grasp released), then the launcher releases the "
                                                            + launcher.body + " with that protocol's release state"}
                task["launcher_protocols"] = launcher.public()
            if "release" in task["action_schema"]:
                task["action_schema"]["release"] = "releases the grasp; no ticks"
            task["realtime"] = {
                "clock": "the world advances continuously at one simulated second per wall-clock second, also while you reason "
                         "or your code runs; it cannot be paused, slowed, stepped or saved",
                "actions": "actions take effect in arrival order at the next physics tick; a timed action (ticks = its duration in "
                           "physics ticks) occupies those ticks and later actions wait for it to finish, so chained actions "
                           "(for example move then release) take effect back to back; each reply comes when its action takes effect",
                "frames": "camera frames exist only every 1/frame_rate simulated seconds since the latest reset or throw (frame 0 is "
                          "the moment of the reset or release); every image is one of them; each carries its frame index, episode "
                          "(count of resets and throws) and time since that episode began; the latest %d s are kept" % round(live.kept.maxlen / sim_config["frame_rate"]),
                "frame_rate_hz": sim_config["frame_rate"], "seconds_per_tick": public_config["timestep"]}
            task["observation_contract"].update(frame="global frame index", episode="resets and throws so far", time="seconds since the episode began")
        if die_metric:
            from .die import public_contract
            task["die"] = public_contract(sim, launcher, {**private_manifest["die"], **config["evaluation"].get("die", {})}, sim_config["frame_rate"])
            task["submission"]["protocol"] = "task die.predictor"
        if predictor_metric:
            task["submission"]["kind"] = "predictor"
            task["action_schema"]["submit"] = {"project_b64": "base64 ZIP of your predictor program (%s and its files); scores stay private"
                                                              % task["submission"]["entrypoint"]}
        if metrics == ["lowrank"]:  # a parameter answer (evaluation._lowrank), not a scene
            task["submission"]["kind"] = "parameters"
            task["action_schema"]["submit"] = {"project_b64": "base64 ZIP holding your parameter answer %s; scores stay private"
                                                              % task["submission"]["entrypoint"]}
        if "channel" in tools:
            task["code_channel"] = {
                "path": CHANNEL_SOCKET, "availability": "only while a bash command runs; one connection at a time",
                "protocol": "connect a Unix stream socket; send one JSON request per line, exactly as a controller action "
                            "({\"op\": ..., fields}), and read one JSON reply per line (the controller's reply, with base64 images)",
                "ops": [op for op in task["actions"] if op in PHYSICAL + ("frames", "throw")],
                "python": "from sim2sim.client import Channel; channel = Channel(); reply = channel.request('observe')"}
        if not sim_config["reference_interaction"]:  # no-action ablation: the reference world can only be watched
            blocked = [op for op in INTERACTIONS + FIXTURE_OPS if op in task["actions"]]
            task["worlds"]["reference"] = ("Observe-only: " + ", ".join(blocked) + " are unavailable in the reference world; "
                                           "observe and reset work there, and every action works in the reconstruction world.")
            for op in blocked:
                task["action_schema"][op] = {**task["action_schema"][op], "world": "reconstruction only"}
        with audit.reserve(len(json.dumps(task)) * 3 + 65536) as token:
            context["token"] = token
            transport.send(task, min(deadline, supervisor.deadline))
        context["token"] = None
        while True:
            # Reserve the entire bounded receive before reading even malformed input.
            with audit.reserve(max_message * 3 + 65536) as incoming:
                context["incoming"] = incoming
                request = transport.receive(min(deadline, supervisor.deadline))
            context["incoming"] = None
            op = request.get("op")
            token = None
            before = dict(charged)
            before_episode = dict(episodes)
            before_candidate_worker_ticks = candidate_worker_ticks
            action_started = False
            outcome = "rejected"
            world = "reference"
            response = None
            try:
                world = canonical_world(request)
                if op == "load_reconstruction":
                    world = "reconstruction"
                context["world"] = world
                if not isinstance(op, str):
                    raise RuntimeFailure("Operation must be a string")
                if op in INTERACTIONS and op not in (ROBOTS[robot][2] if robot else ("grasp", "move", "release")):
                    raise RuntimeFailure("Operation unavailable for configured robot")
                if op in FIXTURE_OPS and (not fixture_table or world != "reference"):
                    raise RuntimeFailure("Unknown operation" if not fixture_table else f"{op} works only in the reference world")
                if world == "reference" and op in INTERACTIONS + FIXTURE_OPS and not sim_config["reference_interaction"]:
                    raise RuntimeFailure("The reference world is observe-only: %s works only in the reconstruction world" % op)
                ticks = request.get("ticks", sim_config["default_ticks"] if op in ("move", "arm_target", "hand_target") else 0)
                if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 0:
                    raise RuntimeFailure("ticks must be a nonnegative integer")
                if op in ("move", "arm_target", "hand_target") and ticks < 1:
                    raise RuntimeFailure(f"{op} requires at least one tick")
                if max_ticks is not None and ticks > max_ticks - sum(charged.values()):
                    raise RuntimeFailure("Shared cumulative physics tick budget exceeded")
                if op not in ("observe", "move", "release", "arm_target", "hand_target", "release_carriage") and ticks:
                    raise RuntimeFailure("This operation does not accept ticks")
                if op == "camera" and not sim_config.get("movable_camera") or op == "frames" and not realtime or op == "throw" and not launcher:
                    raise RuntimeFailure("Unknown operation")
                if realtime and (op == "load_reconstruction" or world != "reference"):
                    raise RuntimeFailure("The real-time world has no reconstruction world")
                if realtime and op in ("observe", "release") and ticks:
                    raise RuntimeFailure("The real-time world advances on its own: ticks only set the duration of a timed action")
                resolved = {k: request[k] for k in ("pixel", "displacement", "joint_targets", "gripper_width_m", "actuator_targets", "pos", "lookat",
                                                    "protocol", "since", "limit") if k in request}
                resolved.update(op=op, world=world, ticks=ticks)
                if op == "load_reconstruction":
                    resolved.update(entrypoint=request.get("entrypoint", "scene.xml"), camera=request.get("camera", "fixed"))
                token = reserve_for(request, world, ticks + (sim.fixture.settle_ticks if op == "reset_carriage" else 0))
                context["token"] = token
                private = cache.get(world) or {}
                audit.record({"kind": "action_begin", "action_id": context["action_id"], "world": world,
                              "resolved_action": resolved, "candidate_version": candidate_version,
                              "before_world_ticks": before, "before_episode": before_episode,
                              "before_state": private.get("state"), "before_observation": private.get("observation")}, reservation=token)
                action_started = True
                if time.monotonic() >= min(deadline, supervisor.deadline):
                    supervisor.check()
                    raise RuntimeFailure("Active wall-time budget exhausted")
                if live and op == "frames":
                    since, count = request.get("since", -1), request.get("limit", 25)
                    if type(since) is not int or type(count) is not int or not 1 <= count <= 100:
                        raise RuntimeFailure("frames needs an integer since and a limit from 1 to 100")
                    entries = live.frames(since, count, min(time.monotonic() + 1, deadline, supervisor.deadline))
                    response = {"ok": True, "world": world, "frames": [public_observation(live.render(entry), sim_config) for entry in entries]}
                elif live and op in ("observe", "grasp", "move", "release", "reset", "arm_target", "hand_target", "camera", "throw"):
                    action = {k: v for k, v in resolved.items() if k not in ("world", "ticks")} | ({"ticks": ticks} if ticks else {})
                    if op == "observe":
                        entry = live.latest()
                    else:
                        state, entry = live.command(action, min(deadline, supervisor.deadline))
                        save_checkpoint(world, state, None, "applied", token)  # the state when the action took effect
                        charged[world] = int(state["total_ticks"])
                    if op in ("reset", "throw"):
                        episodes[world] += 1
                    complete_observation = live.render(entry)
                    save_checkpoint(world, entry["state"], complete_observation, token=token)
                    response = {"ok": True, "world": world, "observation": public_observation(complete_observation, sim_config), "remaining_ticks": None}
                elif op in PHYSICAL:
                    action = {k: v for k, v in resolved.items() if k != "world"}
                    if world == "reference":
                        if op == "reset":
                            if video:
                                video.finish()
                            episodes[world] += 1
                            next_reference_frame = 1 / fps
                        if hasattr(sim, "_model"):
                            sim.action(action, render=False)
                        else:
                            sim.action(action)
                        charged[world] = int(sim.total_ticks)
                        complete_observation = full_observation(sim)
                        public_observation_result = public_observation(complete_observation, sim_config)
                        save_checkpoint(world, snapshot_simulation(sim), complete_observation, token=token)
                    else:
                        if candidate is None or candidate.closed:
                            raise RuntimeFailure("Load a reconstruction before using that world")
                        def on_tick(packet):
                            nonlocal candidate_worker_ticks
                            state = packet["state"]
                            actual = int(state["total_ticks"])
                            delta = actual - candidate_worker_ticks
                            if delta != 1 or actual - before_candidate_worker_ticks > ticks:
                                raise RuntimeFailure("Invalid candidate tick accounting")
                            candidate_worker_ticks = actual
                            charged[world] += delta
                            if packet.get("state_sha256") != state_digest(state):
                                raise RuntimeFailure("Candidate state checksum mismatch")
                            save_checkpoint(world, state, packet.get("full_observation"), "tick", token)
                            if time.monotonic() >= min(deadline, supervisor.deadline):
                                supervisor.check()
                                raise RuntimeFailure("Active wall-time budget exhausted")
                        try:
                            packet = candidate.exchange({"op": "action", "action": action}, min(deadline, supervisor.deadline), token, context["action_id"], on_tick)
                        except BaseException:
                            # A killed native step cannot be proved to have completed. Charge
                            # the entire requested bound; preserve exact confirmed snapshots.
                            charged[world] += max(0, ticks - (charged[world] - before[world]))
                            raise
                        if packet.get("state") is None or packet.get("full_observation") is None:
                            candidate.close()
                            charged[world] += max(0, ticks - (charged[world] - before[world]))
                            raise RuntimeFailure("Candidate final checkpoint is unavailable")
                        if packet.get("state_sha256") != state_digest(packet["state"]) or int(packet["state"]["total_ticks"]) != candidate_worker_ticks:
                            candidate.close()
                            charged[world] += max(0, ticks - (charged[world] - before[world]))
                            raise RuntimeFailure("Candidate final state or tick checksum mismatch")
                        if op == "reset" and packet.get("ok"):
                            episodes[world] += 1
                        if packet.get("state") is not None:
                            save_checkpoint(world, packet["state"], packet["full_observation"], token=token)
                        if not packet.get("ok"):
                            raise RuntimeFailure(packet.get("error", "Candidate operation failed"))
                        public_observation_result = public_observation(packet["observation"], public_config)
                    response = {"ok": True, "world": world, "observation": public_observation_result,
                                "remaining_ticks": None if max_ticks is None else max_ticks - sum(charged.values())}
                elif op == "load_reconstruction":
                    project = decode_submission(request.get("project_b64"), int(limits.get("submission_bytes", 4 * 1024 * 1024)))
                    project_version = hashlib.sha256(project).hexdigest()
                    audit.record({"kind": "candidate_load_intent", "world": world, "action_id": context["action_id"],
                                  "version": project_version, "entrypoint": resolved["entrypoint"],
                                  "camera": resolved["camera"], "public_config": public_config}, reservation=token)
                    if candidate:
                        candidate.close()
                    if candidate_video:
                        candidate_video.finish()
                    supervisor.check()
                    candidate_output = output / ("candidate-" + project_version[:12] + ".stderr.log")
                    candidate = CandidateSession(sandbox, candidate_output, audit) if backend is None else backend.candidate(candidate_output, audit)
                    candidate.supervisor = supervisor
                    if hasattr(candidate, "proc"):
                        supervisor.register(candidate.proc, candidate.name)
                    candidate_worker_ticks = 0
                    own_config = dict(public_config)
                    own_config["recording_fps"] = fps
                    own_config["camera"] = resolved["camera"]
                    packet = candidate.exchange({"op": "load", "project_b64": request["project_b64"],
                                                 "entrypoint": resolved["entrypoint"], "config": own_config},
                                                min(deadline, supervisor.deadline), token, context["action_id"])
                    if not packet.get("ok"):
                        candidate.close()
                        raise RuntimeFailure("Candidate project could not be loaded")
                    if packet.get("state_sha256") != state_digest(packet.get("state")) or packet["state"]["total_ticks"] != 0:
                        candidate.close()
                        raise RuntimeFailure("Candidate initial checkpoint is invalid")
                    candidate_version = project_version
                    candidate_changes += 1
                    candidate_change_wall = clocks()["wall_elapsed"]
                    episodes[world] += 1
                    save_checkpoint(world, packet["state"], packet["full_observation"], token=token)
                    if config.get("evaluate_candidate_changes", False):
                        evaluation = dict(config["evaluation"])
                        evaluation.setdefault("source_dir", str(source_root))
                        evaluation.setdefault("source_entrypoint", Path(scene).resolve().relative_to(source_root).as_posix())
                        evaluation.setdefault("source_config", dict(sim_config))
                        evaluation["entrypoint"] = resolved["entrypoint"]
                        change_output = output / ("candidate-change-%04d" % candidate_changes)
                        with frozen():
                            pause = time.monotonic()
                            timing["pause_start"] = pause
                            try:
                                try:
                                    evaluator = evaluate_in_container if backend is None else backend.evaluate
                                    report = evaluator(project, evaluation, sandbox, change_output, source_bundle, supervisor)
                                except Exception as exc:
                                    report = {"status": "unevaluated" if supervisor.expired.is_set() else "failed", "version": project_version, "scores": None,
                                              "error_type": type(exc).__name__, "error": "Candidate change evaluation failed"}
                                    change_output.mkdir(parents=True, exist_ok=True)
                                    (change_output / "evaluation.json").write_text(json.dumps(report, indent=2))
                                audit.record({"kind": "evaluation", "checkpoint_kind": "candidate_change",
                                              "change_index": candidate_changes, "version": project_version,
                                              "wall_elapsed": candidate_change_wall,
                                              "evaluated_at": datetime.now(timezone.utc).isoformat(),
                                              "action_id": context["action_id"], "world": "researcher", "report": report}, reservation=token)
                            finally:
                                duration = time.monotonic() - pause
                                deadline += duration
                                timing["paused_total"] += duration
                                timing["pause_start"] = None
                    response = {"ok": True, "world": world, "version": candidate_version,
                                "observation": public_observation(packet["observation"], public_config), "remaining_ticks": None if max_ticks is None else max_ticks - sum(charged.values())}
                elif op == "inference":
                    stopped = False
                    if checkpoint is not None and clocks()["wall_elapsed"] >= float(checkpoint):
                        checkpoint, stopped = None, not budget_hold("Wall-time checkpoint reached")
                    failures = 0
                    while not stopped:
                        attempt = time.monotonic()
                        try:
                            inference = broker.call(request, min(deadline, supervisor.deadline) - time.monotonic())
                        except ProviderUnavailable as exc:
                            if provider_hold is None or provider_held >= float(provider_hold):
                                raise
                            provider_pause(str(exc), time.monotonic() - attempt, failures)
                            failures += 1
                            continue
                        except BudgetStop as exc:
                            if hold_seconds is not None and str(exc) in HOLDABLE and budget_hold(str(exc)):
                                continue
                            # Spend control belongs to the orchestrator: the run is suspended at its latest checkpoint.
                            # The agent side learns only that the host stopped inference.
                            suspended = str(exc)
                            audit.record({"kind": "inference_suspended", "reason": suspended})
                            inference = {"budget_stop": True, "reason": "host stop"}
                        break
                    if stopped:
                        suspended = "Wall-time checkpoint stop"
                        audit.record({"kind": "inference_suspended", "reason": suspended})
                        inference = {"budget_stop": True, "reason": "host stop"}
                    if not inference.get("budget_stop"):
                        audit.record({"kind": "inference_accounting", **{key: inference.get(key) for key in INFERENCE_ACCOUNTING}})
                        for summary in inference.get("summaries", []):
                            progress_note(summary, "provider_summary", "provider", inference.get("model"))
                        # Only what the harness needs enters the container: no cost, cap or remaining-budget fields.
                        inference = {"content": inference["content"], "tool_calls": inference["tool_calls"],
                                     "usage": {key: inference["usage"][key] for key in CONTEXT_USAGE} if inference.get("usage") else None}
                    # The harness re-anchors its wall clock on every reply, so a budget hold never shortens the agent's time.
                    inference["deadline_unix"] = time.time() + supervisor.remaining() if supervisor.deadline != float("inf") else None
                    supervisor.check()
                    response = {"ok": True, "inference": inference}
                elif op == "sam3d" and sam3d_relay:
                    result, call = sam3d_relay.call(request, min(deadline, supervisor.deadline), supervisor.check)
                    audit.record(call, reservation=token)
                    if result is None:
                        raise RuntimeFailure(call["error"])
                    response = {"ok": True, "sam3d": result}
                elif op == "progress":
                    origin = request.get("origin", "agent")
                    if origin not in ("agent", "model", "adapter"):
                        raise RuntimeFailure("Invalid public progress origin")
                    progress_note(request.get("text"), "agent", origin)
                    response = {"ok": True, "status": "recorded"}
                elif op == "submit":
                    if max_submissions is not None and submissions >= max_submissions:
                        raise RuntimeFailure("Submission budget exhausted")
                    submission = decode_submission(request.get("project_b64"), int(limits.get("submission_bytes", 4 * 1024 * 1024)))
                    submissions += 1
                    (output / ("submission-%03d.zip" % submissions)).write_bytes(submission)
                    version_output = output / ("version-%03d" % submissions)
                    evaluation = dict(config["evaluation"])
                    evaluation.setdefault("source_dir", str(Path(scene).resolve().parent))
                    evaluation.setdefault("source_entrypoint", Path(scene).resolve().relative_to(source_root).as_posix())
                    evaluation.setdefault("source_config", dict(sim_config))
                    for key in ("camera", "timestep", "grasp_solref", "grasp_solimp"):
                        if key in sim_config:
                            evaluation.setdefault(key, sim_config[key])
                    with frozen():
                        pause = time.monotonic()
                        timing["pause_start"] = pause
                        try:
                            try:
                                scene_evaluator = evaluate_in_container if backend is None else backend.evaluate
                                if predictor_metric:
                                    if backend is not None:
                                        raise RuntimeFailure("Predictor evaluation requires strict_oci")
                                    from .predictor import Job
                                    job = Job(submission, evaluation, output, version_output, output / ("predictor-%03d" % submissions), scene, sim_config,
                                              private_manifest, clocks()["wall_elapsed"], sandbox, supervisor,
                                              lambda scored: scene_evaluator(submission, scored, sandbox, version_output, source_bundle, supervisor))
                                    report = importlib.import_module("." + PREDICTOR_METRICS[predictor_metric], __package__).evaluate_submission(job)
                                else:
                                    report = scene_evaluator(submission, evaluation, sandbox, version_output, source_bundle, supervisor)
                            except Exception as exc:
                                report = {"status": "unevaluated" if supervisor.expired.is_set() else "failed", "version": hashlib.sha256(submission).hexdigest(), "scores": None,
                                          "error_type": type(exc).__name__, "error": str(exc) if isinstance(exc, RuntimeFailure) else "Evaluation worker failed"}
                                version_output.mkdir(parents=True, exist_ok=True)
                                (version_output / "evaluation.json").write_text(json.dumps(report, indent=2))
                            from .plotting import update_metric_history
                            update_metric_history(output, submissions, report, render_plots=config.get("plot_checkpoints", True))
                            audit.record({"kind": "evaluation", "action_id": context["action_id"], "world": "researcher",
                                          "submission": submissions, "report": report}, reservation=token)
                            evaluated = submissions
                        finally:
                            duration = time.monotonic() - pause
                            deadline += duration
                            timing["paused_total"] += duration
                            timing["pause_start"] = None
                    response = {"ok": True, "submission": submissions, "status": report.get("status", "failed")}
                elif op == "finish":
                    response = {"ok": True, "status": "finished"}
                    termination, reason = "finished", "budget_suspended" if suspended else "agent_finish"
                else:
                    raise RuntimeFailure("Unknown operation")
                outcome = "ok"
            except (AuditFailure, SandboxFailure):
                raise
            except Exception as exc:
                if live and isinstance(exc, WorldFailure):
                    raise
                outcome = "error"
                visible = isinstance(exc, RuntimeFailure) or live is not None and isinstance(exc, ValueError)  # real-time input checks are public
                response = {"ok": False, "error": str(exc) if visible and op != "submit" else "Operation failed"}
                if world == "reference" and hasattr(sim, "total_ticks") and not live:
                    charged[world] = int(sim.total_ticks)
                    if action_started and op in PHYSICAL:
                        save_checkpoint(world, snapshot_simulation(sim), None, "interrupted", token)
            finally:
                if response is not None:
                    if token is None:
                        token = audit.reserve(256 * 1024)
                        context["token"] = token
                    private = cache.get(world) or {}
                    audit.record({"kind": "action_end" if action_started else "rejection", "action_id": context["action_id"],
                                  "world": world, "operation": op if isinstance(op, str) else None, "outcome": outcome,
                                  "before_world_ticks": before, "after_world_ticks": dict(charged),
                                  "before_episode": before_episode, "after_episode": dict(episodes),
                                  "candidate_version": candidate_version,
                                  "candidate_available": candidate is not None and not candidate.closed,
                                  "executed_ticks": (candidate_worker_ticks - before_candidate_worker_ticks) if world == "reconstruction" and op != "load_reconstruction" else charged.get(world, 0) - before.get(world, 0),
                                  "charged_ticks": charged.get(world, 0) - before.get(world, 0),
                                  "inflight_ticks_unconfirmed": max(0, charged.get(world, 0) - before.get(world, 0) - (candidate_worker_ticks - before_candidate_worker_ticks)) if world == "reconstruction" and op != "load_reconstruction" else 0,
                                  "after_state": private.get("state"), "response": response}, reservation=token)
            try:
                transport.send(response, min(deadline, supervisor.deadline))
                audit.record({"kind": "delivery_complete", "action_id": context["action_id"], "world": world}, reservation=token)
                events.write(json.dumps({**clocks(), "op": op, "world": world, "ok": response.get("ok"), "total_ticks": sum(charged.values())}) + "\n")
                events.flush()
            finally:
                if token:
                    token.release()
                context["token"] = None
            if termination == "finished":
                break
        return {"output_dir": str(output), "status": termination, "reason": reason,
                "submissions": submissions, "total_ticks": sum(charged.values()), "world_ticks": charged}
    except BaseException as exc:
        if transport is not None and (supervisor.expired.is_set() or supervisor.remaining() <= 0) and isinstance(exc, Exception) and not isinstance(exc, AuditFailure):
            # The wall deadline is a normal end once the agent ran: keep accepted work, and record
            # every accepted submission without a finished private evaluation as unevaluated.
            termination, reason = "finished", "wall_deadline"
            for index in range(evaluated + 1, submissions + 1):
                version_output = output / ("version-%03d" % index)
                report = {"status": "unevaluated", "version": hashlib.sha256((output / ("submission-%03d.zip" % index)).read_bytes()).hexdigest(),
                          "scores": None, "error": "Wall deadline arrived before private evaluation completed"}
                version_output.mkdir(parents=True, exist_ok=True)
                (version_output / "evaluation.json").write_text(json.dumps(report, indent=2))
                audit.record({"kind": "evaluation", "world": "researcher", "submission": index, "report": report}, reservation=emergency)
            return {"output_dir": str(output), "status": termination, "reason": reason,
                    "submissions": submissions, "total_ticks": sum(charged.values()), "world_ticks": charged}
        termination = "failed"
        reason = "Hard wall-time budget exhausted" if supervisor.expired.is_set() or supervisor.remaining() <= 0 else (str(exc) if isinstance(exc, (RuntimeFailure, AuditFailure)) else type(exc).__name__)
        if isinstance(exc, AuditFailure):
            raise RuntimeFailure(reason) from exc
        raise
    finally:
        if context["token"] is not None:
            context["token"].release()
        context["token"] = None
        if candidate:
            candidate.close()
        if transport is not None:
            context["incoming"] = emergency
            transport.audit_pending()
            context["incoming"] = None
        if proc is not None:
            stop_container(proc, name)
            supervisor.unregister(proc)
        for volume, _ in volumes:
            subprocess.run(["podman", "volume", "rm", "--force", volume], capture_output=True, timeout=15, env=podman_env())
        if backend is not None:
            backend.close_agent()
        realtime_stats = live.close() if live else None
        if live:
            charged["reference"] = realtime_stats["total_ticks"]
        supervisor.close()  # Child work has stopped; only private log/video finalization follows.
        if video:
            video.close()
        if candidate_video:
            candidate_video.close()
        status = {"status": termination, "reason": reason, "remaining_active_seconds": None if wall_only else max(0, deadline - time.monotonic()),
                  "total_ticks": sum(charged.values()), "world_ticks": charged, "remaining_ticks": None if max_ticks is None else max_ticks - sum(charged.values()), "submissions": submissions, "wall_elapsed": time.monotonic() - run_started - supervisor.held,
                  "execution_wall_elapsed": supervisor.finished_at - run_started - supervisor.held, "held_seconds": supervisor.held, "holds": holds,
                  "remaining_wall_seconds": supervisor.remaining() if supervisor.deadline != float("inf") else None,
                  "hard_wall_expired": supervisor.expired.is_set() or supervisor.remaining() <= 0,
                  **({"realtime_world": realtime_stats} if live else {})}
        (output / "termination.json").write_text(json.dumps(status, indent=2))
        try:
            audit.record({"kind": "termination", **status}, reservation=emergency)
            for path in sorted(output.rglob("*.mp4")):
                audit.record({"kind": "video", "world": "reconstruction" if path.parent.name == "reconstruction" else "reference",
                              "path": path.relative_to(output).as_posix(), "size": path.stat().st_size,
                              "sha256": file_digest(path)}, reservation=emergency)
        finally:
            emergency.release()
            audit.close()
            sim.close()
            events.close()
            progress.close()
            if stderr is not None:
                stderr.close()
