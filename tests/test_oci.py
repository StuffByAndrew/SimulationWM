"""Opt-in real sandbox check on a Slurm test node after loading the trusted image.

SIM2SIM_OCI_TEST=1 SIM2SIM_PROCESS_LIMIT_BACKEND=slurm_step sbatch -c 4 --mem=8G --export=ALL scripts/test.slurm tests/test_oci.py
"""
import hashlib
import json
import os
from pathlib import Path
import time
import unittest
import uuid
from unittest.mock import patch

BACKEND = os.environ.get("SIM2SIM_PROCESS_LIMIT_BACKEND", "cgroup")
IMAGE = os.environ.get("SIM2SIM_AGENT_IMAGE", "localhost/sim2sim:latest")
# The in-container agent proves the boundary from inside, then drives the public protocol.
AGENT = r'''
import base64, errno, io, os, resource, socket, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from sim2sim.client import Client
client = Client()
observation = client.observe()
assert os.getuid() == 65532 and "SIM2SIM_TEST_SECRET" not in os.environ and not Path("/opt/sim2sim/environments").exists()
assert resource.getrlimit(resource.RLIMIT_NPROC) == (64, 64)
assert "CapEff:\t0000000000000000" in Path("/proc/self/status").read_text()
for attempt in (lambda: resource.setrlimit(resource.RLIMIT_NPROC, (65, 65)), lambda: os.setuid(0),
                lambda: Path("/opt/forbidden").write_text("x"), lambda: socket.create_connection(("1.1.1.1", 53), timeout=.5)):
    try:
        attempt()
        raise AssertionError("sandbox escape succeeded")
    except (ValueError, OSError):
        pass
children = []
while len(children) < 70:
    try:
        pid = os.fork()
    except OSError as exc:
        assert exc.errno == errno.EAGAIN
        break
    if pid == 0:
        time.sleep(10)
        os._exit(0)
    children.append(pid)
assert len(children) < 64
for pid in children:
    os.kill(pid, 9)
    os.waitpid(pid, 0)
if sys.argv[1] == "slurm_step":  # the step cpuset bounds even a widened affinity request
    os.sched_setaffinity(0, range(os.cpu_count()))
    assert len(os.sched_getaffinity(0)) <= 1
image = np.asarray(Image.open(io.BytesIO(base64.b64decode(observation["rgb"]))))
red = (image[:, :, 0].astype(float) > 1.7 * image[:, :, 1]) & (image[:, :, 0].astype(float) > 1.7 * image[:, :, 2]) & (image[:, :, 0] > 90)
y, x = np.nonzero(red)
assert client.grasp([float(np.median(x)), float(np.median(y))])["grasped"]
client.move([0, 0, .1], ticks=10)
client.release(ticks=10)
client.observe(ticks=5)
client.reset()
client.observe(ticks=5)
project = Path("/workspace/project")
project.mkdir()
scene = """<mujoco model="guess"><option timestep="0.002"/><worldbody><light pos="0 0 3"/>
<camera name="fixed" pos="1.2 -1.5 1.2" xyaxes="0.78 0.62 0 -0.35 0.44 0.83"/><geom name="floor" type="plane" size="2 2 0.1"/>
<body name="other" pos="0.15 0.1 0.1"><freejoint/><geom type="box" size="0.07 0.05 0.05" mass="0.3" rgba="0.1 0.3 0.8 1"/></body>
<body name="object" pos="0 0 0.12"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="MASS" rgba="0.8 0.2 0.2 1"/></body>
</worldbody></mujoco>"""
for mass in ("0.5", "0.6"):
    (project / "scene.xml").write_text(scene.replace("MASS", mass))
    client.submit(project)
client.finish()
'''


@unittest.skipUnless(os.environ.get("SIM2SIM_OCI_TEST") == "1", "requires explicit rootless OCI integration opt-in")
class OCIIntegrationTests(unittest.TestCase):
    def test_full_agent_and_private_evaluator(self):
        import imageio.v2 as imageio
        from sim2sim.runtime import run
        root = Path(__file__).resolve().parents[1]
        # Kept for inspection: public test scene only, no credentials.
        output = root / "experiments/validation/sandbox-backend" / ("oci-" + os.environ.get("SLURM_JOB_ID", "local") + "-" + uuid.uuid4().hex[:8])
        config = {
            "source": {"scene": str(root / "environments/blocks/scene.xml")},
            "sandbox": {"process_limit_backend": BACKEND, "image": IMAGE,
                        "evaluator_image": os.environ.get("SIM2SIM_EVALUATOR_IMAGE", IMAGE),
                        "command": ["python", "-c", AGENT, BACKEND], "memory": "1g", "cpus": 1, "pids": 64},
            "output_dir": str(output), "record_video": True, "video_fps": 30,
            "budget": {"active_seconds": 180, "physics_ticks": 300, "submissions": 2},
            "simulation": {"camera": "fixed", "width": 320, "height": 240,
                           "rgb": True, "depth": False, "calibration": True},
            "evaluation": {"source_dir": str(root / "environments/blocks"), "camera": "fixed",
                           "width": 160, "height": 120, "timeout_seconds": 120,
                           "correspondence": {"block_red": "object", "block_blue": "other"}},
        }
        with patch.dict(os.environ, {"SIM2SIM_TEST_SECRET": "host-only"}):
            try:
                result = run(config)
            except Exception:
                for log in output.rglob("*.log"):
                    print(str(log.relative_to(output)) + ":\n" + log.read_text(errors="replace"))
                raise
        self.assertEqual(result["total_ticks"], 30)
        self.assertEqual(result["submissions"], 2)
        termination = json.loads((output / "termination.json").read_text())
        self.assertEqual(termination["status"], "finished")
        manifest = json.loads((output / "manifest.json").read_text())
        if BACKEND == "slurm_step":
            host = manifest["sandbox_verification"]["host"]
            self.assertEqual((host["memory_max"], host["memory_swap_max"], host["cpuset_cpus"], host["nproc"]), (1024 ** 3, 0, 1, ["64", "64"]))
            print("slurm_step verification: " + json.dumps(manifest["sandbox_verification"]))
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        self.assertTrue(all(event.get("ok", True) for event in events))
        self.assertTrue({"grasp", "move", "release", "reset", "submit"}.issubset({event["op"] for event in events}))
        for version in ("version-001", "version-002"):
            report = json.loads((output / version / "evaluation.json").read_text())
            self.assertEqual(report["status"], "ok", report)
            self.assertTrue(report["scores"])
        index = json.loads((output / "video-index.json").read_text())
        self.assertEqual(len(index), 2)
        self.assertEqual(index[0]["end_total_ticks"], 25)
        self.assertEqual(index[1]["start_total_ticks"], 25)
        for segment in index:
            reader = imageio.get_reader(output / segment["file"])
            try:
                self.assertEqual(reader.get_meta_data()["fps"], 30)
                self.assertGreater(reader.get_data(0).size, 0)
            finally:
                reader.close()

    def test_public_panda_fixture_is_hashed_read_only_and_anonymous(self):
        from sim2sim.runtime import (PUBLIC_ROBOT, WallClock, Transport, launch_container, podman_env, public_robot_files,
                                     stage_public_volume, stop_container, paused_containers)
        import subprocess
        files = public_robot_files(Path(__file__).resolve().parents[1] / "environments/panda/robot")
        code = r'''import hashlib, json, os, socket
from pathlib import Path
root = Path(os.environ["ROOT"]) if "ROOT" in os.environ else Path("/opt/public/robot")
try:
    (root / "x").write_text("x"); writable = True
except OSError:
    writable = False
visible = "\n".join([socket.gethostname(), Path("/etc/hosts").read_text(), Path("/proc/self/mountinfo").read_text(), json.dumps(dict(os.environ))])
print(json.dumps({"files": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()},
                  "writable": writable, "visible": visible}), flush=True)
input()
'''
        supervisor = WallClock(300)
        volume = stage_public_volume(files, supervisor)
        name = "sim2sim-fixture-" + uuid.uuid4().hex
        sandbox = {"process_limit_backend": BACKEND, "memory": "1g", "cpus": 1, "pids": 64}
        proc = launch_container(IMAGE, ["python", "-c", code], sandbox, name, Path(os.devnull), supervisor, mounts=((volume, PUBLIC_ROBOT),))
        try:
            result = Transport(proc, 1 << 20).receive(time.monotonic() + 60)
            self.assertEqual(result["files"], {k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
            self.assertFalse(result["writable"])
            for private in ("sibling scenes", "blocks", "random_assembly", "panda_small", "experiments/runs", "INFERENCE"):
                self.assertNotIn(private, result["visible"])
            real_run = subprocess.run
            def no_podman(args, *rest, **options):  # pause/resume must use the launch-time step record, never podman
                assert args[0] != "podman", args
                return real_run(args, *rest, **options)
            with patch("sim2sim.runtime.subprocess.run", no_podman), paused_containers([name], BACKEND):
                pass  # freezer state is verified inside the context manager
            print("fixture container: " + json.dumps(proc.verified))
        finally:
            stop_container(proc, name)
            supervisor.close()
            subprocess.run(["podman", "volume", "rm", "--force", volume], capture_output=True, timeout=15, env=podman_env())


    def test_predictor_runner_is_isolated_bounded_and_timed(self):
        """An agent's predictor program in its own worker container: no source, network, credentials or writable root;
        a per-call alarm it cannot swallow, a 1 MiB value bound, and a session timeout that fails every later call."""
        import io, zipfile
        from sim2sim.predictor import run_predictor
        from sim2sim.runtime import WallClock
        program = r'''
import json, os, signal, socket
from pathlib import Path
def probe():
    hidden = [str(p) for p in ("/opt/sim2sim/environments", "/workspace/source", "/workspace/output", "/opt/public") if Path(p).exists()]
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=1); network = True
    except OSError:
        network = False
    try:
        Path("/opt/x").write_text("x"); writable = True
    except OSError:
        writable = False
    print("printed text never reaches the protocol")
    return {"hidden": hidden, "network": network, "writable": writable, "secret": "SIM2SIM_TEST_SECRET" in os.environ,
            "files": sorted(p.name for p in Path(__file__).parent.iterdir()), "uid": os.getuid()}
def slow():
    try:
        while True:
            pass
    except Exception:
        return "swallowed"
def huge():
    return [0.5] * 400000
def block():
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    while True:
        pass
'''
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("predictor.py", program)
        sandbox = {"process_limit_backend": BACKEND, "evaluator_image": IMAGE, "memory": "1g", "cpus": 1, "pids": 64}
        output = Path(__file__).resolve().parents[1] / "experiments/validation/sandbox-backend" / ("predictor-" + os.environ.get("SLURM_JOB_ID", "local") + "-" + uuid.uuid4().hex[:8])
        supervisor = WallClock(600)
        try:
            with patch.dict(os.environ, {"SIM2SIM_TEST_SECRET": "host-only"}):
                replies = run_predictor(data.getvalue(), "predictor.py", sandbox, output / "a", supervisor,
                                        lambda call: [call("probe"), call("slow"), call("huge"), call("probe")], 120, call_seconds=3)
                blocked = run_predictor(data.getvalue(), "predictor.py", sandbox, output / "b", supervisor,
                                        lambda call: [call("block"), call("probe")], 20, call_seconds=3)
        finally:
            supervisor.close()
        probe = replies[0]["value"]
        self.assertEqual((probe["hidden"], probe["network"], probe["writable"], probe["secret"], probe["uid"]), ([], False, False, False, 65532))
        self.assertIn("predictor.py", probe["files"])
        self.assertIn("CallTimeout", replies[1]["error"])
        self.assertIn("exceeds", replies[2]["error"])
        self.assertTrue(replies[3]["ok"])
        self.assertEqual([r["ok"] for r in blocked], [False, False])
        self.assertTrue(all("predictor worker failed" in r["error"] for r in blocked), blocked)
        self.assertIn("printed text never reaches the protocol", (output / "a/predictor.stderr.log").read_text())
        print("predictor isolation: " + json.dumps(probe))


    def test_fixture_level_predictor_submission_end_to_end(self):
        """A strict run on the rigid fixture: fixture actions, a predictor submission run by the isolated predictor worker and
        scored by the evaluator container against real releases (a reduced private table keeps it short)."""
        import shutil
        from sim2sim.runtime import run
        root = Path(__file__).resolve().parents[1]
        output = root / "experiments/validation/sandbox-backend" / ("fixture-" + os.environ.get("SLURM_JOB_ID", "local") + "-" + uuid.uuid4().hex[:8])
        source = output.with_name(output.name + "-source")
        for name in ("robot", "fixture_rigid"):
            shutil.copytree(root / "environments/panda_small" / name, source / name)
        manifest = (source / "fixture_rigid/manifest.toml").read_text()
        manifest = manifest[:manifest.index("settings = [")] + 'settings = [[8.0, 3.0], [5.0, 5.0]]\nrepeats = 2\nchosen_repeats = 2\nsamples = 20\n' + \
            manifest[manifest.index("max_release_ticks"):]
        (source / "fixture_rigid/manifest.toml").write_text(manifest)
        agent = r'''
import json
from pathlib import Path
from sim2sim.client import Client
client = Client()
task = client.task
assert task["submission"]["entrypoint"] == "predictor.py" and set(task["submission"]["predictor"]["functions"]) == {"predict", "settings"}
cocked = client.request("reset_carriage")["observation"]
assert cocked["fixture"]["carriage_cocked"] and set(cocked["fixture"]) == {"compression_dial", "pressure_dial", "carriage_cocked"}
released = client.request("release_carriage", ticks=300)["observation"]
assert not released["fixture"]["carriage_cocked"] and "seed" not in json.dumps(task) + json.dumps(released)
project = Path("/workspace/project")
project.mkdir()
(project / "predictor.py").write_text("def predict(compression, pressure, n):\n    return [12.0 * compression - 4.0 * pressure] * n\n"
                                      "def settings():\n    return {'target': {'compression': 8, 'pressure': 3}, 'narrow_target': {'compression': 5, 'pressure': 5}}\n")
client.submit(project)
client.finish()
'''
        config = {
            "source": {"scene": str(source / "fixture_rigid/scene.xml")},
            "sandbox": {"process_limit_backend": BACKEND, "image": IMAGE, "evaluator_image": os.environ.get("SIM2SIM_EVALUATOR_IMAGE", IMAGE),
                        "command": ["python", "-c", agent], "memory": "2g", "cpus": 1, "pids": 64},
            "output_dir": str(output), "record_video": False,
            "budget": {"active_seconds": 300, "physics_ticks": 2000, "submissions": 1, "wall_seconds": 1200},
            "simulation": {"robot": "panda", "rgb": False, "depth": False, "calibration": False},
            "evaluation": {"source_dir": str(source), "source_entrypoint": "fixture_rigid/scene.xml", "manifest": "fixture_rigid/manifest.toml",
                           "entrypoint": "predictor.py", "metrics": ["fixture"], "timeout_seconds": 600},
        }
        try:
            result = run(config)
        except Exception:
            for log in output.rglob("*.log"):
                print(str(log.relative_to(output)) + ":\n" + log.read_text(errors="replace"))
            raise
        self.assertEqual((result["status"], result["submissions"]), ("finished", 1))
        report = json.loads((output / "version-001/evaluation.json").read_text())
        self.assertEqual(report["status"], "ok", report)
        scores = report["scores"]["fixture"]
        self.assertEqual(scores["prediction"]["status"], "ok")
        self.assertEqual(scores["targets"]["target"]["trials"], 2)
        calls = json.loads((output / "predictor-001/predictor-calls.json").read_text())
        self.assertEqual([c["function"] for c in calls], ["<load>", "settings", "predict", "predict", "predict", "predict"])
        self.assertTrue((output / "version-001/evaluation" / scores["view"]["path"]).is_file())
        print("fixture end to end: " + json.dumps({k: scores["prediction"][k] for k in ("crps_mm", "oracle_crps_mm")}))


# Runs as the agent in a strict container: a tool is usable exactly when enabled, and otherwise absent.
TOOLS_AGENT = r'''
import base64, importlib.util, json, resource, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from sim2sim.client import Client
client = Client()
enabled = json.loads(sys.argv[1])
report = {"enabled": enabled}
for name in ("segmentation", "planning"):
    assert Path("/opt/public/tools", name).is_dir() == (name in enabled) and ("/opt/public/tools/" + name in client.task["task"]) == (name in enabled), name
assert importlib.util.find_spec("mobile_sam") is None and importlib.util.find_spec("plan") is None
names = [str(p) for root in ("/opt", "/usr", "/workspace", "/tmp", "/home") for p in Path(root).rglob("*") if p.name.startswith("mobile_sam")]
assert bool(names) == ("segmentation" in enabled), names
assert Path("/opt/public/tools").exists() == bool(enabled)
observation = client.observe()
Path("/workspace/reference.png").write_bytes(base64.b64decode(observation["rgb"]))
def tool(*args):
    before, clock = resource.getrusage(resource.RUSAGE_CHILDREN), time.perf_counter()
    out = subprocess.run(["python3", *args], capture_output=True, text=True, timeout=900)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout), {"process_wall_s": round(time.perf_counter() - clock, 2),
                                    "process_cpu_s": round(after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime, 2)}
for name in enabled:
    try:
        Path("/opt/public/tools", name, "x").write_text("x")
        raise AssertionError("tool volume is writable")
    except OSError:
        pass
if "segmentation" in enabled:
    image = np.asarray(Image.open("/workspace/reference.png").convert("RGB")).astype(float)
    red = (image[..., 0] > 1.7 * image[..., 1]) & (image[..., 0] > 1.7 * image[..., 2]) & (image[..., 0] > 90)
    y, x = np.nonzero(red)
    seg = "/opt/public/tools/segmentation/segment.py"
    point, point_cost = tool(seg, "/workspace/reference.png", "/workspace/seg-point", "--point", "%d,%d" % (np.median(x), np.median(y)))
    box, box_cost = tool(seg, "/workspace/reference.png", "/workspace/seg-box", "--box", "%d,%d,%d,%d" % (x.min(), y.min(), x.max(), y.max()))
    fit = {}
    for name in ("point", "box"):  # the strict red threshold misses shaded cube faces: check recall and a tight box
        mask = np.load("/workspace/seg-%s/masks.npz" % name)["masks"][0]
        my, mx = np.nonzero(mask)
        fit[name] = {"red_recall": round(float((mask & red).sum() / red.sum()), 3),
                     "bbox_offset_px": int(np.abs(np.array([mx.min(), my.min(), mx.max(), my.max()]) - [x.min(), y.min(), x.max(), y.max()]).max())}
        assert fit[name]["red_recall"] > .9 and fit[name]["bbox_offset_px"] <= 3, fit
    auto, auto_cost = tool(seg, "/workspace/reference.png", "/workspace/seg-auto", "--auto")
    assert len(auto["masks"]) >= 2, auto
    report["segmentation"] = {"image_size": point["size"], "cpus": len(__import__("os").sched_getaffinity(0)), "red_block_fit": fit,
                              "point": {**point["timing"], **point_cost}, "box": {**box["timing"], **box_cost}, "auto_masks": len(auto["masks"]), "auto": {**auto["timing"], **auto_cost}}
if "planning" in enabled:
    project = Path("/workspace/project")
    shutil.copytree("/opt/public/robot", project / "robot")
    (project / "scene.xml").write_text("""<mujoco model="candidate"><compiler angle="radian" autolimits="true" meshdir="robot/assets"/>
<option timestep="0.002" integrator="implicitfast"/><worldbody><light pos="0 -1 3" dir="0 0 -1"/>
<geom name="floor" type="plane" size="3 3 .1"/><geom name="table" type="box" pos="0 0 .35" size=".65 .5 .05"/>
<geom name="pedestal" type="box" pos="0 -.65 .2" size=".13 .13 .2"/>
<body name="guess" pos="-.15 0 .46"><freejoint/><geom type="box" size=".06 .06 .06" mass=".3"/></body>
</worldbody><include file="robot/panda.xml"/></mujoco>""")
    planner, scene = "/opt/public/tools/planning/plan.py", str(project / "scene.xml")
    ik, ik_cost = tool(planner, scene, "ik", "--target", "-0.15", "0", "0.62", "--quat", "0", "1", "0", "0")
    assert ik["converged"] and ik["collision_free"], ik
    start = observation["proprioception"]["joint_positions_rad"]
    path, path_cost = tool(planner, scene, "plan", "--start", *map(str, start), "--goal", *map(str, ik["joint_targets"]))
    for q in path["joint_targets"]:
        client.arm_target(q, .08, ticks=20)
    final = client.observe(ticks=300)["proprioception"]["joint_positions_rad"]
    error = max(abs(a - b) for a, b in zip(final, ik["joint_targets"]))
    assert error < .05, (final, ik)
    report["planning"] = {"ik": {"position_error_m": ik["position_error_m"], **ik_cost}, "plan": {k: path[k] for k in ("method", "corners", "collision_checks", "seconds")},
                          "plan_process": path_cost, "waypoints": len(path["joint_targets"]), "reference_tracking_error_rad": round(error, 4)}
client.progress(json.dumps(report), origin="agent")
client.finish()
'''


@unittest.skipUnless(os.environ.get("SIM2SIM_TOOLS_OCI_TEST") == "1", "requires the tools image (trusted-image-tools.json) and explicit opt-in")
class AgentToolsOCITests(unittest.TestCase):
    """SIM2SIM_TOOLS_OCI_TEST=1 SIM2SIM_OCI_TEST=1 SIM2SIM_TRUSTED_IMAGE_RECORD=experiments/images/trusted-image-tools.json
    SIM2SIM_PROCESS_LIMIT_BACKEND=slurm_step sbatch -c 8 --mem=16G -t 40 --export=ALL scripts/test.slurm tests/test_oci.py"""
    def run_agent(self, tools):
        from sim2sim.runtime import run
        root = Path(__file__).resolve().parents[1]
        output = root / "experiments/validation/agent-tools" / "-".join([os.environ.get("SLURM_JOB_ID", "local"), *(tools or ["none"]), uuid.uuid4().hex[:8]])
        config = {
            "source": {"scene": str(root / "environments/panda/blocks/scene.xml")}, "tools": {name: True for name in tools},
            "sandbox": {"process_limit_backend": BACKEND, "image": IMAGE, "evaluator_image": IMAGE, "command": ["python", "-c", TOOLS_AGENT, json.dumps(tools)],
                        "memory": "8g", "cpus": 4, "pids": 256, "workspace": "1g"},
            "output_dir": str(output), "record_video": False,
            "budget": {"active_seconds": 1200, "physics_ticks": 20000, "submissions": 1},
            "simulation": {"robot": "panda", "camera": "fixed", "width": 320, "height": 240, "rgb": True, "depth": False, "calibration": True},
            "evaluation": {"source_dir": str(root / "environments/panda"), "source_entrypoint": "blocks/scene.xml", "manifest": "blocks/manifest.toml",
                           "camera": "fixed", "width": 160, "height": 120, "timeout_seconds": 120},
        }
        try:
            result = run(config)
        except Exception:
            for log in output.rglob("*.log"):
                print(str(log.relative_to(output)) + ":\n" + log.read_text(errors="replace"))
            raise
        self.assertEqual(json.loads((output / "termination.json").read_text())["status"], "finished", result)
        report = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines() if '"origin": "agent"' in line][-1]
        print("agent tools %s: %s" % (tools, report["text"]))
        return json.loads(report["text"])

    def test_enabled_tools_work_and_disabled_tools_are_absent(self):
        report = self.run_agent(["segmentation", "planning"])
        self.assertGreater(report["segmentation"]["red_block_fit"]["box"]["red_recall"], .9)
        self.assertLess(report["planning"]["reference_tracking_error_rad"], .05)
        self.assertEqual(self.run_agent([]), {"enabled": []})


SAM3D_AGENT = r'''
import base64, io, json
from pathlib import Path
import numpy as np
from PIL import Image
from sim2sim.client import Client
client = Client()
assert "sam3d" in client.task["actions"] and "sam3d" in client.task["action_schema"] and "controller action sam3d" in client.task["task"]
assert not Path("/opt/public/tools").exists() and not any("spool" in p.name for p in Path("/opt").rglob("*"))
found = sorted(str(p) for p in Path("/opt").rglob("*sam3d*"))
assert all(p.endswith(("/sim2sim/sam3d.py", "/sim2sim/__pycache__/sam3d.cpython-312.pyc")) for p in found), found  # package module only
observation = client.observe()
mask = np.zeros((240, 320), np.uint8)
mask[100:140, 140:180] = 255
data = io.BytesIO()
Image.fromarray(mask).save(data, "PNG")
result = client.request("sam3d", image_png_b64=observation["rgb"], mask_png_b64=base64.b64encode(data.getvalue()).decode(), seed=7)["sam3d"]
report = {"keys": sorted(result), "faces": result["faces"], "ply_header": base64.b64decode(result["mesh_ply_b64"])[:3].decode(), "model": result["model"]}
try:
    client.request("sam3d", image_png_b64=observation["rgb"], mask_png_b64=observation["rgb"])
    raise AssertionError("an RGB mask was accepted")
except RuntimeError as exc:
    report["rejected"] = str(exc)
client.progress(json.dumps(report), origin="agent")
client.finish()
'''


@unittest.skipUnless(os.environ.get("SIM2SIM_OCI_TEST") == "1", "requires explicit rootless OCI integration opt-in")
class Sam3dRelayOCITests(unittest.TestCase):
    """The strict controller relays the agent's sam3d action to a (fake-model) service over a private spool."""
    def test_sam3d_relay_round_trip_is_audited(self):
        import tempfile, threading
        import numpy as np
        from sim2sim.audit import read_audit
        from sim2sim.runtime import run
        from sim2sim.sam3d import serve
        root = Path(__file__).resolve().parents[1]
        output = root / "experiments/validation/sam3d-relay" / ("oci-" + os.environ.get("SLURM_JOB_ID", "local") + "-" + uuid.uuid4().hex[:8])
        spool, stop = Path(tempfile.mkdtemp()), threading.Event()
        model = lambda rgb, mask, seed: (np.eye(3).tolist() + [[0, 0, 0]], [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], [[9, 9, 9]] * 4,
                                         {"rotation_wxyz": [1, 0, 0, 0], "translation": [0, 0, 2], "scale": [seed] * 3})
        service = threading.Thread(target=serve, args=(spool, model, {"name": "fake"}, stop))
        service.start()
        time.sleep(.5)
        config = {"source": {"scene": str(root / "environments/blocks/scene.xml")}, "tools": {"sam3d": True},
                  "sandbox": {"process_limit_backend": BACKEND, "image": IMAGE, "evaluator_image": IMAGE,
                              "command": ["python", "-c", SAM3D_AGENT], "memory": "1g", "cpus": 1, "pids": 64},
                  "output_dir": str(output), "record_video": False, "budget": {"active_seconds": 180, "physics_ticks": 300, "submissions": 1},
                  "simulation": {"camera": "fixed", "width": 320, "height": 240, "rgb": True, "depth": False, "calibration": True},
                  "evaluation": {"source_dir": str(root / "environments/blocks"), "camera": "fixed", "width": 160, "height": 120, "timeout_seconds": 120}}
        try:
            with patch.dict(os.environ, {"SIM2SIM_SAM3D_SPOOL": str(spool)}):
                run(config)
        finally:
            stop.set()
            service.join()
        self.assertEqual(json.loads((output / "termination.json").read_text())["status"], "finished")
        report = json.loads([json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines() if '"origin": "agent"' in line][-1]["text"])
        print("sam3d relay: " + json.dumps(report))
        self.assertEqual(report["keys"], ["faces", "frame", "layout", "mesh_ply_b64", "model", "seed", "vertices"])
        self.assertEqual((report["faces"], report["ply_header"], report["model"]), (4, "ply", "fake"))
        self.assertIn("grayscale", report["rejected"])
        self.assertEqual(json.loads((output / "manifest.json").read_text())["sam3d_service"]["model"], {"name": "fake"})
        calls = [event for _, event in read_audit(output) if event.get("kind") == "sam3d_call"]
        self.assertEqual([(call["ok"], call["seed"]) for call in calls], [(True, 7)])
        self.assertFalse(list(spool.rglob("*.json")))


CAMERA_AGENT = r'''
import numpy as np
from pathlib import Path
from sim2sim.client import Client
client = Client()
assert "camera" in client.task["actions"] and client.task["public_tool_config"]["movable_camera"] is True
project = Path("/workspace/project")
project.mkdir()
(project / "scene.xml").write_text("""<mujoco><worldbody><light pos="0 0 3"/><camera name="fixed" pos="1 -1 1" xyaxes="0.7 0.7 0 -0.4 0.4 0.8"/>
<geom name="floor" type="plane" size="2 2 0.1"/><body name="object" pos="0 0 0.1"><freejoint/><geom type="box" size="0.1 0.1 0.1"/></body></worldbody></mujoco>""")
client.load_reconstruction(project)
for world in ("reference", "reconstruction"):
    observation = client.request("camera", world=world, pos=[0, -1.4, 1.1], lookat=[0, 0, 0.1])["observation"]
    pose = np.array(observation["calibration"]["world_from_camera"])
    assert np.allclose(pose[:3, 3], [0, -1.4, 1.1], atol=1e-6) and observation["total_ticks"] == 0, (world, pose, observation["total_ticks"])
client.submit(project)
client.finish()
'''


@unittest.skipUnless(os.environ.get("SIM2SIM_OCI_TEST") == "1", "requires explicit rootless OCI integration opt-in")
class MovableCameraOCITests(unittest.TestCase):
    """simulation.movable_camera: the agent re-aims both worlds' observation cameras through the strict controller."""
    def test_camera_action_in_both_worlds_without_ticks(self):
        from sim2sim.runtime import run
        root = Path(__file__).resolve().parents[1]
        output = root / "experiments/validation/movable-camera" / ("oci-" + os.environ.get("SLURM_JOB_ID", "local") + "-" + uuid.uuid4().hex[:8])
        config = {"source": {"scene": str(root / "environments/blocks/scene.xml")},
                  "sandbox": {"process_limit_backend": BACKEND, "image": IMAGE, "evaluator_image": os.environ.get("SIM2SIM_EVALUATOR_IMAGE", IMAGE),
                              "command": ["python", "-c", CAMERA_AGENT], "memory": "1g", "cpus": 1, "pids": 64},
                  "output_dir": str(output), "budget": {"active_seconds": 180, "physics_ticks": 100, "submissions": 1},
                  "simulation": {"camera": "fixed", "width": 160, "height": 120, "rgb": True, "calibration": True, "movable_camera": True},
                  "evaluation": {"source_dir": str(root / "environments/blocks"), "camera": "fixed", "width": 80, "height": 60,
                                 "timeout_seconds": 120}}
        result = run(config)
        self.assertEqual((result["total_ticks"], result["submissions"]), (0, 1))
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        cameras = [event for event in events if event.get("op") == "camera"]
        self.assertEqual(len(cameras), 2)
        self.assertTrue(all(event.get("ok", True) for event in events))


# Runs as the agent in a strict Shadow-hand container: the local_controller tool reproduces the reconstruction world exactly,
# and the pausable-bash process control (subreaper, /proc, SIGSTOP/SIGKILL of a detached tree) works without capabilities.
TOUCH_AGENT = r'''
import ctypes, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
sys.path[:0] = ["/opt/public/tools/local_controller", "/opt/agent"]
import localsim, sandbox_agent
from sim2sim.client import Client
from sim2sim.robots import HAND_ACTUATORS, HAND_HOME
client = Client()
assert "/opt/public/tools/local_controller" in client.task["task"]
assert json.loads(Path("/opt/public/tools/local_controller/public_tool_config.json").read_text()) == client.task["public_tool_config"]
project = Path("/workspace/project")
shutil.copytree("/opt/public/robot", project / "robot")
(project / "scene.xml").write_text("""<mujoco model="guess"><compiler angle="radian" autolimits="true" meshdir="robot/assets"/>
<worldbody><camera name="fixed" pos="0.58 -0.42 0.38" xyaxes="0.8229 0.5682 0 -0.3213 0.4654 0.8247"/>
<geom name="floor" type="plane" pos="0 0 -0.1" size="1 1 .05"/>
<body name="guess" pos="0.3 0 0.04"><freejoint/><geom type="box" size=".03 .02 .015" density="500"/></body>
</worldbody><include file="robot/shadow_hand.xml"/></mujoco>""")
client.load_reconstruction(project)
home = [HAND_HOME.get(n.replace("_A_", "_")[:-1] + "1", 0) + HAND_HOME.get(n.replace("_A_", "_")[:-1] + "2", 0) if n.endswith("J0")
        else HAND_HOME.get(n.replace("_A_", "_"), 0) for n in HAND_ACTUATORS]
curl = [1.5708 if n[-3:] == "FJ3" else 3.0 if n[-3:] == "FJ0" else v for n, v in zip(HAND_ACTUATORS, home)]
steps = [{"op": "observe", "ticks": 10}, {"op": "hand_target", "actuator_targets": curl, "ticks": 200}, {"op": "reset"}, {"op": "observe"}]
remote = [client.request(s["op"], world="reconstruction", **{k: v for k, v in s.items() if k != "op"})["observation"] for s in steps]
local = localsim.LocalController(project)
assert [local.act(s) for s in steps] == remote, "local copy differs from the reconstruction world"
cells = localsim.taxels(remote[1]["proprioception"]["joint_positions_rad"])
assert len(cells) == 6 and all(c["surface_point_m"] for c in cells["middle"]["cells"][1])
assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
job = subprocess.Popen(["bash", "-c", "(setsid bash -c 'while :; do echo x >> /tmp/d; sleep .05; done' &); while :; do echo x >> /tmp/f; sleep .05; done"],
                       start_new_session=True)
time.sleep(1)
size = lambda: [os.path.getsize("/tmp/" + n) for n in "fd"]
sandbox_agent.signal_tree(signal.SIGSTOP)
paused, before = sandbox_agent.descendants(), size()
time.sleep(.5)
assert len(paused) >= 3 and all(s in "tT" for s in paused.values()) and size() == before, (paused, before, size())
sandbox_agent.signal_tree(signal.SIGCONT)
time.sleep(.5)
assert all(a > b for a, b in zip(size(), before))
sandbox_agent.signal_tree(signal.SIGKILL)
job.wait()
sandbox_agent.reap()
assert sandbox_agent.descendants() == {}
client.progress(json.dumps({"exact_steps": len(steps), "paused_processes": len(paused)}), origin="agent")
client.finish()
'''


@unittest.skipUnless(os.environ.get("SIM2SIM_OCI_TEST") == "1", "requires explicit rootless OCI integration opt-in")
class TouchHarnessOCITests(unittest.TestCase):
    def test_local_controller_matches_reconstruction_world_and_bash_tree_control(self):
        from sim2sim.runtime import run
        root = Path(__file__).resolve().parents[1]
        output = root / "experiments/validation/touch-harness" / ("oci-" + os.environ.get("SLURM_JOB_ID", "local") + "-" + uuid.uuid4().hex[:8])
        config = {"source": {"scene": str(root / "environments/shadow_hand/hand_prism/scene.xml")}, "tools": {"local_controller": True},
                  "sandbox": {"process_limit_backend": BACKEND, "image": IMAGE, "evaluator_image": os.environ.get("SIM2SIM_EVALUATOR_IMAGE", IMAGE),
                              "command": ["python", "-c", TOUCH_AGENT], "memory": "4g", "cpus": 2, "pids": 128, "workspace": "1g"},
                  "output_dir": str(output), "record_video": False, "budget": {"active_seconds": 600, "physics_ticks": 5000, "submissions": 1},
                  "simulation": {"robot": "shadow_hand", "camera": "fixed", "width": 320, "height": 240, "rgb": False, "depth": False,
                                 "calibration": False, "tactile": True},
                  "evaluation": {"source_dir": str(root / "environments/shadow_hand"), "source_entrypoint": "hand_prism/scene.xml",
                                 "manifest": "hand_prism/manifest.toml", "camera": "fixed", "width": 80, "height": 60, "timeout_seconds": 120}}
        try:
            run(config)
        except Exception:
            for log in output.rglob("*.log"):
                print(str(log.relative_to(output)) + ":\n" + log.read_text(errors="replace"))
            raise
        self.assertEqual(json.loads((output / "termination.json").read_text())["status"], "finished")
        report = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines() if '"origin": "agent"' in line]
        self.assertTrue(report, "agent script did not reach its report; see agent.stderr.log")
        print("touch harness: " + report[-1]["text"])
