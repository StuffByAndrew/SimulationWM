"""Real-time world, launcher, predictor streaming and die scoring (native MuJoCo: run on Slurm)."""
import base64
from contextlib import nullcontext
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import tomllib
from unittest.mock import patch
import zipfile

import numpy as np
import pytest

from sim2sim import die
from sim2sim.launcher import Launcher
from sim2sim.predictor import stream
from sim2sim.realtime import RealtimeWorld
from sim2sim.simulation import Simulation

ROOT = Path(__file__).resolve().parents[1]
DIE = ROOT / "environments/die_biased/scene.xml"
MANIFEST = tomllib.loads((DIE.parent / "manifest.toml").read_text())
CONFIG = {"width": 160, "height": 120, "depth": True, "camera": "fixed"}


def die_pixel(sim):
    body = sim._model.body("die").id
    calibration = sim.calibration()
    pose, k = np.array(calibration["world_from_camera"]), np.array(calibration["intrinsics"])
    c = pose[:3, :3].T @ (sim._data.xpos[body] - pose[:3, 3])
    return [float(k[0, 2] + k[0, 0] * c[0] / -c[2]), float(k[1, 2] - k[1, 1] * c[1] / -c[2])]


def test_world_keeps_wall_clock_rate_chains_actions_and_has_a_fixed_frame_grid():
    launcher = Launcher(MANIFEST["launcher"])
    world = RealtimeWorld(Simulation(DIE, CONFIG), Simulation(DIE, CONFIG), 25, launcher)
    try:
        began, ticks = time.monotonic(), world.sim.total_ticks
        time.sleep(1)
        rate = (world.sim.total_ticks - ticks) / (time.monotonic() - began) / 1000
        assert .9 < rate < 1.1, rate
        with world.lock:
            pixel = die_pixel(world.sim)  # the die rests, so a read during stepping is exact enough
        deadline = time.monotonic() + 10
        grasp, _ = world.command({"op": "grasp", "pixel": pixel}, deadline)
        assert grasp["grasp"] is not None
        move, _ = world.command({"op": "move", "displacement": [0, 0, .1], "ticks": 300}, deadline)
        release, entry = world.command({"op": "release"}, deadline)
        assert release["total_ticks"] - move["total_ticks"] == 300 and release["grasp"] is None  # back to back, whatever the latency
        frames = world.frames(entry["frame"] - 5, 10, time.monotonic() + 1)
        assert [f["frame"] for f in frames] == list(range(entry["frame"] - 4, entry["frame"] + 1))
        times = [f["state"]["time"] for f in frames]
        assert np.allclose(np.diff(times), .04) and all(abs(t / .04 - round(t / .04)) < 1e-6 for t in times)
        observation = world.render(frames[-1])
        assert observation["frame"] == frames[-1]["frame"] and observation["width"] == 160 and "depth" in observation
        state, entry = world.command({"op": "throw", "protocol": "spin"}, deadline)
        assert entry["state"]["time"] == 0 and entry["episode"] == frames[-1]["episode"] + 1
        with pytest.raises(ValueError):
            world.command({"op": "throw", "protocol": "nope"}, deadline)
        with world.paused():
            time.sleep(.05)  # a tick already in progress completes
            before = world.sim.total_ticks
            time.sleep(.3)
            assert world.sim.total_ticks == before
    finally:
        stats = world.close()
    assert stats["error"] is None and stats["launches"] == 2


def test_generic_world_without_launcher_resets_to_the_initial_state():
    scene = ROOT / "environments/blocks/scene.xml"
    world = RealtimeWorld(Simulation(scene, CONFIG), Simulation(scene, CONFIG), 50)
    try:
        deadline = time.monotonic() + 5
        state, entry = world.command({"op": "reset"}, deadline)
        assert entry["state"]["ticks"] == 0 and entry["episode"] == 2
        with pytest.raises(ValueError):
            world.command({"op": "throw", "protocol": "drop"}, deadline)
    finally:
        assert world.close()["error"] is None


def test_launches_are_reproducible_and_streams_differ():
    launcher = Launcher(MANIFEST["launcher"])
    a, b = launcher.state("toss", 1, 3), launcher.state("toss", 1, 3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(launcher.state("toss", 0, 3)[0], a[0])


PREDICT = '''import json, sys, time
for line in sys.stdin:
    m = json.loads(line)
    if m["type"] == "start":
        print(json.dumps({"type": "ready"}), flush=True)
    elif m["type"] == "frame" and m["index"] == 0:
        i = m["trial"]
        time.sleep(.3 if i == 1 else 0)
        p = [1, 0, 0] if i in (0, 1) else [2, 0, 0]
        if i != 3:
            print(json.dumps({"type": "prediction", "trial": i, "probabilities": p, "estimates": {"com_die_frame_m": [0, 0, 1]}}), flush=True)
    elif m["type"] == "end":
        break
'''


def test_predictor_stream_statuses_and_latency(tmp_path):
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("predict.py", PREDICT)
    process = subprocess.Popen([sys.executable, "-m", "sim2sim.predictor", "stream", str(tmp_path / "predictor")], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        trials = [({"type": "trial", "trial": i, "protocol": "drop"}, [{"time": k * .04} for k in range(4)]) for i in range(4)]
        ready, results = stream(process, bundle.getvalue(), "predict.py", {"type": "start", "faces": 3}, trials, 25, .15,
                                lambda m: die.valid_probabilities(m.get("probabilities"), 3) is not None, tmp_path / "log.jsonl")
    finally:
        process.kill()
    assert ready and [r["status"] for r in results] == ["on_time", "late", "malformed", "missing"], results
    assert results[0]["latency_s"] < .15 and results[0]["message"]["probabilities"] == [1, 0, 0] and results[0]["message"]["estimates"] == {"com_die_frame_m": [0, 0, 1]}
    assert results[1]["latency_s"] > .15
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 5  # load reply, ready and three answers


def test_scores_uniform_oracle_and_ties():
    trials = [{"trial": i, "protocol": "drop", "outcome": 1 + i % 3, "mechanics": 1, "oracle": 1 + i % 3} for i in range(3)]
    data = {"meta": {"faces": 3}, "trials": trials, "empirical": {"drop": {"probabilities": [1 / 3] * 3}}}
    results = [{"trial": 0, "status": "on_time", "probabilities": [1, 0, 0], "latency_s": .01, "estimates": None},
               {"trial": 1, "status": "on_time", "probabilities": [.5, .5, 0], "latency_s": .02, "estimates": None},
               {"trial": 2, "status": "late", "probabilities": None, "latency_s": .3, "estimates": None}]
    summary, rows = die.score(data, results)
    assert [row["brier"]["predictor"] for row in rows] == pytest.approx([0, .5, 2 / 3])
    assert [row["accuracy"]["predictor"] for row in rows] == pytest.approx([1, .5, 1 / 3])
    assert summary["baselines"]["oracle"]["brier"] == 0 and summary["baselines"]["uniform"]["brier"] == pytest.approx(2 / 3)
    assert (summary["on_time"], summary["late"], summary["latency_median_s"]) == (2, 1, pytest.approx(.015))


AGENT = r'''
import json, sys, time
task = json.loads(input())
def ask(**request):
    print(json.dumps(request), flush=True)
    return json.loads(input())
assert "load_reconstruction" not in task["actions"] and {"frames", "throw"} <= set(task["actions"]) and task["worlds"]["available"] == ["reference"]
assert task["realtime"]["frame_rate_hz"] == 25 and task["submission"]["kind"] == "predictor" and len(task["die"]["faces"]) == 12
assert set(task["launcher_protocols"]) == {"drop", "roll", "toss", "spin"} and "seed" not in json.dumps(task) and "perturbation" not in json.dumps(task)
assert not ask(op="observe", ticks=5)["ok"] and not ask(op="observe", world="reconstruction")["ok"]
first = ask(op="observe")["observation"]
time.sleep(1)
later = ask(op="observe")["observation"]
thrown = ask(op="throw", protocol="toss")
time.sleep(.3)
frames = ask(op="frames", since=thrown["observation"]["frame"] - 1, limit=3)["frames"]
print(json.dumps({"op": "progress", "text": json.dumps({"advanced": later["frame"] - first["frame"], "times": [f["time"] for f in frames],
                  "episodes": [f["episode"] for f in frames], "keys": sorted(set().union(*frames, thrown["observation"]))})}), flush=True)
input()
ask(op="finish")
'''


def run_realtime_agent(tmp_path, agent, simulation):
    """Run the die level in real time with a local agent script; return the parsed text of its last progress note."""
    from sim2sim.runtime import run
    cmd = [sys.executable, "-u", "-c", agent]
    config = {"source": {"scene": str(DIE)}, "sandbox": {"image": "test", "command": ["agent"]}, "record_video": False,
              "output_dir": str(tmp_path / "run"), "budget": {"wall_only": True, "wall_seconds": 120},
              "simulation": {"realtime": True, **simulation},
              "evaluation": {"source_dir": str(DIE.parent), "metrics": ["die"], "entrypoint": "predict.py"}}
    with patch("sim2sim.runtime.check_rootless"), patch("sim2sim.runtime.check_limits", return_value={}), \
         patch("sim2sim.runtime.container_command", return_value=cmd), \
         patch("sim2sim.runtime.stop_container", side_effect=lambda proc, name: proc.kill()), \
         patch("sim2sim.runtime.subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout="test-image-id")):
        result = run(config)
    assert result["status"] == "finished", (tmp_path / "run/termination.json").read_text()
    progress = [json.loads(line) for line in (tmp_path / "run/progress.jsonl").read_text().splitlines()]
    return json.loads(progress[-1]["text"])


def test_runtime_realtime_contract_and_no_state_leaks(tmp_path):
    report = run_realtime_agent(tmp_path, AGENT, {**CONFIG, "frame_rate": 25})
    assert 24 <= report["advanced"] <= 40 and report["times"] == pytest.approx([0, .04, .08]) and len(set(report["episodes"])) == 1
    assert set(report["keys"]) <= {"width", "height", "time", "ticks", "total_ticks", "frame", "episode", "grasped", "rgb", "rgb_format",
                                   "depth", "depth_format", "calibration"}, report
    termination = json.loads((tmp_path / "run/termination.json").read_text())
    assert termination["realtime_world"]["error"] is None and termination["realtime_world"]["launches"] == 2


FRAME_BATCH = r'''
import json, time
input()
def ask(**request):
    print(json.dumps(request), flush=True)
    return json.loads(input())
thrown = ask(op="throw", protocol="drop")["observation"]
time.sleep(1.2)
frames = ask(op="frames", since=thrown["frame"] - 1, limit=100)["frames"]
print(json.dumps({"op": "progress", "text": json.dumps([f["frame"] for f in frames])}), flush=True)
input()
ask(op="finish")
'''


def test_runtime_records_a_full_frame_batch(tmp_path):
    """Regression (w24 fps100 runs): a 100-frame batch at 320x240 overflowed an audit reservation sized for 5 frames."""
    report = run_realtime_agent(tmp_path, FRAME_BATCH, {"width": 320, "height": 240, "depth": True, "camera": "fixed", "frame_rate": 100})
    assert report == list(range(report[0], report[0] + 100)), report
