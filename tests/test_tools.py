"""Optional agent tools: configuration, hashed staging and the Panda planner (MuJoCo; run on a Slurm test node)."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from sim2sim import runtime
from sim2sim.experiments import load_experiment
from sim2sim.runtime import AGENT_TOOLS, RuntimeFailure, enabled_tools, public_tool_files
from sim2sim.simulation import Simulation

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "tools/planning/plan.py"
SCENE = ROOT / "environments/panda/blocks/scene.xml"
SMALL = ROOT / "environments/panda_small/blocks/scene.xml"  # blocks narrow enough for the 0.08 m jaws
spec = importlib.util.spec_from_file_location("agent_plan", PLAN)
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)
DOWN = [0, 1, 0, 0]


def test_tools_config_defaults_off_and_rejects_invalid():
    assert enabled_tools({}) == () and enabled_tools({"tools": {"segmentation": False, "planning": False}}) == ()
    assert enabled_tools({"tools": {"segmentation": True}}) == ("segmentation",)
    assert enabled_tools({"tools": {"planning": True, "segmentation": True}, "simulation": {"robot": "panda"}}) == ("segmentation", "planning")
    for bad in ({"tools": {"segmentation": 1}}, {"tools": {"depth": True}}, {"tools": {"planning": True}}, {"tools": []}):
        with pytest.raises(RuntimeFailure):
            enabled_tools(bad)
    for name, (text, pins) in AGENT_TOOLS.items():  # service tools (pins None, e.g. sam3d) mount nothing
        assert ("/opt/public/tools/" + name in text) == (pins is not None) and text.startswith("Optional tool: ")


def test_tool_volumes_are_exact_hashed_files(tmp_path, monkeypatch):
    assert set(public_tool_files("planning")) == {"plan.py", "NOTICE", "LICENSE"}
    files = public_tool_files("segmentation")
    assert {"segment.py", "vendor/LICENSE", "vendor/mobile_sam.pt", "vendor/mobile_sam/__init__.py"} <= set(files)
    assert not [name for name in files if "__pycache__" in name or name.endswith(".pyc")]
    assert hashlib.sha256(files["vendor/mobile_sam.pt"]).hexdigest() == AGENT_TOOLS["segmentation"][1]["vendor/mobile_sam.pt"]
    (tmp_path / "segmentation/vendor").mkdir(parents=True)
    (tmp_path / "segmentation/vendor/mobile_sam.pt").write_bytes(b"tampered")
    monkeypatch.setattr(runtime, "TOOLS_DIR", tmp_path)
    with pytest.raises(RuntimeFailure, match="Pinned agent tool file"):
        public_tool_files("segmentation")
    with pytest.raises(RuntimeFailure, match="unavailable"):
        public_tool_files("planning")


def test_experiment_tools_require_strict_oci(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    text = (ROOT / "experiments/configs/wave3b-20261104/w3-ps-assembly-fixed.toml").read_text() + "\n[tools]\nplanning = true\n"
    (tmp_path / "strict.toml").write_text(text)
    assert enabled_tools(load_experiment(tmp_path / "strict.toml")[1]) == ("planning",)
    (tmp_path / "development.toml").write_text(text.replace('execution_mode = "strict_oci"', 'execution_mode = "development_shared_filesystem"'))
    with pytest.raises(ValueError, match="strict_oci"):
        load_experiment(tmp_path / "development.toml")


def test_ik_reaches_top_down_fingertip_target():
    panda = plan.Panda(SCENE)
    assert not panda.contacts(plan.HOME)
    result = panda.ik([-.15, 0, .62], DOWN)
    assert result["converged"] and result["collision_free"], result
    position, rotation = panda.pose(np.array(result["joint_targets"]))
    assert np.linalg.norm(position - [-.15, 0, .62]) < 1e-3 and rotation[:, 2] @ [0, 0, -1] > .999
    hand = panda.ik([.2, -.1, .7], frame="hand")
    assert hand["converged"] and np.linalg.norm(panda.pose(np.array(hand["joint_targets"]), "hand")[0] - [.2, -.1, .7]) < 1e-3


def test_plan_is_collision_free_bounded_and_rrt_detours_around_wall(tmp_path):
    wall = SCENE.read_text().replace("../robot/", str(ROOT / "environments/panda/robot") + "/").replace(
        '<geom name="panda_pedestal"', '<geom name="wall" type="box" pos="0 .1 .75" size=".01 .3 .35"/>\n    <geom name="panda_pedestal"')
    (tmp_path / "scene.xml").write_text(wall)
    for scene, method in ((SCENE, "straight_line"), (tmp_path / "scene.xml", "rrt_connect")):
        panda = plan.Panda(scene, ignore_bodies=())
        start, goal = (panda.ik([x, -.1, .7], DOWN)["joint_targets"] for x in (-.35, .35))
        result = panda.plan(start, goal, max_step=.1)
        assert result["method"] == method, result
        path = np.array([start] + result["joint_targets"])
        assert np.allclose(path[-1], goal, atol=1e-5) and np.abs(np.diff(path, axis=0)).max() <= .1 + 1e-6
        assert all(panda.within_limits(q) for q in path) and all(panda.segment_free(a, b) for a, b in zip(path, path[1:]))
    panda = plan.Panda(SCENE)
    blocked = panda.ik([-.15, 0, .5], DOWN, collision_free=False)["joint_targets"]  # fingertips inside the red block
    with pytest.raises(AssertionError, match="goal configuration is in collision"):
        panda.plan(plan.HOME, blocked)
    assert not plan.Panda(SCENE, ignore_bodies=["block_red"]).contacts(np.array(blocked))


def test_cli_prints_json_waypoints():
    out = subprocess.run([sys.executable, "-B", str(PLAN), str(SCENE), "plan", "--target", "-0.15", "0", "0.62", "--quat", *map(str, DOWN)],
                         capture_output=True, text=True, check=True)
    result = json.loads(out.stdout)
    assert result["goal_ik"]["converged"] and result["joint_targets"][-1] == pytest.approx(result["goal_ik"]["joint_targets"], abs=1e-5)
    help_text = subprocess.run([sys.executable, "-B", str(PLAN), "--help"], capture_output=True, text=True, check=True).stdout
    assert "RRT-connect" in help_text and "arm_target" in help_text and "grasp" in help_text and "linear" in help_text
    run = lambda *args: subprocess.run([sys.executable, "-B", str(PLAN), str(SCENE), *args], capture_output=True, text=True)
    fk = json.loads(run("fk").stdout)
    assert not fk["contacts"] and 0 < fk["clearance_m"] <= .2 and len(fk["fingertips"]["quat"]) == 4, fk
    start = [str(v) for v in result["joint_targets"][-1]]
    line = json.loads(run("linear", "--start", *start, "--target", "-0.15", "0", "0.7").stdout)
    assert line["reached"] and line["continuous"] and line["collision_free"] and len(line["joint_targets"]) == 8, line
    failed = run("linear", "--start", *start, "--target", "2", "0", "0.6")  # out of reach: fails loudly, no JSON
    assert failed.returncode != 0 and "no continuous straight line" in failed.stderr and not failed.stdout


def test_linear_tracks_line_and_turns_to_quat():
    panda = plan.Panda(SCENE)
    start = panda.ik([-.3, -.1, .7], DOWN)["joint_targets"]
    a, b, yawed = np.array([-.3, -.1, .7]), np.array([.2, 0, .6]), [0, np.cos(np.pi / 8), np.sin(np.pi / 8), 0]  # yaw 45 deg
    result = panda.linear(start, b, yawed, max_step=.1)
    assert result["reached"] and result["continuous"] and result["collision_free"], {k: v for k, v in result.items() if k != "joint_targets"}
    path = np.array(result["joint_targets"])
    assert len(path) == result["waypoints"] >= 51 and np.abs(np.diff(np.vstack([start, path]), axis=0)).max() <= .1
    direction = (b - a) / np.linalg.norm(b - a)
    for q in path:
        offset = panda.pose(q)[0] - a
        assert np.linalg.norm(offset - (offset @ direction) * direction) < 2e-3 and -1e-3 < offset @ direction < np.linalg.norm(b - a) + 1e-3
    position, rotation = panda.pose(path[-1])
    assert np.linalg.norm(position - b) < 1e-3 and np.abs(rotation - plan._matrix(yawed)).max() < 1e-2


def test_grasp_lifts_block_in_simulation():
    assert plan.Panda(SCENE).grasp("block_red")["pairs"] == 0  # a 0.12 m cube does not fit the 0.08 m jaws
    panda = plan.Panda(SMALL)
    result = panda.grasp(near=[.0675, -.12, .44], count=2)
    assert result["body"] == "block_blue" and len(result["candidates"]) == 2, result
    best = result["candidates"][0]
    assert best["top_down"] and best["cone_deg"] <= 20 and .0375 - 1e-6 <= best["width_m"] <= .07 and best["close_width_m"] == 0
    assert best["joint_targets"] == best["approach_joint_targets"][-1] and not panda.contacts(np.array(best["pregrasp_joint_targets"]))
    sim = Simulation(SMALL, {"robot": "panda", "rgb": False, "depth": False, "calibration": False})
    try:
        body = mujoco.mj_name2id(sim._model, mujoco.mjtObj.mjOBJ_BODY, "block_blue")
        height = sim._data.xpos[body][2]

        def send(targets, width, ticks=20):
            for q in targets:
                sim.action({"op": "arm_target", "joint_targets": q, "gripper_width_m": width, "ticks": ticks}, render=False)
        send(panda.plan(plan.HOME, best["pregrasp_joint_targets"])["joint_targets"], best["open_width_m"])
        send(best["approach_joint_targets"], best["open_width_m"])
        send([best["joint_targets"]], best["close_width_m"], ticks=300)
        lift = plan.Panda(SMALL, ignore_bodies=["block_blue"]).linear(best["joint_targets"], np.add(best["position"], [0, 0, .1]), best["quat"])
        assert lift["reached"] and lift["continuous"] and lift["collision_free"], lift
        send(lift["joint_targets"], best["close_width_m"])
        send(lift["joint_targets"][-1:], best["close_width_m"], ticks=300)
        assert sim._data.xpos[body][2] - height > .08 and sim.observe()["proprioception"]["gripper_width_m"] > .03
    finally:
        sim.close()
