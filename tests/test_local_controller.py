"""Agent local_controller tool: exact agreement with the candidate worker (the reconstruction world) and tactile cell
geometry against the simulator's private contacts. Native MuJoCo physics: run on a Slurm test node."""
import base64
import importlib.util
import io
from pathlib import Path
import zipfile
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from sim2sim.candidate_worker import main as worker
from sim2sim.robots import HAND_ACTUATORS, HAND_HOME, HAND_JOINTS
from sim2sim.runtime import AGENT_TOOLS, RuntimeFailure, enabled_tools, public_tool_files
from sim2sim.simulation import TACTILE, Simulation

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "environments/shadow_hand"
spec = importlib.util.spec_from_file_location("localsim", ROOT / "tools/local_controller/localsim.py")
localsim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(localsim)
# The public settings the runtime gives the reconstruction world for the wave-21 hand levels.
CONFIG = {"robot": "shadow_hand", "rgb": False, "depth": False, "calibration": False, "wrench": False, "tactile": True, "default_ticks": 20,
          "width": 320, "height": 240, "timestep": .002, "grasp_solref": [.01, 1], "grasp_solimp": [.9, .95, .001, .5, 2]}


def project(tmp_path, level):
    """The level's scene as an agent project: its own copy of the public fixture under robot/."""
    root = tmp_path / "project"
    (root / "robot").mkdir(parents=True)
    (root / "scene.xml").write_text((FAMILY / level / "scene.xml").read_text().replace("../robot/", "robot/"))
    for path in (FAMILY / "robot").rglob("*"):
        if path.is_file():
            (root / "robot" / path.relative_to(FAMILY / "robot")).parent.mkdir(parents=True, exist_ok=True)
            (root / "robot" / path.relative_to(FAMILY / "robot")).write_bytes(path.read_bytes())
    return root


def targets(**changes):
    values = []  # actuator targets at the reset pose: joint angles, coupled J0 = J1 + J2
    for name in HAND_ACTUATORS:
        joint = name.replace("_A_", "_")
        values.append(HAND_HOME.get(joint[:-1] + "1", 0) + HAND_HOME.get(joint[:-1] + "2", 0) if joint.endswith("J0") else HAND_HOME.get(joint, 0))
    for name, value in changes.items():
        values[HAND_ACTUATORS.index("rh_A_" + name)] = value
    return values


STEPS = [{"op": "observe", "ticks": 30},
         {"op": "hand_target", "actuator_targets": targets(FFJ3=1.5708, FFJ0=3.0, MFJ3=1.5708, MFJ0=3.0, RFJ3=1.5708, RFJ0=3.0), "ticks": 300},
         {"op": "observe", "ticks": 50},
         {"op": "reset"},
         {"op": "hand_target", "actuator_targets": targets(WRJ1=-0.5), "ticks": 200},
         {"op": "observe"}]


def test_local_copy_matches_the_reconstruction_world_exactly(tmp_path):
    root = project(tmp_path, "hand_prism")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(root).as_posix())
    messages = [{"op": "load", "project_b64": base64.b64encode(archive.getvalue()).decode(), "entrypoint": "scene.xml",
                 "config": dict(CONFIG, camera="fixed", recording_fps=30)}]
    for step in STEPS:
        messages += [{"op": "action", "action": step}] + [{"continue": True}] * step.get("ticks", CONFIG["default_ticks"] if step["op"] == "hand_target" else 0)
    messages.append({"op": "close"})
    packets = []
    with patch("sim2sim.candidate_worker._read", side_effect=messages), patch("sim2sim.candidate_worker._emit", side_effect=packets.append):
        worker(project_dir=tmp_path / "worker")
    remote = [packet["observation"] for packet in packets if packet["type"] == "result" and packet.get("observation")]
    assert len(remote) == len(STEPS) + 1 and all(packet["ok"] for packet in packets if packet["type"] == "result")
    local = localsim.LocalController(root, config=CONFIG)
    try:
        assert local.observe() == remote[0]
        assert [local.act(step) for step in STEPS] == remote[1:]
        assert sum(np.sum(remote[2]["tactile"][f]["pad_normal_force_n"]) for f in ("first", "middle", "ring")) > 0  # fingers touched it
        fresh = localsim.LocalController(root, config=CONFIG)  # total_ticks never resets, so compare run() from a fresh load
        records = fresh.run(STEPS)
        fresh.close()
        assert [r["op"] for r in records] == [s["op"] for s in STEPS] and [r["observation"] for r in records] == remote[1:]
        for bad in ({"op": "grasp", "pixel": [1, 1]}, {"op": "wait", "ticks": 1}, {"op": "observe", "world": "reference"}):
            with pytest.raises(ValueError, match="an action is an object"):
                local.act(bad)
    finally:
        local.close()


def test_taxel_cells_locate_the_contacts_they_count(tmp_path):
    sim = Simulation(project(tmp_path, "hand_prism") / "scene.xml", CONFIG)
    try:
        cells = localsim.taxels(robot_dir=FAMILY / "robot")
        assert list(cells) == [pad[0] for pad in TACTILE["shadow_hand"][2]]
        # Every cell centre has palmar surface below it except the two distal corners of each rounded finger tip.
        assert {name: [[cell["surface_point_m"] is not None for cell in row] for row in pad["cells"]] for name, pad in cells.items() if name != "palm"} == \
            {"thumb": [[True] * 3] * 3, **{f: [[False, True, False]] + [[True] * 3] * 2 for f in ("first", "middle", "ring", "little")}}
        observation = sim.action(STEPS[1])
        data, joints = sim._proprioception_data, observation["proprioception"]["joint_positions_rad"]
        cells = localsim.taxels(joints, robot_dir=FAMILY / "robot")
        checked = 0
        for name, body, *_ in TACTILE["shadow_hand"][2]:
            pad, body_id = cells[name], sim._model.body(body).id
            np.testing.assert_allclose(pad["body_position_m"], data.xpos[body_id], atol=1e-9)
            np.testing.assert_allclose(pad["body_rotation"], data.xmat[body_id].reshape(3, 3), atol=1e-9)
            for row in pad["cells"]:
                for cell in row:
                    if cell["surface_normal"] is not None:  # outward on the palmar (-y) side
                        assert np.dot(cell["surface_normal"], np.array(pad["body_rotation"])[:, 1]) < 0
            for i in range(data.ncon):
                contact = data.contact[i]
                if body_id not in (sim._model.geom_bodyid[contact.geom1], sim._model.geom_bodyid[contact.geom2]):
                    continue
                x, y, z = data.xmat[body_id].reshape(3, 3).T @ (contact.pos - data.xpos[body_id])
                hits = [(r, c) for r, row in enumerate(pad["cells"]) for c, cell in enumerate(row)
                        if cell["x_m"][0] <= x < cell["x_m"][1] and cell["z_m"][0] < z <= cell["z_m"][1] and (pad["y_max_m"] is None or y <= pad["y_max_m"])]
                force = np.zeros(6)
                mujoco.mj_contactForce(sim._model, data, i, force)
                if not hits or abs(force[0]) == 0:
                    continue
                (r, c), = hits
                assert observation["tactile"][name]["pad_normal_force_n"][r][c] > 0  # the cell this contact falls in reports force
                if pad["cells"][r][c]["surface_point_m"] is not None:
                    assert np.linalg.norm(np.array(pad["cells"][r][c]["surface_point_m"]) - contact.pos) < (.025 if name == "palm" else .01)
                    checked += 1
        assert checked >= 3
    finally:
        sim.close()


def test_local_controller_tool_is_hand_only_and_mounts_its_files():
    assert enabled_tools({"tools": {"local_controller": True}, "simulation": {"robot": "shadow_hand"}}) == ("local_controller",)
    with pytest.raises(RuntimeFailure, match="shadow_hand"):
        enabled_tools({"tools": {"local_controller": True}, "simulation": {"robot": "panda"}})
    assert AGENT_TOOLS["local_controller"][1] == {} and set(public_tool_files("local_controller")) == {"localsim.py"}
    assert "/opt/public/tools/local_controller/localsim.py" in AGENT_TOOLS["local_controller"][0]
