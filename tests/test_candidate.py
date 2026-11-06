"""Candidate protocol tests; actual trusted-fixture physics runs on Slurm only."""
import base64
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

from sim2sim.candidate_worker import main, state_digest, public_observation
from sim2sim.client import Client


SCENE = '''<mujoco><option timestep="0.002"/><worldbody>
<camera name="own" pos="1.4 -2 1" xyaxes=".819 .573 0 -.2 .28 .94"/>
<geom name="floor" type="plane" size="2 2 .1"/>
<body name="owned_cube" pos="0 0 .3"><freejoint/><geom name="cube" type="box" size=".1 .1 .1" mass="1"/></body>
</worldbody></mujoco>'''


def test_grasp_overlay_projection_is_private_and_does_not_step(tmp_path):
    import numpy as np
    from sim2sim.simulation import Simulation
    from sim2sim.candidate_worker import snapshot_simulation
    scene=tmp_path/'scene.xml';scene.write_text(SCENE)
    sim=Simulation(scene,{'camera':'own','width':96,'height':64})
    try:
        body=sim._model.body('owned_cube').id
        sim._grasp_controller.attach(body,sim._data.xpos[body].copy())
        before=sim.total_ticks,float(sim._data.time)
        first=snapshot_simulation(sim)['grasp_overlay']
        assert np.allclose(first['anchor'],first['handle'])
        assert (sim.total_ticks,float(sim._data.time))==before
        sim.action({'op':'move','displacement':[.1,0,0],'ticks':10},render=False)
        second=snapshot_simulation(sim)['grasp_overlay']
        assert not np.allclose(first['handle'],second['handle'])
        assert 'grasp_overlay' not in sim.observe(full=True)
        assert 'grasp_overlay' not in public_observation({'grasp_overlay':second},sim.config)
        sim.action({'op':'release'},render=False)
        assert snapshot_simulation(sim)['grasp_overlay'] is None
    finally:
        sim.close()


def test_persistent_worker_trusted_fixture_partial_stop_and_reset(tmp_path):
    import numpy as np
    from sim2sim.simulation import Simulation
    project = io.BytesIO()
    with zipfile.ZipFile(project, "w") as archive:
        archive.writestr("scene.xml", SCENE)
    messages = [{"op": "load", "project_b64": base64.b64encode(project.getvalue()).decode(),
                 "config": {"camera": "own", "width": 96, "height": 64, "timestep": .002,
                            "rgb": True, "depth": False, "calibration": True, "recording_fps": 30}},
                {"op": "action", "action": {"op": "observe", "ticks": 20}}]
    messages += [{"continue": True} for _ in range(18)] + [{"continue": False},
                {"op": "action", "action": {"op": "reset", "ticks": 0}}, {"op": "close"}]
    reference_path = tmp_path / "reference.xml"
    reference_path.write_text(SCENE)
    reference = Simulation(reference_path, {"camera": "own", "width": 96, "height": 64})
    initial = reference._data.qpos.copy()
    packets = []
    try:
        with patch("sim2sim.candidate_worker._read", side_effect=messages), patch("sim2sim.candidate_worker._emit", side_effect=packets.append):
            # Explicit trusted-fixture injection only: production main() always uses
            # /workspace/project inside the resource-verified OCI worker.
            main(project_dir=tmp_path / "candidate_project")
        np.testing.assert_array_equal(reference._data.qpos, initial)
        assert reference.total_ticks == 0
    finally:
        reference.close()
    ticks = [packet for packet in packets if packet["type"] == "tick"]
    assert len(ticks) == 19
    assert [packet["state"]["total_ticks"] for packet in ticks] == list(range(1, 20))
    assert sum("full_observation" in packet for packet in ticks) == 1
    results = [packet for packet in packets if packet["type"] == "result"]
    assert results[0]["ok"]
    assert not results[1]["ok"] and results[1]["executed_ticks"] == 19
    assert results[2]["ok"] and results[2]["state"]["ticks"] == 0 and results[2]["state"]["total_ticks"] == 19
    for packet in packets:
        if "state" in packet:
            assert packet["state_sha256"] == state_digest(packet["state"])
    assert "depth" not in results[0]["observation"]
    assert "depth" in results[0]["full_observation"]


def test_candidate_refuses_private_source_configuration(tmp_path):
    project = io.BytesIO()
    with zipfile.ZipFile(project, "w") as archive:
        archive.writestr("scene.xml", SCENE)
    messages = [{"op": "load", "project_b64": base64.b64encode(project.getvalue()).decode(),
                 "config": {"camera": "own", "graspable_bodies": ["secret_reference_body"]}}, {"op": "close"}]
    packets = []
    with patch("sim2sim.candidate_worker._read", side_effect=messages), patch("sim2sim.candidate_worker._emit", side_effect=packets.append), \
         patch("sim2sim.simulation.Simulation") as simulation:
        main(project_dir=tmp_path / "candidate_project")
    simulation.assert_not_called()
    assert not packets[0]["ok"]


def test_client_routes_identical_actions_and_loads_independently(tmp_path):
    writer = io.StringIO()
    reader = io.StringIO(json.dumps({"submission": {"max_bytes": 4096}}) + "\n" +
                         '{"ok":true,"observation":{}}\n' * 5 + '{"ok":true,"version":"own"}\n')
    client = Client(reader, writer)
    client.observe(2, world="reconstruction")
    client.grasp([1, 2], world="reconstruction")
    client.move([0, 0, .1], ticks=3, world="reconstruction")
    client.release(4, world="reconstruction")
    client.reset(world="reconstruction")
    (tmp_path / "scene.xml").write_text(SCENE)
    client.load_reconstruction(tmp_path, camera="own")
    requests = [json.loads(line) for line in writer.getvalue().splitlines()]
    assert all(request["world"] == "reconstruction" for request in requests)
    assert [request["op"] for request in requests] == ["observe", "grasp", "move", "release", "reset", "load_reconstruction"]
    assert requests[-1]["camera"] == "own" and requests[-1]["entrypoint"] == "scene.xml"


def test_public_redaction_keeps_sensing_independent():
    complete = {"rgb": "r", "rgb_format": "png_rgb8", "depth": "d", "depth_format": "png_uint16_mm",
                "calibration": {}, "wrench": {"force": [1, 2, 3]}, "ticks": 1,
                "original_mass": 1234, "body_hits": [1,2], "qpos": [0,0,1]}
    assert public_observation(complete, {"rgb": False, "depth": True, "calibration": False, "wrench": False}) == {
        "depth": "d", "depth_format": "png_uint16_mm", "ticks": 1}


def test_transport_preserves_partial_json_on_deadline():
    import subprocess
    import sys
    import time
    from sim2sim.runtime import Transport, RuntimeFailure
    records = []
    proc = subprocess.Popen([sys.executable, "-u", "-c", "import sys,time;sys.stdout.write('{\"op\":');sys.stdout.flush();time.sleep(2)"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        transport = Transport(proc, 1024, on_receive=lambda raw, complete: records.append((raw, complete)))
        try:
            transport.receive(time.monotonic()+.1)
            raise AssertionError("Incomplete JSON must time out")
        except RuntimeFailure:
            pass
        assert records == [(b'{"op":', False)]
    finally:
        proc.kill()
        proc.wait()
        proc.stdin.close()
        proc.stdout.close()


def test_blind_panda_observation_curates_only_robot_sensors():
    complete = {"width": 640, "height": 480, "rgb": "private", "rgb_format": "png_rgb8",
                "depth": "private", "depth_format": "png_uint16_mm", "calibration": {},
                "grasped": False, "wrench": {}, "time": 1., "ticks": 2, "total_ticks": 3,
                "qpos": [999], "proprioception": {"joint_positions_rad": [0]*7,
                "gripper_width_m": .04, "object_mass": 123, "body_com": [1,2,3]}}
    assert public_observation(complete, {"robot": "panda", "rgb": False, "depth": False,
                                       "calibration": False, "wrench": True}) == {
        "time": 1., "ticks": 2, "total_ticks": 3,
        "proprioception": {"joint_positions_rad": [0]*7, "gripper_width_m": .04}}
