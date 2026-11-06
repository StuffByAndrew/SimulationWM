"""Panda wrist observation privacy and kinematics; run on Slurm test."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from sim2sim.candidate_worker import public_observation
from sim2sim.simulation import Simulation

SCENE = Path(__file__).resolve().parents[1] / 'environments/panda_small/blocks/scene.xml'


def test_wrist_camera_tracks_hand_without_observation_ticks():
    sim = Simulation(SCENE, dict(robot='panda', wrist_camera='wrist', width=64, height=48,
                                 rgb=True, depth=True))
    try:
        before = sim.observe()
        assert {'rgb', 'depth', 'calibration', 'wrist_rgb', 'wrist_depth', 'wrist_calibration'} <= before.keys()
        sim._data.qpos[sim._arm_qpos[0]] += .3
        mujoco.mj_forward(sim._model, sim._data)
        after = sim.observe()
        assert sim.ticks == sim.total_ticks == 0 and sim._data.time == 0
        assert before['calibration'] == after['calibration']
        assert not np.allclose(before['wrist_calibration']['world_from_camera'], after['wrist_calibration']['world_from_camera'])
        assert before['wrist_rgb'] != after['wrist_rgb']
        assert public_observation(after, sim.config)['wrist_depth'] == after['wrist_depth']
        blind = dict(sim.config, rgb=False, depth=False, calibration=False)
        public = public_observation(sim.observe(full=True), blind)
        assert not any(key in public for key in ('rgb', 'depth', 'calibration', 'wrist_rgb', 'wrist_depth', 'wrist_calibration'))
        global_only = public_observation(after, dict(sim.config, wrist_camera=None))
        assert not any(key.startswith('wrist_') for key in global_only)

    finally:
        sim.close()


@pytest.mark.parametrize('config,match', [
    (dict(wrist_camera='wrist'), 'requires'),
    (dict(robot='panda', wrist_camera='fixed'), 'Panda hand'),
    (dict(robot='panda', wrist_camera='absent'), 'Panda hand'),
    (dict(robot='panda', camera='wrist'), 'worldbody'),
])
def test_rejects_incorrect_camera_attachments(config, match):
    with pytest.raises(ValueError, match=match):
        Simulation(SCENE, config)


@pytest.mark.parametrize('attachment', ['object', 'tracking'])
def test_rejects_object_and_tracking_wrist_cameras(monkeypatch, attachment):
    import sim2sim.simulation as module
    root = module._load_xml(SCENE)
    camera = root.find('.//camera[@name="wrist"]')
    hand = root.find('.//body[@name="hand"]')
    if attachment == 'object':
        hand.remove(camera)
        root.find('.//body[@name="block_red"]').append(camera)
    else:
        camera.set('mode', 'track')
    monkeypatch.setattr(module, '_load_xml', lambda _: root)
    with pytest.raises(ValueError, match='Panda hand'):
        Simulation(SCENE, dict(robot='panda', wrist_camera='wrist'))


def test_wrist_camera_matches_integrated_arm_state_without_mutating_live_data():
    sim = Simulation(SCENE, dict(robot='panda', wrist_camera='wrist', width=64, height=48,
                                 rgb=True, depth=True))
    try:
        target = sim._data.qpos[sim._arm_qpos].copy()
        target[0] += .5
        observation = sim.action(dict(op='arm_target', joint_targets=target.tolist(),
                                      gripper_width_m=.04, ticks=20))
        fresh = mujoco.MjData(sim._model)
        mujoco.mj_copyData(fresh, sim._model, sim._data)
        mujoco.mj_forward(sim._model, fresh)
        pose = np.eye(4)
        pose[:3, :3] = fresh.cam_xmat[sim._wrist_camera_id].reshape(3, 3)
        pose[:3, 3] = fresh.cam_xpos[sim._wrist_camera_id]
        np.testing.assert_allclose(observation['wrist_calibration']['world_from_camera'], pose, atol=1e-12)
        np.testing.assert_array_equal(observation['proprioception']['joint_positions_rad'], fresh.qpos[sim._arm_qpos])
        state = mujoco.MjData(sim._model)
        mujoco.mj_copyData(state, sim._model, sim._data)
        ticks = (sim.ticks, sim.total_ticks)
        sim.observe()
        assert (sim.ticks, sim.total_ticks) == ticks and sim._data.time == state.time
        for key in ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart', 'qacc', 'cam_xpos', 'cam_xmat'):
            np.testing.assert_array_equal(getattr(sim._data, key), getattr(state, key))
    finally:
        sim.close()
