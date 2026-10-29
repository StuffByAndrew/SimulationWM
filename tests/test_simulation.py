"""Physics and image boundary checks (require a working MuJoCo render backend)."""
import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from sim2sim.simulation import Simulation

ROOT = Path(__file__).resolve().parents[1]


def project(sim, point):
    camera = sim._camera_id
    local = sim._data.cam_xmat[camera].reshape(3, 3).T @ (point - sim._data.cam_xpos[camera])
    scale = np.tan(np.deg2rad(sim._model.cam_fovy[camera]) / 2)
    return [(local[0] / -local[2] / scale * sim.height / sim.width + 1) * sim.width / 2 - .5,
            (1 - local[1] / -local[2] / scale) * sim.height / 2 - .5]


def test_observation_grasp_motion_and_reset():
    sim = Simulation(ROOT / 'environments/blocks/scene.xml', {'width': 160, 'height': 120})
    try:
        obs = sim.observe()
        rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(obs['rgb']))))
        depth = np.asarray(Image.open(io.BytesIO(base64.b64decode(obs['depth']))))
        assert rgb.shape == (120, 160, 3)
        assert depth.shape == (120, 160)
        assert np.min(depth) > 0
        body = sim._body_names.index('block_red')
        body_id = list(sim._constraints)[body]
        before = sim._data.xpos[body_id].copy()
        pixel = project(sim, before + [0, 0, .04])
        assert sim.action({'op': 'grasp', 'pixel': pixel})['grasped']
        sim.action({'op': 'move', 'displacement': [0, 0, .15], 'ticks': 150})
        assert sim._data.xpos[body_id, 2] > before[2] + .10
        sim.action({'op': 'release'})
        assert not sim.observe()['grasped']
        sim.action({'op': 'reset'})
        assert sim.total_ticks == 150
        assert sim.ticks == 0
        np.testing.assert_allclose(sim._data.xpos[body_id], before)
        with pytest.raises(ValueError):
            sim.action({'op': 'move', 'displacement': [0, 0, 1]})
    finally:
        sim.close()


def test_spring_loads_and_private_raycast_misses_table():
    sim = Simulation(ROOT / 'environments/spring/scene.xml', {'width': 160, 'height': 120})
    try:
        assert sim._body_names == ['lever']
        assert not sim.action({'op': 'grasp', 'pixel': [0, 0]})['grasped']
        sim.action({'op': 'wait', 'ticks': 10})
        assert sim.total_ticks == 10
    finally:
        sim.close()


def test_modalities_and_off_center_point_force():
    sim = Simulation(ROOT / 'environments/blocks/scene.xml',
                     {'width': 160, 'height': 120, 'rgb': False, 'depth': True, 'calibration': True})
    try:
        obs = sim.observe()
        assert 'rgb' not in obs and 'depth' in obs and 'calibration' in obs
        assert 'rgb' in sim.observe(full=True)
        body = next(b for b in sim._constraints if sim._model.body(b).name == 'block_red')
        point = sim._data.xpos[body].copy() + [.05, 0, .04]
        sim._grasp_controller.attach(body, point)
        # A connect constraint supplies three translational equations, never a weld.
        eq = sim._grasp[0]
        assert int(sim._model.eq_type[eq]) == 0
        qadr = int(sim._model.body_jntadr[body])
        qposadr = int(sim._model.jnt_qposadr[qadr])
        before_quat = sim._data.qpos[qposadr + 3:qposadr + 7].copy()
        sim.action({'op': 'move', 'displacement': [0, 0, .15], 'ticks': 150})
        assert not np.allclose(sim._data.qpos[qposadr + 3:qposadr + 7], before_quat, atol=.01)
        sim.action({'op': 'observe', 'ticks': 2})
        assert sim.total_ticks == 152
    finally:
        sim.close()
