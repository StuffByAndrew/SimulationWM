"""Physics and image boundary checks (require a working MuJoCo render backend)."""
import base64
import io
from pathlib import Path

import mujoco
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
        obs = sim.observe(full=True)
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
        assert sim.action({'op': 'reset'}, render=False) is None
        with pytest.raises(ValueError):
            sim.action({'op': 'move', 'displacement': [0, 0, 1]})
    finally:
        sim.close()


def test_spring_loads_and_private_raycast_misses_table():
    sim = Simulation(ROOT / 'environments/spring/scene.xml', {'width': 160, 'height': 120})
    try:
        assert sim._body_names == ['lever']
        defaults = sim.observe()
        assert 'rgb' in defaults and 'calibration' in defaults and 'depth' not in defaults
        assert 'wrench' not in defaults and 'wrench' not in sim.observe(full=True)
        assert not sim.action({'op': 'grasp', 'pixel': [0, 0]})['grasped']
        sim.action({'op': 'wait', 'ticks': 10})
        assert sim.total_ticks == 10
    finally:
        sim.close()


def test_modalities_and_off_center_point_force():
    sim = Simulation(ROOT / 'environments/blocks/scene.xml',
                     {'width': 160, 'height': 120, 'rgb': False, 'depth': True, 'calibration': True, 'wrench': True})
    try:
        obs = sim.observe()
        assert 'rgb' not in obs and 'depth' in obs and 'calibration' in obs
        assert 'rgb' in sim.observe(full=True)
        assert sim.observe()['wrench']['torque'] == [0, 0, 0]
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
        assert sim.observe()['wrench']['torque'] == [0, 0, 0]
    finally:
        sim.close()


def test_included_bodies_and_fixed_camera_validation(tmp_path):
    (tmp_path / 'parts').mkdir()
    (tmp_path / 'parts' / 'object.xml').write_text('''<mujocoinclude>
      <body name="included" pos="0 0 1"><freejoint/>
      <geom type="sphere" size=".1" mass="1"/></body></mujocoinclude>''')
    (tmp_path / 'world.xml').write_text('''<mujocoinclude>
      <camera name="fixed" pos="0 -3 1" xyaxes="1 0 0 0 0 1"/>
      <include file="parts/object.xml"/></mujocoinclude>''')
    scene = tmp_path / 'scene.xml'
    scene.write_text('<mujoco><worldbody><include file="world.xml"/></worldbody></mujoco>')
    sim = Simulation(scene, {'width': 80, 'height': 60})
    try:
        assert sim._body_names == ['included']
    finally:
        sim.close()
    scene.write_text('''<mujoco><worldbody><body name="moving"><freejoint/>
      <geom type="sphere" size=".1"/><camera name="fixed"/></body></worldbody></mujoco>''')
    with pytest.raises(ValueError, match='fixed directly in worldbody'):
        Simulation(scene)


def test_optional_wrench_hanging_load_sign_and_release(tmp_path):
    scene = tmp_path / 'hanging.xml'
    scene.write_text('''<mujoco><option timestep=".002"/>
      <worldbody><camera name="fixed" pos="0 -3 1" xyaxes="1 0 0 0 0 1"/>
      <geom type="plane" size="3 3 .1"/>
      <body name="load" pos="0 0 1"><freejoint/>
        <geom type="sphere" size=".1" mass="2"/></body>
      <body name="other" pos="1 0 .1"><freejoint/>
        <geom type="sphere" size=".1" mass="5"/></body>
      </worldbody></mujoco>''')
    sim = Simulation(scene, {'width': 80, 'height': 60, 'wrench': True})
    try:
        idle = sim.observe()['wrench']
        assert idle['force'] == idle['torque'] == [0, 0, 0]
        assert not idle['active']
        body = next(b for b in sim._constraints if sim._model.body(b).name == 'load')
        sim._grasp_controller.attach(body, sim._data.xpos[body].copy())
        sim._advance(500)
        before = (sim.total_ticks, sim.ticks, float(sim._data.time))
        observation = sim.observe()
        wrench = observation['wrench']
        assert wrench['active']
        assert wrench['frame'] == 'world'
        assert wrench['reference'] == 'grasp_point'
        assert wrench['sign'] == 'object_on_handle'
        assert wrench['force_unit'] == 'N' and wrench['torque_unit'] == 'N m'
        np.testing.assert_allclose(wrench['force'], [0, 0, -2 * 9.81], atol=.05)
        assert wrench['torque'] == [0, 0, 0]
        assert (sim.total_ticks, sim.ticks, float(sim._data.time)) == before
        assert sim.observe()['wrench']['force'] == pytest.approx(wrench['force'])
        released = sim.action({'op': 'release'})['wrench']
        assert not released['active'] and released['force'] == released['torque'] == [0, 0, 0]
        reset = sim.action({'op': 'reset'})['wrench']
        assert not reset['active'] and reset['force'] == reset['torque'] == [0, 0, 0]
    finally:
        sim.close()


def test_wrench_toggle_does_not_change_dynamics():
    disabled = Simulation(ROOT / 'environments/blocks/scene.xml', {'width': 80, 'height': 60})
    enabled = Simulation(ROOT / 'environments/blocks/scene.xml', {'width': 80, 'height': 60, 'wrench': True})
    try:
        body = next(b for b in disabled._constraints if disabled._model.body(b).name == 'block_red')
        pixel = project(disabled, disabled._data.xpos[body] + [.04, 0, .04])
        actions = [
            {'op': 'grasp', 'pixel': pixel},
            {'op': 'move', 'displacement': [0, 0, .12], 'ticks': 80},
            {'op': 'observe', 'ticks': 10},
            {'op': 'move', 'displacement': [.08, .02, 0], 'ticks': 50},
            {'op': 'release', 'ticks': 30},
            {'op': 'reset'},
        ]
        for action in actions:
            for sim in (disabled, enabled):
                sim.action(action)
                for _ in range(3):
                    sim.observe()
            for field in ('qpos', 'qvel', 'qacc_warmstart'):
                np.testing.assert_allclose(getattr(enabled._data, field), getattr(disabled._data, field),
                                           rtol=0, atol=1e-10)
            assert enabled.ticks == disabled.ticks
            assert enabled.total_ticks == disabled.total_ticks
            assert enabled._data.time == disabled._data.time
    finally:
        disabled.close()
        enabled.close()


def test_panda_named_control_reset_and_private_proprioception(tmp_path):
    # Deliberately put an object free joint before the robot: indices must be
    # resolved by names, never by assuming robot state starts at qpos zero.
    scene = tmp_path / 'panda.xml'
    bodies = ''.join(f'<body pos="0 0 .1"><joint name="joint{i}" range="-3 3"/>'
                     '<geom type="sphere" size=".03" mass="1"/>' for i in range(1, 8))
    fingers = ''.join(f'<body pos="{sign} 0 .1"><joint name="finger_joint{i}" '
                      'type="slide" axis="1 0 0" range="0 .04"/>'
                      '<geom type="sphere" size=".02" mass=".1"/></body>'
                      for i, sign in ((1, '.04'), (2, '-.04')))
    actuators = ''.join(f'<position name="actuator{i}" joint="joint{i}" kp="10"/>'
                        for i in range(1, 8))
    scene.write_text('<mujoco><compiler angle="radian"/><worldbody>'
                     '<camera name="fixed" pos="0 -3 1" xyaxes="1 0 0 0 0 1"/>'
                     '<body name="secret_object" pos="1 0 1"><freejoint/>'
                     '<geom type="sphere" size=".1" mass="1"/></body>'
                     + bodies + fingers + '</body>' * 7 + '</worldbody><actuator>'
                     + actuators + '<position name="actuator8" joint="finger_joint1" kp="1"/>'
                     '</actuator></mujoco>')
    controls = []
    sim = Simulation(scene, {'robot': 'panda', 'rgb': False, 'depth': False,
                            'calibration': False, 'recording_callback':
                            lambda s: controls.append(s._data.ctrl.copy())})
    try:
        obs = sim.observe()
        assert sim._renderer is None
        assert not sim._constraints and sim._model.neq == 0
        assert set(obs) == {'width', 'height', 'time', 'ticks', 'total_ticks', 'grasped', 'proprioception'}
        home = np.array([0, 0, 0, -1.57079, 0, 1.57079, -.7853])
        np.testing.assert_allclose(obs['proprioception']['joint_positions_rad'], home)
        assert obs['proprioception']['gripper_opening'] == pytest.approx(1)
        object_initial = sim._data.qpos[:7].copy()
        target = home + .02
        sim.action({'op': 'arm_target', 'joint_targets': target.tolist(),
                    'gripper_width_m': .04, 'ticks': 4}, render=False)
        assert sim.total_ticks == sim.ticks == len(controls) == 4
        np.testing.assert_allclose(controls[0][:7], home + .005)
        np.testing.assert_allclose(controls[-1][:7], target)
        assert controls[-1][7] == pytest.approx(127.5)
        before = {field: getattr(sim._data, field).copy() for field in ('qpos', 'qvel', 'qacc_warmstart', 'ctrl')}
        sim.observe()
        for field, value in before.items():
            np.testing.assert_array_equal(getattr(sim._data, field), value)
        for op in ('grasp', 'move', 'release'):
            with pytest.raises(ValueError, match='disabled'):
                sim.action({'op': op}, render=False)
        for bad in (0, -1, 100001, True, 1.5):
            with pytest.raises(ValueError, match='ticks'):
                sim.action({'op': 'arm_target', 'joint_targets': home.tolist(),
                            'gripper_width_m': .08, 'ticks': bad}, render=False)
        sim.action({'op': 'reset'}, render=False)
        assert sim.total_ticks == 4 and sim.ticks == 0
        np.testing.assert_allclose(sim._data.qpos[sim._arm_qpos], home)
        np.testing.assert_allclose(sim._data.qpos[:7], object_initial)
        np.testing.assert_allclose(sim._data.ctrl[:7], home)
        assert sim._data.ctrl[7] == 255
        assert 'rgb' in sim.observe(full=True)
    finally:
        sim.close()


def test_movable_camera_reaims_without_physics_and_is_opt_in():
    scene = ROOT / 'environments/blocks/scene.xml'
    sim = Simulation(scene, {'width': 64, 'height': 48, 'movable_camera': True})
    before = sim.observe()
    state, ticks, time = sim._data.qpos.copy(), sim.total_ticks, sim._data.time
    moved = sim.action({'op': 'camera', 'pos': [0, -1.5, 1.2], 'lookat': [0, 0, .4]})
    pose = np.array(moved['calibration']['world_from_camera'])
    forward = -pose[:3, 2]
    assert np.allclose(pose[:3, 3], [0, -1.5, 1.2]) and np.allclose(forward, np.array([0, 1.5, -.8]) / np.linalg.norm([0, 1.5, -.8]))
    assert pose[2, 1] > 0 and moved['rgb'] != before['rgb']  # camera up has +z; the view changed
    assert np.array_equal(state, sim._data.qpos) and sim.total_ticks == ticks and sim._data.time == time
    sim.action({'op': 'reset'})
    assert np.allclose(np.array(sim.observe()['calibration']['world_from_camera'])[:3, 3], [0, -1.5, 1.2])  # persists
    for bad in ({'pos': [0, 0, 9], 'lookat': [0, 0, 0]}, {'pos': [0, 0, 1], 'lookat': [0, 0, 1.01]}, {'pos': [0, 0], 'lookat': [0, 0, 0]}):
        with pytest.raises(ValueError):
            sim.action({'op': 'camera', **bad})
    with pytest.raises(ValueError, match='movable_camera'):
        Simulation(scene, {'width': 64, 'height': 48}).action({'op': 'camera', 'pos': [0, -1, 1], 'lookat': [0, 0, 0]})


def test_movable_camera_cannot_clip_into_a_sealed_box():
    # A camera whose near plane could cut a wall, or one inside a geom or the box cavity, would see the hidden contents.
    sim = Simulation(ROOT / 'environments/panda_small/box_marble_equal/scene.xml', {'robot': 'panda', 'width': 64, 'height': 48, 'movable_camera': True})
    box, lookat = sim._data.xpos[sim._model.body('box').id].copy(), [-.3, -.22, .42]
    sim.action({'op': 'camera', 'pos': (box + [.15, 0, .1]).tolist(), 'lookat': lookat})  # 12 cm from the wall, 12 cm above the table: allowed
    for offset in ([.0285 + .01, 0, 0], [.027, 0, 0], [0, 0, 0]):  # 1 cm outside the wall, inside the wall, inside the cavity
        with pytest.raises(ValueError, match='every surface'):
            sim.action({'op': 'camera', 'pos': (box + offset).tolist(), 'lookat': lookat})
    # Objects carried to a camera aimed at a clear spot cover its lens: black image and zero depth, no view inside.
    sim = Simulation(ROOT / 'environments/panda_small/box_marble_equal/scene.xml',
                     {'robot': 'panda', 'width': 64, 'height': 48, 'movable_camera': True, 'depth': True})
    pos = (box + [.15, 0, .1]).tolist()
    seen = sim.action({'op': 'camera', 'pos': pos, 'lookat': lookat})
    assert np.asarray(Image.open(io.BytesIO(base64.b64decode(seen['rgb'])))).any()
    free = sim._model.body('box').jntadr[0]
    sim._data.qpos[sim._model.jnt_qposadr[free]:sim._model.jnt_qposadr[free] + 3] = np.asarray(pos) - [.0285 + .01, 0, 0]
    mujoco.mj_forward(sim._model, sim._data)
    covered = sim.observe()
    for key in ('rgb', 'depth'):
        assert not np.asarray(Image.open(io.BytesIO(base64.b64decode(covered[key])))).any(), key
