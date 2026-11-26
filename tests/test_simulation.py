"""Physics and image boundary checks (require a working MuJoCo render backend)."""
import base64
import json
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


def test_two_point_grasps_hold_move_and_sense_independently():
    one = Simulation(ROOT / 'environments/blocks/scene.xml', {'width': 160, 'height': 120})
    sim = Simulation(ROOT / 'environments/blocks/scene.xml', {'width': 160, 'height': 120, 'wrench': True, 'point_grasps': 2})
    try:
        assert sim._model.neq - one._model.neq == len(sim._body_names) and sim._model.nmocap == 2 * one._model.nmocap
        assert sim._model.equality(one._model.neq).name == '_grasp_connect_0_1'
        red, blue = (next(b for b in sim._constraints if sim._model.body(b).name == name) for name in ('block_red', 'block_blue'))
        starts = sim._data.xpos[[red, blue]].copy()
        for index, body in enumerate((red, blue)):
            assert sim.action({'op': 'grasp', 'pixel': project(sim, sim._data.xpos[body] + [0, 0, .04]), 'grasp_index': index}, render=False) is None
        observation = sim.observe()
        assert observation['grasped'] == [True, True] and len(observation['wrench']) == 2
        from sim2sim.candidate_worker import snapshot_simulation
        state = snapshot_simulation(sim)
        assert state['grasp'] == state['grasps'][0] and all(o['anchor'] for o in state['grasp_overlays'])
        assert not {'grasps', 'grasp_overlays'} & set(snapshot_simulation(one))
        sim.action({'op': 'move', 'displacement': [0, 0, .15], 'ticks': 150, 'grasp_index': 1}, render=False)
        assert sim._data.xpos[blue, 2] > starts[1, 2] + .1 and abs(sim._data.xpos[red, 2] - starts[0, 2]) < .01
        held, lifted = sim.observe()['wrench']
        mass = sim._model.body_subtreemass[blue]
        assert -1.2 * mass * 9.81 < lifted['force'][2] < -.8 * mass * 9.81  # grasp 1 carries the (still swinging) lifted block
        assert held['active'] and abs(held['force'][2]) < lifted['force'][2] * -.5 and held['torque'] == lifted['torque'] == [0, 0, 0]
        with pytest.raises(ValueError, match='release the current grasp'):
            sim.action({'op': 'grasp', 'pixel': [0, 0], 'grasp_index': 1})
        for bad in (2, -1, True, 1.0):
            with pytest.raises(ValueError, match='grasp_index'):
                sim.action({'op': 'release', 'grasp_index': bad})
        sim.action({'op': 'release', 'grasp_index': 1})
        assert sim.observe()['grasped'] == [True, False]
        sim.action({'op': 'reset'})
        assert sim.observe()['grasped'] == [False, False] and sim.total_ticks == 150
        with pytest.raises(ValueError, match='point_grasps'):
            Simulation(ROOT / 'environments/blocks/scene.xml', {'point_grasps': 3})
        assert one.observe()['grasped'] is False
        with pytest.raises(ValueError, match='grasp_index'):
            one.action({'op': 'grasp', 'pixel': [0, 0], 'grasp_index': 1})
    finally:
        one.close()
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


@pytest.mark.parametrize('grasp', ['connect', 'weld'])
def test_wrench_on_a_jointed_body_excludes_blocked_directions(tmp_path, grasp):
    """A hinged lever held at its tip, the grip pushed 1 mm outward along the blocked radial direction: the reading is the
    force that balances the lever (0.5 m g down at the tip), not the 10^11 N the soft constraint showed along the blocked
    direction (gear-train hand, wave 28)."""
    scene = tmp_path / 'lever.xml'
    scene.write_text('''<mujoco><option timestep=".001"/>
      <worldbody><camera name="fixed" pos="0 -3 1" xyaxes="1 0 0 0 0 1"/>
      <body name="lever" pos="0 0 1"><joint type="hinge" axis="0 1 0" damping="1"/>
        <geom type="box" size=".2 .01 .01" pos=".2 0 0" mass="1"/></body>
      </worldbody></mujoco>''')
    sim = Simulation(scene, {'width': 80, 'height': 60, 'wrench': True, 'grasp': grasp})
    try:
        lever = mujoco.mj_name2id(sim._model, mujoco.mjtObj.mjOBJ_BODY, 'lever')
        sim._grasp_controller.attach(lever, sim._data.xpos[lever] + [.4, 0, 0])
        sim.action({'op': 'move', 'displacement': [.001, 0, 0], 'ticks': 50})
        sim._advance(2000)
        wrench = sim.observe()['wrench']
        assert wrench['active']
        if grasp == 'connect':
            np.testing.assert_allclose(wrench['force'], [0, 0, -.5 * 9.81], atol=.05)
            assert wrench['torque'] == [0, 0, 0]
        else:
            assert np.linalg.norm(wrench['force']) + np.linalg.norm(wrench['torque']) < 10
            assert_wrench_is_the_weld_rows(sim)
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


WELD_SCENE = '''<mujoco><option timestep=".002"/>
  <worldbody><camera name="fixed" pos="0 -2 .5" xyaxes="1 0 0 0 0 1"/>
  <body name="box" pos="0 0 .5"><freejoint/>
    <inertial pos=".04 0 0" mass=".5" diaginertia=".001 .002 .002"/>
    <geom type="box" size=".06 .03 .02" contype="0" conaffinity="0"/></body>
  <body name="bar" pos=".5 0 .5"><joint name="hinge" type="hinge" axis="0 0 1"/>
    <geom type="box" size=".1 .02 .02" mass=".3" contype="0" conaffinity="0"/></body>
  </worldbody></mujoco>'''
WELD = {'width': 64, 'height': 48, 'rgb': False, 'calibration': False, 'wrench': True, 'grasp': 'weld'}


def weld_sim(tmp_path, **config):
    (tmp_path / 'weld.xml').write_text(WELD_SCENE)
    sim = Simulation(tmp_path / 'weld.xml', {**WELD, **config})
    return sim, *(mujoco.mj_name2id(sim._model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ('box', 'bar'))


def assert_wrench_is_the_weld_rows(sim, index=0):
    """The reported object-on-handle force and couple at the pinned point reproduce, through the body Jacobian, exactly the
    generalized force of that grasp's own weld rows (J^T efc_force)."""
    model, data = sim._model, sim._data
    eq = sim._grasps[index][0]
    rows = np.flatnonzero((data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (data.efc_id == eq))
    vector, qfrc = np.zeros(data.nefc), np.zeros(model.nv)
    vector[rows] = data.efc_force[rows]
    mujoco.mj_mulJacTVec(model, data, qfrc, vector)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, sim.grasp_anchor(sim._grasps[index]), model.eq_obj1id[eq])
    wrench = sim._wrench(index)
    assert len(rows) == 6 and wrench['active']
    np.testing.assert_allclose(qfrc, -(jacp.T @ wrench['force'] + jacr.T @ wrench['torque']), atol=1e-9)


def twist(a, b):
    error = np.zeros(4)
    mujoco.mju_mulQuat(error, a, np.array([b[0], -b[1], -b[2], -b[3]]))
    return 2 * np.arctan2(np.linalg.norm(error[1:]), abs(error[0]))


def test_weld_grasp_holds_still_rotates_and_reports_offset_com_torque(tmp_path):
    sim, box, _ = weld_sim(tmp_path)
    try:
        assert sim._model.eq_type[sim._constraints[box][0]] == mujoco.mjtEq.mjEQ_WELD
        assert sim._model.equality(sim._constraints[box][0]).name.startswith('_grasp_weld_')
        assert sim.action({'op': 'grasp', 'pixel': project(sim, sim._data.xpos[box] + [0, -.03, 0])})['grasped']
        sim.action({'op': 'release'})
        sim.action({'op': 'reset'})
        sim._grasp_controller.attach(box, sim._data.xpos[box] + [0, 0, .02])  # top centre; the COM is .04 m off along body x
        sim._advance(1000)
        settled = sim._data.qpos.copy()
        sim._advance(500)
        drift, speed = np.abs(sim._data.qpos - settled).max(), np.abs(sim._data.qvel[:6]).max()
        print(f'weld hold: drift {drift:.2e}, speed {speed:.2e}, stretch {sim.grasp_stretch():.2e} m, twist {sim.grasp_twist():.2e} rad')
        assert drift < 1e-5 and speed < 1e-4 and sim.grasp_stretch() < 2e-3 and sim.grasp_twist() < .02
        weight = .5 * np.array([0, 0, -9.81])  # object on handle: the object's weight, and its moment about the grasp point
        arm = sim._data.xipos[box] - sim.grasp_anchor(sim._grasp)
        wrench = sim.observe()['wrench']
        print('weld wrench', wrench['force'], wrench['torque'], 'analytic', weight.tolist(), np.cross(arm, weight).tolist())
        np.testing.assert_allclose(wrench['force'], weight, atol=1e-4)
        np.testing.assert_allclose(wrench['torque'], np.cross(arm, weight), atol=1e-4)
        assert abs(wrench['torque'][1]) > .19 and wrench['frame'] == 'world' and wrench['reference'] == 'grasp_point'
        assert_wrench_is_the_weld_rows(sim)
        start, anchor = sim._data.xquat[box].copy(), sim.grasp_anchor(sim._grasp).copy()
        turn = np.zeros(4)
        mujoco.mju_axisAngle2Quat(turn, [.6, 0, .8], 1.2)
        target = np.zeros(4)
        mujoco.mju_mulQuat(target, turn, start)
        sim.action({'op': 'move', 'displacement': [0, 0, 0], 'rotation': (1.2 * np.array([.6, 0, .8])).tolist(), 'ticks': 300}, render=False)
        assert_wrench_is_the_weld_rows(sim)  # also mid-motion, rotated off the axes
        sim._advance(300)
        error, moved = twist(sim._data.xquat[box], target), np.linalg.norm(sim.grasp_anchor(sim._grasp) - anchor)
        print(f'weld rotation 1.2 rad: orientation error {error:.3e} rad, grasp point moved {moved:.2e} m')
        assert error < .03 and moved < 3e-3
        sim.action({'op': 'move', 'displacement': [.1, 0, .05], 'rotation': [0, 0, -1], 'ticks': 300}, render=False)  # a helical move
        sim._advance(300)
        np.testing.assert_allclose(sim.grasp_anchor(sim._grasp), anchor + [.1, 0, .05], atol=3e-3)
        mujoco.mju_axisAngle2Quat(turn, [0, 0, 1], -1)
        mujoco.mju_mulQuat(target, turn, target.copy())
        assert twist(sim._data.xquat[box], target) < .03
        with pytest.raises(ValueError, match='rotation'):
            sim.action({'op': 'move', 'displacement': [0, 0, 0], 'rotation': [0, 1]})
        sim.action({'op': 'reset'})
        assert sim.observe()['wrench'] == {**wrench, 'force': [0, 0, 0], 'torque': [0, 0, 0], 'active': False}
    finally:
        sim.close()


def test_weld_grasp_configuration_and_connect_unchanged(tmp_path):
    (tmp_path / 'weld.xml').write_text(WELD_SCENE)
    for config in ({'grasp': 'hinge'}, {'grasp': 'weld', 'robot': 'panda'}, {'grasp_slip_rad': .2}, {'grasp': 'weld', 'grasp_slip_rad': 4},
                   {'grasp': 'weld', 'grasp_slip_rad': True}, {'grasp_slip_m': 0}):
        with pytest.raises(ValueError, match='grasp'):
            Simulation(tmp_path / 'weld.xml', config)
    with pytest.raises(ValueError, match='flex'):
        Simulation(ROOT / 'environments/cloth_quadrants/scene.xml', {'grasp': 'weld'})
    sim = Simulation(tmp_path / 'weld.xml', {**WELD, 'grasp': 'connect'})
    try:
        box = mujoco.mj_name2id(sim._model, mujoco.mjtObj.mjOBJ_BODY, 'box')
        assert sim.grasp_type == 'connect' and sim._model.equality(sim._constraints[box][0]).name == '_grasp_connect_0'
        sim._grasp_controller.attach(box, sim._data.xpos[box] + [0, 0, .02])
        with pytest.raises(ValueError, match='rotation'):
            sim.action({'op': 'move', 'displacement': [0, 0, 0], 'rotation': [0, 0, 1]})
        sim._advance(10)
        assert sim.observe()['wrench']['torque'] == [0, 0, 0]
    finally:
        sim.close()


def test_weld_grasp_slips_on_stretch_and_on_twist(tmp_path):
    sim, _, bar = weld_sim(tmp_path, grasp_slip_m=.01, grasp_slip_rad=.2)
    hinge = sim._model.jnt_qposadr[sim._model.body_jntadr[bar]]
    try:
        on_axis = sim._data.xpos[bar] + [0, 0, .02]
        sim._grasp_controller.attach(bar, on_axis)
        sim.action({'op': 'move', 'displacement': [0, 0, 0], 'rotation': [0, 0, 1], 'ticks': 300}, render=False)  # about the hinge
        sim._advance(200)
        assert sim._grasp is not None and abs(sim._data.qpos[hinge] - 1) < .02, sim._data.qpos[hinge]
        readings = []
        sim._callback = lambda s: readings.append((s.grasp_stretch(), s.grasp_twist()) if s._grasp is not None else None)
        sim.action({'op': 'move', 'displacement': [0, 0, 0], 'rotation': [.6, 0, 0], 'ticks': 100}, render=False)  # the hinge resists
        held = [reading for reading in readings if reading is not None]
        assert sim._grasp is None and None in readings and len(held) < 100, held[-3:]
        assert .15 < held[-1][1] <= .2 and max(stretch for stretch, _ in held) < .01, held[-3:]  # the twist let go
        assert sim.observe()['grasped'] is False and sim.observe()['wrench']['active'] is False
        sim._callback = None
        sim._grasp_controller.attach(bar, sim._data.xpos[bar] + [0, 0, .02])
        sim.action({'op': 'move', 'displacement': [0, 0, .05], 'ticks': 100}, render=False)  # the hinge resists lifting
        assert sim._grasp is None
        with pytest.raises(ValueError, match='active grasp'):
            sim.action({'op': 'move', 'displacement': [0, 0, 0], 'rotation': [0, 0, 1]})
    finally:
        sim.close()
    sim, _, bar = weld_sim(tmp_path)  # without slip settings the weld never lets go
    try:
        sim._grasp_controller.attach(bar, sim._data.xpos[bar] + [0, 0, .02])
        sim.action({'op': 'move', 'displacement': [0, 0, .05], 'rotation': [.6, 0, 0], 'ticks': 100}, render=False)
        assert sim._grasp is not None and np.isfinite(sim._data.qpos).all()
    finally:
        sim.close()


def test_weld_probe_rotation_is_split_over_the_probe_moves(tmp_path):
    from sim2sim.evaluation import _probe_actions
    sim, box, _ = weld_sim(tmp_path)
    try:
        actions = _probe_actions(sim, {'body': 'box', 'local_point': [0, -.03, 0], 'displacement': [0, 0, .04], 'rotation': [0, 0, .8]})
        moves = [action for action in actions if action['op'] == 'move']
        assert len(moves) == 4 and all(move['rotation'] == [0, 0, .2] for move in moves)
        assert 'rotation' not in json.dumps(_probe_actions(sim, {'body': 'box', 'local_point': [0, -.03, 0]}))
        start = sim._data.xquat[box].copy()
        for action in actions[:6] + [{'op': 'wait', 'ticks': 200}]:
            sim.action(action, render=False)
        assert sim._grasp is not None and abs(twist(sim._data.xquat[box], start) - .8) < .05
    finally:
        sim.close()


def test_two_welds_on_one_body_share_the_load_without_blowing_up(tmp_path):
    sim, box, _ = weld_sim(tmp_path, point_grasps=2)
    try:
        points = [sim._data.xpos[box] + [-.05, 0, .02], sim._data.xpos[box] + [.05, 0, .02]]
        for index, point in enumerate(points):
            sim._grasp_controller.attach(box, point, index)
        sim._advance(1000)
        settled = sim._data.qpos.copy()
        sim._advance(300)
        assert np.abs(sim._data.qpos - settled).max() < 1e-5 and np.abs(sim._data.qvel[:6]).max() < 1e-4
        first, second = sim.observe()['wrench']
        anchors = [sim.grasp_anchor(grasp) for grasp in sim._grasps]
        weight = .5 * np.array([0, 0, -9.81])
        np.testing.assert_allclose(np.add(first['force'], second['force']), weight, atol=1e-4)
        moment = np.add(first['torque'], second['torque']) + np.cross(anchors[1] - anchors[0], second['force'])
        np.testing.assert_allclose(moment, np.cross(sim._data.xipos[box] - anchors[0], weight), atol=1e-4)
        for index in range(2):
            assert_wrench_is_the_weld_rows(sim, index)
        # Closed loop: moving and turning only grasp 0 fights grasp 1; the soft welds strain but stay finite and settle.
        sim.action({'op': 'move', 'displacement': [.03, 0, .03], 'rotation': [0, .3, 0], 'ticks': 200, 'grasp_index': 0}, render=False)
        sim._advance(1000)
        assert np.isfinite(sim._data.qpos).all() and np.abs(sim._data.qvel[:6]).max() < 1e-2
        print('two welds after a conflicting move: stretches', [sim.grasp_stretch(i) for i in range(2)],
              'twists', [sim.grasp_twist(i) for i in range(2)])
        assert all(np.isfinite(reading['torque']).all() for reading in sim.observe()['wrench'])
        sim.action({'op': 'move', 'displacement': [.05, 0, 0], 'ticks': 100, 'grasp_index': 1}, render=False)
        sim._advance(100)
        assert np.isfinite(sim._data.qpos).all()
    finally:
        sim.close()
