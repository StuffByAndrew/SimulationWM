"""Shadow-hand tactile levels and researcher-only scores; run on a Slurm test node."""
from __future__ import annotations

import io
from pathlib import Path
import zipfile

import numpy as np
import pytest

from sim2sim.candidate_worker import public_observation
from sim2sim.evaluation import evaluate_submission
from sim2sim.robots import HAND_ACTUATORS, HAND_JOINTS
from sim2sim.simulation import Simulation

FAMILY = Path(__file__).resolve().parents[1] / 'environments' / 'shadow_hand'
LEVELS = ('hand_prism', 'hand_prism_offset_com', 'hand_smooth_mesh', 'hand_spiky')
CONFIG = dict(robot='shadow_hand', rgb=False, depth=False, calibration=False, tactile=True, width=64, height=48)


def _candidate(level, edit=None, robot_edit=None):
    scene = (FAMILY/level/'scene.xml').read_text().replace('../robot/', 'robot/')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('scene.xml', edit(scene) if edit else scene)
        for path in sorted((FAMILY/'robot').rglob('*')):
            if path.is_file():
                data = path.read_bytes()
                archive.writestr('robot/'+path.relative_to(FAMILY/'robot').as_posix(),
                                 robot_edit(data) if robot_edit and path.name == 'shadow_hand.xml' else data)
    return buffer.getvalue()


def _evaluate(level, tmp_path, edit=None, robot_edit=None, metrics=('chamfer', 'center_of_mass', 'dynamics')):
    return evaluate_submission(_candidate(level, edit, robot_edit), dict(source_dir=str(FAMILY), source_entrypoint=f'{level}/scene.xml',
        manifest=f'{level}/manifest.toml', output_dir=str(tmp_path), metrics=list(metrics), source_config=CONFIG,
        camera='fixed', width=64, height=48))


@pytest.mark.parametrize('level', LEVELS)
def test_hand_holds_feels_drops_and_resets(level):
    sim = Simulation(FAMILY/level/'scene.xml', CONFIG)
    try:
        start = sim._data.qpos.copy()
        assert sim._model.jnt_type[0] == 0 and sim._model.jnt_qposadr[0] == 0  # the object's free joint leads qpos
        observation = public_observation(sim.observe(), dict(CONFIG, effort=True))
        assert set(observation) == {'time', 'ticks', 'total_ticks', 'proprioception', 'tactile'}
        assert [len(observation['proprioception'][k]) for k in ('joint_positions_rad', 'joint_velocities_rad_s', 'actuator_effort_nm')] == \
            [len(HAND_JOINTS), len(HAND_JOINTS), len(HAND_ACTUATORS)]
        assert observation['tactile']['palm']['total_normal_force_n'] > 0  # the object rests on the palm
        with pytest.raises(ValueError, match='point grasp'):
            sim.action({'op': 'grasp', 'pixel': [1, 1]})
        home = sim._data.ctrl[sim._hand_actuators].copy()
        sim.action({'op': 'observe', 'ticks': 250}, render=False)  # hand and object sag together after reset
        sagged = sim._data.qpos[:3].copy()
        assert np.linalg.norm(sagged - start[:3]) < 5e-3
        sim.action({'op': 'observe', 'ticks': 500}, render=False)
        assert np.linalg.norm(sim._data.qpos[:3] - sagged) < 1e-3  # held still
        curl = home.copy()
        for i, name in enumerate(HAND_ACTUATORS):
            if name[-2:] in ('J3', 'J0') and name[5:7] in ('FF', 'MF', 'RF', 'LF'):
                curl[i] = 1.5708 if name.endswith('3') else 3.0
        felt = sim.action({'op': 'hand_target', 'actuator_targets': curl.tolist(), 'ticks': 500})
        assert sum(np.sum(felt['tactile'][f]['pad_normal_force_n']) for f in ('first', 'middle', 'ring', 'little')) > 0
        tilt = home.copy()
        tilt[HAND_ACTUATORS.index('rh_A_WRJ1')] = -0.698132
        sim.action({'op': 'reset'}, render=False)
        sim.action({'op': 'hand_target', 'actuator_targets': tilt.tolist(), 'ticks': 500}, render=False)
        sim.action({'op': 'observe', 'ticks': 1500}, render=False)
        assert sim._data.qpos[2] < -0.04  # dropped out of the hand (palm face at +0.022, floor at -0.1)
        total = sim.total_ticks
        sim.action({'op': 'reset'}, render=False)
        assert np.array_equal(sim._data.qpos, start) and sim.ticks == 0 and sim.total_ticks == total
        with pytest.raises(ValueError, match='control ranges'):
            sim.action({'op': 'hand_target', 'actuator_targets': [5.0]*len(HAND_ACTUATORS)})
    finally:
        sim.close()


@pytest.mark.parametrize('level', LEVELS)
def test_hand_identity_evaluation_is_zero(level, tmp_path):
    report = _evaluate(level, tmp_path)
    assert report['status'] == 'ok', report.get('error')
    scores = report['scores']
    assert scores['reconstruction']['chamfer_distance_m'] == pytest.approx(0, abs=1e-9)
    assert scores['reconstruction']['center_of_mass_error_m'] == pytest.approx(0, abs=1e-9)
    assert all(p['joint_position_mae_rad'] == 0 and p['tactile_mae_n'] == 0 and p['source_target_contact_ticks'] > 0
               for p in scores['arm_probes'])


def test_hand_wrong_shape_and_fixture_integrity(tmp_path):
    def sphere(scene):
        start = scene.index('<geom name="pebble"')
        return scene[:start] + '<geom type="sphere" size=".025" density="500"/>' + scene[scene.index('/>', start) + 2:]
    scores = _evaluate('hand_smooth_mesh', tmp_path/'sphere', sphere)['scores']
    assert scores['reconstruction']['chamfer_distance_m'] > 0.003 and scores['reconstruction']['center_of_mass_error_m'] > 0.001
    uniform = _evaluate('hand_prism_offset_com', tmp_path/'uniform', lambda s: s.replace('density="3880"', 'density="0"'))['scores']
    assert uniform['reconstruction']['chamfer_distance_m'] == pytest.approx(0, abs=1e-9)
    assert uniform['reconstruction']['center_of_mass_error_m'] > 0.008
    report = _evaluate('hand_prism', tmp_path/'tampered', robot_edit=lambda d: d.replace(b'kp="10"', b'kp="20"'), metrics=('chamfer',))
    assert report['status'] == 'failed' and 'robot fixture' in report['error']
