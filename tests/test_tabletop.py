"""Franka + Shadow hand fixture and the touch tabletop levels' private metrics (src/sim2sim/tabletop.py); run on Slurm."""
from __future__ import annotations

import io
import json
from pathlib import Path
import tomllib
import zipfile

import numpy as np
import pytest

from sim2sim.candidate_worker import allowed_actions, public_observation
from sim2sim.evaluation import evaluate_submission
from sim2sim.robots import HAND_ACTUATORS, joints
from sim2sim.runtime import RuntimeFailure, enabled_tools, public_robot_files
from sim2sim.simulation import Simulation
from sim2sim.tabletop import grab

FAMILY = Path(__file__).resolve().parents[1] / 'environments' / 'panda_shadow'
CONFIG = dict(robot='panda_shadow', rgb=False, depth=False, calibration=False, tactile=True, width=64, height=48)


def _manifest(level):
    return tomllib.loads((FAMILY/level/'manifest.toml').read_text())


def _candidate(level, scene='scene.xml', edit=None, parameters=None):
    text = (FAMILY/level/scene).read_text().replace('../robot/', 'robot/')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('scene.xml', edit(text) if edit else text)
        if parameters is not None:
            archive.writestr('parameters.json', parameters)
        for path in sorted((FAMILY/'robot').rglob('*')):
            if path.is_file():
                archive.writestr('robot/'+path.relative_to(FAMILY/'robot').as_posix(), path.read_bytes())
    return buffer.getvalue()


def _scores(level, tmp_path, metrics, **candidate):
    report = evaluate_submission(_candidate(level, **candidate), dict(source_dir=str(FAMILY), source_entrypoint=f'{level}/scene.xml',
        manifest=f'{level}/manifest.toml', output_dir=str(tmp_path), metrics=metrics, source_config=CONFIG, camera='fixed', width=64, height=48,
        reconstruction={'samples': 1024, 'seed': 1}))
    assert report['status'] == 'ok', report.get('error')
    return report['scores']


def test_arm_and_hand_actions_proprioception_and_palm_down_home():
    sim = Simulation(FAMILY/'bin_spiky'/'scene.xml', CONFIG)
    try:
        observation = public_observation(sim.observe(), dict(CONFIG, effort=True))
        assert set(observation) == {'time', 'ticks', 'total_ticks', 'proprioception', 'tactile'}
        assert [len(observation['proprioception'][k]) for k in ('joint_positions_rad', 'joint_velocities_rad_s', 'actuator_effort_nm')] == [31, 31, 27]
        assert joints('panda_shadow')[:8] == ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'rh_WRJ2')
        assert list(observation['tactile'])[:6] == ['thumb', 'first', 'middle', 'ring', 'little', 'palm']
        assert sim._data.xmat[sim._model.body('rh_palm').id].reshape(3, 3)[2, 1] > np.cos(np.radians(1))  # palm faces down at home
        assert allowed_actions(CONFIG) == ('observe', 'reset', 'arm_target', 'hand_target')
        home = sim._data.ctrl[sim._arm_actuators].copy()
        with pytest.raises(ValueError, match='no gripper'):
            sim.action({'op': 'arm_target', 'joint_targets': home.tolist(), 'gripper_width_m': .04})
        with pytest.raises(ValueError, match='point grasp'):
            sim.action({'op': 'grasp', 'pixel': [1, 1]})
        lifted = home + [0, -.1, 0, 0, 0, 0, 0]
        sim.action({'op': 'arm_target', 'joint_targets': lifted.tolist(), 'ticks': 200}, render=False)
        sim.action({'op': 'hand_target', 'actuator_targets': [0.] * len(HAND_ACTUATORS), 'ticks': 50}, render=False)
        np.testing.assert_allclose(sim._data.ctrl[sim._arm_actuators], lifted)
        sim.action({'op': 'observe', 'ticks': 500}, render=False)
        assert np.abs(sim._data.qpos[sim._arm_qpos] - lifted).max() < .005  # static gravity error of the raised gains
        sim.action({'op': 'reset'}, render=False)
        np.testing.assert_allclose(sim._data.ctrl[sim._arm_actuators], home)
    finally:
        sim.close()
    assert 'panda_shadow.xml' in public_robot_files(FAMILY/'robot', 'panda_shadow')
    assert enabled_tools({'tools': {'local_controller': True}, 'simulation': {'robot': 'panda_shadow'}}) == ('local_controller',)
    with pytest.raises(RuntimeFailure, match='hand robot'):
        enabled_tools({'tools': {'local_controller': True}, 'simulation': {'robot': 'panda'}})


def test_objects_metric_identity_prior_and_unassigned(tmp_path):
    identity = _scores('table_objects', tmp_path/'identity', ['objects'])['objects']
    assert identity['found_fraction'] == 1 and identity['label_accuracy'] == 1 and identity['mean_assigned_chamfer_m'] == 0
    assert all(r['centroid_error_m'] == 0 and all(d['absolute_error_m'] == 0 for d in r['dimensions']) for r in identity['per_object'])
    prior = {r['name']: r for r in _scores('table_objects', tmp_path/'prior', ['objects'], scene='prior.xml')['objects']['per_object']}
    assert prior['mug']['dimensions'][0]['name'] == 'mug_height'
    assert prior['mug']['dimensions'][0]['absolute_error_m'] == pytest.approx(.025, abs=1e-6)  # 95 mm typical vs 120 mm true
    assert prior['hammer']['dimensions'][0]['absolute_error_m'] == pytest.approx(.120, abs=1e-6)  # 280 mm vs 160 mm handle
    assert prior['banana']['dimensions'][0]['absolute_error_m'] > .01 and all(r['label_correct'] for r in prior.values())

    def only_mug(text):  # the banana and hammer bodies removed; the mug renamed to something that is not a label
        for name in ('banana', 'hammer'):
            start = text.index(f'<body name="{name}"')
            text = text[:start] + text[text.index('</body>', start) + len('</body>'):]
        return text.replace('body name="mug"', 'body name="vessel"')
    partial = _scores('table_objects', tmp_path/'partial', ['objects'], edit=only_mug)['objects']
    rows = {r['name']: r for r in partial['per_object']}
    assert rows['mug']['candidate_body'] == 'vessel' and rows['mug']['found'] and rows['mug']['label_correct'] is False
    assert rows['banana']['chamfer_m'] is None and rows['banana']['unassigned_reason'] and not rows['banana']['found']
    assert partial['found_fraction'] == pytest.approx(1 / 3) and partial['label_accuracy'] == 0


def test_identification_and_grab(tmp_path):
    manifest = _manifest('bin_spiky')
    spiky, smooth = manifest['tabletop']['spiky'], next(n for n in manifest['object_bodies'] if n != manifest['tabletop']['spiky'])
    sim = Simulation(FAMILY/'bin_spiky'/'scene.xml', CONFIG)
    xyz = {n: sim._data.xpos[sim._model.body(n).id].tolist() for n in manifest['object_bodies']}
    for name, parameters, expected in (('correct', {'spiky_ball_xyz': xyz[spiky]}, True), ('wrong', {'spiky_ball_xyz': xyz[smooth]}, False)):
        scores = _scores('bin_spiky', tmp_path/name, ['objects', 'identification'], parameters=json.dumps(parameters))
        assert scores['identification']['status'] == 'ok' and scores['identification']['correct'] is expected
        assert scores['identification']['distance_error_m'] == pytest.approx(0 if expected else np.linalg.norm(np.subtract(xyz[smooth], xyz[spiky])))
    for name, parameters, reason in (('missing', None, 'no parameters.json'), ('malformed', '{"spiky_ball_xyz": [0, 1]}', 'malformed'),
                                     ('not_json', 'the third one', 'malformed')):
        scores = _scores('bin_spiky', tmp_path/name, ['objects', 'identification'], parameters=parameters)
        assert scores['identification'] == {'status': 'unavailable', 'reason': scores['identification']['reason']}
        assert reason in scores['identification']['reason'] and scores['objects']['status'] == 'ok'
    try:
        assert not grab(sim, manifest)['success']
        palm = sim._model.body('rh_palm').id
        ball = sim._model.jnt_qposadr[sim._model.body(spiky).jntadr[0]]
        sim._data.qpos[ball:ball + 3] = sim._data.xpos[palm] + sim._data.xmat[palm].reshape(3, 3) @ [0, -.0375, .05]  # 2 mm into the palm
        import mujoco
        mujoco.mj_forward(sim._model, sim._data)
        result = grab(sim, manifest)
        assert result['success'] and 'rh_palm' in result['spiky_touching_robot_bodies'] and all(result['smooth_balls_in_bin'].values())
    finally:
        sim.close()
