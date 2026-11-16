"""Friction metric, tactile sensing, expert-video and policy-transfer metrics, and trivial baselines (Slurm test node)."""
import io
from pathlib import Path
import zipfile

import numpy as np
import pytest
import tomllib

from sim2sim.baselines import baseline_archive, normalize
from sim2sim.candidate_worker import public_observation
from sim2sim.evaluation import evaluate_submission
from sim2sim.simulation import Simulation

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = ROOT / 'environments/blocks'


def _zip(scene_text, extra=()):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('scene.xml', scene_text)
        for name, data in extra:
            archive.writestr(name, data)
    return buffer.getvalue()


def _evaluate(tmp_path, archive, metrics, source=BLOCKS, **config):
    report = evaluate_submission(archive, {'source_dir': str(source), 'output_dir': str(tmp_path), 'width': 64, 'height': 48,
                                           'metrics': metrics, 'reconstruction': {'samples': 128}, **config})
    assert report['status'] == 'ok', report.get('error')
    return report['scores']


@pytest.mark.parametrize('mu,geom_error,floor_error', [('0.5', .3, 0.0), ('1.2', .4, .4)])
def test_friction_compares_geom_and_effective_support_mu_by_geometry(tmp_path, mu, geom_error, floor_error):
    # Only the blocks change: MuJoCo mixes a contact's mu as the max of both geoms, so a lower block mu hides behind the 0.8 table.
    text = (BLOCKS / 'scene.xml').read_text().replace('<geom name="red" ', f'<geom name="red" friction="{mu} .02 .002" ')
    friction = _evaluate(tmp_path, _zip(text), ['friction'])['reconstruction']['friction']
    records = {r['source_name']: r for r in friction['per_primitive']}
    assert friction['status'] == 'ok' and friction['coverage'] == 1 and set(records) == {'red', 'blue'}
    assert records['red']['sliding_mu_absolute_error'] == pytest.approx(geom_error)
    assert records['red']['support_sliding_mu_absolute_error'] == pytest.approx(floor_error)
    assert records['blue']['support_sliding_mu_absolute_error'] == 0 and records['red']['source_support_sliding_mu'] == pytest.approx(.8)


def test_friction_is_a_gap_when_an_explicit_pair_overrides_the_geom(tmp_path):
    text = (BLOCKS / 'scene.xml').read_text().replace('</worldbody>', '</worldbody><contact><pair geom1="red" geom2="floor"/></contact>')
    red = [r for r in _evaluate(tmp_path, _zip(text), ['friction'])['reconstruction']['friction']['per_primitive'] if r['source_name'] == 'red'][0]
    assert red['status'] == 'unavailable' and 'pair' in red['unavailable_reason'] and red['support_sliding_mu_absolute_error'] is None


def test_tactile_senses_a_closed_grasp_without_advancing_time():
    manifest = tomllib.loads((ROOT / 'environments/panda/blocks/manifest.toml').read_text())
    sim = Simulation(ROOT / 'environments/panda/blocks/scene.xml', {'robot': 'panda', 'tactile': True, 'rgb': False, 'calibration': False})
    quiet = sim.observe()['tactile']
    assert quiet['left']['finger_normal_force_n'] == 0 and np.array(quiet['left']['pad_normal_force_n']).shape == (3, 3)
    for step in manifest['arm_probes'][0]['steps'][:2]:  # reach the red block, then close on it
        sim.action({'op': 'arm_target', **step}, render=False)
    state, ticks = sim._data.qpos.copy(), sim.total_ticks
    touch = sim.observe()['tactile']
    assert np.array_equal(state, sim._data.qpos) and ticks == sim.total_ticks
    assert all(touch[side]['finger_normal_force_n'] > 0 and np.sum(touch[side]['pad_normal_force_n']) > 0 for side in ('left', 'right')), touch
    full = {**sim.observe(), 'width': 64}
    assert 'tactile' in public_observation(full, {'robot': 'panda', 'tactile': True})
    assert 'tactile' not in public_observation(full, {'robot': 'panda'})
    assert 'actuator_effort_nm' in public_observation(full, {'robot': 'panda'})['proprioception']
    assert set(public_observation(full, {'robot': 'panda', 'effort': False})['proprioception']) == \
        set(public_observation(full, {'robot': 'panda'})['proprioception']) - {'actuator_effort_nm'}
    for bad in ({'tactile': True}, {'robot': 'panda', 'tactile': 1}):
        with pytest.raises(ValueError, match='tactile'):
            Simulation(ROOT / ('environments/panda/blocks/scene.xml' if 'robot' in bad else 'environments/blocks/scene.xml'), bad)


def test_expert_video_and_policy_transfer_are_exact_for_the_source_and_differ_for_wrong_physics(tmp_path):
    scores = _evaluate(tmp_path / 'same', _zip((BLOCKS / 'scene.xml').read_text()), ['expert', 'policy_transfer'])
    expert, transfer = scores['expert'][0], scores['policy_transfer'][0]
    assert expert['frames'] > 50 and expert['rgb_mse'] < 1e-9 and expert['silhouette_error'] < 1e-9 and expert['source_grasp_success']
    assert abs(transfer['transfer_regret_m']) < 1e-9 and transfer['candidate_predicted_error_m'] == pytest.approx(transfer['oracle_error_m'], abs=1e-9)
    assert transfer['oracle_error_m'] < transfer['naive_error_m']
    assert any((tmp_path / 'same/artifacts').rglob('side_by_side.mp4'))
    heavy = (BLOCKS / 'scene.xml').read_text().replace('mass=".35"', 'mass="1.4"').replace('<geom name="table" ', '<geom name="table" friction="0.2 .02 .002" ')
    scores = _evaluate(tmp_path / 'heavy', _zip(heavy), ['expert', 'policy_transfer'])
    assert scores['expert'][0]['rgb_mse'] > 0 and scores['policy_transfer'][0]['transfer_regret_m'] > 0, scores


@pytest.mark.parametrize('environment,source', [('blocks', 'environments/blocks'), ('blocks/scene.xml', 'environments/panda')])
def test_trivial_baselines_evaluate_and_normalize(tmp_path, environment, source):
    source, entry = ROOT / source, environment if environment.endswith('.xml') else 'scene.xml'
    simconfig = {'robot': 'panda'} if 'panda' in str(source) else {}
    metrics = ['image_mse', 'chamfer', 'center_of_mass']
    reports = {kind: _evaluate(tmp_path / kind, baseline_archive(source, entry, simconfig, kind), metrics, source,
                               source_entrypoint=entry, manifest=entry.replace('scene.xml', 'manifest.toml'), source_config=simconfig)
               for kind in ('empty', 'bounding_box')}
    assert reports['empty']['reconstruction']['chamfer_distance_m'] is None  # a gap, not a zero error
    box = reports['bounding_box']['reconstruction']
    assert box['chamfer_distance_m'] > 0 and box['center_of_mass_error_m'] < .2
    ratios = normalize(reports['bounding_box'], reports['bounding_box'])
    assert ratios['reset.rgb_mse'] == 1 and ratios['reconstruction.chamfer_distance_m'] == 1
    assert normalize(reports['bounding_box'], reports['empty'])['reconstruction.chamfer_distance_m'] is None
