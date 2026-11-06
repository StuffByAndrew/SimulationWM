"""Seeded private-source perturbation: actual MJCF edits, records and native verification."""
import io
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import pytest
from sim2sim.evaluation import evaluate_submission
from sim2sim.experiments import prepare_experiment
from sim2sim.perturbation import COLOR_RULE, PALETTE, nonround, verify

ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path, environment, perturbation):
    root, _, sub = environment.partition('/')
    prefix, path = sub and sub + '/', tmp_path / 'experiment.toml'
    path.write_text(f'''[experiment]
name="Perturbation fixture"
environment="{environment}"
evaluation_metrics=["dynamics"]
[experiment.physical_parameters]
scene="Seeded perturbation of the authored scene."
[source]
scene="{ROOT / 'environments' / root / prefix / 'scene.xml'}"
[budget]
wall_seconds=60
active_seconds=60
physics_ticks=10
submissions=1
[evaluation]
source_dir="{ROOT / 'environments' / root}"
manifest="{prefix}manifest.toml"
{'[simulation]' + chr(10) + 'robot="panda"' if root == 'panda' else ''}
[perturbation]
seed=7
{perturbation}
''')
    return path


@pytest.mark.parametrize('environment,mode', [('blocks', 'random'), ('panda/tetrahedral_cluster', 'uniform_gray')])
def test_color_perturbation_is_applied_recorded_private_and_verified(tmp_path, environment, mode):
    config = _config(tmp_path, environment, f'colors="{mode}"')
    runs = [Path(prepare_experiment(config, tmp_path / 'store', f'run-{i}')['run_dir']) for i in range(2)]
    metadata = [json.loads((run / 'experiment.json').read_text()) for run in runs]
    record = metadata[0]['perturbation']
    assert record['colors'] == metadata[1]['perturbation']['colors']  # seeded
    assert json.loads(metadata[0]['physical_parameters']['perturbation.colors']) == record['colors']
    private = json.loads((runs[0] / 'config-private.json').read_text())
    scene = ET.parse(private['source']['scene']).getroot()
    for name, item in record['colors']['geoms'].items():
        assert [float(v) for v in scene.find(f".//geom[@name='{name}']").get('rgba').split()] == item['value']
        assert all(abs(v * 20 - round(v * 20)) > .01 for v in item['value'][:3])  # non-round canary
        assert mode == 'uniform_gray' or min(math.dist(item['value'][:3], c) for c in PALETTE.values()) >= .35
    assert (record['colors'].get('rule') == COLOR_RULE) == (mode == 'random')
    assert 'perturbation' not in private and 'rgba' not in private['task']
    assert verify(runs[0])['geoms'] == len(record['object_geoms'])


@pytest.mark.parametrize('environment,metrics', [('blocks', ['chamfer', 'center_of_mass']), ('damped_lever', ['hinges'])])
def test_nonround_physics_is_seeded_verified_and_visible_to_the_evaluator(tmp_path, environment, metrics):
    config = _config(tmp_path, environment, 'physics="nonround"')
    runs = [Path(prepare_experiment(config, tmp_path / 'store', f'run-{i}')['run_dir']) for i in range(2)]
    records = [json.loads((run / 'experiment.json').read_text()) for run in runs]
    physics = records[0]['perturbation']['physics']
    assert physics == records[1]['perturbation']['physics'] and 'colors' not in records[0]['perturbation']
    assert json.loads(records[0]['physical_parameters']['perturbation.physics']) == physics
    values = [v for item in [*physics['geoms'].values(), *physics['joints'].values()] for p in item.values()
              for v in (p['value'] if isinstance(p['value'], list) else [p['value']]) if v]
    assert len(values) >= 4 and all(map(nonround, values))
    assert set(physics['placed_bodies']) == ({'block_red', 'block_blue'} if environment == 'blocks' else set())
    assert verify(runs[0])['perturbed_initial_penetration_m'] <= 1e-3
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:  # the unperturbed original as a candidate must score nonzero error
        archive.write(ROOT / 'environments' / environment / 'scene.xml', 'scene.xml')
    private = json.loads((runs[0] / 'config-private.json').read_text())
    report = evaluate_submission(buffer.getvalue(), dict(private['evaluation'], metrics=metrics, output_dir=str(tmp_path / 'eval')))
    assert report['status'] == 'ok', report.get('error')
    if environment == 'blocks':
        assert report['scores']['reconstruction']['chamfer_distance_m'] > 0
        assert report['scores']['reconstruction']['center_of_mass_error_m'] > 0
    else:
        assert report['scores']['hinges']['hinge']['damping_absolute_error'] > 0
        assert report['scores']['hinges']['hinge']['frictionloss_absolute_error'] > 0


@pytest.mark.parametrize('perturbation', ['colors="pink"', '[perturbation.pose]\nxy_m=0.02\nyaw_deg=0'])
def test_unsupported_perturbations_fail_before_copy(tmp_path, perturbation):
    with pytest.raises(ValueError, match='perturbation takes seed and colors'):
        prepare_experiment(_config(tmp_path, 'blocks', perturbation), tmp_path / 'store', 'run')
    assert not (tmp_path / 'store/runs/run').exists()


def test_panda_tetrahedral_cluster_starts_clear_and_prepares_nonround(tmp_path):
    run = Path(prepare_experiment(_config(tmp_path, 'panda/tetrahedral_cluster', 'physics="nonround"'), tmp_path / 'store', 'run')['run_dir'])
    report = verify(run)
    assert report['original_initial_penetration_m'] <= 1e-3 and report['perturbed_initial_penetration_m'] <= 1e-3, report
    assert all(after > 0 for _, after in report['arm_probe_contact_ticks'].values()), report


@pytest.mark.parametrize('perturbation', ['pose="jitter"', 'pose="jitter"\nphysics="nonround"'])
def test_pose_jitter_moves_the_object_group_rigidly_and_is_verified(tmp_path, perturbation):
    runs = [Path(prepare_experiment(_config(tmp_path, 'blocks', perturbation), tmp_path / 'store', f'run-{i}')['run_dir']) for i in range(2)]
    records = [json.loads((run / 'experiment.json').read_text())['perturbation'] for run in runs]
    pose = records[0]['pose']
    assert pose == records[1]['pose'] and abs(pose['yaw_deg']) <= 15 and max(map(abs, pose['offset_m'])) <= .03  # seeded, bounded
    (a, b), (c, d) = ((item['pos']['original'][:2], item['pos']['value'][:2]) for item in pose['bodies'].values())
    assert math.isclose(math.dist(a, c), math.dist(b, d), rel_tol=1e-9) and a != b  # rigid: the blocks keep their separation
    report = verify(runs[0])
    assert report['pose_yaw_deg'] == pose['yaw_deg'] and report['perturbed_initial_penetration_m'] <= 1e-3, report


@pytest.mark.parametrize('environment', ['damped_lever', 'panda/blocks'])
def test_pose_jitter_refuses_anchored_objects_and_panda_scenes(tmp_path, environment):
    with pytest.raises((ValueError, AssertionError), match='hinged|Panda'):
        prepare_experiment(_config(tmp_path, environment, 'pose="jitter"'), tmp_path / 'store', 'run')
