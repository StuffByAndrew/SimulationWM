"""Rendering/physics integration tests: run only in the Slurm test allocation."""
import io
from pathlib import Path
import zipfile
import pytest
from sim2sim.evaluation import evaluate_submission

ROOT = Path(__file__).resolve().parents[1]


def evaluate(tmp_path, environment, mutate=lambda text: text, **overrides):
    source = ROOT / 'environments' / environment
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as archive:
        archive.writestr('scene.xml', mutate((source / 'scene.xml').read_text()))
    config = dict(source_dir=str(source), output_dir=str(tmp_path), width=96, height=72)
    config.update(overrides)
    report = evaluate_submission(data.getvalue(), config)
    assert report['status'] == 'ok', report
    return report['scores']


def motion_error(scores):
    return sum(probe['image_trajectory_error'] + probe['rgb_error'] + probe['depth_error']
               for probe in scores['probes'])


def test_identical_submission_zero(tmp_path):
    scores = evaluate(tmp_path, 'blocks')
    assert all(value == pytest.approx(0, abs=1e-9) for value in scores['reset'].values())
    assert motion_error(scores) == pytest.approx(0, abs=1e-9)
    assert all(value == 0 for body in scores['parameters']['bodies'].values() for value in body.values())
    assert list((tmp_path / 'artifacts').rglob('*_rgb.png'))
    assert list((tmp_path / 'artifacts').rglob('trajectory.json'))


def test_geometry_perturbation_worsens_pixels(tmp_path):
    scores = evaluate(tmp_path, 'blocks', lambda s: s.replace('size=".06 .06 .06"', 'size=".11 .11 .11"'))
    assert scores['reset']['rgb_error'] > 0
    assert scores['reset']['silhouette_error'] > 0


def test_friction_direct_and_slide_dynamics(tmp_path):
    scores = evaluate(tmp_path, 'blocks', lambda s: s.replace('friction="0.8 0.02 0.002"', 'friction="0.01 0.001 0.0001"'),
                      probes=[{'body': 'block_blue', 'local_point': [0, 0, .05],
                               'displacement': [-.25, 0, 0], 'ticks': 80}])
    assert scores['parameters']['bodies']['block_blue']['friction'] > .5
    assert motion_error(scores) > 1e-7


@pytest.mark.parametrize('old,new,metric', [('stiffness="2.5"', 'stiffness="20"', 'hinge_stiffness'),
                                           ('damping=".12"', 'damping="2"', 'hinge_damping')])
def test_spring_parameters_and_probes_distinguish(tmp_path, old, new, metric):
    scores = evaluate(tmp_path, 'spring', lambda s: s.replace(old, new))
    assert scores['parameters']['bodies']['lever'][metric] > 0
    assert motion_error(scores) > 1e-7


def test_no_correspondence_retains_generic_metrics(tmp_path):
    scores = evaluate(tmp_path, 'blocks', correspondence={})
    assert scores['parameters']['status'] == 'unavailable'
    assert scores['reset']['rgb_error'] == pytest.approx(0, abs=1e-9)
    assert scores['probes']


def test_candidate_camera_ignored(tmp_path):
    scores = evaluate(tmp_path, 'blocks', lambda s: s.replace('pos="1.4 -1.8 1.5"', 'pos="8 8 8"').replace('fovy="45"', 'fovy="10"'))
    assert all(value == pytest.approx(0, abs=1e-9) for value in scores['reset'].values())
    assert motion_error(scores) == pytest.approx(0, abs=1e-9)
