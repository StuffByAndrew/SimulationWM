"""The childproof bottle levels' private metrics (Slurm test node): the `bottle` mechanism probes and the point-grasp
`objects` contents breakdown. experiments/validation/bottle-check.py proves the level's premises."""
import io
from pathlib import Path
import zipfile

from sim2sim.evaluation import evaluate_submission

ROOT = Path(__file__).resolve().parents[1] / 'environments'


def _zip(text):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('scene.xml', text)
    return buffer.getvalue()


def _evaluate(level, text, metrics, tmp_path):
    report = evaluate_submission(_zip(text), {'source_dir': str(ROOT / level), 'output_dir': str(tmp_path), 'metrics': metrics,
                                              'camera': 'fixed', 'width': 320, 'height': 240})
    assert report['status'] == 'ok', report.get('error')
    return report['scores']


def test_missing_contents_are_unassigned_gaps_not_zeros(tmp_path):
    scores = _evaluate('bottle_contents', (ROOT / 'bottle_empty/scene.xml').read_text(), ['objects'], tmp_path)['objects']
    records = {r['name']: r for r in scores['per_object']}
    assert records['bottle']['chamfer_m'] < 1e-9 and records['cap']['chamfer_m'] < 1e-9
    for name in ('cube', 'rod', 'ball'):
        assert records[name]['candidate_body'] is None and records[name]['chamfer_m'] is None and records[name]['unassigned_reason']
    assert scores['found_fraction'] == 2 / 5


def test_bottle_identity_agrees_on_every_probe(tmp_path):
    scores = _evaluate('bottle_empty', (ROOT / 'bottle_empty/scene.xml').read_text(), ['bottle'], tmp_path)['bottle']
    assert scores['outcome_agreement'] == 1 and scores['cannot_answer'] == 0
    assert scores['mean_cap_lift_rms_m'] < 1e-9 and scores['engage']['engage_force_error_n'] == 0
    assert scores['angle']['opening_angle_error_deg'] == 0
    assert [p['source_opened'] for p in scores['probes']][0] is False and any(p['source_opened'] for p in scores['probes'])
