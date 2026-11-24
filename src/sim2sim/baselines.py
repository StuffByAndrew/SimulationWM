"""Researcher-only trivial-baseline candidates and metric normalization, so error metrics compare across environments.

`empty` keeps the source's world fixtures and removes every object body; `bounding_box` replaces the objects with one
free box filling their reset-pose world bounding box, with MuJoCo's built-in density and friction; `prior` (only where the
source level has a private prior.xml beside its scene, e.g. the touch tabletop objects) is that prior scene as written. Each run's baselines
are evaluated by the same trusted configuration and hash-verified source snapshot as its own checkpoints
(`python -m sim2sim.baselines RUN_DIR ...` on a Slurm test node), into RUN_DIR/baseline-scores.json. `normalize` then
divides an error metric by the baseline's: 1 means no better than the trivial scene, 0 exact. A metric the baseline cannot
produce (e.g. Chamfer of an empty scene, or hinges without a hinge) stays a gap, never a fabricated ratio.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import io
import itertools
import json
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from .audit import read_audit
from .robots import ROBOTS
from .candidate_history import _save, evaluate_archive, recover_changes, require_compute_allocation, trusted_evaluation
from .dashboard import _scalars
from .runtime import extract_bundle

BASELINES = ('empty', 'bounding_box')
ERROR_FIELD = re.compile(r'error|mse|_mae_|chamfer_distance_m|difference')
DEFAULT_BOX = {'density': '1000', 'friction': '1 0.005 0.0001', 'rgba': '.5 .5 .5 1'}  # MuJoCo's built-in geom values


def _object_box(scene, simconfig):
    """Center and half extents of the world bounding box of every non-robot object geom at the reset pose."""
    import numpy as np
    from .primitive_metrics import _robot_bodies
    from .simulation import Simulation
    sim = Simulation(scene, simconfig)
    m, d = sim._model, sim._data
    robot = _robot_bodies(sim) if simconfig.get('robot') else set()
    signs = np.array(list(itertools.product((-1, 1), repeat=3)))
    corners = [(m.geom_aabb[g][:3] + signs * m.geom_aabb[g][3:]) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
               for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and m.geom_bodyid[g] not in robot]
    sim.close()
    assert corners, 'source has no object geoms'
    points = np.concatenate(corners)
    return (points.min(0) + points.max(0)) / 2, (points.max(0) - points.min(0)) / 2


def baseline_archive(source_dir, entrypoint, simconfig, kind):
    """ZIP bytes of one trivial candidate project built from the trusted source scene (or its private prior scene)."""
    assert kind in BASELINES + ('prior',), kind
    scene = Path(source_dir) / entrypoint
    root = ET.parse(scene.parent / 'prior.xml' if kind == 'prior' else scene).getroot()
    world = root.find('worldbody')
    if kind != 'prior':  # the trivial baselines keep only the world fixtures
        for body in world.findall('body'):
            world.remove(body)
        for tag in ('equality', 'contact', 'tendon', 'sensor', 'actuator', 'keyframe'):
            for node in root.findall(tag):
                root.remove(node)
    assert not [n for n in root.iter() if n.tag in ('mesh', 'texture', 'hfield', 'skin') and n.get('file')], 'baselines support file-free source assets only'
    files, includes, robot = {}, root.findall('include'), Path(source_dir) / 'robot'
    if simconfig.get('robot'):
        fixture = ROBOTS[simconfig['robot']][0]
        assert len(includes) == 1 and (scene.parent / includes[0].get('file')).resolve() == (robot / fixture).resolve(), 'robot source must include the public robot once'
        includes[0].set('file', 'robot/' + fixture)
        root.find('compiler').set('meshdir', 'robot/assets')
        files = {'robot/' + p.relative_to(robot).as_posix(): p.read_bytes() for p in sorted(robot.rglob('*')) if p.is_file()}
    else:
        assert not includes, 'baselines support only the public robot include'
    if kind == 'bounding_box':
        center, half = _object_box(scene, simconfig)
        body = ET.SubElement(world, 'body', name='baseline_box', pos=' '.join(map(repr, center.tolist())))
        ET.SubElement(body, 'freejoint')
        ET.SubElement(body, 'geom', type='box', size=' '.join(map(repr, half.tolist())), **DEFAULT_BOX)
    files['scene.xml'] = ET.tostring(root, encoding='unicode').encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    return buffer.getvalue()


def normalize(scores, baseline_scores):
    """Error leaves divided by the same leaf of a baseline's scores; None where either is missing or the baseline is 0."""
    base = dict(_scalars(baseline_scores))
    return {path: (value / base[path] if value is not None and base.get(path) else None)
            for path, value in _scalars(scores) if ERROR_FIELD.search([p for p in path.split('.') if not p.isdigit()][-1])}


def evaluate_baselines(run_dir, *, timeout=600):
    """Evaluate both trivial baselines of one run with its recorded trusted evaluation configuration."""
    require_compute_allocation()
    run = Path(run_dir).resolve()
    config = {}
    _, _, digest = recover_changes(read_audit(run), config)
    evaluation, source = trusted_evaluation(run, config, digest, timeout)
    result = {'schema_version': 1, 'baselines': {}}
    with tempfile.TemporaryDirectory() as temp:
        extract_bundle(source, temp)
        for kind in BASELINES + (('prior',) if (Path(temp) / evaluation['source_entrypoint']).with_name('prior.xml').is_file() else ()):
            archive = baseline_archive(temp, evaluation['source_entrypoint'], evaluation['source_config'], kind)
            # An empty candidate has no object to fling: policy transfer is a gap for it, not a failed evaluation.
            trial = {**evaluation, 'metrics': [m for m in evaluation['metrics'] if m != 'policy_transfer']} \
                if kind == 'empty' and 'policy_transfer' in (evaluation.get('metrics') or ()) else evaluation
            report = evaluate_archive(config, trial, source, archive, 'scene.xml', run / 'baselines' / kind, timeout)
            result['baselines'][kind] = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'status': report['status'], 'report': report}
            print(json.dumps({'run': run.name, 'baseline': kind, 'status': report['status']}), flush=True)
    _save(run / 'baseline-scores.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    for run in args.runs:
        evaluate_baselines(run, timeout=args.timeout)


if __name__ == '__main__':
    main()
