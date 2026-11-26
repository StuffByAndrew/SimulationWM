"""Researcher-only childproof bottle metric (selector `bottle`, manifest [bottle] KEYS; levels environments/bottle_*) and
post-run behaviour analysis (docs/level-designs/childproof-bottle-20261125.md, experiments/validation/bottle-scenes.py).

Metric `bottle`: fresh source and candidate worlds, both with two weld grasps whatever the run's grasp count (the probes are
the same for every condition), at the evaluator's timestep. Pixels come from the source camera at reset: `shell_point` on the
cap top and `bottle_point` on the bottle wall (body-frame points of the source's `cap_shell` and `bottle`). In each world the
"cap" and "bottle" are the bodies those pixels' grasp rays hit at reset (geometry only, never names); a world whose ray misses
cannot answer that probe (a gap, never an error value).
- `probes`: for each depth in `push_m` (0 = twist only, below and above the engage threshold) with both grasps, and once with
  the cap grasp only at `one_grasp_push_m`: grasp, push the cap down by the depth, turn it by `turn_deg` about the vertical
  through the grasp point, unpush, release the bottle grasp, lift the cap by `lift_m`, release, settle. Recorded (reusing
  gear_train.rollout) at reset and every step: the world rotation of cap and bottle about the vertical and the lift of
  their grasped points. Per probe: RMS rotation (rad) and lift (m) differences, per body, and the outcome `opened` (the cap
  point rose more than `open_lift_m` above the bottle point at some time). `outcome_agreement` is the share of answered probes
  whose outcome matches the source's.
- `engage`: per world, its smallest two-grasp depth that opens and the cap grasp's vertical wrench force at the end of that
  push (`engage_force_n`), with the force at the deepest push that did not open (`closed_force_n`): the push force the model
  implies, bracketed by the probe grid. A world that never opens has no engage force (a gap with a reason).
- `angle`: per world at its own engage depth, the smallest turn in `angles_deg` that opens (`opening_angle_deg`).
Artifacts: probes.json (per probe both worlds' samples every TRAJECTORY_SECONDS) and scores.json.

Post-run (`python -m sim2sim.bottle RUN_DIR ...` on a Slurm test node; reads the verified audit and source snapshot, writes
RUN/bottle.json): at every recorded reference-world state, whether the cap is off (inner cap moved more than
`open_separation_m` from its closed pose in the bottle frame), whether any content is outside the bottle (centre outside the
`interior` cylinder in the bottle frame), and, at open states, whether contents are visible in the agent's own recorded camera
view (at least MIN_PIXELS segmentation pixels, the pose from the observation's calibration). Reported: first opening (action
id, reference episode and ticks, the latest audited wall time at or before it), states open, contents seen and tipped out.
Nothing here reaches the agent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KEYS = {'shell_point', 'bottle_point', 'push_m', 'one_grasp_push_m', 'turn_deg', 'angles_deg', 'lift_m', 'push_ticks', 'turn_ticks',
        'lift_ticks', 'settle_ticks', 'open_lift_m', 'open_separation_m', 'interior', 'contents'}
UP = np.array([0., 0., 1.])
TRAJECTORY_SECONDS = .01
MIN_PIXELS = 4
PROBE_CONFIG = {'grasp': 'weld', 'point_grasps': 2}  # every world, every condition: two weld grasps


def _pixel(source, body, point):
    from .evaluation import _probe_pixel
    return [float(v) for v in _probe_pixel(source, {'body': body, 'local_point': point})]


def pixels(source, table):
    source.action({'op': 'reset'}, render=False)
    return _pixel(source, 'cap_shell', table['shell_point']), _pixel(source, 'bottle', table['bottle_point'])


def actions(table, shell_px, bottle_px, push, turn_deg, two=True):
    """One probe: grasp, push, turn about the vertical through the grasp point, unpush, release the bottle, lift, release, settle."""
    move = lambda d, ticks, **extra: {'op': 'move', 'displacement': [0., 0., float(d)], 'ticks': int(table[ticks]), 'grasp_index': 0, **extra}  # noqa: E731
    return ([{'op': 'reset'}] + [{'op': 'grasp', 'pixel': bottle_px, 'grasp_index': 1}] * two + [{'op': 'grasp', 'pixel': shell_px, 'grasp_index': 0}]
            + [move(-push, 'push_ticks'), move(0, 'turn_ticks', rotation=[0., 0., float(np.deg2rad(turn_deg))]), move(push, 'push_ticks')]
            + [{'op': 'release', 'grasp_index': 1}] * two + [move(table['lift_m'], 'lift_ticks'), {'op': 'release', 'grasp_index': 0},
                                                             {'op': 'wait', 'ticks': int(table['settle_ticks'])}])


def targets(sim, shell_px, bottle_px):
    """(cap body, bottle body) hit by the two pixels' grasp rays at reset, with the hit points in body frames; None on a miss."""
    sim.action({'op': 'reset'}, render=False)
    hits = [sim._raycast(px) for px in (shell_px, bottle_px)]
    if None in hits or hits[0][0] == hits[1][0]:
        return None
    return [b for b, _ in hits], [sim._data.xmat[b].reshape(3, 3).T @ (p - sim._data.xpos[b]) for b, p in hits]


def run(sim, probe_actions, aim, open_lift):
    """Samples (times, [rot cap, rot bottle, lift cap, lift bottle]), opened, the cap grasp's push force (N, +up on the
    handle) at the end of the push, and whether every grasp held."""
    from .gear_train import rollout
    dt = float(sim._model.opt.timestep)
    push_end = probe_actions.index(next(a for a in probe_actions if a['op'] == 'move')) + 1
    rollout(sim, probe_actions[:push_end], dt, UP, *aim)
    force = float(sim._wrench(0)['force'][2])
    times, samples, held = rollout(sim, probe_actions, dt, UP, *aim)
    return times, samples, bool(np.max(samples[:, 2] - samples[:, 3]) > open_lift), force, held


def sweep(sim, table, shell_px, bottle_px):
    """All probes of one world (None where its rays miss), its engage record and its angle sweep."""
    aim = targets(sim, shell_px, bottle_px)
    probes = [(f'push_{1000 * d:.1f}mm', d, True) for d in table['push_m']] + [('one_grasp', table['one_grasp_push_m'], False)]
    if aim is None:
        return {name: None for name, _, _ in probes}, {'status': 'cannot_answer', 'reason': 'grasp ray misses cap or bottle'}, None
    lift = float(table['open_lift_m'])
    runs = {name: run(sim, actions(table, shell_px, bottle_px, d, table['turn_deg'], two), aim, lift) for name, d, two in probes}
    two = [(d, runs[name]) for name, d, both in probes if both]
    opening = next(((d, r) for d, r in two if r[2]), None)
    if opening is None:
        return runs, {'status': 'never_opens', 'push_forces_n': [r[3] for _, r in two]}, None
    closed = [r[3] for d, r in two if d < opening[0] and not r[2]]
    engage = {'status': 'ok', 'engage_push_m': opening[0], 'engage_force_n': opening[1][3], 'closed_force_n': closed[-1] if closed else None,
              'push_forces_n': [r[3] for _, r in two]}
    angle = next((a for a in table['angles_deg'] if run(sim, actions(table, shell_px, bottle_px, opening[0], a), aim, lift)[2]), None)
    return runs, engage, angle


def evaluate(source_path, candidate_path, simconfig, candidate_config, config, artifacts):
    """All `bottle` scores of one candidate (module docstring)."""
    from .evaluation import _trusted_camera
    from .simulation import Simulation
    table = config['bottle']
    assert set(table) == KEYS, f'[bottle] needs exactly {sorted(KEYS)}'
    source = Simulation(source_path, {**simconfig, **PROBE_CONFIG})
    candidate = Simulation(candidate_path, {**candidate_config, **PROBE_CONFIG})
    try:
        _trusted_camera(source, candidate)
        shell_px, bottle_px = pixels(source, table)
        truth, truth_engage, truth_angle = sweep(source, table, shell_px, bottle_px)
        assert all(r is not None and r[4] for r in truth.values()), 'invalid reference probe: a source grasp missed or slipped'
        model, model_engage, model_angle = sweep(candidate, table, shell_px, bottle_px)
    finally:
        source.close()
        candidate.close()
    records, trajectories = [], []
    for name, (at, a, a_open, a_force, _) in truth.items():
        record = {'name': name, 'source_opened': a_open, 'source_push_force_n': a_force}
        if model[name] is None:
            record.update(status='cannot_answer', candidate_opened=None)
        else:
            bt, b, b_open, b_force, held = model[name]
            b = np.column_stack([np.interp(at, bt, b[:, k]) for k in range(4)])
            rms = np.sqrt(np.mean((a - b) ** 2, axis=0))
            record.update(status='ok', candidate_opened=b_open, outcome_agrees=a_open == b_open, candidate_push_force_n=b_force,
                          candidate_grasps_held=held, cap_rotation_rms_rad=float(rms[0]), bottle_rotation_rms_rad=float(rms[1]),
                          cap_lift_rms_m=float(rms[2]), bottle_lift_rms_m=float(rms[3]))
            every = max(1, round(TRAJECTORY_SECONDS / float(at[1] - at[0])))
            trajectories.append({'name': name, 'time_s': at[::every].tolist(), 'source': a[::every].tolist(), 'candidate': b[::every].tolist()})
        records.append(record)
    answered = [r for r in records if r['status'] == 'ok']
    mean = lambda key: float(np.mean([r[key] for r in answered])) if answered else None  # noqa: E731
    engage = {'source': truth_engage, 'candidate': model_engage, 'engage_force_error_n': None, 'reason': None}
    if model_engage['status'] == 'ok':
        engage['engage_force_error_n'] = abs(model_engage['engage_force_n'] - truth_engage['engage_force_n'])
    else:
        engage['reason'] = model_engage['status']
    angle = {'source_opening_angle_deg': truth_angle, 'candidate_opening_angle_deg': model_angle,
             'opening_angle_error_deg': None if model_angle is None else abs(model_angle - truth_angle),
             'reason': None if model_angle is not None else model_engage['status'] if model_engage['status'] != 'ok' else 'no tested angle opens'}
    scores = {'probes': records, 'answered': len(answered), 'cannot_answer': len(records) - len(answered),
              'outcome_agreement': sum(r['outcome_agrees'] for r in answered) / len(answered) if answered else None,
              'mean_cap_rotation_rms_rad': mean('cap_rotation_rms_rad'), 'mean_bottle_rotation_rms_rad': mean('bottle_rotation_rms_rad'),
              'mean_cap_lift_rms_m': mean('cap_lift_rms_m'), 'mean_bottle_lift_rms_m': mean('bottle_lift_rms_m'),
              'engage': engage, 'angle': angle}
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / 'probes.json').write_text(json.dumps(trajectories, allow_nan=False))
    (artifacts / 'scores.json').write_text(json.dumps(scores, allow_nan=False))
    return scores


def state_flags(sim, table, closed):
    """(cap off, contents outside) in sim's current state; `closed` is the cap's closed pose in the bottle frame."""
    m, d = sim._model, sim._data
    bottle, cap = m.body('bottle').id, m.body('cap').id
    R, p = d.xmat[bottle].reshape(3, 3), d.xpos[bottle]
    off = bool(np.linalg.norm(R.T @ (d.xpos[cap] - p) - closed) > float(table['open_separation_m']))
    radius, top = table['interior']
    out = []
    for name in table['contents']:
        local = R.T @ (d.xpos[m.body(name).id] - p)
        if np.linalg.norm(local[:2]) > radius or local[2] > top:
            out.append(name)
    return off, out


def analyse(run_dir):
    """Researcher-only post-run pass over a verified audit (module docstring); writes RUN/bottle.json."""
    import hashlib
    import tempfile
    import tomllib
    import mujoco
    from .audit import read_audit
    from .candidate_history import _save, source_entrypoint
    from .realtime import restore
    from .runtime import extract_bundle
    from .simulation import Simulation
    run = Path(run_dir).resolve()
    manifest, states, wall = None, [], None
    for _, event in read_audit(run):
        if isinstance(event.get('wall_elapsed'), (int, float)):
            wall = float(event['wall_elapsed'])
        if event.get('kind') == 'run_start':
            manifest = event['manifest']
        if event.get('world') == 'reference' and event.get('kind') in ('checkpoint', 'tick', 'interrupted') and (event.get('state') or {}).get('arrays'):
            states.append((event['action_id'], event.get('episode'), event.get('world_total_ticks'), wall, event['state'],
                           ((event.get('observation') or {}).get('calibration') or {}).get('world_from_camera')))
    source = (run / 'source-snapshot.zip').read_bytes()
    assert hashlib.sha256(source).hexdigest() == manifest['source_bundle_sha256'], 'source snapshot hash mismatch'
    with tempfile.TemporaryDirectory() as temp:
        extract_bundle(source, temp)
        entry = Path(temp) / source_entrypoint(manifest)
        config = tomllib.loads((Path(temp) / manifest['evaluation'].get('manifest', 'manifest.toml')).read_text())
        sim = Simulation(entry, {**manifest['resolved_source_config'], 'rgb': False, 'depth': False, 'calibration': False})
    table, m, d = config['bottle'], sim._model, sim._data
    bottle, cap = m.body('bottle').id, m.body('cap').id
    closed = d.xmat[bottle].reshape(3, 3).T @ (d.xpos[cap] - d.xpos[bottle])
    contents = {g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name in table['contents']}
    renderer = mujoco.Renderer(m, height=sim.height, width=sim.width)
    renderer.enable_segmentation_rendering()
    camera = (m.cam_pos[sim._camera_id].copy(), m.cam_quat[sim._camera_id].copy())
    records, first = [], None
    for action, episode, ticks, at_wall, state, pose in states:
        if pose is not None:
            pose = np.asarray(pose)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, pose[:3, :3].ravel())
            camera = (pose[:3, 3], quat)
        restore(sim, state, camera)
        off, out = state_flags(sim, table, closed)
        seen = 0
        if off and contents:
            renderer.update_scene(d, camera=sim._camera_id)
            segment = renderer.render()
            ids = np.where(segment[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), segment[..., 0], -1)
            seen = int(sum(np.count_nonzero(ids == g) for g in contents))
        if off and first is None:
            first = {'action_id': action, 'episode': episode, 'reference_ticks': ticks, 'wall_elapsed_at_or_before_s': at_wall}
        records.append({'action_id': action, 'off': off, 'outside': out, 'content_pixels': seen})
    renderer.close()
    sim.close()
    result = {'schema_version': 1, 'states': len(records), 'opened': first is not None, 'first_opening': first,
              'states_open': sum(r['off'] for r in records),
              'contents_seen': any(r['content_pixels'] >= MIN_PIXELS for r in records) if table['contents'] else None,
              'contents_tipped_out': sorted({n for r in records for n in r['outside']}) if table['contents'] else None,
              'open_actions': sorted({r['action_id'] for r in records if r['off']})}
    _save(run / 'bottle.json', result)
    return result


def main():
    import argparse
    from .candidate_history import require_compute_allocation
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('runs', nargs='+')
    args = parser.parse_args()
    require_compute_allocation()
    for run_dir in args.runs:
        print(json.dumps({'run': run_dir, **{k: v for k, v in analyse(run_dir).items() if k != 'open_actions'}}), flush=True)


if __name__ == '__main__':
    main()
