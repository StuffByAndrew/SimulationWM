"""Researcher-only gear-train metric (selector `gear_train`, manifest [gear_train] KEYS; level environments/gear_train).

Scores a candidate scene against the deep source (every gear its own hinged body, pairwise joint-equality couplings) and the
private shallow reference (`shallow_reference`: crank and hand only, one coupling, the gears' inertia and damping lumped into
the crank joint). The candidate must name its crank and dial hand bodies `crank` and `hand` (the task says so).

- `accuracy`: each private [[probes]] entry runs `evaluation._probe_actions` plus `settle_ticks` of waiting, from reset, in
  the fresh source and the candidate. At reset and after every tick the world rotation of `crank` and `hand` about the panel
  normal (the source crank's hinge axis) relative to their reset orientation is recorded and unwrapped, so a flipped hinge
  axis or another zero angle is not penalized. Per probe and body: RMS difference (rad); `probe_error_rad` is their mean.
- `structure`: hinge bodies other than crank and hand whose axis is within AXIS_DEG of the panel normal are axles; source
  axles (the gears) are matched one-to-one to candidate axles by axle position in the panel plane within
  `axle_tolerance_m`, nearest first (geometry only: names, sizes and masses play no part). Depth class: `shallow` (no gear
  matched), `deep` (at least DEEP_FRACTION matched) or `partial`. Coupling counts are equality constraints by type and tendons.
- `sim_cost`: each model compiled as submitted, at the source timestep, from rest with a constant `cost_torque_nm` on the
  crank about the panel normal: wall seconds per simulated second of `mj_step` over `cost_seconds`, `cost_runs` interleaved
  runs of candidate, shallow and deep reference, median; plus body, joint, equality, tendon, geom and peak contact and
  constraint-row counts.
- `cheap` (G-cheap): over the gate (`probe_error_rad` > `cheap_gate_rad`) the status is `fails_accuracy_gate` with no score
  (never zero); otherwise `cost_relative_to_shallow`, the candidate's median step cost over the shallow reference's. The
  gate is the worst probe error of the shallow reference with its lumped armature and damping each scaled by 0.9 or 1.1
  (computed by experiments/validation/gear-scenes.py); `probe_error_rad` is always kept, so another gate can be applied later.
- `removal` (G-deep): for each source gear in `removals`, that body is deleted from the source and its matched body from the
  candidate (with its subtree and every equality, tendon, actuator, sensor and contact element naming anything inside it,
  repeated to a fixpoint; keyframes are dropped) and the probes rerun. A gear without a candidate match is `cannot_answer`
  (a gap, never an error value); a candidate whose edited model cannot run is `failed` with its error.
Artifacts: probes.json (both worlds' rotations every TRAJECTORY_EVERY ticks for the main probes) and the scores.
"""
from __future__ import annotations

import json
import statistics
import time

import numpy as np

KEYS = {'removals', 'axle_tolerance_m', 'cheap_gate_rad', 'cost_runs', 'cost_seconds', 'cost_torque_nm', 'shallow_reference'}
BODIES = ('crank', 'hand')
AXIS_DEG = 5.0             # a candidate hinge is a gear axle when its axis is within this angle of the panel normal
DEEP_FRACTION = .8         # matched share of source gears for the `deep` class
TRAJECTORY_EVERY = 5       # ticks between saved trajectory samples
COUPLING_SECTIONS = ('equality', 'tendon', 'actuator', 'sensor', 'contact')


def _ids(model):
    import mujoco
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in BODIES]
    if min(ids) < 0:
        raise ValueError('the model must name its crank body `crank` and its dial hand body `hand`')
    return ids


def compile_plain(path):
    """The model as submitted (no evaluation grasp targets), with its rest state."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def panel_normal(model, data):
    """World axis of the crank's single hinge."""
    joints = np.flatnonzero(model.jnt_bodyid == _ids(model)[0])
    assert len(joints) == 1, 'the source crank has exactly one hinge'
    return data.xaxis[joints[0]].copy()


def probe_actions(source, probe):
    from .evaluation import _probe_actions
    return _probe_actions(source, probe) + [{'op': 'wait', 'ticks': int(probe['settle_ticks'])}]


def rollout(sim, actions, normal):
    """Run probe actions from reset (a missed grasp's moves advance as waits). Returns the unwrapped world rotation (rad) of
    crank and hand about `normal` relative to reset, at reset and after every tick, and whether every grasp held."""
    ids = _ids(sim._model)
    u = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
    u /= np.linalg.norm(u)
    assert actions[0] == {'op': 'reset'}
    sim.action({'op': 'reset'}, render=False)
    start = [sim._data.xmat[b].reshape(3, 3).copy() for b in ids]
    samples, grasped = [], []

    def record(current):
        row = []
        for b, rotation in zip(ids, start):
            v = current._data.xmat[b].reshape(3, 3) @ rotation.T @ u
            row.append(np.arctan2(normal @ np.cross(u, v), u @ v))
        samples.append(row)
    record(sim)
    old, sim._callback = sim._callback, record
    try:
        for action in actions[1:]:
            if action['op'] == 'move' and sim._grasp is None:
                sim.action({'op': 'wait', 'ticks': action['ticks']}, render=False)
            else:
                sim.action(action, render=False)
            if action['op'] == 'grasp':
                grasped.append(sim._grasp is not None)
            if not np.isfinite(sim._data.qpos).all() or not np.isfinite(sim._data.qvel).all():
                raise ValueError('nonfinite simulation state')
    finally:
        sim._callback = old
    return np.unwrap(np.asarray(samples), axis=0), all(grasped)


def reference_runs(source, probes, normal):
    runs = []
    for probe in probes:
        actions = probe_actions(source, probe)
        angles, grasped = rollout(source, actions, normal)
        if not grasped:
            raise ValueError(f'invalid reference probe: grasp missed in {probe["name"]}')
        runs.append((probe['name'], actions, angles))
    return runs


def probe_errors(runs, candidate, normal, trajectories=None):
    """Candidate rotations against reference runs: per-probe RMS (rad) per body and their mean `probe_error_rad`."""
    records = []
    for name, actions, a in runs:
        b, grasped = rollout(candidate, actions, normal)
        assert a.shape == b.shape, 'probe sample count mismatch'
        rms = np.sqrt(np.mean((a - b) ** 2, axis=0))
        records.append({'name': name, 'candidate_grasp_success': grasped, 'crank_rms_rad': float(rms[0]), 'hand_rms_rad': float(rms[1]),
                        'source_crank_range_rad': float(np.ptp(a[:, 0])), 'source_hand_range_rad': float(np.ptp(a[:, 1]))})
        if trajectories is not None:
            trajectories.append({'name': name, 'actions': actions, 'every_ticks': TRAJECTORY_EVERY,
                                 'source_rad': a[::TRAJECTORY_EVERY].tolist(), 'candidate_rad': b[::TRAJECTORY_EVERY].tolist()})
    return {'probe_error_rad': float(np.mean([[r['crank_rms_rad'], r['hand_rms_rad']] for r in records])), 'probes': records}


def _axles(model, data, normal):
    """Hinge bodies other than crank and hand: {body: world axle point} for those whose axis lies along the normal, and the
    count of all of them."""
    import mujoco
    exclude, axles, hinged = set(_ids(model)), {}, set()
    for j in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE):
        body = int(model.jnt_bodyid[j])
        if body in exclude:
            continue
        hinged.add(body)
        if abs(data.xaxis[j] @ normal) >= np.cos(np.deg2rad(AXIS_DEG)):
            axles.setdefault(body, data.xanchor[j].copy())
    return axles, len(hinged)


def structure(source, candidate, normal, tolerance):
    """Geometry-only one-to-one gear matching by axle position in the panel plane; returns (record, {source: candidate body})."""
    import mujoco
    (sm, sd), (cm, cd) = source, candidate
    sources, _ = _axles(sm, sd, normal)
    candidates, hinged = _axles(cm, cd, normal)
    plane = np.eye(3) - np.outer(normal, normal)
    pairs = sorted((float(np.linalg.norm(plane @ (sp - cp))), sb, cb) for sb, sp in sources.items() for cb, cp in candidates.items())
    matched, distance = {}, {}
    for d, sb, cb in pairs:
        if d <= tolerance and sb not in matched and cb not in matched.values():
            matched[sb], distance[sb] = cb, d
    fraction = len(matched) / len(sources)
    equalities = {}
    for e in range(cm.neq):
        kind = mujoco.mjtEq(int(cm.eq_type[e])).name.removeprefix('mjEQ_').lower()
        equalities[kind] = equalities.get(kind, 0) + 1
    return {'source_gears': len(sources), 'candidate_hinge_bodies': hinged, 'candidate_axle_bodies': len(candidates),
            'matched': len(matched), 'matched_fraction': fraction, 'unmatched_candidate_axles': len(candidates) - len(matched),
            'depth_class': 'shallow' if not matched else 'deep' if fraction >= DEEP_FRACTION else 'partial',
            'couplings': {'equalities': equalities, 'tendons': int(cm.ntendon)},
            'gears': [{'source': sm.body(sb).name, 'candidate': cm.body(matched[sb]).name if sb in matched else None,
                       'distance_m': distance.get(sb)} for sb in sources]}, matched


def _cost_runner(model, normal, torque, seconds, timestep):
    """Counts of a model and a timing function (wall seconds per simulated second under a constant crank torque)."""
    import mujoco
    model.opt.timestep = timestep
    data = mujoco.MjData(model)
    crank, steps = _ids(model)[0], round(seconds / timestep)

    def start():
        mujoco.mj_resetData(model, data)
        data.xfrc_applied[crank, 3:] = torque * normal
    start()
    contacts = rows = 0
    for _ in range(steps):
        mujoco.mj_step(model, data)
        contacts, rows = max(contacts, data.ncon), max(rows, data.nefc)

    def run():
        start()
        begin = time.perf_counter()
        mujoco.mj_step(model, data, nstep=steps)
        return (time.perf_counter() - begin) / (steps * timestep)
    return {'bodies': model.nbody - 1, 'joints': int(model.njnt), 'dofs': int(model.nv), 'equalities': int(model.neq),
            'tendons': int(model.ntendon), 'geoms': int(model.ngeom), 'max_contacts': int(contacts), 'max_constraint_rows': int(rows)}, run


def sim_cost(models, normal, table, timestep):
    """{name: counts + median and runs of wall seconds per simulated second} for compiled models, timed interleaved."""
    runners = {name: _cost_runner(model, normal, float(table['cost_torque_nm']), float(table['cost_seconds']), timestep)
               for name, model in models.items()}
    runs = {name: [] for name in models}
    for _ in range(int(table['cost_runs'])):
        for name, (_, run) in runners.items():
            runs[name].append(run())
    return {name: {**counts, 'seconds_per_simulated_second': statistics.median(runs[name]), 'runs': runs[name]}
            for name, (counts, _) in runners.items()}


def remove_body(model, body):
    """Simulation `scene_edit`: delete compiled body `body` (the matching <body> element in document preorder), its subtree,
    every equality, tendon, actuator, sensor and contact element naming anything inside it (to a fixpoint) and keyframes."""
    def edit(root):
        bodies = list(root.iter('body'))
        assert len(bodies) == model.nbody - 1 and all(e.get('name', '') == model.body(i).name for i, e in enumerate(bodies, 1)), \
            'body elements do not map one-to-one onto compiled bodies'
        target = bodies[body - 1]
        parents = {child: parent for parent in root.iter() for child in parent}
        parents[target].remove(target)
        names, changed = {e.get('name') for e in target.iter() if e.get('name')}, True
        while changed:
            changed = False
            for section in [s for s in root if s.tag in COUPLING_SECTIONS]:
                for element in list(section):
                    if any(v in names for e in element.iter() for k, v in e.attrib.items() if k not in ('name', 'class')):
                        section.remove(element)
                        names.add(element.get('name'))
                        changed = True
        for keyframe in root.findall('keyframe'):
            root.remove(keyframe)
    return edit


def removals(source_path, candidate_path, simconfig, candidate_config, plain, matched, probes, normal, names):
    from .evaluation import _trusted_camera
    from .simulation import Simulation
    import mujoco
    (sm, _), (cm, _) = plain
    results = []
    for name in names:
        sb = mujoco.mj_name2id(sm, mujoco.mjtObj.mjOBJ_BODY, name)
        assert sb >= 0, f'unknown removal gear {name}'
        record = {'gear': name}
        results.append(record)
        if sb not in matched:
            record['status'] = 'cannot_answer'
            continue
        source = Simulation(source_path, {**simconfig, 'scene_edit': remove_body(sm, sb)})
        candidate = None
        try:
            runs = reference_runs(source, probes, normal)
            record['candidate_body'] = cm.body(matched[sb]).name
            try:
                candidate = Simulation(candidate_path, {**candidate_config, 'scene_edit': remove_body(cm, matched[sb])})
                _trusted_camera(source, candidate)
                record.update(status='ok', **probe_errors(runs, candidate, normal))
            except Exception as exc:  # the candidate's edited model: a recorded gap, never a score
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}'[:500])
        finally:
            for sim in (source, candidate):
                if sim is not None:
                    sim.close()
    answered = [r['probe_error_rad'] for r in results if r['status'] == 'ok']
    return {'gears': results, 'answered': len(answered), 'cannot_answer': sum(r['status'] == 'cannot_answer' for r in results),
            'failed': sum(r['status'] == 'failed' for r in results), 'mean_probe_error_rad': float(np.mean(answered)) if answered else None}


def evaluate(source, candidate, source_path, candidate_path, simconfig, candidate_config, config, artifacts):
    table, probes = config['gear_train'], config.get('probes')
    assert set(table) == KEYS, f'[gear_train] needs exactly {sorted(KEYS)}'
    assert probes and all('settle_ticks' in p and 'name' in p for p in probes), 'gear_train needs named [[probes]] with settle_ticks'
    plain = {'candidate': compile_plain(candidate_path), 'shallow_reference': compile_plain(source_path.parent / table['shallow_reference']),
             'deep_reference': compile_plain(source_path)}
    normal = panel_normal(*plain['deep_reference'])
    shape, matched = structure(plain['deep_reference'], plain['candidate'], normal, float(table['axle_tolerance_m']))
    trajectories = []
    accuracy = probe_errors(reference_runs(source, probes, normal), candidate, normal, trajectories)
    cost = sim_cost({name: model for name, (model, _) in plain.items()}, normal, table, float(source._model.opt.timestep))
    shallow = cost['shallow_reference']['seconds_per_simulated_second']
    for record in cost.values():
        record['relative_to_shallow'] = record['seconds_per_simulated_second'] / shallow
    gate, error = float(table['cheap_gate_rad']), accuracy['probe_error_rad']
    cheap = {'gate_rad': gate, 'probe_error_rad': error, 'status': 'ok' if error <= gate else 'fails_accuracy_gate',
             'cost_relative_to_shallow': cost['candidate']['relative_to_shallow'] if error <= gate else None}
    removal = removals(source_path, candidate_path, simconfig, candidate_config, (plain['deep_reference'], plain['candidate']),
                       matched, probes, normal, table['removals'])
    scores = {'structure': shape, 'accuracy': accuracy, 'sim_cost': cost, 'cheap': cheap, 'removal': removal}
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / 'probes.json').write_text(json.dumps(trajectories, allow_nan=False))
    (artifacts / 'scores.json').write_text(json.dumps(scores, allow_nan=False))
    return scores
