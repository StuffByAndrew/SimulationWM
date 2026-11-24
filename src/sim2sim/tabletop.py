"""Private metrics and researcher view of the touch-only tabletop levels (environments/panda_shadow; manifest [tabletop]).

Metric `objects` (fresh reset state, both worlds): every source object body (manifest object_bodies) is assigned at most one
candidate object, i.e. a top-level non-robot candidate body with its descendants, by geometry only: a one-to-one assignment
minimising the symmetric Chamfer distance between the two objects' exposed primitive-union surfaces (never names, color,
density or mass). Per source object: the assigned candidate body, its Chamfer distance, the distance between the two
surface-sample centroids, found (assigned with the centroid error within the source object's bounding radius about its own
centroid), the label check (candidate body name contains a private synonym of [tabletop.labels], only where a list exists)
and the prior-defeating dimensions ([[tabletop.prior_dimensions]]: exact extent of both objects along a world axis).
An unassigned object has no Chamfer, centroid or label: gaps with a reason, never zeros.

Metric `identification` (L3): the submitted project's root parameters.json {"spiky_ball_xyz": [x, y, z]} (world m, the
ball's initial position) against the initial ball positions: correct when the nearest ball is [tabletop] spiky, and the
distance to it. A missing or malformed file leaves identification unavailable, with the reason; the rest of the submission
is scored as usual.

Post-run (researcher-only, `python -m sim2sim.tabletop RUN_DIR ...` on a Slurm test node; reads the verified audit and
source snapshot, writes RUN/tabletop.json and RUN/touch-view.html): grab success on the reference world's last recorded state
(L3: spiky ball centre of mass at least GRAB_LIFT_M above the bin rim and touching the robot, every smooth ball inside the
bin), and the contact coverage map: every taxel contact (a contact on a tactile pad body that its taxel grid counts) in
world coordinates at every recorded reference-world state, drawn over the true scene, with each object's covered surface
fraction (surface samples within COVER_M of a contact). Nothing here reaches the agent.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

GRAB_LIFT_M = .02
COVER_M = .01


def support(primitive, axis):
    """Largest projection of a compiled primitive (primitive_metrics._primitives entry) onto a unit world axis."""
    a, kind, s = np.asarray(axis, float), primitive['kind'], primitive['size']
    local, centre = primitive['rotation'].T @ a, float(primitive['pos'] @ a)
    if kind == 'mesh':
        return centre + float(np.max(s.points[s.vertices] @ local))
    if kind == 'box':
        return centre + float(np.abs(local) @ s[:3])
    if kind == 'sphere':
        return centre + float(s[0])
    if kind == 'ellipsoid':
        return centre + float(np.linalg.norm(local * s[:3]))
    if kind == 'capsule':
        return centre + float(s[1] * abs(local[2]) + s[0])
    if kind == 'cylinder':
        return centre + float(s[1] * abs(local[2]) + s[0] * math.sqrt(max(0., 1 - local[2] ** 2)))
    raise ValueError(f'unsupported primitive: {kind}')


def extent(primitives, axis):
    """Exact length of a primitive union's shadow on a unit axis."""
    return max(support(p, axis) for p in primitives) + max(support(p, -np.asarray(axis, float)) for p in primitives)


def _objects(sim, primitives):
    """Primitives grouped by object: the top-level body under the world that holds each geom."""
    groups = {}
    for p in primitives:
        root = int(sim._model.body_rootid[sim._model.geom_bodyid[p['id']]])
        groups.setdefault(sim._model.body(root).name or f'body_{root}', []).append(p)
    return groups


def objects(source, candidate, config):
    """Metric `objects`: per source object assignment, Chamfer, centroid, found, label and prior-defeating dimensions."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree
    from .primitive_metrics import _primitives, exposed_surface
    reconstruction, table = config.get('reconstruction') or {}, config.get('tabletop') or {}
    samples, seed = int(reconstruction.get('samples', 2048)), int(reconstruction.get('seed', 0))
    for sim in (source, candidate):
        sim.action({'op': 'reset'}, render=False)
    truth = _objects(source, _primitives(source, geom_names=config['object_geoms']))
    assert sorted(truth) == sorted(config['object_bodies']), 'object_geoms must cover exactly the top-level object_bodies'
    try:
        found = _objects(candidate, _primitives(candidate, exclude_robot=True))
    except ValueError as exc:
        return {'status': 'unavailable', 'reason': f'candidate objects: {exc}'}
    names, others = list(truth), list(found)
    a = {n: exposed_surface(truth[n], samples, seed) for n in names}
    b = {n: exposed_surface(found[n], samples, seed) for n in others}
    cost = np.array([[(cKDTree(a[s]).query(b[c])[0].mean() + cKDTree(b[c]).query(a[s])[0].mean()) / 2 for c in others] for s in names])
    assignment = dict(zip(*linear_sum_assignment(cost)))
    records = []
    for i, name in enumerate(names):
        centre = a[name].mean(axis=0)
        record = {'name': name, 'source_centroid_m': centre.tolist(), 'source_radius_m': float(np.linalg.norm(a[name] - centre, axis=1).max())}
        synonyms = (table.get('labels') or {}).get(name)
        if i in assignment:
            other = others[assignment[i]]
            error = float(np.linalg.norm(b[other].mean(axis=0) - centre))
            record.update(candidate_body=other, chamfer_m=float(cost[i, assignment[i]]), centroid_error_m=error,
                          candidate_centroid_m=b[other].mean(axis=0).tolist(), found=error <= record['source_radius_m'],
                          label_correct=None if synonyms is None else any(s in other.lower() for s in synonyms))
        else:
            record.update(candidate_body=None, chamfer_m=None, centroid_error_m=None, found=False, label_correct=None,
                          unassigned_reason='fewer candidate objects than source objects')
        record['dimensions'] = []
        for d in (d for d in table.get('prior_dimensions', []) if d['body'] == name):
            truth_m = extent(truth[name], d['axis'])
            guess = extent(found[record['candidate_body']], d['axis']) if record['candidate_body'] else None
            record['dimensions'].append({'name': d['name'], 'source_m': truth_m, 'candidate_m': guess,
                                         'absolute_error_m': None if guess is None else abs(guess - truth_m)})
        records.append(record)
    labelled = [r for r in records if (table.get('labels') or {}).get(r['name']) is not None]
    chamfers = [r['chamfer_m'] for r in records if r['chamfer_m'] is not None]
    return {'status': 'ok', 'matching': 'automatic_geometry_only', 'per_object': records, 'candidate_objects': others,
            'found_fraction': sum(r['found'] for r in records) / len(records),
            'mean_assigned_chamfer_m': float(np.mean(chamfers)) if chamfers else None,
            'label_accuracy': sum(r['label_correct'] is True for r in labelled) / len(labelled) if labelled else None,
            'surface_samples': samples, 'surface_seed': seed}


def identification(source, project, config):
    """Metric `identification`: parameters.json spiky_ball_xyz against the initial ball positions (see module docstring)."""
    path = Path(project) / 'parameters.json'
    if not path.is_file():
        return {'status': 'unavailable', 'reason': 'no parameters.json at the project root'}
    try:
        xyz = json.loads(path.read_text())['spiky_ball_xyz']
        if not (isinstance(xyz, list) and len(xyz) == 3 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in xyz)):
            raise ValueError('spiky_ball_xyz must be three finite numbers')
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
        return {'status': 'unavailable', 'reason': f'malformed parameters.json: {type(exc).__name__}: {exc}'}
    source.action({'op': 'reset'}, render=False)
    distances = {name: float(np.linalg.norm(np.asarray(xyz) - source._data.xpos[source._model.body(name).id])) for name in config['object_bodies']}
    nearest = min(distances, key=distances.get)
    spiky = config['tabletop']['spiky']
    return {'status': 'ok', 'spiky_ball_xyz': xyz, 'nearest_ball': nearest, 'correct': nearest == spiky,
            'distance_error_m': distances[spiky], 'ball_distances_m': distances}


def grab(sim, config):
    """L3 grab success in sim's current state (module docstring)."""
    from .primitive_metrics import _robot_bodies
    m, d = sim._model, sim._data
    box, spiky = config['tabletop']['bin'], config['tabletop']['spiky']
    robot, ball = _robot_bodies(sim), m.body(spiky).id
    touching = sorted({m.body(int(m.geom_bodyid[g])).name for c in d.contact[:d.ncon] for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1))
                       if m.geom_bodyid[other] == ball and m.geom_bodyid[g] in robot})
    inside = {name: bool(np.all(np.abs(d.xipos[m.body(name).id][:2] - box['center']) < box['inner_half'])
                         and box['floor_z'] < d.xipos[m.body(name).id][2] < box['rim_z']) for name in config['object_bodies'] if name != spiky}
    lift = float(d.xipos[ball][2] - box['rim_z'])
    return {'success': lift >= GRAB_LIFT_M and bool(touching) and all(inside.values()), 'spiky_com_above_rim_m': lift,
            'spiky_touching_robot_bodies': touching, 'smooth_balls_in_bin': inside}


def analyse(run_dir):
    """Researcher-only post-run pass over a verified audit: taxel contacts at every recorded reference-world state (checked
    against the recorded tactile observation), per-object coverage, and grab success at the last state. Writes
    RUN/tabletop.json and RUN/touch-view.html."""
    import hashlib
    import tempfile
    import tomllib
    from .audit import read_audit
    from .candidate_history import _save, source_entrypoint
    from .primitive_metrics import _primitives, _robot_bodies, exposed_surface
    from .realtime import restore
    from .runtime import extract_bundle
    from .simulation import Simulation
    run = Path(run_dir).resolve()
    manifest, states, last = None, {}, None
    for _, event in read_audit(run):
        if event.get('kind') == 'run_start':
            manifest = event['manifest']
        state = event.get('state')
        if event.get('world') == 'reference' and event.get('kind') in ('checkpoint', 'tick', 'interrupted') and (state or {}).get('arrays'):
            last = (event['action_id'], state)
            states.setdefault(event['state_sha256'], (*last, (event.get('observation') or {}).get('tactile')))
    source = (run / 'source-snapshot.zip').read_bytes()
    assert hashlib.sha256(source).hexdigest() == manifest['source_bundle_sha256'], 'source snapshot hash mismatch'
    with tempfile.TemporaryDirectory() as temp:
        extract_bundle(source, temp)
        entry = Path(temp) / source_entrypoint(manifest)
        config = tomllib.loads((Path(temp) / manifest['evaluation'].get('manifest', 'manifest.toml')).read_text())
        sim = Simulation(entry, {**manifest['resolved_source_config'], 'rgb': False, 'depth': False, 'calibration': False, 'tactile': True})
    m, d = sim._model, sim._data
    initial = {b: (d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()) for b in range(m.nbody)}
    truth = _objects(sim, _primitives(sim, geom_names=config['object_geoms']))
    surfaces = {name: exposed_surface(prims, 2048, 0) for name, prims in truth.items()}
    camera = (m.cam_pos[sim._camera_id].copy(), m.cam_quat[sim._camera_id].copy())
    robot, pad_body = _robot_bodies(sim), {name: body for name, body, *_ in sim._pads}
    contacts, checked, self_contacts = [], 0, 0
    for action, state, observed in states.values():
        restore(sim, state, camera)
        sim._proprioception()
        if observed is not None:  # the recomputed taxels must equal what the agent observed at this state
            current = sim._tactile()
            assert all(np.allclose(current[name][key], value, rtol=0, atol=1e-9) for name, pad in observed.items() if isinstance(pad, dict)
                       for key, value in pad.items()), f'recorded tactile differs from the restored state (action {action})'
            checked += 1
        data = sim._proprioception_data
        for pad, cell, force, i in sim._pad_contacts(data):
            c = data.contact[i]
            body = next(int(m.geom_bodyid[g]) for g in (c.geom1, c.geom2) if int(m.geom_bodyid[g]) != pad_body[pad])
            if cell is None or force <= 0 or body in robot:
                self_contacts += cell is not None and body in robot
                continue
            mapped = initial[body][1] @ data.xmat[body].reshape(3, 3).T @ (c.pos - data.xpos[body]) + initial[body][0]
            contacts.append({'action': action, 'pad': pad, 'cell': list(cell), 'force_n': round(force, 5), 'body': m.body(body).name or 'world',
                             'world_m': np.round(c.pos, 5).tolist(), 'initial_m': np.round(mapped, 5).tolist()})
    coverage = {}
    for name, points in surfaces.items():
        touched = np.array([c['initial_m'] for c in contacts if c['body'] == name]).reshape(-1, 3)
        near = np.zeros(len(points), bool) if not len(touched) else np.min(np.linalg.norm(points[:, None] - touched[None], axis=2), axis=1) <= COVER_M
        coverage[name] = {'contacts': len(touched), 'covered_surface_fraction': float(near.mean())}
    result = {'schema_version': 1, 'states': len(states), 'states_with_tactile_checked': checked, 'contacts': len(contacts),
              'robot_self_taxel_contacts': self_contacts, 'coverage': coverage}
    if 'bin' in config.get('tabletop', {}):
        restore(sim, last[1], camera)
        result['grab_at_last_reference_state'] = {'action': last[0], **grab(sim, config)}
    sim.close()
    _save(run / 'tabletop.json', result)
    rng = np.random.default_rng(0)
    payload = {'run': run.name, 'result': result, 'contacts': contacts, 'bin': config.get('tabletop', {}).get('bin'),
               'objects': {n: np.round(p[rng.choice(len(p), 400, replace=False)], 4).tolist() for n, p in surfaces.items()}}
    (run / 'touch-view.html').write_text(VIEW.replace('__DATA__', json.dumps(payload, separators=(',', ':')).replace('</', '<\\/')))
    return result


VIEW = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Touch coverage</title><style>
:root{--bg:#fbfbfa;--card:#fff;--fg:#1f2329;--mute:#6a717c;--line:#e2e4e8;--obj:#9aa0a8}
@media (prefers-color-scheme:dark){:root{--bg:#121418;--card:#1a1d22;--fg:#e4e6e9;--mute:#99a0aa;--line:#2d3138;--obj:#5d636c}}
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--mute);font-size:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr));gap:12px;margin-top:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;min-width:0}canvas{width:100%;height:auto;display:block}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}td,th{padding:3px 8px;border-bottom:1px solid var(--line);text-align:right}td:first-child,th:first-child{text-align:left}
.key span{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 4px 0 10px;vertical-align:-1px}</style></head><body>
<h1>Touch coverage · <span id="run"></span></h1>
<div class="sub">Researcher-only. Every taxel contact (a contact counted by a tactile pad's taxel grid) at every recorded reference-world state,
drawn over the true objects at their initial poses. <label><input type="checkbox" id="raw"> show where contacts happened (moved objects
not mapped back)</label></div><div class="key sub" id="key"></div>
<div class="grid"><div class="card"><b>Top (x, y)</b><canvas id="xy" width="520" height="520"></canvas></div>
<div class="card"><b>Front (x, z)</b><canvas id="xz" width="520" height="300"></canvas></div>
<div class="card"><b>Side (y, z)</b><canvas id="yz" width="520" height="300"></canvas></div>
<div class="card" id="summary"></div></div>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);$('run').textContent=D.run;
const pads=['thumb','first','middle','ring','little','palm'],colors=['#d23b3b','#2f6bd0','#1f9d6a','#d0842f','#8b52c4','#c43d9a'],css=n=>getComputedStyle(document.documentElement).getPropertyValue(n);
$('key').innerHTML=pads.map((p,i)=>`<span style="background:${colors[i]}"></span>${p}`).join('');
const pts=Object.values(D.objects).flat(),all=pts.concat(D.contacts.map(c=>c.initial_m));
const lo=[0,1,2].map(i=>Math.min(...all.map(p=>p[i]))-.03),hi=[0,1,2].map(i=>Math.max(...all.map(p=>p[i]))+.03);
function draw(id,a,b){const cv=$(id),ctx=cv.getContext('2d'),s=Math.min(cv.width/(hi[a]-lo[a]),cv.height/(hi[b]-lo[b])),
 X=p=>(p[a]-lo[a])*s,Y=p=>cv.height-(p[b]-lo[b])*s;ctx.clearRect(0,0,cv.width,cv.height);ctx.fillStyle=css('--obj');
 for(const p of pts){ctx.fillRect(X(p)-1,Y(p)-1,2,2)}
 if(D.bin&&a===0&&b===1){const c=D.bin.center,h=D.bin.inner_half;ctx.strokeStyle=css('--mute');ctx.strokeRect(X([c[0]-h,0]),Y([0,c[1]+h]),2*h*s,2*h*s)}
 for(const c of D.contacts){const p=$('raw').checked?c.world_m:c.initial_m;ctx.fillStyle=colors[pads.indexOf(c.pad)];ctx.globalAlpha=.6;
  ctx.beginPath();ctx.arc(X(p),Y(p),1.5+Math.min(4,Math.sqrt(c.force_n)),0,7);ctx.fill()}ctx.globalAlpha=1}
function all3(){draw('xy',0,1);draw('xz',0,2);draw('yz',1,2)}$('raw').onchange=all3;all3();
const R=D.result,f=v=>v===null||v===undefined?'—':(+v).toFixed(3);
$('summary').innerHTML=`<b>Summary</b><div class="sub">${R.states} recorded states (${R.states_with_tactile_checked} checked against the observed taxels), ${R.contacts} taxel contacts</div>
<table><tr><th>object</th><th>contacts</th><th>surface within 1 cm</th></tr>${Object.entries(R.coverage).map(([n,c])=>`<tr><td>${n}</td><td>${c.contacts}</td><td>${(100*c.covered_surface_fraction).toFixed(1)}%</td></tr>`).join('')}</table>`+
(R.grab_at_last_reference_state?`<p><b>Grab at the last reference state:</b> ${R.grab_at_last_reference_state.success?'success':'failure'} · spiky COM ${f(R.grab_at_last_reference_state.spiky_com_above_rim_m)} m above the rim · touching ${R.grab_at_last_reference_state.spiky_touching_robot_bodies.join(', ')||'nothing'} · smooth balls in bin ${Object.values(R.grab_at_last_reference_state.smooth_balls_in_bin).filter(x=>x).length}/${Object.keys(R.grab_at_last_reference_state.smooth_balls_in_bin).length}</p>`:'');
</script></body></html>
"""


def main():
    import argparse
    from .candidate_history import require_compute_allocation
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('runs', nargs='+')
    args = parser.parse_args()
    require_compute_allocation()
    for run in args.runs:
        print(json.dumps({'run': run, **{k: v for k, v in analyse(run).items() if k != 'coverage'}}), flush=True)


if __name__ == '__main__':
    main()
