"""Seeded, recorded color, non-round physics and pose-jitter perturbations of a prepared run's private MJCF source copy."""
from __future__ import annotations

import colorsys
import hashlib
import json
import math
from pathlib import Path
import random
import tomllib
import xml.etree.ElementTree as ET

GRAY = [0.4537, 0.4537, 0.4537, 1.0]  # non-round: rounding pixel fits cannot land on it exactly
FACTORS = {'size': (.8, 1.25), 'mass': (.6, 1.6), 'density': (.6, 1.6),
           'damping': (.6, 1.6), 'frictionloss': (.6, 1.6), 'stiffness': (.6, 1.6)}
SIZE_COUNT = {'sphere': 1, 'capsule': 2, 'cylinder': 2, 'box': 3, 'ellipsoid': 3}
PENETRATION_M = 1e-3
JITTER = {'xy_m': .03, 'yaw_deg': 15.0}  # one rigid planar offset and yaw of the whole object group about its centroid


def validate(spec):
    """Reject malformed private [perturbation] tables before any source is copied."""
    if not isinstance(spec, dict) or not isinstance(spec.get('seed'), int) or isinstance(spec['seed'], bool):
        raise ValueError('perturbation.seed must be an integer')
    families = set(spec) - {'seed'}
    if not families or not families <= {'colors', 'physics', 'pose'} or spec.get('colors', 'random') not in ('random', 'uniform_gray') \
            or spec.get('physics', 'nonround') != 'nonround' or spec.get('pose', 'jitter') != 'jitter':
        raise ValueError('perturbation takes seed and colors = "random" or "uniform_gray", physics = "nonround" and/or pose = "jitter"')


# Stereotypical colors an agent might guess without seeing the scene (pure and CSS variants; gray = MuJoCo default).
PALETTE = {'red': [1, 0, 0], 'green': [0, 1, 0], 'css_green': [0, .502, 0], 'blue': [0, 0, 1], 'yellow': [1, 1, 0],
           'orange': [1, .5, 0], 'css_orange': [1, .647, 0], 'purple': [.502, 0, .502], 'violet': [.5, 0, 1], 'cyan': [0, 1, 1],
           'magenta': [1, 0, 1], 'white': [1, 1, 1], 'black': [0, 0, 0], 'gray_mujoco_default': [.5, .5, .5]}
COLOR_RULE = {'palette': PALETTE, 'min_rgb_distance': {'palette': .35, 'original': .35, 'fixtures': .25, 'each_other': .25},
              'hsv': {'saturation': [.5, 1], 'value': [.5, 1]}, 'decimals': 4, 'channel_min_offset_from_0.05_multiple': .0005}


def _random_colors(rng, original, avoid):
    """4-decimal HSV draws, no channel near a 0.05 multiple, far from stereotypical, authored, fixture and other canary colors."""
    d = COLOR_RULE['min_rgb_distance']
    for _ in range(100):
        colors = {}
        for name in original:
            for _ in range(1000):
                c = [round(v, 4) for v in colorsys.hsv_to_rgb(rng.random(), rng.uniform(.5, 1), rng.uniform(.5, 1))]
                if all(abs(v * 20 - round(v * 20)) > .01 for v in c) and math.dist(c, original[name][:3]) >= d['original'] \
                        and all(math.dist(c, p) >= d['palette'] for p in PALETTE.values()) \
                        and all(math.dist(c, a[:3]) >= d['fixtures'] for a in avoid) \
                        and all(math.dist(c, o[:3]) >= d['each_other'] for o in colors.values()):
                    colors[name] = [*c, 1.0]
                    break
            else:
                break
        if len(colors) == len(original):
            return colors
    raise ValueError(f'cannot choose {len(original)} distinguishable object colors')


def nonround(value):
    """Positive 4-significant-digit value whose mantissa m (1000-9999) has a nonzero, non-5 last digit and lies
    more than 5 from any multiple of 50 and 50 from any multiple of 500 (in [0.01, 0.1): not within 0.0005 of 0.005k)."""
    m = int(f'{value:.3e}'[:5].replace('.', ''))
    return value > 0 and value == float(f'{value:.4g}') and m % 5 != 0 and 5 < m % 50 < 45 and 50 < m % 500 < 450


def _resample(rng, value, family):
    assert value >= 0, f'negative {family} {value}'
    if value == 0:
        return 0.0  # absent damping/friction/stiffness stays absent
    for _ in range(1000):
        sample = float(f'{value * rng.uniform(*FACTORS[family]):.4g}')
        if nonround(sample):
            return sample
    raise AssertionError('non-round resampling exhausted')


def _lowest(model, data, root):
    """Lowest world z over the primitive geoms of a top-level body's subtree."""
    import mujoco
    import numpy as np
    low = math.inf
    for g in np.flatnonzero(model.body_rootid[model.geom_bodyid] == root):
        axis, s = data.geom_xmat[g].reshape(3, 3)[2], model.geom_size[g]
        extent = {mujoco.mjtGeom.mjGEOM_SPHERE: lambda: s[0],
                  mujoco.mjtGeom.mjGEOM_CAPSULE: lambda: abs(axis[2]) * s[1] + s[0],
                  mujoco.mjtGeom.mjGEOM_CYLINDER: lambda: abs(axis[2]) * s[1] + s[0] * math.sqrt(max(0., 1 - axis[2] ** 2)),
                  mujoco.mjtGeom.mjGEOM_BOX: lambda: np.abs(axis) @ s,
                  mujoco.mjtGeom.mjGEOM_ELLIPSOID: lambda: np.linalg.norm(axis * s)}[mujoco.mjtGeom(model.geom_type[g])]()
        low = min(low, data.geom_xpos[g][2] - extent)
    return float(low)


def _penetration(model, data, roots):
    """Deepest current contact penetration (m) involving any object subtree."""
    hits = [-float(d) for (a, b), d in zip(data.contact.geom[:data.ncon], data.contact.dist[:data.ncon])
            if {int(model.body_rootid[model.geom_bodyid[a]]), int(model.body_rootid[model.geom_bodyid[b]])} & roots]
    return max(hits, default=0.0)


def _simconfig(config, manifest):
    """The trusted source simulation settings that preparation, verification and evaluation reset with."""
    evaluation = config['evaluation']
    simconfig = dict(config.get('simulation', {}), **{k: evaluation[k] for k in ('camera', 'width', 'height') if k in evaluation})
    for key in ('camera', 'graspable_bodies'):
        if key in manifest:
            simconfig.setdefault(key, manifest[key])
    return simconfig


def _compile(scene, simconfig):
    """Compiled model and forwarded state at the simulator's reset pose (e.g. the Panda home pose)."""
    from .simulation import Simulation
    sim = Simulation(scene, simconfig)
    return sim._model, sim._data


def _probes_hit(sim, manifest):
    """Every private grasp probe's pixel still raycasts onto its declared body."""
    import mujoco
    from .evaluation import _probe_actions
    return all((hit := sim._raycast(_probe_actions(sim, probe)[1]['pixel'])) is not None
               and hit[0] == mujoco.mj_name2id(sim._model, mujoco.mjtObj.mjOBJ_BODY, probe['body']) for probe in manifest.get('probes', []))


def _pose(scene, tree, top, names, rng, simconfig, manifest):
    """Rigidly offset and yaw all free object bodies together; keep heights, contacts between objects and probe hits."""
    import mujoco
    import numpy as np
    from .simulation import Simulation
    assert simconfig.get('robot') is None, 'pose jitter refuses Panda scenes: arm probes target fixed object poses'
    sim = Simulation(scene, simconfig)
    model = sim._model
    ids = [model.body(n).id for n in names]
    joints = [j for j in range(model.njnt) if model.body_rootid[model.jnt_bodyid[j]] in ids]
    if len(joints) != len(ids) or any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_bodyid[j] not in ids for j in joints):
        raise ValueError('pose jitter requires one free joint on each top-level object body (hinged or welded objects are anchored)')
    original = {n: (model.body_pos[i].tolist(), model.body_quat[i].tolist()) for n, i in zip(names, ids)}
    centroid = np.mean([model.body_pos[i][:2] for i in ids], axis=0)
    sim.close()
    for attempt in range(1, 101):
        offset, yaw = [rng.uniform(-JITTER['xy_m'], JITTER['xy_m']) for _ in range(2)], rng.uniform(-JITTER['yaw_deg'], JITTER['yaw_deg'])
        c, s, values = math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), {}
        turn = np.array([math.cos(math.radians(yaw) / 2), 0, 0, math.sin(math.radians(yaw) / 2)])
        for n in names:
            pos, quat = np.array(original[n][0]), np.zeros(4)
            x, y = pos[:2] - centroid
            pos[:2] = centroid + [c * x - s * y, s * x + c * y] + offset
            mujoco.mju_mulQuat(quat, turn, np.array(original[n][1]))
            values[n] = (pos.tolist(), quat.tolist())
            for key in ('euler', 'axisangle', 'xyaxes', 'zaxis'):
                top[n].attrib.pop(key, None)
            top[n].set('pos', ' '.join(map(repr, values[n][0])))
            top[n].set('quat', ' '.join(map(repr, values[n][1])))
        tree.write(scene, encoding='unicode')
        sim = Simulation(scene, simconfig)
        penetration, hit = _penetration(sim._model, sim._data, set(ids)), _probes_hit(sim, manifest)
        sim.close()
        if penetration <= PENETRATION_M and hit:
            break
    else:
        raise ValueError(f'no pose jitter without >{PENETRATION_M} m penetration and with every probe on its body in 100 attempts')
    return {'mode': 'jitter', 'range': JITTER, 'attempts': attempt, 'offset_m': offset, 'yaw_deg': yaw, 'centroid_xy_m': centroid.tolist(),
            'initial_penetration_m': penetration,
            'bodies': {n: {'pos': {'original': original[n][0], 'value': values[n][0]}, 'quat': {'original': original[n][1], 'value': values[n][1]}}
                       for n in names}}


def _physics(scene, tree, top, names, geoms, spec, rng, simconfig):  # spec: MuJoCo's resolved MjSpec of the unperturbed copy
    """Resample object sizes, mass/density and hinge parameters; keep placed bodies' lowest points; reject penetration."""
    import mujoco
    if any(top[n].find('.//inertial') is not None for n in names):
        raise ValueError('physics perturbation refuses explicit object inertials: geom mass/density would not apply')
    targets = []  # (element, attribute, record key, family, original values)
    for geom in geoms:
        g, name = spec.geom(geom.get('name')), geom.get('name')
        kind = g.type.name.removeprefix('mjGEOM_').lower()
        if kind not in SIZE_COUNT:
            raise ValueError(f'physics perturbation supports primitive object geoms only, not {kind} {name}')
        count = 1 if not math.isnan(g.fromto[0]) else SIZE_COUNT[kind]  # fromto endpoints are anchors; only radius changes
        targets.append((geom, 'size', ('geoms', name), 'size', [float(v) for v in g.size[:count]]))
        family = 'density' if math.isnan(g.mass) else 'mass'  # explicit mass overrides density
        targets.append((geom, family, ('geoms', name), family, [float(getattr(g, family))]))
    for n in names:
        for joint in top[n].iter('joint'):
            if joint.get('name') is None:
                raise ValueError(f'object joints must be named for physics perturbation ({n})')
            j = spec.joint(joint.get('name'))
            if j.type == mujoco.mjtJoint.mjJNT_FREE:
                continue
            if j.type != mujoco.mjtJoint.mjJNT_HINGE or any(j.damping[1:]) or any(j.stiffness[1:]):
                raise ValueError(f'physics perturbation supports scalar-parameter hinges only ({joint.get("name")})')
            for family, value in (('damping', j.damping[0]), ('frictionloss', j.frictionloss), ('stiffness', j.stiffness[0])):
                targets.append((joint, family, ('joints', joint.get('name')), family, [float(value)]))
    original, data = _compile(scene, simconfig)
    roots = {original.body(n).id for n in names}
    authored = _penetration(original, data, roots)
    if authored > PENETRATION_M:
        raise ValueError(f'physics perturbation refused: the authored scene already has {authored:.4f} m object contact penetration at reset')
    # Free and welded object roots are placed on supports; hinged roots stay anchored at their axes.
    placed = [n for n in names if all(original.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                                      for j in range(original.njnt) if original.body_rootid[original.jnt_bodyid[j]] == original.body(n).id)]
    lows = {n: _lowest(original, data, original.body(n).id) for n in placed}
    poses = {n: [float(v) for v in spec.body(n).pos] for n in placed}
    for attempt in range(1, 101):
        values = [[_resample(rng, v, family) for v in vals] for *_, family, vals in targets]
        for (element, attribute, *_), vals in zip(targets, values):
            element.set(attribute, ' '.join(f'{v:.4g}' for v in vals))
        for n in placed:
            top[n].set('pos', ' '.join(repr(v) for v in poses[n]))
        tree.write(scene, encoding='unicode')
        model, data = _compile(scene, simconfig)
        shifted = {n: [*poses[n][:2], float(poses[n][2] + lows[n] - _lowest(model, data, model.body(n).id))] for n in placed}
        for n in placed:
            top[n].set('pos', ' '.join(repr(v) for v in shifted[n]))
        tree.write(scene, encoding='unicode')
        model, data = _compile(scene, simconfig)
        penetration = _penetration(model, data, roots)
        if penetration <= PENETRATION_M:
            break
    else:
        raise ValueError(f'no physics perturbation without >{PENETRATION_M} m initial penetration in 100 attempts')
    record = {'mode': 'nonround', 'factors': FACTORS, 'attempts': attempt, 'initial_penetration_m': penetration,
              'geoms': {}, 'joints': {}, 'placed_bodies': {n: {'pos': {'original': poses[n], 'value': shifted[n]}} for n in placed}}
    for (element, attribute, (group, name), family, vals), new in zip(targets, values):
        record[group].setdefault(name, {})[attribute] = {'original': vals if family == 'size' else vals[0],
                                                         'value': new if family == 'size' else new[0]}
    return record


def apply(scene, manifest_path, spec, original_scene, config):
    """Perturb object geoms/joints in the private scene copy and return every original and resolved value."""
    import mujoco
    scene = Path(scene)
    manifest = tomllib.loads(Path(manifest_path).read_text())
    original_sha256 = hashlib.sha256(scene.read_bytes()).hexdigest()
    assert hashlib.sha256(Path(original_scene).read_bytes()).hexdigest() == original_sha256, 'source copy differs from original'
    tree = ET.parse(scene)
    top = {b.get('name'): b for b in tree.getroot().find('worldbody').findall('body')}
    names = manifest.get('object_bodies') or list(manifest.get('correspondence', {}))
    if not names or not set(names) <= set(top):
        raise ValueError('manifest object_bodies/correspondence must name top-level scene bodies')
    geoms = [g for name in names for g in top[name].iter('geom')]
    geom_names = [g.get('name') for g in geoms]
    if None in geom_names or len(set(geom_names)) != len(geoms):
        raise ValueError('object geoms must be uniquely named')
    if set(manifest.get('object_geoms', geom_names)) != set(geom_names):
        raise ValueError('manifest object_geoms differ from geoms under the object bodies')
    model = mujoco.MjSpec.from_file(str(scene))  # MuJoCo's own default-class resolution, no compilation
    record = {'seed': spec['seed'], 'specification': spec, 'original_scene': str(Path(original_scene).resolve()),
              'original_scene_sha256': original_sha256, 'object_bodies': names, 'object_geoms': geom_names}
    if 'colors' in spec:
        if any(model.geom(n).material for n in geom_names):
            raise ValueError('object geom materials would override or tint perturbed rgba')
        original = {n: [float(v) for v in model.geom(n).rgba] for n in geom_names}
        avoid = [list(g.rgba) for g in model.geoms if g.name not in original and not g.material]
        values = {n: GRAY for n in geom_names} if spec['colors'] == 'uniform_gray' else \
            _random_colors(random.Random(f"{spec['seed']}/colors"), original, avoid)
        for geom in geoms:
            geom.set('rgba', ' '.join(repr(float(v)) for v in values[geom.get('name')]))
        record['colors'] = {'mode': spec['colors'], 'geoms': {n: {'original': original[n], 'value': values[n]} for n in geom_names},
                            **({'rule': COLOR_RULE} if spec['colors'] == 'random' else {})}
    tree.write(scene, encoding='unicode')
    if 'physics' in spec:
        record['physics'] = _physics(scene, tree, top, names, geoms, model, random.Random(f"{spec['seed']}/physics"),
                                     _simconfig(config, manifest))
    if 'pose' in spec:
        record['pose'] = _pose(scene, tree, top, names, random.Random(f"{spec['seed']}/pose"), _simconfig(config, manifest), manifest)
    return record


def _volume(kind, s):
    return {'sphere': 4 / 3 * math.pi * s[0] ** 3, 'box': 8 * s[0] * s[1] * s[2], 'ellipsoid': 4 / 3 * math.pi * s[0] * s[1] * s[2],
            'cylinder': 2 * math.pi * s[0] ** 2 * s[1], 'capsule': 2 * math.pi * s[0] ** 2 * s[1] + 4 / 3 * math.pi * s[0] ** 3}[kind]


def _verify_physics(o, m, physics, sims, manifest, report, pose=None):
    """Compiled physics values, untouched anchors, non-penetration and still-valid private probes."""
    import mujoco
    import numpy as np
    from .evaluation import _panda_probe
    geom = {name: (mujoco.mj_name2id(o, mujoco.mjtObj.mjOBJ_GEOM, name), mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name))
            for name in physics['geoms']}
    expected_mass = {}
    for name, item in physics['geoms'].items():
        og, g = geom[name]
        size, count = item['size'], len(item['size']['value'])
        assert og >= 0 and g >= 0, f'{name} missing'
        assert np.allclose(m.geom_size[g][:count], size['value'], rtol=1e-12) and np.allclose(o.geom_size[og][:count], size['original'], rtol=1e-12), f'{name} size'
        assert np.array_equal(m.geom_size[g][count:], o.geom_size[og][count:]), f'{name} anchored extent changed'
        kind = mujoco.mjtGeom(m.geom_type[g]).name.removeprefix('mjGEOM_').lower()
        family = 'mass' if 'mass' in item else 'density'
        assert all(map(nonround, size['value'] + [item[family]['value']])), f'{name} round value'
        assert size['value'] != size['original'] and item[family]['value'] != item[family]['original'], f'{name} unchanged'
        body = int(m.geom_bodyid[g])
        expected_mass[body] = expected_mass.get(body, 0) + (item['mass']['value'] if family == 'mass' else
                                                            item['density']['value'] * _volume(kind, m.geom_size[g]))
    for body, mass in expected_mass.items():
        assert math.isclose(m.body_mass[body], mass, rel_tol=1e-9), f'{m.body(body).name} compiled mass {m.body_mass[body]} != {mass}'
    for name, item in physics['joints'].items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        oj = mujoco.mj_name2id(o, mujoco.mjtObj.mjOBJ_JOINT, name)
        for family, compiled in (('damping', lambda x, i: x.dof_damping[x.jnt_dofadr[i]]),
                                 ('frictionloss', lambda x, i: x.dof_frictionloss[x.jnt_dofadr[i]]),
                                 ('stiffness', lambda x, i: x.jnt_stiffness[i])):
            value = item[family]
            assert compiled(m, j) == value['value'] and compiled(o, oj) == value['original'], f'{name} {family} not applied'
            assert (value['value'] == 0) if value['original'] == 0 else nonround(value['value']), f'{name} {family} round'
    moved = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n): item['pos']['value'] for n, item in physics['placed_bodies'].items()}
    expected, quats = o.body_pos.copy(), o.body_quat.copy()
    for body, pos in moved.items():
        assert np.array_equal(expected[body][:2], pos[:2]), 'placed body moved horizontally'
        expected[body] = pos
    for name, item in (pose or {}).get('bodies', {}).items():
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        assert np.allclose(expected[body], item['pos']['original'], rtol=0, atol=1e-12), f'{name} pose jitter started from another pose'
        expected[body], quats[body] = item['pos']['value'], item['quat']['value']
    others = [g for g in range(m.ngeom) if m.geom(g).name not in physics['geoms']]
    assert m.nbody == o.nbody and m.njnt == o.njnt and m.ngeom == o.ngeom, 'model structure changed'
    assert np.allclose(m.body_pos, expected, rtol=0, atol=1e-12) and np.allclose(m.body_quat, quats, rtol=0, atol=1e-12), 'bodies moved'
    assert np.array_equal(m.geom_pos, o.geom_pos) and np.array_equal(m.geom_quat, o.geom_quat), 'geom placement changed'
    assert np.array_equal(m.jnt_pos, o.jnt_pos) and np.array_equal(m.jnt_axis, o.jnt_axis), 'joint anchors/axes changed'
    assert np.array_equal(m.geom_size[others], o.geom_size[others]), 'non-object geom sizes changed'
    original, perturbed = sims
    if perturbed.robot is not None:
        for probe in manifest.get('arm_probes', []):
            targets = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in [probe['target_body']]] if 'target_body' in probe \
                else [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in manifest['object_bodies']]
            before, after = (_panda_probe(sim, probe, targets, False)[1] for sim in sims)
            assert after > 0 or before == 0, f'arm probe {probe["name"]} lost target contact'
            report.setdefault('arm_probe_contact_ticks', {})[probe['name']] = [before, after]
    else:
        assert _probes_hit(perturbed, manifest), 'a private probe misses its body'
    report.update(physics_geoms=len(physics['geoms']), physics_joints=len(physics['joints']))


def _verify_pose(o, m, pose, perturbed, manifest, report, with_physics):
    """Recorded rigid jitter applied exactly to the object bodies, everything else unmoved, probes still on their bodies."""
    import mujoco
    import numpy as np
    ids = []
    for name, item in pose['bodies'].items():
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        ids.append(body)
        assert np.allclose(m.body_pos[body], item['pos']['value'], rtol=0, atol=1e-12) \
            and np.allclose(m.body_quat[body], item['quat']['value'], rtol=0, atol=1e-12), f'{name} jitter not applied'
        assert with_physics or (np.allclose(o.body_pos[body], item['pos']['original'], rtol=0, atol=1e-12)
                                and np.allclose(o.body_quat[body], item['quat']['original'], rtol=0, atol=1e-12)), f'{name} original pose misrecorded'
        assert item['pos']['value'][2] == item['pos']['original'][2], f'{name} height changed'
    others = [b for b in range(m.nbody) if b not in ids]
    assert np.array_equal(m.body_pos[others], o.body_pos[others]) and np.array_equal(m.body_quat[others], o.body_quat[others]), 'non-object bodies moved'
    assert _probes_hit(perturbed, manifest), 'a private probe misses its body after pose jitter'
    report.update(pose_offset_m=pose['offset_m'], pose_yaw_deg=pose['yaw_deg'])


def verify(run_dir):
    """Compile original and perturbed sources (Slurm test node) and fail loudly on any mismatch."""
    import mujoco
    import numpy as np
    from .simulation import Simulation
    run_dir = Path(run_dir)
    record = json.loads((run_dir / 'experiment.json').read_text())['perturbation']
    config = json.loads((run_dir / 'config-private.json').read_text())
    evaluation = config['evaluation']
    manifest = tomllib.loads((Path(evaluation['source_dir']) / evaluation.get('manifest', 'manifest.toml')).read_text())
    original_scene = Path(record['original_scene'])
    assert hashlib.sha256(original_scene.read_bytes()).hexdigest() == record['original_scene_sha256'], 'original scene changed'
    simconfig = _simconfig(config, manifest)
    sims = [Simulation(path, simconfig) for path in (original_scene, config['source']['scene'])]
    (o, m), report = (sim._model for sim in sims), {'geoms': len(record['object_geoms'])}
    for name, item in record.get('colors', {}).get('geoms', {}).items():
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert g >= 0 and g == mujoco.mj_name2id(o, mujoco.mjtObj.mjOBJ_GEOM, name), f'{name} missing'
        assert np.allclose(m.geom_rgba[g], item['value'], atol=1e-6) and m.geom_matid[g] == -1, f'{name} color not applied'
        assert np.allclose(o.geom_rgba[g], item['original'], atol=1e-6), f'{name} original color misrecorded'
        assert not np.allclose(m.geom_rgba[g], o.geom_rgba[g]), f'{name} color unchanged'
    others = [g for g in range(m.ngeom) if 'colors' not in record or m.geom(g).name not in record['colors']['geoms']]
    assert m.ngeom == o.ngeom and np.array_equal(m.geom_rgba[others], o.geom_rgba[others]), 'non-object colors changed'
    if 'physics' in record:
        _verify_physics(o, m, record['physics'], sims, manifest, report, record.get('pose'))
    if 'pose' in record:
        _verify_pose(o, m, record['pose'], sims[1], manifest, report, 'physics' in record)
    roots = {m.body(n).id for n in record['object_bodies']}
    for label, sim in zip(('original', 'perturbed'), sims):
        sim.action({'op': 'reset'}, render=False)
        report[f'{label}_initial_penetration_m'] = _penetration(sim._model, sim._data, roots)
        start = sim._data.xpos[sorted(roots)].copy()
        sim.action({'op': 'wait', 'ticks': 500}, render=False)
        assert np.isfinite(sim._data.qpos).all() and np.isfinite(sim._data.qvel).all(), f'{label} settle diverged'
        report[f'{label}_settle_displacement_m'] = float(np.linalg.norm(sim._data.xpos[sorted(roots)] - start, axis=1).max())
        report[f'{label}_settled_qpos'] = sim._data.qpos.copy()
        sim.close()
    settled = [report.pop(f'{k}_settled_qpos') for k in ('original', 'perturbed')]
    if record.keys() & {'physics', 'pose'}:
        assert report['perturbed_initial_penetration_m'] <= PENETRATION_M, 'perturbed initial penetration'
        # Resized supports may tilt a resting body by centimeters; an explosion moves it far more.
        assert report['perturbed_settle_displacement_m'] <= report['original_settle_displacement_m'] + .1, 'perturbed scene unstable'
    else:
        assert np.allclose(*settled, atol=1e-9), 'colors changed dynamics'
    return report
