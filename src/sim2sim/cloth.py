"""Researcher-only cloth metric (selector `cloth`), model-agnostic over cloth resolution and MJCF layout.

The cloth of a model is the union of its 2D flexes; their vertex bodies must each move on three orthogonal slide joints
(flexcomp `dof="full"` or explicit vertex bodies). A candidate without a 2D flex is `unavailable`. The manifest `[cloth]`
table (KEYS) drives three probes, run identically in fresh source and candidate worlds that gain two static fixtures far
from everything else: a horizontal bar (capsule along y) at -FIXTURE_X and a cone at +FIXTURE_X from the source cloth's
reset centre, their tops FIXTURE_UP above it. Placements and probe points come from the source and are applied to both
worlds in the world frame, so a candidate cloth built off-place keeps that error.

- Stiffness map: on a `lift_grid` x `lift_grid` grid over the source cloth's reset bounding box (inset `lift_margin_m`), a
  vertical ray from above grasps the cloth exactly as an agent's point grasp does and lifts it `lift_height_m` at
  `lift_speed_m_s`, recording the grasp's vertical reaction every `sample_ticks`. A stiffer cloth raises a wider tent, so
  the lift force at a height and its slope (N/m) map local bending stiffness (plus the lifted weight). Points where the
  source ray misses its cloth (holes, outside the outline) are gaps; a candidate miss there is counted, never zero.
- Drops: each `[[cloth.drops]]` entry rotates the source reset layout by yaw about z, then tilt about a horizontal axis at
  `tilt_axis_deg`, and releases it at rest with its lowest point DROP_CLEARANCE above the bar or the cone apex; the same
  world-frame motion is applied to the candidate cloth. Every `frame_ticks` over `drop_ticks` both cloths' surfaces are
  sampled (`samples` area-weighted barycentric points, fixed on each mesh) and compared by symmetric mean Chamfer distance.
- `rest_chamfer_m`: the same distance between the two flat cloths at reset.
Artifacts: stiffness_map.{json,png} and drop_NNN/side_by_side.mp4 (source left, candidate right, real time).
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np

KEYS = {'samples', 'seed', 'lift_grid', 'lift_margin_m', 'lift_height_m', 'lift_speed_m_s', 'sample_ticks', 'drop_ticks',
        'frame_ticks', 'drops'}
BAR_RADIUS, BAR_HALF_LENGTH = .02, .5       # m
CONE_RADIUS, CONE_HEIGHT = .08, .16         # m
FIXTURE_X, FIXTURE_UP = 1.5, .6             # fixture offsets from the source cloth reset centre, m
DROP_CLEARANCE, RAY_UP = .05, .3            # m
RAY_JITTER = np.array([3.1e-4, 1.7e-4])     # m: lift rays avoid exact mesh edges, where mj_rayFlex can miss
VIDEO = 240, 320                            # height, width


def _fixtures(centre):
    """scene_edit hook: a bar and a cone, colliding with any cloth collision bits."""
    bar = centre + [-FIXTURE_X, 0, FIXTURE_UP - BAR_RADIUS]
    cone = centre + [FIXTURE_X, 0, FIXTURE_UP - CONE_HEIGHT]
    ring = [f'{CONE_RADIUS * np.cos(a):.6f} {CONE_RADIUS * np.sin(a):.6f} 0' for a in np.linspace(0, 2 * np.pi, 32, endpoint=False)]
    def edit(root):
        asset = root.find('asset')
        if asset is None:
            asset = ET.SubElement(root, 'asset')
        ET.SubElement(asset, 'mesh', name='_cloth_probe_cone', vertex=f'0 0 {CONE_HEIGHT} ' + ' '.join(ring))
        world = root.find('worldbody')
        common = dict(contype='65535', conaffinity='65535', condim='3', rgba='.55 .57 .6 1')
        ET.SubElement(world, 'geom', name='_cloth_probe_bar', type='capsule', size=f'{BAR_RADIUS} {BAR_HALF_LENGTH}',
                      pos=' '.join(map(str, bar)), quat='0.7071068 0.7071068 0 0', **common)
        ET.SubElement(world, 'geom', name='_cloth_probe_cone', type='mesh', mesh='_cloth_probe_cone', pos=' '.join(map(str, cone)), **common)
    return edit, {'bar': centre + [-FIXTURE_X, 0, FIXTURE_UP], 'cone': centre + [FIXTURE_X, 0, FIXTURE_UP]}


class _Cloth:
    """Triangles (global flex vertex ids), vertex bodies and their slide-joint qpos addresses of a model's 2D flexes."""
    def __init__(self, sim):
        import mujoco
        m, d = sim._model, sim._data
        flexes = [f for f in range(m.nflex) if m.flex_dim[f] == 2]
        self.ok = bool(flexes)
        if not self.ok:
            return
        self.tris = np.concatenate([m.flex_elem[m.flex_elemdataadr[f]:m.flex_elemdataadr[f] + 3 * m.flex_elemnum[f]].reshape(-1, 3)
                                    + m.flex_vertadr[f] for f in flexes])
        self.bodies = np.unique(np.concatenate([m.flex_vertbodyid[m.flex_vertadr[f]:m.flex_vertadr[f] + m.flex_vertnum[f]] for f in flexes]))
        self.qpos = []
        for b in self.bodies:
            joints = np.flatnonzero(m.jnt_bodyid == b)
            assert len(joints) == 3 and all(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE for j in joints), \
                'cloth vertex bodies must move on exactly three slide joints'
            axes = d.xaxis[joints]
            assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-6), 'cloth vertex slide axes must be orthonormal'
            self.qpos.append((m.jnt_qposadr[joints], axes))

    def place(self, sim, target):
        """Move every vertex body to its target world position, at rest."""
        import mujoco
        d = sim._data
        delta = target - d.xpos[self.bodies]
        for (adr, axes), step in zip(self.qpos, delta):
            d.qpos[adr] += axes @ step
        d.qvel[:] = 0
        mujoco.mj_forward(sim._model, d)

    def sampler(self, sim, n, seed):
        """Fixed area-weighted barycentric samples on the current mesh: points(data) gives their world positions."""
        x = sim._data.flexvert_xpos[self.tris]
        area = .5 * np.linalg.norm(np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), axis=1)
        rng = np.random.default_rng(seed)
        tri = rng.choice(len(self.tris), n, p=area / area.sum())
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        ids, w = self.tris[tri], np.stack([1 - u - v, u, v], axis=1)
        return lambda data: np.einsum('nk,nkj->nj', w, data.flexvert_xpos[ids])


def _chamfer(a, b):
    from scipy.spatial import cKDTree
    return float((cKDTree(a).query(b)[0].mean() + cKDTree(b).query(a)[0].mean()) / 2)


def _rotation(yaw, tilt, axis):
    from scipy.spatial.transform import Rotation
    horizontal = [np.cos(np.radians(axis)), np.sin(np.radians(axis)), 0]
    return (Rotation.from_rotvec(np.radians(tilt) * np.asarray(horizontal)) * Rotation.from_euler('z', yaw, degrees=True)).as_matrix()


def _spec(spec):
    assert isinstance(spec, dict) and set(spec) == KEYS, f'cloth needs exactly {sorted(KEYS)}'
    assert spec['drops'] and all(d['obstacle'] in ('bar', 'cone') for d in spec['drops']), 'cloth drops need bar or cone obstacles'
    return spec


def _plan(source_path, simconfig, spec):
    """Source-derived probe geometry shared by both worlds: reset centre, lift grid and drop placements."""
    from .simulation import Simulation
    sim = Simulation(source_path, {**simconfig, 'rgb': False, 'depth': False, 'calibration': False, 'recording_callback': None})
    try:
        cloth = _Cloth(sim)
        assert cloth.ok, 'source has no 2D flex cloth'
        pos = sim._data.xpos[cloth.bodies].copy()
    finally:
        sim.close()
    centre = pos.mean(axis=0)
    lo, hi = pos[:, :2].min(axis=0) + spec['lift_margin_m'], pos[:, :2].max(axis=0) - spec['lift_margin_m']
    k = int(spec['lift_grid'])
    grid = np.array([[x, y] for y in np.linspace(lo[1], hi[1], k) for x in np.linspace(lo[0], hi[0], k)]) + RAY_JITTER
    _, tops = _fixtures(centre)
    drops = []
    for drop in spec['drops']:
        rotation = _rotation(drop['yaw_deg'], drop['tilt_deg'], drop['tilt_axis_deg'])
        local = (pos - centre) @ rotation.T
        drops.append((rotation, tops[drop['obstacle']] + [0, 0, DROP_CLEARANCE - local[:, 2].min()]))
    return centre, grid, drops


def measure(scene_path, simconfig, spec, plan):
    """All cloth probes in one world; None when the model has no 2D flex cloth."""
    import mujoco
    from .simulation import Simulation
    centre, grid, drops = plan
    edit, tops = _fixtures(centre)
    sim = Simulation(scene_path, {**simconfig, 'rgb': False, 'depth': False, 'calibration': False, 'wrench': True,
                                  'recording_callback': None, 'scene_edit': edit})
    renderer = None
    try:
        cloth = _Cloth(sim)
        if not cloth.ok:
            return None
        dt = float(sim._model.opt.timestep)
        sim.action({'op': 'reset'}, render=False)
        rest = sim._data.xpos[cloth.bodies].copy()
        points = cloth.sampler(sim, int(spec['samples']), int(spec['seed']))
        result = {'rest': points(sim._data), 'lift': [], 'drops': []}
        ticks, every = round(spec['lift_height_m'] / spec['lift_speed_m_s'] / dt), int(spec['sample_ticks'])
        assert ticks > 0 and ticks % every == 0, 'the lift must last a whole number of sample_ticks'
        top = centre[2] + RAY_UP
        for x, y in grid:
            sim.action({'op': 'reset'}, render=False)
            hit = sim._ray_hit(np.array([x, y, top]), np.array([0., 0., -1.]))
            if hit is None or hit[0] not in cloth.bodies:
                result['lift'].append(None)
                continue
            sim._grasp_controller.attach(*hit)
            force = []
            for _ in range(ticks // every):
                sim._advance(every, [0, 0, spec['lift_height_m'] * every / ticks])
                force.append(-sim._wrench()['force'][2])
            sim._grasp_controller.release()
            result['lift'].append(force)
        renderer = mujoco.Renderer(sim._model, *VIDEO)
        camera = mujoco.MjvCamera()
        camera.type, camera.distance, camera.azimuth, camera.elevation = mujoco.mjtCamera.mjCAMERA_FREE, 1.0, 60, -20
        for (rotation, target), drop in zip(drops, spec['drops']):
            sim.action({'op': 'reset'}, render=False)
            cloth.place(sim, (rest - centre) @ rotation.T + target)
            camera.lookat[:] = tops[drop['obstacle']] - [0, 0, .12]
            frames, clouds = [], []
            for tick in range(0, int(spec['drop_ticks']) + 1, int(spec['frame_ticks'])):
                if tick:
                    sim._advance(int(spec['frame_ticks']))
                    assert np.isfinite(sim._data.qpos).all(), 'nonfinite cloth state'
                clouds.append(points(sim._data))
                renderer.update_scene(sim._data, camera)
                frames.append(renderer.render().copy())
            result['drops'].append({'clouds': clouds, 'frames': frames})
        return result
    finally:
        if renderer is not None:
            renderer.close()
        sim.close()


def _lift_summary(lifts, spec):
    """Final lift force (N) and least-squares force/height slope (N/m) per grid point, None for gaps."""
    final, slope = [], []
    for force in lifts:
        if force is None:
            final.append(None); slope.append(None)
            continue
        height = spec['lift_height_m'] * np.arange(1, len(force) + 1) / len(force)
        final.append(float(force[-1]))
        slope.append(float(np.polyfit(height, force, 1)[0]))
    return final, slope


def _errors(source, candidate):
    pairs = [(a, b) for a, b in zip(source, candidate) if a is not None and b is not None]
    if not pairs:
        return None, None
    e = np.array([b - a for a, b in pairs])
    return float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))


def compare(source, candidate, spec, plan, artifacts, timestep):
    import imageio.v2 as imageio
    centre, grid, _ = plan
    k = int(spec['lift_grid'])
    src_final, src_slope = _lift_summary(source['lift'], spec)
    if candidate is None:
        return {'status': 'unavailable', 'reason': 'candidate has no 2D flex cloth', 'source_lift_force_n': src_final}
    cand_final, cand_slope = _lift_summary([c if s is not None else None for s, c in zip(source['lift'], candidate['lift'])], spec)
    probed = [s is not None for s in source['lift']]
    force_mae, force_rms = _errors(src_final, cand_final)
    slope_mae, slope_rms = _errors(src_slope, cand_slope)
    valid = [abs(f) for f in src_final if f is not None]
    stiffness = {'grid_xy_m': grid.tolist(), 'grid_shape': [k, k], 'lift_height_m': spec['lift_height_m'],
                 'source_lift_force_n': src_final, 'candidate_lift_force_n': cand_final,
                 'source_stiffness_n_m': src_slope, 'candidate_stiffness_n_m': cand_slope,
                 'source_force_curves_n': source['lift'], 'candidate_force_curves_n': candidate['lift'],
                 'probed_points': int(sum(probed)),
                 'candidate_missed_points': int(sum(p and c is None for p, c in zip(probed, candidate['lift']))),
                 'lift_force_mae_n': force_mae, 'lift_force_rms_error_n': force_rms,
                 'lift_force_relative_error': None if force_mae is None else force_mae / float(np.mean(valid)),
                 'stiffness_mae_n_m': slope_mae, 'stiffness_rms_error_n_m': slope_rms}
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / 'stiffness_map.json').write_text(json.dumps(stiffness, allow_nan=False))
    _plot_map(artifacts / 'stiffness_map.png', grid, k, src_final, cand_final)
    drops = []
    for index, (a, b, drop) in enumerate(zip(source['drops'], candidate['drops'], spec['drops'])):
        series = [_chamfer(x, y) for x, y in zip(a['clouds'], b['clouds'])]
        path = artifacts / f'drop_{index:03d}'
        path.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(path / 'side_by_side.mp4', [np.hstack([x, y]) for x, y in zip(a['frames'], b['frames'])],
                         fps=max(1, round(1 / (int(spec['frame_ticks']) * timestep))), macro_block_size=1)
        drops.append({**drop, 'frames': len(series), 'chamfer_series_m': series, 'mean_chamfer_m': float(np.mean(series)),
                      'final_chamfer_m': series[-1], 'max_chamfer_m': float(np.max(series))})
    return {'status': 'ok', 'rest_chamfer_m': _chamfer(source['rest'], candidate['rest']), 'stiffness_map': stiffness, 'drops': drops,
            'drop_mean_chamfer_m': float(np.mean([d['mean_chamfer_m'] for d in drops])),
            'drop_final_chamfer_m': float(np.mean([d['final_chamfer_m'] for d in drops]))}


def _plot_map(path, grid, k, source, candidate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    grids = [np.array([np.nan if v is None else v for v in values]).reshape(k, k) for values in (source, candidate)]
    grids.append(np.abs(grids[1] - grids[0]))
    extent = [grid[:, 0].min(), grid[:, 0].max(), grid[:, 1].min(), grid[:, 1].max()]
    vmax = np.nanmax(grids[:2]) if np.isfinite(grids[:2]).any() else 1
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, values, title in zip(axes, grids, ('source lift force (N)', 'candidate lift force (N)', '|candidate - source| (N)')):
        image = ax.imshow(values, origin='lower', extent=extent, vmin=0, vmax=vmax, cmap='viridis')
        ax.set(title=title, xlabel='x (m)', ylabel='y (m)')
        fig.colorbar(image, ax=ax, shrink=.85)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def evaluate(source_path, candidate_path, source_config, candidate_config, spec, artifacts):
    spec = _spec(spec)
    plan = _plan(source_path, source_config, spec)
    source = measure(source_path, source_config, spec, plan)
    try:
        candidate = measure(candidate_path, candidate_config, spec, plan)
    except Exception as exc:
        return {'status': 'unavailable', 'error': f'{type(exc).__name__}: {exc}'[:1000]}
    return compare(source, candidate, spec, plan, artifacts, float(candidate_config['timestep']))
