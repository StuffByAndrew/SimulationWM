"""Researcher-only force-displacement profile of a pushed mechanism (metric `force_profile`).

The manifest's `[force_profile]` table describes one model-agnostic probe, run identically on the source and on a
candidate: the trusted-camera ray through `grasp_pixel` (or the nearest pixel within `grasp_radius_px` that hits one)
picks the first movable body (the stick handle), exactly as an
agent's own pixel grasp does, so a candidate built in a shifted world frame that still matches the image is grasped on
its handle; an ideal point grasp attaches there at time 0 (the scene the image shows, before a candidate that is not at
rest can drift off the pixel) and holds still for `settle_ticks`, and the grasp target is driven along the polyline `path` (handle displacements from the
grasp point, starting at [0, 0, 0]) through successive `strokes` = [[arc_length_m, speed_m_s], ...]. Every
`sample_ticks` the grasp reaction is recorded; `resistance_n` is its component against the local path tangent, so
pushing into a spring reads positive. Loading and unloading strokes stay separate samples, so hysteresis (a latch)
and rate dependence (a damper, via a fast stroke) remain visible. A candidate whose handle the ray misses is a gap.
"""
from __future__ import annotations

import numpy as np

KEYS = {'grasp_pixel', 'grasp_radius_px', 'path', 'strokes', 'sample_ticks', 'settle_ticks'}


def _spec(spec):
    assert isinstance(spec, dict) and set(spec) == KEYS, f'force_profile needs exactly {sorted(KEYS)}'
    path = np.asarray(spec['path'], dtype=float)
    assert path.ndim == 2 and path.shape[1] == 3 and len(path) >= 2 and not path[0].any(), 'path starts at [0, 0, 0]'
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert lengths.min() > 0, 'path segments must have positive length'
    arc = np.concatenate([[0.], np.cumsum(lengths)])
    strokes = [(float(s), float(v)) for s, v in spec['strokes']]
    assert strokes and all(0 <= s <= arc[-1] + 1e-12 and v > 0 for s, v in strokes), 'strokes stay on the path at positive speed'
    return path, arc, strokes, int(spec['sample_ticks']), int(spec['settle_ticks'])


def _at(path, arc, s):
    """Point and unit tangent at arc length s of the polyline."""
    i = min(int(np.searchsorted(arc, s, side='right')) - 1, len(arc) - 2)
    tangent = (path[i + 1] - path[i]) / (arc[i + 1] - arc[i])
    return path[i] + (s - arc[i]) * tangent, tangent


def measure(scene_path, simconfig, spec):
    """Run the probe on one scene; simconfig is the evaluation simulation configuration of that scene."""
    from .simulation import Simulation
    path, arc, strokes, every, settle = _spec(spec)
    config = {**simconfig, 'rgb': False, 'depth': False, 'calibration': False, 'wrench': True, 'recording_callback': None}
    if config.get('robot') == 'panda':
        config['panda_point_grasps'] = True
    sim = Simulation(scene_path, config)
    try:
        # The movable body at grasp_pixel, else at the nearest pixel within grasp_radius_px: a candidate's handle may sit a
        # pixel or two off the reference's (the reference itself is hit exactly; tunnel-check asserts that).
        u, v, r = *spec['grasp_pixel'], int(spec['grasp_radius_px'])
        rings = sorted(((du, dv) for du in range(-r, r + 1) for dv in range(-r, r + 1) if du * du + dv * dv <= r * r), key=lambda d: d[0] ** 2 + d[1] ** 2)
        hit = next((h for du, dv in rings if (h := sim._raycast([u + du, v + dv])) is not None), None)
        if hit is None:
            return {'status': 'missed_grasp'}
        sim._grasp_controller.attach(*hit)
        if settle:
            sim._advance(settle)
        dt, s, rows = float(sim._model.opt.timestep), 0., []
        for index, (target, speed) in enumerate(strokes):
            ticks = max(1, round(abs(target - s) / speed / dt))
            done = 0
            while done < ticks:
                n = min(every, ticks - done)
                nxt = s + (target - s) * n / (ticks - done)
                delta = _at(path, arc, nxt)[0] - _at(path, arc, s)[0]
                sim._advance(n, delta)
                done, s = done + n, nxt
                force = np.asarray(sim._wrench()['force'])
                rows.append((index, s, float(-force @ _at(path, arc, s)[1]), *force.tolist()))
        rows = np.asarray(rows)
        return {'status': 'ok', 'stroke': rows[:, 0].astype(int).tolist(), 'arc_length_m': rows[:, 1].tolist(),
                'resistance_n': rows[:, 2].tolist(), 'force_world_n': rows[:, 3:].tolist()}
    finally:
        sim.close()


def compare(source, candidate):
    """Sample-by-sample resistance error; the probe schedule is identical, so samples align by index."""
    assert source['status'] == 'ok', 'reference force probe must grasp the handle'
    result = {'source': source, 'candidate': candidate}
    if candidate['status'] != 'ok':
        return {'status': candidate['status'], 'rms_resistance_error_n': None, 'max_resistance_error_n': None, **result}
    a, b = np.asarray(source['resistance_n']), np.asarray(candidate['resistance_n'])
    assert a.shape == b.shape, 'identical probe schedules must give aligned samples'
    error = b - a
    # No median: most samples lie in the free travel before the mechanism, where any candidate reads ~0, so it hides a missing mechanism.
    return {'status': 'ok', 'rms_resistance_error_n': float(np.sqrt(np.mean(error ** 2))),
            'max_resistance_error_n': float(np.abs(error).max()),
            'source_rms_resistance_n': float(np.sqrt(np.mean(a ** 2))), **result}


def evaluate(source_path, candidate_path, source_config, candidate_config, spec):
    return compare(measure(source_path, source_config, spec), measure(candidate_path, candidate_config, spec))
