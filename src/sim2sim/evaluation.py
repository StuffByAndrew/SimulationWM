"""Private evaluator. Call only inside the resource-limited evaluation container."""
from __future__ import annotations
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile
import xml.etree.ElementTree as ET

MAX_ZIP = 8 * 1024 * 1024
MAX_EXPANDED = 32 * 1024 * 1024
MAX_FILES = 256


def decode_submission(value: str) -> bytes:
    if not isinstance(value, str) or len(value) > (MAX_ZIP * 4 // 3 + 8):
        raise ValueError('submission exceeds base64 size limit')
    try:
        data = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError('invalid base64 submission') from exc
    if len(data) > MAX_ZIP:
        raise ValueError('submission exceeds ZIP size limit')
    return data


def _relative(value):
    if not value or '\\' in value or ':' in value or '\x00' in value:
        raise ValueError('unsafe project path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or value.startswith('/'):
        raise ValueError('unsafe project path')
    return path


def extract_submission(data: bytes, destination: Path, entrypoint='scene.xml'):
    if len(data) > MAX_ZIP:
        raise ValueError('submission exceeds ZIP size limit')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_FILES or sum(i.file_size for i in infos) > MAX_EXPANDED:
            raise ValueError('expanded submission exceeds limits')
        seen = set()
        for info in infos:
            path = _relative(info.filename)
            if str(path) in seen:
                raise ValueError('duplicate ZIP path')
            seen.add(str(path))
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError('special files and symlinks are forbidden')
            target = destination.joinpath(*path.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
    scene = destination.joinpath(*_relative(entrypoint).parts)
    if not scene.is_file():
        raise ValueError('missing scene entrypoint')
    pending = [scene] + [p for p in destination.rglob('*') if p.is_file() and p.suffix.lower() == '.xml']
    checked = set()
    while pending:
        xml = pending.pop()
        if xml in checked:
            continue
        checked.add(xml)
        raw = xml.read_bytes()
        if b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
            raise ValueError('XML entities are forbidden')
        root = ET.fromstring(raw)
        for element in root.iter():
            if element.tag in ('plugin', 'extension'):
                raise ValueError('MJCF plugins are forbidden')
            for key in ('file', 'meshdir', 'texturedir', 'assetdir'):
                value = element.get(key)
                if value:
                    rel = _relative(value)
                    resolved = (xml.parent / str(rel)).resolve()
                    if not resolved.is_relative_to(destination.resolve()):
                        raise ValueError('external MJCF reference')
                    if element.tag == 'include' and key == 'file':
                        pending.append(resolved)
    return scene


def _pixels(observation, key):
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(observation[key])))).astype(float)


def _capture(sim):
    import numpy as np
    obs = sim.observe(full=True)
    sim._renderer.enable_segmentation_rendering()
    sim._renderer.update_scene(sim._data, camera=sim.camera)
    ids = sim._renderer.render().copy()
    sim._renderer.disable_segmentation_rendering()
    import mujoco
    valid = (ids[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids[:, :, 0] >= 0)
    mask = np.zeros(valid.shape, dtype=bool)
    mask[valid] = sim._model.geom_bodyid[ids[:, :, 0][valid]] != 0
    obs['_mask'] = mask
    return obs


def _visual(a, b):
    import numpy as np
    rgb = float(np.mean(np.abs(_pixels(a, 'rgb') - _pixels(b, 'rgb'))) / 255)
    da, db = _pixels(a, 'depth'), _pixels(b, 'depth')
    depth = float(np.mean(np.minimum(np.abs(da - db) / 1000, 1)))
    # Finite rendered geometry against the clipped far background.
    ma, mb = a['_mask'], b['_mask']
    union = np.logical_or(ma, mb).sum()
    silhouette = float(np.logical_xor(ma, mb).sum() / max(1, union))
    return {'rgb_error': rgb, 'depth_error': depth, 'silhouette_error': silhouette}


def _parameters(source, candidate, correspondence):
    import mujoco
    import numpy as np
    errors = {}
    for sname, cname in correspondence.items():
        si = mujoco.mj_name2id(source._model, mujoco.mjtObj.mjOBJ_BODY, sname)
        ci = mujoco.mj_name2id(candidate._model, mujoco.mjtObj.mjOBJ_BODY, cname)
        if si < 0 or ci < 0:
            raise ValueError('missing corresponding body')
        sm, cm = source._model, candidate._model
        def relative(a, b):
            a, b = np.asarray(a), np.asarray(b)
            if a.shape != b.shape:
                return 1.0
            return float(np.mean(np.minimum(np.abs(a-b) / np.maximum(np.abs(a), .01), 1))) if a.size else 0.0
        errors[sname] = {'mass': relative(sm.body_mass[si], cm.body_mass[ci]),
                        'friction': relative(sm.geom_friction[sm.geom_bodyid == si], cm.geom_friction[cm.geom_bodyid == ci])}
        sf, cf = sm.geom_friction[sm.geom_bodyid == si, 0], cm.geom_friction[cm.geom_bodyid == ci, 0]
        errors[sname]['sliding_mu_absolute_error'] = float(np.mean(np.abs(sf-cf))) if sf.shape == cf.shape and sf.size else 1.0
        sj = np.flatnonzero((sm.jnt_bodyid == si) & (sm.jnt_type == mujoco.mjtJoint.mjJNT_HINGE))
        cj = np.flatnonzero((cm.jnt_bodyid == ci) & (cm.jnt_type == mujoco.mjtJoint.mjJNT_HINGE))
        for name, sa, ca in [('hinge_stiffness', sm.jnt_stiffness[sj], cm.jnt_stiffness[cj]),
                             ('hinge_damping', sm.dof_damping[sm.jnt_dofadr[sj]], cm.dof_damping[cm.jnt_dofadr[cj]]),
                             ('hinge_range', sm.jnt_range[sj], cm.jnt_range[cj])]:
            errors[sname][name] = relative(sa, ca)
    return errors


def _probe(sim, actions):
    import numpy as np
    frames, trajectory = [], []
    grasp_success = False
    for action in actions:
        if action['op'] == 'move' and sim._grasp is None:
            sim.action({'op': 'wait', 'ticks': action['ticks']})
        else:
            sim.action(action)
        if not np.isfinite(sim._data.qpos).all() or not np.isfinite(sim._data.qvel).all():
            raise ValueError('nonfinite simulation state')
        if action['op'] == 'grasp':
            grasp_success = sim._grasp is not None
        frame = _capture(sim)
        frames.append(frame)
        ys, xs = np.nonzero(frame['_mask'])
        depth = _pixels(frame, 'depth')
        trajectory.append([float(xs.mean()) / sim.width, float(ys.mean()) / sim.height,
                           float(depth[ys, xs].mean()) / 1000] if len(xs) else [0., 0., 65.535])
    return frames, np.asarray(trajectory), grasp_success


def _probe_actions(source, probe):
    import mujoco
    import numpy as np
    source.action({'op': 'reset'})
    if 'pixel' in probe:
        pixel = probe['pixel']
    else:
        bid = mujoco.mj_name2id(source._model, mujoco.mjtObj.mjOBJ_BODY, probe['body'])
        if bid < 0:
            raise ValueError('unknown private probe body')
        local = np.asarray(probe.get('local_point', [0, 0, 0]), dtype=float)
        world = source._data.xpos[bid] + source._data.xmat[bid].reshape(3, 3) @ local
        camera = source._camera_id
        point = source._data.cam_xmat[camera].reshape(3, 3).T @ (world-source._data.cam_xpos[camera])
        if point[2] >= 0:
            raise ValueError('probe behind camera')
        scale = np.tan(np.deg2rad(source._model.cam_fovy[camera]) / 2)
        pixel = [(point[0]/(-point[2]*scale)*source.height/source.width+1)*source.width/2-.5,
                 (1-point[1]/(-point[2]*scale))*source.height/2-.5]
    ticks = int(probe.get('ticks', 120))
    if not 4 <= ticks <= 100000:
        raise ValueError('probe ticks outside limits')
    delta = (np.asarray(probe.get('displacement', [0, 0, .1]), dtype=float)/4).tolist()
    return [{'op': 'reset'}, {'op': 'grasp', 'pixel': pixel}] + [
        {'op': 'move', 'ticks': ticks//4, 'displacement': delta} for _ in range(4)] + [
        {'op': 'release', 'ticks': ticks//4}] + [{'op': 'wait', 'ticks': ticks//4} for _ in range(3)]


def _trusted_camera(source, candidate):
    import mujoco
    sc, cc = source._camera_id, candidate._camera_id
    candidate._model.cam_bodyid[cc] = 0
    candidate._model.cam_pos[cc] = source._data.cam_xpos[sc]
    mujoco.mju_mat2Quat(candidate._model.cam_quat[cc], source._data.cam_xmat[sc])
    for key in ('cam_fovy', 'cam_projection', 'cam_intrinsic', 'cam_sensorsize', 'cam_resolution'):
        if hasattr(source._model, key):
            getattr(candidate._model, key)[cc] = getattr(source._model, key)[sc]
    mujoco.mj_forward(candidate._model, candidate._data)


def _save_frames(path, frames):
    from PIL import Image
    path.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        for key in ('rgb', 'depth'):
            (path / f'{i:03d}_{key}.png').write_bytes(base64.b64decode(frame[key]))
        Image.fromarray(frame['_mask'].astype('uint8')*255).save(path / f'{i:03d}_mask.png')


def evaluate_submission(zip_bytes: bytes, config: dict) -> dict:
    """Archive an immutable version and return private metrics or explicit failure."""
    version = hashlib.sha256(zip_bytes).hexdigest()
    report = {'status': 'failed', 'version': version, 'scores': None}
    source = candidate = None
    try:
        import numpy as np
        from .simulation import Simulation
        config = dict(config)
        manifest = Path(config['source_dir']) / config.get('manifest', 'manifest.toml')
        if manifest.is_file():
            import tomllib
            metadata = tomllib.loads(manifest.read_text())
            for key in ('probes', 'correspondence', 'camera', 'graspable_bodies'):
                if key not in config and key in metadata:
                    config[key] = metadata[key]
        output = Path(config['output_dir'])
        versions = output / 'versions'
        versions.mkdir(parents=True, exist_ok=True)
        target = versions / version
        with tempfile.TemporaryDirectory(dir=versions) as temp:
            staging = Path(temp)
            extract_submission(zip_bytes, staging / 'project', config.get('entrypoint', 'scene.xml'))
            (staging / 'submission.zip').write_bytes(zip_bytes)
            if not target.exists():
                staging.rename(target)
        report['project_path'] = f'versions/{version}/project'
        simconfig = dict(config.get('source_config', {}))
        for key in ('camera', 'width', 'height', 'timestep', 'grasp_solref', 'grasp_solimp', 'default_ticks', 'graspable_bodies'):
            if key in config:
                simconfig[key] = config[key]
        source = Simulation(Path(config['source_dir']) / config.get('source_entrypoint', config.get('entrypoint', 'scene.xml')), simconfig)
        import mujoco
        camera_quat = np.zeros(4)
        mujoco.mju_mat2Quat(camera_quat, source._data.cam_xmat[source._camera_id])
        candidate_config = dict(simconfig, timestep=float(source._model.opt.timestep), trusted_camera={'pos': source._data.cam_xpos[source._camera_id].tolist(),
            'quat': camera_quat.tolist(), 'fovy': float(source._model.cam_fovy[source._camera_id])})
        candidate_config.pop('graspable_bodies', None)
        candidate = Simulation(target / 'project' / config.get('entrypoint', 'scene.xml'), candidate_config)
        _trusted_camera(source, candidate)
        correspondence = config.get('correspondence')
        source_reset, candidate_reset = _capture(source), _capture(candidate)
        reset = _visual(source_reset, candidate_reset)
        artifacts = output / 'artifacts' / version
        _save_frames(artifacts / 'reset' / 'source', [source_reset])
        _save_frames(artifacts / 'reset' / 'candidate', [candidate_reset])
        if correspondence:
            try:
                params = {'status': 'ok', 'bodies': _parameters(source, candidate, correspondence)}
            except ValueError as exc:
                params = {'status': 'failed', 'error': str(exc)}
        else:
            params = {'status': 'unavailable', 'reason': 'no declared correspondence'}
        probes = config.get('probes') or [{'body': name} for name in source._body_names]
        results = []
        for probe in probes:
            body = probe.get('body', 'pixel')
            actions = _probe_actions(source, probe)
            af, at, source_grasp = _probe(source, actions)
            if not source_grasp:
                raise ValueError(f'invalid reference probe: grasp missed for {body}')
            bf, bt, candidate_grasp = _probe(candidate, actions)
            probe_path = artifacts / f'probe_{len(results):03d}'
            _save_frames(probe_path / 'source', af)
            _save_frames(probe_path / 'candidate', bf)
            (probe_path / 'trajectory.json').write_text(json.dumps({'actions': actions, 'source': at.tolist(), 'candidate': bt.tolist(),
                'source_grasp_success': source_grasp, 'candidate_grasp_success': candidate_grasp,
                'missed_grasp_move_policy': 'advance equivalent ticks without applied displacement'}))
            visuals = [_visual(a, b) for a, b in zip(af, bf)]
            results.append({'name': probe.get('name', body), 'source_grasp_success': source_grasp,
                            'candidate_grasp_success': candidate_grasp, 'status': 'ok' if candidate_grasp else 'missed_grasp', 'image_trajectory_error': float(np.mean(np.linalg.norm(at-bt, axis=1))),
                            **{key: float(np.mean([v[key] for v in visuals])) for key in visuals[0]}})
        scores = {'reset': reset, 'probes': results, 'parameters': params,
                  'probe_grasp_failure_rate': sum(not p['candidate_grasp_success'] for p in results) / max(1, len(results))}
        hook = config.get('metrics_hook')
        if hook:
            import importlib
            module, name = hook.split(':', 1)
            function = getattr(importlib.import_module(module), name)
            scores['custom'] = function(source, candidate, {'scores': scores, 'artifact_dir': str(artifacts), 'config': config})
        json.dumps(scores, allow_nan=False)
        report.update(status='ok', scores=scores)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'[:1000]
    finally:
        for sim in (source, candidate):
            if sim is not None:
                try:
                    sim.close()
                except Exception as exc:
                    report.update(status='failed', scores=None, error=f'cleanup failed: {type(exc).__name__}: {exc}'[:1000])
    return report
