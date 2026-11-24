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

from .robots import ROBOTS

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
    # Every XML file is checked as MJCF, and so is the entrypoint unless it is a predictor program (src/sim2sim/predictor.py)
    # or a low-rank parameter answer (JSON, metric `lowrank`).
    pending = [scene] * (scene.suffix not in ('.py', '.json')) + [p for p in destination.rglob('*') if p.is_file() and p.suffix.lower() == '.xml']
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
    return {'rgb_error': rgb, 'rgb_mse': _rgb_mse(_pixels(a, 'rgb'), _pixels(b, 'rgb')), 'depth_error': depth, 'silhouette_error': silhouette}


def _rgb_mse(source_rgb, candidate_rgb):
    """Full-image mean squared error in normalized RGB, without masking."""
    import numpy as np
    delta = (np.asarray(source_rgb, dtype=float) - np.asarray(candidate_rgb, dtype=float)) / 255.0
    return float(np.mean(delta * delta))


def _reset_visual_scores(source, candidate, selection, artifacts):
    if not selection & {'appearance', 'image_mse'}:
        return {}
    source.action({'op': 'reset'}, render=False)
    candidate.action({'op': 'reset'}, render=False)
    _trusted_camera(source, candidate)
    if 'appearance' in selection:
        a, b = _capture(source), _capture(candidate)
        _save_frames(artifacts / 'reset' / 'source', [a])
        _save_frames(artifacts / 'reset' / 'candidate', [b])
        return {'reset': _visual(a, b)}
    import mujoco
    frames = []
    for sim in (source, candidate):
        if sim._renderer is None:
            sim._renderer = mujoco.Renderer(sim._model, height=sim.height, width=sim.width)
        sim._renderer.disable_depth_rendering()
        sim._renderer.disable_segmentation_rendering()
        sim._renderer.update_scene(sim._data, camera=sim.camera)
        frames.append(sim._renderer.render().copy())
    return {'reset': {'rgb_mse': _rgb_mse(*frames)}}


def _parameters(source, candidate, correspondence, hinge_correspondence=None):
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
        if hinge_correspondence is not None:
            pairs=[]
            for s_joint,c_joint in hinge_correspondence.items():
                s_id=mujoco.mj_name2id(sm,mujoco.mjtObj.mjOBJ_JOINT,s_joint)
                c_id=mujoco.mj_name2id(cm,mujoco.mjtObj.mjOBJ_JOINT,c_joint)
                if s_id<0 or c_id<0 or sm.jnt_type[s_id]!=mujoco.mjtJoint.mjJNT_HINGE or cm.jnt_type[c_id]!=mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError('missing or invalid corresponding named hinge')
                if sm.jnt_bodyid[s_id]==si:
                    pairs.append((s_id,c_id))
            sj=np.asarray([pair[0] for pair in pairs],dtype=int)
            cj=np.asarray([pair[1] for pair in pairs],dtype=int)
        for name, sa, ca in [('hinge_stiffness', sm.jnt_stiffness[sj], cm.jnt_stiffness[cj]),
                             ('hinge_damping', sm.dof_damping[sm.jnt_dofadr[sj]], cm.dof_damping[cm.jnt_dofadr[cj]]),
                             ('hinge_frictionloss', sm.dof_frictionloss[sm.jnt_dofadr[sj]], cm.dof_frictionloss[cm.jnt_dofadr[cj]]),
                             ('hinge_range', sm.jnt_range[sj], cm.jnt_range[cj])]:
            errors[sname][name] = relative(sa, ca)
    return errors


LOWRANK_KEYS = {'mass', 'stiffness', 'com', 'inertia'}


def _lowrank(source, answer_path, table):
    """Errors of a parameter answer {mass (kg), stiffness (N/m), com (world m at the initial state), inertia (3x3 world-axes
    tensor about the COM at the initial state, kg m^2)} against manifest [lowrank] body and spring joint of the fresh source."""
    import mujoco
    import numpy as np
    answer = json.loads(answer_path.read_text())
    assert isinstance(answer, dict) and set(answer) == LOWRANK_KEYS, f'answer keys must be exactly {sorted(LOWRANK_KEYS)}'
    mass, stiffness = (float(answer[k]) for k in ('mass', 'stiffness'))
    com, inertia = np.asarray(answer['com'], dtype=float), np.asarray(answer['inertia'], dtype=float)
    assert com.shape == (3,) and inertia.shape == (3, 3) and np.isfinite([mass, stiffness, *com, *inertia.ravel()]).all(), 'malformed answer'
    model = source._model
    assert model.nkey == 0, 'the initial state of a lowrank scene is qpos0 (no keyframes)'
    data = mujoco.MjData(model)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, table['body'])
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, table['joint'])
    assert body >= 0 and joint >= 0, 'manifest [lowrank] names a missing body or joint'
    mujoco.mj_forward(model, data)
    rotation = data.ximat[body].reshape(3, 3)
    truth = {'mass': float(model.body_mass[body]), 'stiffness': float(model.jnt_stiffness[joint]), 'com': data.xipos[body].tolist(),
             'inertia': (rotation @ np.diag(model.body_inertia[body]) @ rotation.T).tolist()}
    true_inertia = np.asarray(truth['inertia'])
    return {'answer': {'mass': mass, 'stiffness': stiffness, 'com': com.tolist(), 'inertia': inertia.tolist()}, 'truth': truth,
            'mass_relative_error': abs(mass - truth['mass']) / truth['mass'],
            'stiffness_relative_error': abs(stiffness - truth['stiffness']) / truth['stiffness'],
            'com_error_m': float(np.linalg.norm(com - np.asarray(truth['com']))),
            'inertia_relative_error': float(np.linalg.norm(inertia - true_inertia) / np.linalg.norm(true_inertia))}


def _hinges(source, candidate, correspondence, matching=None, *, object_only=False):
    import mujoco
    import numpy as np
    sm,cm=source._model,candidate._model
    if correspondence is None and matching=='single':
        source_ids=np.flatnonzero(sm.jnt_type==mujoco.mjtJoint.mjJNT_HINGE)
        candidate_ids=np.flatnonzero(cm.jnt_type==mujoco.mjtJoint.mjJNT_HINGE)
        if object_only:
            from .primitive_metrics import _robot_bodies
            source_robot=_robot_bodies(source);candidate_robot=_robot_bodies(candidate)
            source_ids=np.asarray([i for i in source_ids if int(sm.jnt_bodyid[i]) not in source_robot])
            candidate_ids=np.asarray([i for i in candidate_ids if int(cm.jnt_bodyid[i]) not in candidate_robot])
        if len(source_ids)!=1 or len(candidate_ids)!=1:
            key=sm.joint(int(source_ids[0])).name if len(source_ids)==1 else 'single_hinge'
            return {key or 'single_hinge':{'status':'unavailable','matching':'single',
                'source_hinge_count':len(source_ids),'candidate_hinge_count':len(candidate_ids),
                'error':'single-hinge matching requires exactly one hinge in each model'}}
        si,ci=int(source_ids[0]),int(candidate_ids[0])
        pairs=[(sm.joint(si).name or f'joint_{si}',cm.joint(ci).name or f'joint_{ci}',si,ci)]
    else:
        pairs=[(sname,cname,mujoco.mj_name2id(sm,mujoco.mjtObj.mjOBJ_JOINT,sname),
            mujoco.mj_name2id(cm,mujoco.mjtObj.mjOBJ_JOINT,cname)) for sname,cname in (correspondence or {}).items()]
    result={}
    for sname,cname,si,ci in pairs:
        record={'status':'unavailable','source_joint':sname,'candidate_joint':cname,
            'matching':'declared' if correspondence is not None else matching}
        result[sname]=record
        if si<0 or ci<0 or sm.jnt_type[si]!=mujoco.mjtJoint.mjJNT_HINGE or cm.jnt_type[ci]!=mujoco.mjtJoint.mjJNT_HINGE:
            record['error']='missing or non-hinge named joint'
            continue
        for parameter,a,b in [('damping',sm.dof_damping[sm.jnt_dofadr[si]],cm.dof_damping[cm.jnt_dofadr[ci]]),
                ('frictionloss',sm.dof_frictionloss[sm.jnt_dofadr[si]],cm.dof_frictionloss[cm.jnt_dofadr[ci]]),
                ('stiffness',sm.jnt_stiffness[si],cm.jnt_stiffness[ci])]:
            a,b=float(a),float(b)
            record.update({f'source_{parameter}':a,f'predicted_{parameter}':b,
                f'{parameter}_absolute_error':abs(b-a),f'{parameter}_relative_error':abs(b-a)/abs(a) if a else None})
        record['status']='ok'
    return result


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
        trajectory.append(_centroid(sim, frame))
    return frames, np.asarray(trajectory), grasp_success


def _centroid(sim, frame):
    """Normalized mask centroid and mean masked depth (m), or the far background when nothing is visible."""
    import numpy as np
    ys, xs = np.nonzero(frame['_mask'])
    depth = _pixels(frame, 'depth')
    return [float(xs.mean()) / sim.width, float(ys.mean()) / sim.height, float(depth[ys, xs].mean()) / 1000] if len(xs) else [0., 0., 65.535]


def _probe_pixel(source, probe):
    """A probe's grasp pixel, or its source body and local point projected through the source camera at the current state."""
    import mujoco
    import numpy as np
    if 'pixel' in probe:
        return probe['pixel']
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
    return [(point[0]/(-point[2]*scale)*source.height/source.width+1)*source.width/2-.5,
            (1-point[1]/(-point[2]*scale))*source.height/2-.5]


def _film(sim, actions, every):
    """Run actions from reset, capturing a frame at reset and every `every` ticks; missed-grasp moves advance as waits.
    Body grasps resolve to the pixel of that body point at that moment; the resolved actions are returned for replay."""
    import numpy as np
    sim.action({'op': 'reset'}, render=False)
    frames, grasped, resolved, old = [_capture(sim)], [], [], sim._callback
    sim._callback = lambda current: frames.append(_capture(current)) if current.ticks % every == 0 else None
    try:
        for action in actions:
            action = {'op': 'grasp', 'pixel': _probe_pixel(sim, action)} if action['op'] == 'grasp' else dict(action)
            resolved.append(action)
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
    return frames, all(grasped) if grasped else None, resolved


def _expert(source, candidate, trajectories, artifacts):
    """Scripted expert manipulations replayed from reset in both worlds, compared frame by frame; side-by-side video."""
    import imageio.v2 as imageio
    import numpy as np
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError('expert metric requires private expert_trajectories')
    results = []
    for index, trajectory in enumerate(trajectories):
        every = int(trajectory.get('frame_ticks', 10))
        if not 1 <= every <= 1000 or not trajectory.get('actions'):
            raise ValueError('expert trajectory needs actions and frame_ticks within 1..1000')
        af, source_grasp, actions = _film(source, trajectory['actions'], every)
        bf, candidate_grasp, _ = _film(candidate, actions, every)
        if source_grasp is False:
            raise ValueError(f'invalid expert trajectory: source grasp missed in {trajectory.get("name", index)}')
        if len(af) != len(bf):
            raise ValueError('expert trajectory frame count mismatch')
        visuals = [_visual(a, b) for a, b in zip(af, bf)]
        at, bt = np.asarray([_centroid(source, f) for f in af]), np.asarray([_centroid(candidate, f) for f in bf])
        path = artifacts / f'expert_{index:03d}'
        path.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(path / 'side_by_side.mp4', [np.hstack([_pixels(a, 'rgb'), _pixels(b, 'rgb')]).astype(np.uint8) for a, b in zip(af, bf)],
                         fps=max(1, round(1 / (every * float(source._model.opt.timestep)))), macro_block_size=1)
        (path / 'trajectory.json').write_text(json.dumps({'actions': actions, 'frame_ticks': every, 'source': at.tolist(), 'candidate': bt.tolist()}))
        results.append({'name': trajectory.get('name', f'expert_{index}'), 'frames': len(af), 'frame_ticks': every,
                        'source_grasp_success': source_grasp, 'candidate_grasp_success': candidate_grasp,
                        'image_trajectory_error': float(np.mean(np.linalg.norm(at - bt, axis=1))),
                        'max_rgb_mse': max(v['rgb_mse'] for v in visuals),
                        **{key: float(np.mean([v[key] for v in visuals])) for key in visuals[0]}})
    return results


def _transfer_outcome(sim, pixel, displacement, task):
    """Grasp, fling by displacement over move_ticks, release and settle; the object's compiled COM afterwards."""
    import numpy as np
    from .primitive_metrics import _com
    sim.action({'op': 'reset'}, render=False)
    sim.action({'op': 'grasp', 'pixel': pixel}, render=False)
    move = {'op': 'move', 'displacement': list(displacement)} if sim._grasp is not None else {'op': 'wait'}
    sim.action({**move, 'ticks': int(task.get('move_ticks', 40))}, render=False)
    sim.action({'op': 'release', 'ticks': int(task.get('settle_ticks', 400))}, render=False)
    if not np.isfinite(sim._data.qpos).all():
        raise ValueError('nonfinite simulation state')
    return _com(sim)[0]


def _cem(sim, pixel, task, goal, seed):
    """Cross-entropy search over the fling displacement in one world: best (cost, displacement), from the naive goal offset."""
    import numpy as np
    rng, bound = np.random.default_rng(seed), float(task.get('max_displacement_m', .3))
    mean, std, best = np.clip(np.asarray(task['goal_offset_m'], float), -bound, bound), np.full(3, bound / 2), (np.inf, None)
    for iteration in range(int(task.get('iterations', 6))):
        samples = np.clip(rng.normal(mean, std, (16, 3)), -bound, bound)
        if iteration == 0:
            samples[0] = mean
        costs = np.array([np.linalg.norm(_transfer_outcome(sim, pixel, s, task) - goal) for s in samples])
        order = np.argsort(costs)
        if costs[order[0]] < best[0]:
            best = (float(costs[order[0]]), samples[order[0]].copy())
        mean, std = samples[order[:4]].mean(axis=0), samples[order[:4]].std(axis=0) + 1e-3
    return best


def _policy_transfer(source, candidate, tasks):
    """Train an open-loop fling policy in the candidate and execute it in the source, against a source-trained oracle."""
    import numpy as np
    from .primitive_metrics import _com
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('policy_transfer metric requires private transfer_tasks')
    results = []
    for index, task in enumerate(tasks):
        source.action({'op': 'reset'}, render=False)
        pixel, start = _probe_pixel(source, task), _com(source)[0]
        goal = start + np.asarray(task['goal_offset_m'], float)
        naive = np.linalg.norm(_transfer_outcome(source, pixel, task['goal_offset_m'], task) - goal)
        predicted, policy = _cem(candidate, pixel, task, goal, index)
        oracle, oracle_policy = _cem(source, pixel, task, goal, index)
        transfer = float(np.linalg.norm(_transfer_outcome(source, pixel, policy, task) - goal))
        candidate.action({'op': 'reset'}, render=False)
        candidate.action({'op': 'grasp', 'pixel': pixel}, render=False)
        results.append({'name': task.get('name', f'transfer_{index}'), 'goal_m': goal.tolist(), 'candidate_grasp_success': candidate._grasp is not None,
                        'candidate_policy_displacement_m': policy.tolist(), 'oracle_policy_displacement_m': oracle_policy.tolist(),
                        'candidate_predicted_error_m': predicted, 'transfer_error_m': transfer, 'oracle_error_m': oracle,
                        'naive_error_m': float(naive), 'transfer_regret_m': transfer - oracle})
    return results


def _probe_actions(source, probe):
    import numpy as np
    source.action({'op': 'reset'})
    pixel = _probe_pixel(source, probe)
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


def _verify_panda_fixture(source_root, candidate_root, robot='panda'):
    """The public robot (Panda or Shadow hand) is a fixed experimental instrument, not a target."""
    expected=Path(source_root)/'robot'
    supplied=Path(candidate_root)/'robot'
    if not expected.is_dir() or not supplied.is_dir():
        raise ValueError(f'{robot} submission requires the supplied public robot fixture')
    # Physics files must match; attribution docs are excluded because the public IMPORT.md is sanitized.
    docs={'IMPORT.md','LICENSE'}
    source_files={p.relative_to(expected).as_posix():hashlib.sha256(p.read_bytes()).digest()
                  for p in expected.rglob('*') if p.is_file() and p.relative_to(expected).as_posix() not in docs}
    candidate_files={p.relative_to(supplied).as_posix():hashlib.sha256(p.read_bytes()).digest()
                     for p in supplied.rglob('*') if p.is_file() and p.relative_to(supplied).as_posix() not in docs}
    if not source_files or source_files!=candidate_files:
        raise ValueError(f'{robot} robot fixture must be copied unchanged')
    root=ET.parse(Path(candidate_root)/'scene.xml').getroot()
    includes=[node.get('file') for node in root.iter('include')]
    if includes.count('robot/'+ROBOTS[robot][0])!=1:
        raise ValueError(f'{robot} candidate scene must include robot/{ROBOTS[robot][0]} exactly once')


def _panda_contact_ticks(sim, target_bodies):
    from .primitive_metrics import _robot_bodies
    robot=_robot_bodies(sim)
    model,data=sim._model,sim._data
    target=set(target_bodies)
    for i in range(data.ncon):
        contact=data.contact[i]
        a=int(model.geom_bodyid[contact.geom1]);b=int(model.geom_bodyid[contact.geom2])
        if (a in robot and b in target) or (b in robot and a in target):
            return 1
    return 0


COM_SAMPLE_TICKS = 10  # composite object COM sampling interval along an arm probe (20 ms at the 2 ms step)


def _free_object_bodies(sim):
    """Model-agnostic moving object: every non-robot body not welded to the world."""
    from .primitive_metrics import _robot_bodies
    robot=_robot_bodies(sim)
    return [b for b in range(1,sim._model.nbody) if b not in robot and int(sim._model.body_weldid[b])!=0]


def _panda_probe(sim, probe, target_bodies, with_rgb):
    """Robot probe (each step holds the robot action fields: Panda arm_target or Shadow hand_target). Observations after each step,
    robot-target contact ticks, and the composite COM of the free object bodies at ticks
    0, COM_SAMPLE_TICKS, ... (MuJoCo's xipos from the step's own forward pass is the state at the start of that step);
    the COM samples are None without positive free mass."""
    import numpy as np
    sim.action({'op':'reset'},render=False)
    observations=[];contacts=[0];com=[]
    free=_free_object_bodies(sim)
    mass=sim._model.body_mass[free]
    def record(current):
        contacts[0]+=_panda_contact_ticks(current,target_bodies)
        if (current.ticks-1)%COM_SAMPLE_TICKS==0:
            com.append((mass@current._data.xipos[free]/mass.sum()).tolist() if mass.sum()>0 else None)
    old_callback=sim._callback
    sim._callback=record
    try:
        for step in probe['steps']:
            (op,)=ROBOTS[sim.robot][2]  # probes of a single-action robot
            sim.action({**step,'op':op},render=False)
            if not np.isfinite(sim._data.qpos).all() or not np.isfinite(sim._data.qvel).all():
                raise ValueError('nonfinite robot probe state')
            observations.append(sim.observe() if with_rgb else sim.observe())
    finally:
        sim._callback=old_callback
    return observations,contacts[0],{'bodies':[sim._model.body(b).name for b in free],'mass_kg':float(mass.sum()),'com_m':com}


def _com_comparison(source, candidate):
    """Composite object COM path error and total free-mass difference; unavailable COM stays None, never zero."""
    import numpy as np
    assert len(source['com_m'])==len(candidate['com_m']), 'COM sample count mismatch'
    record={'sample_ticks':COM_SAMPLE_TICKS,'source_mass_kg':source['mass_kg'],'candidate_mass_kg':candidate['mass_kg'],
            'mass_absolute_error_kg':abs(source['mass_kg']-candidate['mass_kg'])}
    if source['mass_kg']<=0 or candidate['mass_kg']<=0:
        return {**record,'status':'unavailable','reason':'no free object mass in '+('source' if source['mass_kg']<=0 else 'candidate'),
                'trajectory_mean_error_m':None,'trajectory_final_error_m':None,'trajectory_max_error_m':None}
    error=np.linalg.norm(np.asarray(source['com_m'])-np.asarray(candidate['com_m']),axis=1)
    return {**record,'status':'ok','trajectory_mean_error_m':float(error.mean()),
            'trajectory_final_error_m':float(error[-1]),'trajectory_max_error_m':float(error.max())}


def _panda_dynamics(source,candidate,config,artifacts):
    import mujoco
    import numpy as np
    probes=config.get('arm_probes')
    if not isinstance(probes,list) or not probes:
        raise ValueError('Robot dynamics requires private arm_probes')
    names=config.get('object_bodies')
    if not isinstance(names,list) or not names:
        raise ValueError('Robot dynamics requires target object bodies')
    source_ids=[]
    for name in names:
        body=mujoco.mj_name2id(source._model,mujoco.mjtObj.mjOBJ_BODY,name)
        if body<0: raise ValueError('unknown robot target body')
        source_ids.append(body)
    from .primitive_metrics import _robot_bodies
    candidate_robot=_robot_bodies(candidate)
    candidate_ids=[i for i in range(1,candidate._model.nbody) if i not in candidate_robot]
    results=[]
    with_rgb=bool(config.get('source_config',{}).get('rgb',True))
    for index,probe in enumerate(probes):
        named_target=probe.get('target_body')
        if named_target is None:
            probe_source_ids=source_ids
        else:
            body=mujoco.mj_name2id(source._model,mujoco.mjtObj.mjOBJ_BODY,named_target)
            if body not in source_ids:
                raise ValueError('Robot probe target must be a declared object body')
            probe_source_ids=[body]
        a,source_contacts,source_com=_panda_probe(source,probe,probe_source_ids,with_rgb)
        b,candidate_contacts,candidate_com=_panda_probe(candidate,probe,candidate_ids,with_rgb)
        if len(a)!=len(b): raise ValueError('Robot probe trajectory length mismatch')
        keys={'joint_positions_rad':'joint_position_mae_rad',
              'joint_velocities_rad_s':'joint_velocity_mae_rad_s',
              'actuator_effort_nm':'actuator_effort_mae_nm',
              'finger_positions_m':'finger_position_mae_m',
              'finger_velocities_m_s':'finger_velocity_mae_m_s',
              'gripper_width_m':'gripper_width_mae_m'}
        record={'name':probe.get('name',f'arm_probe_{index}'),'status':'ok',
                'source_target_contact_ticks':source_contacts,
                'candidate_target_contact_ticks':candidate_contacts,
                'target_contact_tick_difference':abs(source_contacts-candidate_contacts),
                'object_com':_com_comparison(source_com,candidate_com)}
        for field,label in keys.items():
            if field in a[0]['proprioception']:
                record[label]=float(np.mean([np.mean(np.abs(np.asarray(x['proprioception'][field])-np.asarray(y['proprioception'][field])))
                                             for x,y in zip(a,b)]))
        if 'tactile' in a[0]:
            # Every taxel and body total of every pad, as one flat vector per step.
            def taxels(o): return np.concatenate([np.ravel(pad[key]) for pad in o['tactile'].values() if isinstance(pad,dict) for key in sorted(pad)])
            record['tactile_mae_n']=float(np.mean([np.mean(np.abs(taxels(x)-taxels(y))) for x,y in zip(a,b)]))
        if with_rgb:
            record['rgb_error']=float(np.mean([np.mean(np.abs(_pixels(x,'rgb')-_pixels(y,'rgb')))/255 for x,y in zip(a,b)]))
        path=artifacts/f'arm_probe_{index:03d}'
        path.mkdir(parents=True,exist_ok=True)
        (path/'trajectory.json').write_text(json.dumps({'steps':probe['steps'],
            'source_proprioception':[x['proprioception'] for x in a],
            'candidate_proprioception':[x['proprioception'] for x in b],
            **({'source_tactile':[x['tactile'] for x in a],'candidate_tactile':[x['tactile'] for x in b]} if 'tactile' in a[0] else {}),
            'source_target_contact_ticks':source_contacts,
            'candidate_target_contact_ticks':candidate_contacts,
            'com_sample_ticks':COM_SAMPLE_TICKS,'source_object_com':source_com,'candidate_object_com':candidate_com},allow_nan=False))
        from .plotting import plot_com_trajectories
        plot_com_trajectories(path/'com.png',record['name'],COM_SAMPLE_TICKS*float(source._model.opt.timestep),
                              source_com['com_m'],candidate_com['com_m'])
        results.append(record)
    return results


def _evaluate_panda(source,candidate,config,selection,artifacts,source_path,candidate_path,project):
    """Researcher-only object and arm metrics for a shared public robot."""
    scores = _reset_visual_scores(source, candidate, selection, artifacts)
    if selection & {'objects','identification'}:
        from . import tabletop
        if 'objects' in selection:
            scores['objects']=tabletop.objects(source,candidate,config)
        if 'identification' in selection:
            scores['identification']=tabletop.identification(source,project,config)
    if 'dynamics' in selection:
        scores['arm_probes']=_panda_dynamics(source,candidate,config,artifacts)
    if 'expert' in selection:
        scores['expert']=_expert(source,candidate,config.get('expert_trajectories'),artifacts)
    if 'policy_transfer' in selection:
        raise ValueError('policy_transfer supports point-grasp scenes only')
    if 'parameters' in selection:
        correspondence=config.get('correspondence')
        if correspondence:
            try:
                scores['parameters']={'status':'ok','bodies':_parameters(source,candidate,correspondence,config.get('hinge_correspondence'))}
            except ValueError as exc:
                scores['parameters']={'status':'failed','error':str(exc)}
        else:
            scores['parameters']={'status':'unavailable','reason':'no declared correspondence'}
    if 'hinges' in selection:
        scores['hinges']=_hinges(source,candidate,config.get('hinge_correspondence'),
                                 config.get('hinge_matching'),object_only=True)
    components=selection & {'chamfer','center_of_mass','density','friction'}
    if 'reconstruction' in selection or components:
        from .primitive_metrics import reconstruction_metrics
        reconstruction=dict(config.get('reconstruction') or {})
        reconstruction.update(panda_objects=True,
            source_object_geoms=config.get('object_geoms'),
            source_object_bodies=config.get('object_bodies'))
        if 'reconstruction' not in selection:
            reconstruction['components']=sorted(components)
        try:
            scores['reconstruction']=reconstruction_metrics(source,candidate,source_path,candidate_path,reconstruction)
        except Exception as exc:
            scores['reconstruction']={'status':'unavailable','error':f'{type(exc).__name__}: {exc}',
                'chamfer_distance_m':None,'center_of_mass_error_m':None,
                'density':{'status':'unavailable','per_primitive':[]}}
    hook=config.get('metrics_hook')
    if hook:
        import importlib
        module,name=hook.split(':',1)
        function=getattr(importlib.import_module(module),name)
        scores['custom']=function(source,candidate,{'scores':scores,'artifact_dir':str(artifacts),'config':config})
    return scores


SUPPORTED_METRICS = frozenset({'appearance', 'dynamics', 'parameters', 'reconstruction', 'hinges',
                              'chamfer', 'center_of_mass', 'density', 'friction', 'image_mse',
                               'expert', 'policy_transfer', 'force_profile', 'cloth', 'fixture', 'die', 'lowrank',
                               'objects', 'identification'})  # die: host-side, src/sim2sim/die.py; objects, identification: tabletop.py


def selected_metrics(config):
    value = config.get('metrics')
    if value is None:
        return {'appearance', 'dynamics', 'parameters', 'reconstruction', 'hinges'}
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError('evaluation.metrics must be an array of supported metric names')
    unknown = set(value) - SUPPORTED_METRICS
    if unknown:
        raise ValueError('unsupported evaluation metrics: ' + ', '.join(sorted(unknown)))
    return set(value)


def evaluate_submission(zip_bytes: bytes, config: dict) -> dict:
    """Archive an immutable version and return private metrics or explicit failure."""
    version = hashlib.sha256(zip_bytes).hexdigest()
    report = {'status': 'failed', 'version': version, 'scores': None}
    source = candidate = None
    try:
        config = dict(config)
        selection = selected_metrics(config)
        explicit_selection = 'metrics' in config
        import numpy as np
        from .simulation import Simulation
        manifest = Path(config['source_dir']) / config.get('manifest', 'manifest.toml')
        if manifest.is_file():
            import tomllib
            metadata = tomllib.loads(manifest.read_text())
            for key in ('probes', 'correspondence', 'camera', 'graspable_bodies', 'reconstruction', 'hinge_correspondence', 'hinge_matching',
                        'object_bodies', 'object_geoms', 'arm_probes', 'expert_trajectories', 'transfer_tasks', 'force_profile', 'cloth', 'fixture',
                        'lowrank', 'tabletop'):
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
        if 'fixture' in selection:
            # A predictor submission (src/sim2sim/predictor.py ran it in isolation; config['predictions'] holds its answers):
            # scored against fresh source releases only, with no candidate scene.
            assert selection == {'fixture'}, 'the fixture metric scores a predictor program and combines with no scene metric'
            from .fixture import evaluate as fixture
            scores = {'fixture': fixture(source, config['fixture'], config['predictions'], output / 'artifacts' / version / 'fixture')}
            json.dumps(scores, allow_nan=False)
            report.update(status='ok', scores=scores)
            return report
        if 'lowrank' in selection:
            # A low-rank answer (parameters JSON, no scene) scored against the source's named body and spring joint.
            assert selection == {'lowrank'}, 'the lowrank metric scores a parameter answer and combines with no scene metric'
            scores = {'lowrank': _lowrank(source, target / 'project' / config['entrypoint'], config['lowrank'])}
            json.dumps(scores, allow_nan=False)
            report.update(status='ok', scores=scores)
            return report
        import mujoco
        camera_quat = np.zeros(4)
        mujoco.mju_mat2Quat(camera_quat, source._data.cam_xmat[source._camera_id])
        candidate_config = dict(simconfig, timestep=float(source._model.opt.timestep), trusted_camera={'pos': source._data.cam_xpos[source._camera_id].tolist(),
            'quat': camera_quat.tolist(), 'fovy': float(source._model.cam_fovy[source._camera_id])})
        candidate_config.pop('graspable_bodies', None)
        candidate = Simulation(target / 'project' / config.get('entrypoint', 'scene.xml'), candidate_config)
        _trusted_camera(source, candidate)
        correspondence = config.get('correspondence')
        artifacts = output / 'artifacts' / version
        source_path = Path(config['source_dir']) / config.get('source_entrypoint', config.get('entrypoint', 'scene.xml'))
        candidate_path = target / 'project' / config.get('entrypoint', 'scene.xml')
        def force_profile():
            from .force_profile import evaluate
            assert config.get('force_profile'), 'force_profile metric requires a private [force_profile] probe'
            return evaluate(source_path, candidate_path, simconfig, candidate_config, config['force_profile'])
        if simconfig.get('robot'):
            _verify_panda_fixture(config['source_dir'],target/'project',simconfig['robot'])
            scores=_evaluate_panda(source,candidate,config,selection,artifacts,source_path,candidate_path,target/'project')
            if 'force_profile' in selection:
                scores['force_profile']=force_profile()
            json.dumps(scores,allow_nan=False)
            report.update(status='ok',scores=scores)
            return report
        scores = _reset_visual_scores(source, candidate, selection, artifacts)
        if 'parameters' in selection:
            if correspondence:
                try:
                    params = {'status': 'ok', 'bodies': _parameters(source, candidate, correspondence, config.get('hinge_correspondence'))}
                except ValueError as exc:
                    params = {'status': 'failed', 'error': str(exc)}
            else:
                params = {'status': 'unavailable', 'reason': 'no declared correspondence'}
            scores['parameters'] = params
        probes = (config.get('probes') or [{'body': name} for name in source._body_names]) if 'dynamics' in selection else []
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
        if 'dynamics' in selection:
            scores.update(probes=results, probe_grasp_failure_rate=sum(not p['candidate_grasp_success'] for p in results) / max(1, len(results)))
        if 'hinges' in selection:
            scores['hinges'] = _hinges(source, candidate, config.get('hinge_correspondence'), config.get('hinge_matching'))
        if 'expert' in selection:
            scores['expert'] = _expert(source, candidate, config.get('expert_trajectories'), artifacts)
        if 'policy_transfer' in selection:
            scores['policy_transfer'] = _policy_transfer(source, candidate, config.get('transfer_tasks'))
        reconstruction = config.get('reconstruction')
        components = selection & {'chamfer', 'center_of_mass', 'density', 'friction'}
        if explicit_selection and ('reconstruction' in selection or components):
            reconstruction = {**(reconstruction if isinstance(reconstruction, dict) else {}), 'enabled': True}
            if 'reconstruction' not in selection:
                reconstruction['components'] = sorted(components)
        if ('reconstruction' in selection or components) and reconstruction and (not isinstance(reconstruction, dict) or reconstruction.get('enabled', True)):
            from .primitive_metrics import reconstruction_metrics
            try:
                scores['reconstruction'] = reconstruction_metrics(source, candidate,
                    Path(config['source_dir']) / config.get('source_entrypoint', config.get('entrypoint', 'scene.xml')),
                    target / 'project' / config.get('entrypoint', 'scene.xml'),
                    reconstruction if isinstance(reconstruction, dict) else {})
            except Exception as exc:
                scores['reconstruction'] = {'status': 'unavailable', 'error': f'{type(exc).__name__}: {exc}',
                    'chamfer_distance_m': None, 'center_of_mass_error_m': None,
                    'density': {'status': 'unavailable', 'per_primitive': []}}
        if 'force_profile' in selection:
            scores['force_profile'] = force_profile()
        if 'cloth' in selection:
            from .cloth import evaluate as cloth
            assert config.get('cloth'), 'cloth metric requires a private [cloth] probe table'
            scores['cloth'] = cloth(source_path, candidate_path, simconfig, candidate_config, config['cloth'], artifacts / 'cloth')
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
