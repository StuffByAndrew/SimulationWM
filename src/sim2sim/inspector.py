"""Researcher-only interactive physics inspector for recorded world-model versions.

Runs only on a Slurm ``test`` allocation. Two Viser viewers show the private
source scene and one recorded reconstruction version. Versions are recovered
from the verified audit; native MJCF parsing and physics run in resource-limited
development workers. Viewer state is private and never reaches agents.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import re
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .audit import AuditFailure, read_audit
from .candidate_history import recover_changes, source_entrypoint
from .runtime import RuntimeFailure

MAX_WORKER_LINE = 160 * 1024 * 1024
MAX_REQUEST = 64 * 1024
SCORE_FIELDS = (('reconstruction', 'chamfer_distance_m'), ('reconstruction', 'center_of_mass_error_m'), ('reset', 'rgb_mse'))
PLANE, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH = 0, 2, 3, 4, 5, 6, 7
WORLDS = ('reference', 'candidate')
ENVIRONMENT = 'environment'  # the Environments view's own world: never coupled or camera-synced with the run worlds


def _lathe(profile, segments=32):
    """Revolve bottom-to-top (radius, z) profile points about +z."""
    profile = np.asarray(profile, dtype=float)
    angle = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    vertices = np.stack([np.outer(profile[:, 0], np.cos(angle)), np.outer(profile[:, 0], np.sin(angle)),
                         np.repeat(profile[:, 1:2], segments, axis=1)], axis=-1).reshape(-1, 3)
    ring, column = np.meshgrid(np.arange(len(profile) - 1), np.arange(segments), indexing='ij')
    a = ring * segments + column
    b = ring * segments + (column + 1) % segments
    faces = np.concatenate([np.stack([a, b, b + segments], -1).reshape(-1, 3),
                            np.stack([a, b + segments, a + segments], -1).reshape(-1, 3)])
    return vertices, faces


def _arc(radius, start, stop, offset=0.0, count=9):
    phi = np.linspace(start, stop, count)
    return [(radius * math.sin(p), radius * math.cos(p) + offset) for p in phi]


def primitive_mesh(kind, size, plane_extent=5.0):
    """Body-frame triangle mesh for a MuJoCo primitive, or None if unsupported."""
    size = [float(x) for x in size]
    if kind == PLANE:
        sx, sy = (value if value > 0 else plane_extent for value in size[:2])
        return np.array([[-sx, -sy, 0], [sx, -sy, 0], [sx, sy, 0], [-sx, sy, 0]]), np.array([[0, 1, 2], [0, 2, 3]])
    if kind == BOX:
        vertices, faces = [], []
        for axis in range(3):
            u, v = [k for k in range(3) if k != axis]
            for sign in (-1, 1):
                start = len(vertices)
                for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    corner = [0.0, 0.0, 0.0]
                    corner[axis], corner[u], corner[v] = sign * size[axis], a * size[u], b * size[v]
                    vertices.append(corner)
                faces += [[start, start + 1, start + 2], [start, start + 2, start + 3]]
        vertices, faces = np.array(vertices), np.array(faces)
    elif kind in (SPHERE, ELLIPSOID):
        vertices, faces = _lathe(_arc(1.0, np.pi, 0, count=17))
        vertices = vertices * (size[0] if kind == SPHERE else np.array(size[:3]))
    elif kind == CAPSULE:
        r, h = size[0], size[1]
        vertices, faces = _lathe(_arc(r, np.pi, np.pi / 2, -h) + _arc(r, np.pi / 2, 0, h))
    elif kind == CYLINDER:
        r, h = size[0], size[1]
        vertices, faces = _lathe([(0, -h), (r, -h), (r, -h), (r, h), (r, h), (0, h)])
    else:
        return None
    # Convex, origin-centred primitives: orient every triangle outward.
    triangles = vertices[faces]
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    inward = np.einsum('ij,ij->i', normal, triangles.mean(axis=1)) < 0
    faces[inward] = faces[inward][:, ::-1]
    return vertices, faces


def _decode(value, dtype, width):
    data = np.frombuffer(base64.b64decode(value, validate=True), dtype=dtype)
    if data.size % width:
        raise ValueError('worker mesh array has an invalid shape')
    return data.reshape(-1, width)


def _metadata(path, limit=8 * 1024 * 1024):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > limit:
        return {}
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _scores(report):
    scores = (report or {}).get('scores') or {}
    result = {}
    for group, field in SCORE_FIELDS:
        value = (scores.get(group) or {}).get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            result[f'{group}.{field}'] = float(value)
    return result


def recorded_versions(run):
    """Verified source bundle and successfully loaded/submitted candidate archives."""
    run = Path(run).resolve()
    if not (run / 'audit/manifest.json').is_file():
        raise AuditFailure('run has no verified audit, so no recorded world-model versions')
    config, duration = {}, [0.0]

    def timed(records):
        for pointer, event in records:
            wall = event.get('wall_elapsed')
            if isinstance(wall, (int, float)) and not isinstance(wall, bool) and math.isfinite(wall):
                duration[0] = max(duration[0], float(wall))
            yield pointer, event
    changes, unavailable, digest = recover_changes(timed(read_audit(run, snapshot=True)), config)
    snapshot = run / 'source-snapshot.zip'
    if snapshot.is_symlink() or not snapshot.is_file():
        raise AuditFailure('recorded source snapshot is unavailable')
    source = snapshot.read_bytes()
    if not digest or hashlib.sha256(source).hexdigest() != digest:
        raise AuditFailure('recorded source snapshot hash mismatch')
    sidecar = _metadata(run / 'candidate-change-history.json')
    scores = {(row.get('change_index'), row.get('version')): _scores(row.get('report'))
              for row in sidecar.get('checkpoints', []) if isinstance(row, dict) and sidecar.get('schema_version') == 1}
    versions = []
    for row in changes:
        wall = row.get('wall_elapsed')
        if not isinstance(wall, (int, float)) or not math.isfinite(wall):
            unavailable.append(dict(action_id=row.get('action_id'), reason='load_without_wall_timestamp'))
            continue
        versions.append(dict(change_index=row['change_index'], kind=row['checkpoint_kind'], wall_elapsed=float(wall),
                             version=row.get('version'), entrypoint=row.get('entrypoint', 'scene.xml'),
                             archive=row.get('archive'), error=row.get('recovery_error'),
                             scores=scores.get((row['change_index'], row.get('version')), {})))
    def simulation(values):
        return {key: values[key] for key in ('robot', 'graspable_bodies', 'timestep') if key in values}
    return dict(source=source, source_entrypoint=source_entrypoint(config),
                source_simulation=simulation(config.get('resolved_source_config', config.get('simulation', {}))),
                candidate_simulation=simulation(config.get('simulation', {})),
                camera=config.get('evaluation', {}).get('camera', 'fixed'), versions=versions,
                unavailable=unavailable, duration_wall=max([duration[0], *(v['wall_elapsed'] for v in versions)]))


def version_at(versions, wall):
    """Index of the latest recorded version at or before ``wall``, or None."""
    index = None
    for position, version in enumerate(versions):
        if version['wall_elapsed'] <= wall:
            index = position
    return index


def _job_end():
    """Unix end time of this Slurm job, or None when Slurm cannot report it."""
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        return None
    try:
        from datetime import datetime
        text = subprocess.run(['squeue', '-h', '-j', job, '-o', '%e'], capture_output=True, text=True, timeout=10).stdout.strip()
        return datetime.fromisoformat(text).timestamp()  # Slurm reports local time
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class _Session:
    """Supervisor contract for development workers: whole-session CPU limit."""
    def __init__(self, seconds):
        self.deadline = time.monotonic() + seconds

    def remaining(self):
        return self.deadline - time.monotonic()

    def check(self):
        if self.remaining() <= 0:
            raise RuntimeFailure('inspector session expired')


class WorldProcess:
    """One limited native physics worker; messages arrive on a reader thread."""
    def __init__(self, root, setup, on_message, seconds):
        from .devrunner import native_process
        self.closed, self.lock, self.stderr = False, threading.Lock(), bytearray()
        self.workspace = Path(root) / uuid.uuid4().hex
        config = {'development_worker': {'render_backend': 'disable', 'pin_cpu': False}}
        self.proc = native_process('inspect', self.workspace, config, _Session(seconds))
        threading.Thread(target=self._read, args=(on_message,), daemon=True).start()
        threading.Thread(target=self._drain, daemon=True).start()
        self.send(setup)

    def _drain(self):
        for chunk in iter(lambda: self.proc.stderr.read(4096), b''):
            self.stderr = (self.stderr + chunk)[-8192:]

    def _read(self, on_message):
        try:
            for line in iter(lambda: self.proc.stdout.readline(MAX_WORKER_LINE + 1), b''):
                if len(line) > MAX_WORKER_LINE:
                    raise ValueError('worker message exceeds byte limit')
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError('worker message must be an object')
                on_message(message)
        except (OSError, ValueError) as exc:
            if not self.closed:
                on_message({'type': 'error', 'error': 'invalid worker output: ' + str(exc)})
            self.close()
            return
        if not self.closed:
            self.proc.wait()
            tail = self.stderr.decode(errors='replace').strip().splitlines()[-3:]
            on_message({'type': 'error', 'error': f'physics worker exited ({self.proc.returncode})' + (': ' + ' | '.join(tail) if tail else '')})

    def send(self, message):
        with self.lock:
            if self.closed:
                return
            try:
                self.proc.stdin.write(json.dumps(message, separators=(',', ':')).encode() + b'\n')
                self.proc.stdin.flush()
            except OSError:
                pass

    def close(self):
        from .devrunner import stop_native
        with self.lock:
            if self.closed:
                return
            self.closed = True
        stop_native(self.proc)
        shutil.rmtree(self.workspace, ignore_errors=True)


def _rgb(rgba):
    return tuple(int(round(255 * min(max(float(x), 0.0), 1.0))) for x in rgba[:3])


def _add_model(scene, model, root, *, opacity=None, keep=None, on_click=None):
    """Add a worker model description under ``root``; returns ({body: frame}, nodes, unsupported geom count)."""
    extent = float(model.get('extent') or 1.0)
    meshes = {key: (_decode(value['vertices'], '<f4', 3), _decode(value['faces'], '<i4', 3))
              for key, value in model['meshes'].items()}
    frames, nodes, skipped = {}, [], 0
    for index, geom in enumerate(model['geoms']):
        body = int(geom['body'])
        if not 0 <= body < len(model['bodies']):
            raise ValueError('geom references an unknown body')
        if keep is not None and not keep(geom):
            continue
        if body not in frames:
            frames[body] = scene.add_frame(f'{root}/b{body}', show_axes=False)
            nodes.append(frames[body])
        if int(geom['type']) == MESH:
            vertices, faces = meshes[str(geom['mesh'])]
            if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
                raise ValueError('mesh face index out of range')
            mesh = (vertices, faces)
        else:
            mesh = primitive_mesh(int(geom['type']), geom['size'], plane_extent=max(3 * extent, 5.0))
        if mesh is None or not np.isfinite(mesh[0]).all():
            skipped += 1
            continue
        alpha = float(geom['rgba'][3]) * (opacity or 1.0)
        handle = scene.add_mesh_simple(
            f'{root}/b{body}/g{index}', mesh[0].astype(np.float32), mesh[1].astype(np.uint32),
            color=_rgb(geom['rgba']), opacity=None if alpha >= 1 else alpha,
            flat_shading=int(geom['type']) in (BOX, PLANE, MESH),
            side='double' if int(geom['type']) == PLANE else 'front',
            position=tuple(geom['pos']), wxyz=tuple(geom['quat']))
        nodes.append(handle)
        if body and on_click is not None:
            handle.on_click(on_click)
    return frames, nodes, skipped


def _pose(frames, model, state):
    """Move body frames to a worker state; False when the state is malformed or non-finite."""
    bodies = len(model['bodies'])
    xpos, xquat = np.asarray(state['xpos'], dtype=float), np.asarray(state['xquat'], dtype=float)
    if xpos.shape != (bodies, 3) or xquat.shape != (bodies, 4) or not (np.isfinite(xpos).all() and np.isfinite(xquat).all()):
        return False
    for body, frame in frames.items():
        if body:
            frame.position, frame.wxyz = xpos[body], xquat[body]
    return xpos


def _object_geom(model):
    """Overlay filter: no floors, and no robot links (which are not graspable) in Panda scenes."""
    graspable = [body['graspable'] for body in model['bodies']]
    return lambda geom: int(geom['type']) != PLANE and (not model.get('robot') or int(geom['body']) == 0 or graspable[int(geom['body'])])


def _server(port, label):
    import viser
    server = viser.ViserServer(host='127.0.0.1', port=port, label=label, verbose=False)
    if port and server.get_port() != port:
        server.stop()
        raise OSError(f'viewer port {port} is in use')
    server.scene.set_up_direction('+z')
    server.gui.configure_theme(dark_mode=True, control_width='small', show_share_button=False)
    return server


class Viewer:
    """One Viser server showing one world, or both worlds overlaid with the reconstruction translucent.

    Scene nodes follow worker state messages. The app owns grasp anchors and places this viewer's gizmo.
    """
    def __init__(self, app, name, label, port, worlds):
        self.app, self.name, self.label, self.primary = app, name, label, worlds[0]
        self.single = len(worlds) == 1
        self.lock = threading.RLock()
        self.server = _server(port, label)
        self.port = self.server.get_port()
        gui = self.server.gui
        self.status = gui.add_markdown(f'**{label}**  \nWaiting for a recorded version.')
        target = self.primary if self.single else 'both'
        for text, op in (('Play', 'play'), ('Pause', 'pause'), ('Step 0.1 s', 'step'), ('Reset', 'reset'), ('Release grasp', 'release')):
            gui.add_button(text).on_click(lambda _, op=op: self.app.control(op, target))
        self.speed = gui.add_slider('Speed', min=0.05, max=2.0, step=0.05, initial_value=1.0)
        self.speed.on_update(lambda _: self.app.control('speed', target, self.speed.value))
        gui.add_markdown('Click a movable body to grasp that point; drag the gizmo to move the handle. Physics runs '
                         'while playing; grasping starts playback. With coupled grasping on, pick the part in each world '
                         'and one drag moves both handles by the same displacement.' if self.single else
                         'Reference opaque, reconstruction objects translucent. Click a part of either world to pick it '
                         'for the coupled grasp; drag the gizmo to move both handles together.')
        self.server.on_client_connect(self._connected)
        self.layers = {world: dict(frames={}, nodes=[], markers=None, model=None, positions=None) for world in worlds}
        self.gizmo = self.gizmo_origin = self.description = None
        self.summary = dict(status='waiting', message='Waiting for a recorded version.')
        self.last_status = 0.0

    @property
    def model(self):
        return self.layers[self.primary]['model']

    @property
    def positions(self):
        return self.layers[self.primary]['positions']

    def _connected(self, client):
        client.camera.on_update(lambda camera: self.app.camera_moved(self, camera))

    def clear(self, world, status, message, description=None):
        with self.lock:
            layer = self.layers[world]
            with self.server.atomic():
                for node in reversed(layer['nodes']):
                    node.remove()
            layer.update(frames={}, nodes=[], markers=None, model=None, positions=None)
            if self.single:
                self.description = description
                self.summary = dict(status=status, message=message)
                self.status.content = f'**{self.label}**  \n' + (f'{description}  \n' if description else '') + message

    def show(self, world, model):
        scene, layer = self.server.scene, self.layers[world]
        extent = float(model.get('extent') or 1.0)
        translucent = not self.single and world == 'candidate'
        with self.lock, self.server.atomic():
            frames, nodes, skipped = _add_model(scene, model, f'/{world}', opacity=.4 if translucent else None,
                                                keep=_object_geom(model) if translucent else None,
                                                on_click=lambda event: self._clicked(world, event))
            radius = max(.006, .012 * extent)
            markers = (scene.add_icosphere(f'/{world}/grasp/anchor', radius=radius, color=(105, 226, 255), visible=False),
                       scene.add_icosphere(f'/{world}/grasp/handle', radius=radius * .8, color=(255, 212, 107), visible=False))
            layer.update(frames=frames, nodes=nodes + list(markers), markers=markers, model=model)
            if self.single:
                self.summary = dict(status='ready', message='Loaded' + (f'; {skipped} unsupported geoms hidden' if skipped else ''))

    def set_camera(self, camera, center, extent, move_clients):
        if camera:
            position = np.asarray(camera['pos'])
            forward = np.asarray(camera['forward'])
            look = position + forward * max(.3, float(np.dot(np.asarray(center) - position, forward)))
            up, fov = camera['up'], math.radians(camera['fovy'])
        else:
            look = np.asarray(center)
            position, up, fov = look + extent * np.array([1.2, -1.6, 1.0]), (0, 0, 1), math.radians(50)
        initial = self.server.initial_camera
        initial.position, initial.look_at, initial.up, initial.fov = position, look, up, fov
        if move_clients:
            for client in self.server.get_clients().values():
                with client.atomic():
                    client.camera.position, client.camera.look_at = position, look
                    client.camera.up_direction, client.camera.fov = up, fov

    def mirror_camera(self, camera):
        for client in self.server.get_clients().values():
            with client.atomic():
                client.camera.position, client.camera.look_at = camera.position, camera.look_at
                client.camera.up_direction = camera.up_direction

    def _clicked(self, world, event):
        self.app.control('grasp', world, {'origin': list(event.ray_origin), 'direction': list(event.ray_direction)})

    def place_gizmo(self, origin):
        """Show the handle gizmo at ``origin``; None removes it. An unchanged origin keeps a dragged gizmo."""
        with self.lock:
            if origin is not None and self.gizmo is not None and np.allclose(origin, self.gizmo_origin):
                return
            if self.gizmo is not None:
                self.gizmo.remove()
            self.gizmo = self.gizmo_origin = None
            if origin is None:
                return
            extent = float((self.model or {}).get('extent') or 1.0)
            gizmo = self.server.scene.add_transform_controls('/grasp/target', scale=max(.08, .25 * extent),
                                                             disable_rotations=True, position=tuple(origin))
            gizmo.on_update(lambda _: self.app.drag(self, np.asarray(gizmo.position, dtype=float)))
            self.gizmo, self.gizmo_origin = gizmo, np.asarray(origin, dtype=float)

    def move_gizmo(self, offset):
        with self.lock:
            if self.gizmo is not None:
                self.gizmo.position = tuple(self.gizmo_origin + offset)

    def notice(self, text):
        self.summary['notice'] = str(text)
        for client in self.server.get_clients().values():
            client.add_notification('Inspector', str(text), auto_close_seconds=4.0)

    def update(self, world, state):
        with self.lock:
            layer = self.layers[world]
            if layer['model'] is None:
                return
            grasp = state.get('grasp')
            with self.server.atomic():
                xpos = _pose(layer['frames'], layer['model'], state)
                if xpos is False:
                    self.summary['message'] = 'Worker state is non-finite or malformed; display held.'
                    return
                anchor, handle = layer['markers']
                anchor.visible = handle.visible = grasp is not None
                if grasp is not None:
                    anchor.position, handle.position = tuple(grasp['anchor']), tuple(grasp['handle'])
            layer['positions'] = xpos
            if not self.single:
                return
            self.summary.update(time=float(state['time']), playing=bool(state.get('playing')), speed=state.get('speed'),
                                grasp=grasp['body'] if grasp else None, lagging=bool(state.get('lagging')),
                                unstable=int(state.get('unstable') or 0))
            now = time.monotonic()
            if now - self.last_status > .25:
                self.last_status = now
                notes = [f"t = {state['time']:.3f} s · {'playing' if state.get('playing') else 'paused'} ×{state.get('speed', 1):g}",
                         f"grasp: {grasp['body']}" if grasp else 'no grasp']
                if state.get('lagging'):
                    notes.append('slower than requested speed')
                if state.get('unstable'):
                    notes.append(f"MuJoCo unstable-acceleration warnings: {state['unstable']}")
                self.status.content = f'**{self.label}**  \n' + (f'{self.description}  \n' if self.description else '') + '  \n'.join(notes)


def environment_setup(root, path):
    """Physics-worker setup for one authored environment scene (e.g. ``blocks`` or ``panda/blocks``) under ``root``."""
    from .dashboard import environment_scenes
    from .runtime import zip_directory
    root = Path(root)
    if path not in environment_scenes(root):
        raise ValueError(f'unknown environment {path!r}')
    family = path.split('/')[0]
    return dict(kind='source', archive_b64=base64.b64encode(zip_directory(root / family)).decode(),
                entrypoint=path.split('/', 1)[1] + '/scene.xml' if '/' in path else 'scene.xml',
                simulation={'robot': 'panda'} if (root / family / 'robot').is_dir() else {})


class InspectorApp:
    """Owns the viewers, the selected run/version, the two physics workers and the coupled grasp."""
    def __init__(self, root, ports=(0, 0, 0, 0), session_seconds=2 * 3600, workspace=None, environments=None):
        self.roots = [Path(r).resolve() for r in (root if isinstance(root, (list, tuple)) else [root])]
        self.workspace = Path(workspace or tempfile.mkdtemp(prefix='sim2sim-inspect-'))
        self.session_seconds = session_seconds
        self.expires_at = _job_end() or time.time() + session_seconds
        self.lock = threading.RLock()
        self.viewers, self.environments_root, self.environment = {}, environments, None
        try:
            self.viewers['reference'] = Viewer(self, 'reference', 'Reference · private source scene', ports[0], ('reference',))
            self.viewers['candidate'] = Viewer(self, 'candidate', 'Reconstruction · recorded version', ports[1], ('candidate',))
            self.viewers['overlay'] = Viewer(self, 'overlay', 'Overlay · reference and reconstruction', ports[2], WORLDS)
            if environments is not None:  # one authored environment at a time, independent of the run viewers
                self.viewers[ENVIRONMENT] = Viewer(self, ENVIRONMENT, 'Environment', ports[3], (ENVIRONMENT,))
                self.viewers[ENVIRONMENT].summary = dict(status='waiting', message='Pick an environment.')
        except Exception:
            for viewer in self.viewers.values():
                viewer.server.stop()
            raise
        self.workers, self.tokens = {}, {}
        self.run = self.recorded = self.selected = self.error = None
        self.loading, self.generation, self.sync, self.pending = False, 0, True, (None, None)
        self.source_camera, self.camera_owner = None, (None, 0.0)
        self.coupled, self.anchors = False, {}
        self.verified = {}

    # Environment selection -------------------------------------------------------
    def select_environment(self, path):
        if self.environments_root is None:
            raise ValueError('this inspector serves no environments')
        setup = environment_setup(self.environments_root, path)  # validates the path against the authored scenes
        with self.lock:
            self._stop(ENVIRONMENT)
            self.environment = path
            self._start(ENVIRONMENT, setup, f'Authored environment `{path}` at its initial state')

    # Run and version selection ------------------------------------------------
    def select(self, request):
        directory = request.get('run')
        wall = request.get('time')
        index = request.get('index')
        if wall is not None and (isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall)):
            raise ValueError('time must be a finite number')
        if index is not None and (isinstance(index, bool) or not isinstance(index, int)):
            raise ValueError('index must be an integer')
        with self.lock:
            if directory != self.run:
                self._open_run(directory, wall, index)
            elif self.recorded is not None:
                self._select_version(wall, index)
            else:
                self.pending = (wall, index)

    def _open_run(self, directory, wall, index):
        if not isinstance(directory, str) or not directory or directory != Path(directory).name or directory.startswith('.'):
            raise ValueError('run must be one run directory name')
        paths = [r / 'runs' / directory for r in self.roots if (r / 'runs' / directory).exists() or (r / 'runs' / directory).is_symlink()]
        if len(paths) != 1 or paths[0].is_symlink() or not paths[0].is_dir():
            raise ValueError('unknown or ambiguous run directory')
        path = paths[0]
        self.generation += 1
        for world in tuple(self.workers):
            self._stop(world)
        self.run, self.recorded, self.selected, self.error, self.loading = directory, None, None, None, True
        self.pending = (wall, index)
        for world in WORLDS:
            self._clear(world, 'loading', 'Verifying the recorded audit and recovering versions.')
        threading.Thread(target=self._load_run, args=(self.generation, path), daemon=True).start()

    def _load_run(self, generation, path):
        try:
            # Every audit append rewrites the manifest, so an unchanged manifest
            # and sidecar mean the earlier verified recovery is still exact.
            key = tuple((item.stat().st_mtime_ns, item.stat().st_size) if item.is_file() else None
                        for item in (path / 'audit/manifest.json', path / 'candidate-change-history.json'))
            cached = self.verified.get(path.name)
            recorded = cached[1] if cached and cached[0] == key else recorded_versions(path)
            self.verified[path.name] = (key, recorded)
        except Exception as exc:
            with self.lock:
                if generation == self.generation:
                    self.loading, self.error = False, f'{type(exc).__name__}: {exc}'
                    for world in WORLDS:
                        self._clear(world, 'error', 'No interactive versions: ' + self.error)
            return
        with self.lock:
            if generation != self.generation:
                return
            self.recorded, self.loading, self.source_camera = recorded, False, None
            self._start('reference', dict(kind='source', archive_b64=base64.b64encode(recorded['source']).decode(),
                                          entrypoint=recorded['source_entrypoint'], simulation=recorded['source_simulation'],
                                          camera=recorded['camera']), 'Private source scene (verified snapshot)')
            wall, index = self.pending
            if wall is None and index is None:
                index = len(recorded['versions']) - 1 if recorded['versions'] else None
                wall = 0.0 if index is None else None
            self._select_version(wall, index)

    def _select_version(self, wall, index):
        versions = self.recorded['versions']
        if index is None:
            index = version_at(versions, float(wall or 0))
        elif not 0 <= index < len(versions):
            raise ValueError('version index out of range')
        if index == self.selected and 'candidate' in self.workers:
            return
        self.selected = index
        self._stop('candidate')
        if index is None:
            self._clear('candidate', 'empty', 'No reconstruction had been loaded by this wall time.')
            return
        version = versions[index]
        description = (f"Version {index + 1} of {len(versions)} · {version['kind'].replace('_', ' ')} at "
                       f"{version['wall_elapsed']:.1f} s · `{str(version['version'])[:12]}`")
        if version['archive'] is None:
            self._clear('candidate', 'error', 'Archive unavailable: ' + str(version['error']), description)
            return
        self._start('candidate', dict(kind='candidate', archive_b64=base64.b64encode(version['archive']).decode(),
                                      entrypoint=version['entrypoint'], simulation=self.recorded['candidate_simulation'],
                                      camera=self.recorded['camera']), description)

    def _start(self, world, setup, description):
        token = uuid.uuid4().hex
        self.tokens[world] = token
        self._clear(world, 'loading', 'Compiling in a limited physics worker.', description)
        try:
            self.workers[world] = WorldProcess(self.workspace, setup, lambda message: self._message(world, token, message),
                                               self.session_seconds)
        except Exception as exc:
            self._clear(world, 'error', f'Physics worker failed to start: {exc}', description)

    def _stop(self, world):
        self.tokens.pop(world, None)
        worker = self.workers.pop(world, None)
        if worker is not None:
            threading.Thread(target=worker.close, daemon=True).start()

    def _showing(self, world):
        return [viewer for viewer in self.viewers.values() if world in viewer.layers]

    def _clear(self, world, status, message, description=None):
        for viewer in self._showing(world):
            viewer.clear(world, status, message, description)
        with self.lock:
            self.anchors.pop(world, None)
            self._place_gizmos()

    def _place_gizmos(self):
        """Uncoupled: each world's own viewer drags its grasp. Coupled: once both parts are picked, every viewer drags both."""
        ready = all(world in self.anchors for world in WORLDS)
        for viewer in self.viewers.values():
            if viewer.primary == ENVIRONMENT:
                viewer.place_gizmo(self.anchors.get(ENVIRONMENT))
            elif self.coupled:
                viewer.place_gizmo(self.anchors[viewer.primary] if ready else None)
            else:
                viewer.place_gizmo(self.anchors.get(viewer.primary) if viewer.single else None)

    def _message(self, world, token, message):
        with self.lock:
            if self.tokens.get(world) != token:
                return
            viewers = self._showing(world)
        kind = message.get('type')
        try:
            if kind == 'model' and world == ENVIRONMENT:
                for viewer in viewers:
                    viewer.show(world, message)
                    viewer.set_camera(message.get('camera'), message['center'], float(message['extent']), move_clients=True)
            elif kind == 'model':
                for viewer in viewers:
                    viewer.show(world, message)
                with self.lock:
                    if world == 'reference':
                        self.source_camera = (message.get('camera'), message['center'], float(message['extent']))
                    camera = self.source_camera
                if camera:
                    # Every viewer starts at the recorded source camera pose; a
                    # new version keeps the researcher's current view.
                    for other in self.viewers.values():
                        if world == 'reference' or other.primary == world:
                            other.set_camera(*camera, move_clients=world == 'reference')
            elif kind == 'state':
                for viewer in viewers:
                    viewer.update(world, message)
                with self.lock:
                    if message.get('grasp') is None and world in self.anchors:
                        self.anchors.pop(world)
                        self._place_gizmos()
            elif kind == 'grasp':
                self._grasped(world, message, viewers)
            elif kind == 'notice':
                for viewer in viewers:
                    viewer.notice(message.get('text'))
            elif kind == 'error':
                with self.lock:
                    if self.tokens.get(world) == token:
                        self._stop(world)
                self._clear(world, 'error', 'Model unavailable: ' + str(message.get('error')), self.viewers[world].description)
        except (KeyError, TypeError, ValueError) as exc:
            self._clear(world, 'error', 'Invalid worker message: ' + str(exc), self.viewers[world].description)

    def _grasped(self, world, message, viewers):
        if not message.get('ok'):
            for viewer in viewers:
                viewer.notice(message.get('reason', 'No graspable body under the click.'))
            return
        self.viewers[world].summary['message'] = f"Grasping {message.get('body')}"
        with self.lock:
            self.anchors[world] = np.asarray(message['point'], dtype=float)
            both = self.coupled and all(name in self.anchors for name in WORLDS)
            workers = [self.workers.get(name) for name in WORLDS] if both else []
            self._place_gizmos()
        for worker in workers:  # coupled worlds start moving together once both parts are held
            if worker is not None:
                worker.send({'op': 'play'})

    # Interaction ---------------------------------------------------------------
    def control(self, op, world='both', value=None):
        if op == 'sync':
            self.sync = bool(value)
            return
        if op == 'couple':
            with self.lock:
                self.coupled = bool(value)
                self.anchors.clear()
                self._place_gizmos()
            return self.control('release', 'both')
        if op not in ('play', 'pause', 'reset', 'step', 'release', 'speed', 'grasp', 'target'):
            raise ValueError('unknown inspector control')
        if world not in (*WORLDS, 'both', ENVIRONMENT):
            raise ValueError('world must be reference, candidate, both or environment')
        coupled = self.coupled and world != ENVIRONMENT
        # Coupled worlds share every playback control; grasps and targets stay per world.
        worlds = WORLDS if world == 'both' or (coupled and op not in ('grasp', 'target')) else (world,)
        command = {'op': op}
        if op == 'speed':
            command['value'] = float(value)
        elif op == 'grasp':
            command.update(value, play=not coupled)
        elif op == 'target':
            command['position'] = value
        with self.lock:
            workers = [self.workers.get(name) for name in worlds]
            if op == 'grasp' and coupled:  # hold both worlds until the counterpart is picked
                workers = [self.workers.get(name) for name in WORLDS]
                commands = [{'op': 'pause'}] * len(workers)
                commands[WORLDS.index(world)] = command
            else:
                commands = [command] * len(workers)
        for worker, item in zip(workers, commands):
            if worker is not None:
                worker.send(item)

    def drag(self, viewer, position):
        """Move the grasp handle(s) by the gizmo's displacement from its grasp point."""
        with self.lock:
            if viewer.gizmo_origin is None:
                return
            offset = position - viewer.gizmo_origin
            coupled = self.coupled and viewer.primary != ENVIRONMENT
            worlds = WORLDS if coupled else (viewer.primary,)
            commands = [(self.workers.get(name), (self.anchors[name] + offset).tolist()) for name in worlds if name in self.anchors]
            others = [other for other in self.viewers.values() if other is not viewer and other.primary != ENVIRONMENT] if coupled else []
        for worker, target in commands:
            if worker is not None:
                worker.send({'op': 'target', 'position': target})
        for other in others:
            other.move_gizmo(offset)

    def camera_moved(self, viewer, camera):
        # The viewer being moved owns synchronization briefly, so echoes of
        # mirrored poses from the other viewer cannot feed back.
        now = time.monotonic()
        with self.lock:
            owner, since = self.camera_owner
            if viewer.primary == ENVIRONMENT or not self.sync or (owner is not None and owner is not viewer and now - since < .4):
                return
            self.camera_owner = (viewer, now)
        for other in self.viewers.values():
            if other is not viewer and other.primary != ENVIRONMENT:
                other.mirror_camera(camera)

    def state(self):
        with self.lock:
            recorded = self.recorded or {}
            versions = [{key: version[key] for key in ('change_index', 'kind', 'wall_elapsed', 'version', 'entrypoint', 'scores')}
                        | {'available': version['archive'] is not None, 'error': version['error']}
                        for version in recorded.get('versions', [])]
            return dict(run=self.run, loading=self.loading, error=self.error, selected=self.selected, sync=self.sync,
                        coupled=self.coupled, node=socket.gethostname(), job=os.environ.get('SLURM_JOB_ID'),
                        expires_at_unix=self.expires_at, server_time_unix=time.time(),
                        duration_wall=recorded.get('duration_wall', 0.0), versions=versions,
                        unavailable=recorded.get('unavailable', []),
                        worlds={name: dict(self.viewers[name].summary, port=self.viewers[name].port) for name in WORLDS},
                        overlay_port=self.viewers['overlay'].port,
                        environment=dict(self.viewers[ENVIRONMENT].summary, path=self.environment, port=self.viewers[ENVIRONMENT].port)
                        if ENVIRONMENT in self.viewers else None)

    def close(self):
        with self.lock:
            self.generation += 1
            workers = list(self.workers.values())
            self.workers, self.tokens = {}, {}
        for worker in workers:
            worker.close()
        for viewer in self.viewers.values():
            viewer.server.stop()
        shutil.rmtree(self.workspace, ignore_errors=True)


def _handler(app, directory, live=None):
    from .dashboard_view import HTML, SCRIPT
    # Current view code is served from memory so the physics tab needs no
    # snapshot regeneration; recorded data still comes from the snapshot.
    views = {'/': HTML, '/index.html': HTML, '/dashboard.js': SCRIPT}

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, *args):
            pass

        def _send(self, status, body, content_type):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = self.path.split('?', 1)[0]
            if route.startswith('/api/') and app is None:
                return self._send(503, json.dumps({'error': 'this dashboard server runs no physics inspector'}).encode(), 'application/json')
            if route == '/api/inspector/state':
                return self._send(200, json.dumps(app.state(), allow_nan=False).encode(), 'application/json')
            if route.startswith('/api/'):
                return self.send_error(404)
            if route in views:
                kind = 'text/html; charset=utf-8' if route != '/dashboard.js' else 'text/javascript; charset=utf-8'
                return self._send(200, views[route].encode(), kind)
            if live is not None and route == '/catalog.js':
                try:
                    body = live.catalog_script()
                except AuditFailure as exc:  # e.g. a run directory present in two roots: say so, keep the last list
                    body = ('window.__dashboardCatalogError=' + json.dumps(str(exc)) + ';').encode()
                return self._send(200, body, 'text/javascript; charset=utf-8')
            match = re.fullmatch(r'/runs/([0-9a-f]{16})\.js', route)
            if live is not None and match:
                try:
                    return self._send(200, live.run_script(match.group(1)), 'text/javascript; charset=utf-8')
                except KeyError:
                    return self.send_error(404)
            return super().do_GET()

        def do_POST(self):
            route = self.path.split('?', 1)[0]
            if app is None:
                return self._send(503, json.dumps({'error': 'this dashboard server runs no physics inspector'}).encode(), 'application/json')
            # JSON content type forces a CORS preflight that is never granted.
            if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                return self.send_error(415)
            length = int(self.headers.get('Content-Length') or 0)
            if length > MAX_REQUEST:
                return self.send_error(413)
            try:
                body = json.loads(self.rfile.read(length) or b'{}')
                if not isinstance(body, dict):
                    raise ValueError('request must be a JSON object')
                if route == '/api/inspector/select':
                    app.select(body)
                elif route == '/api/inspector/environment':
                    app.select_environment(body.get('path'))
                elif route == '/api/inspector/control':
                    app.control(body.get('op'), body.get('world', 'both'), body.get('value'))
                else:
                    return self.send_error(404)
            except (ValueError, TypeError, KeyError) as exc:
                return self._send(400, json.dumps({'error': str(exc)}).encode(), 'application/json')
            return self._send(200, json.dumps(app.state(), allow_nan=False).encode(), 'application/json')
    return Handler


PORTS = ('HTTP', 'REFERENCE', 'CANDIDATE', 'OVERLAY', 'ENVIRONMENT')


def serve(root, dashboard, ports, session_seconds, ready_file=None, physics=True):
    """Live dashboard over every root; with ``physics`` also the inspector viewers (Slurm test node only)."""
    from .dashboard import ENVIRONMENTS, LiveDashboard
    if physics and (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'test'):
        raise RuntimeFailure('The interactive inspector runs physics; start it on a Slurm test allocation')
    roots = root if isinstance(root, (list, tuple)) else [root]
    live = LiveDashboard(roots, dashboard)
    app = InspectorApp(roots, ports[1:], session_seconds, environments=ENVIRONMENTS) if physics else None
    try:
        server = ThreadingHTTPServer(('127.0.0.1', ports[0]), _handler(app, live.output, live))
    except OSError:
        if app:
            app.close()
        raise
    viewers = list(app.viewers.values()) if app else []  # reference, candidate, overlay, environment: the order of --ports
    ready = dict(node=socket.gethostname(), ports=[server.server_address[1], *(v.port for v in viewers)],
                 job=os.environ.get('SLURM_JOB_ID'))
    if ready_file:
        Path(ready_file).write_text(json.dumps(ready))
    print('SIM2SIM_INSPECTOR_READY ' + json.dumps(ready), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if app:
            app.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, action='append', help='experiment root; repeat for several (default experiments)')
    parser.add_argument('--dashboard', type=Path)
    parser.add_argument('--ports', type=int, nargs=5, default=(8000, 8001, 8002, 8003, 8004), metavar=PORTS)
    parser.add_argument('--session-seconds', type=float, default=2 * 3600)
    parser.add_argument('--ready-file', type=Path)
    args = parser.parse_args(argv)
    roots = args.root or [Path('experiments')]
    serve(roots, args.dashboard or roots[0] / 'dashboard', args.ports, args.session_seconds, args.ready_file)


if __name__ == '__main__':
    main()
