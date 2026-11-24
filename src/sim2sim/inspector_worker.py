"""Interactive inspector physics worker; native MJCF parsing stays in this limited process.

Reads one JSON setup line (base64 project archive), then JSON command lines.
Writes one ``model`` description followed by real-time ``state`` messages.
The researcher-only inspector is not part of the agent protocol.
"""
from __future__ import annotations
import base64
import json
import os
import select
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from .submission import _relative, extract_submission
from .inspector import PROPERTIES
from .runtime import decode_submission, extract_bundle
from .robots import ROBOTS
from .simulation import Simulation, _load_xml

STATE_HZ = 30
MAX_SETUP = 96 * 1024 * 1024
MAX_COMMAND = 64 * 1024
MAX_VERTICES = 2_000_000
HANDLE_SPEED_M_S = 1.5
INSPECTOR_CAMERA = "_sim2sim_inspector"
# Groups 0-2 are displayed; hidden collision geometry must not block clicks.
VISIBLE_GROUPS = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)


def _non_robot_bodies(scene):
    """Jointed named bodies outside the top-level subtree that owns a robot fixture's root body (Panda link0, Shadow rh_forearm).

    Expanded includes can contribute several worldbody sections (MuJoCo merges
    them), e.g. a robot include placed before the scene's own worldbody.
    """
    worlds = _load_xml(scene).findall("worldbody")
    tops = [top for world in worlds for top in world.findall("body")]
    roots = {entry[1] for entry in ROBOTS.values()}
    robot = {id(body) for top in tops if any(node.get("name") in roots for node in top.iter("body"))
             for body in top.iter("body")}
    return [body.get("name") for world in worlds for body in world.iter("body") if body.get("name") and id(body) not in robot
            and (body.find("joint") is not None or body.find("freejoint") is not None)]


def load(workspace, setup):
    project = Path(workspace) / "project"
    archive = decode_submission(setup["archive_b64"], 64 * 1024 * 1024)
    if setup["kind"] == "candidate":
        scene = extract_submission(archive, project, setup.get("entrypoint", "scene.xml"))
    else:
        extract_bundle(archive, project)
        scene = project.joinpath(*_relative(setup["entrypoint"]).parts)
        if not scene.is_file():
            raise ValueError("missing source scene entrypoint")
    simulation = setup.get("simulation") or {}
    robot = simulation.get("robot")
    names = simulation.get("graspable_bodies")
    if names is None and robot:
        names = _non_robot_bodies(scene)
    # A private camera satisfies the Simulation contract; nothing is rendered.
    config = {"robot": robot, "camera": INSPECTOR_CAMERA, "rgb": False, "calibration": False,
              "trusted_camera": {"pos": [0, 0, 1], "quat": [1, 0, 0, 0], "fovy": 45},
              "panda_point_grasps": True}
    if names is not None:
        config["graspable_bodies"] = list(names)
    if "timestep" in simulation:
        config["timestep"] = simulation["timestep"]
    return Simulation(scene, config)


def _array(values, dtype):
    return base64.b64encode(np.ascontiguousarray(values, dtype=dtype).tobytes()).decode("ascii")


def _visible_tendons(m):
    """Spatial tendons MuJoCo would draw (fixed tendons such as the Panda's finger coupling have no path)."""
    return [t for t in range(m.ntendon) if m.wrap_type[m.tendon_adr[t]] != mujoco.mjtWrap.mjWRAP_JOINT
            and m.tendon_group[t] < 3 and m.tendon_rgba[t, 3] > 0 and m.tendon_width[t] > 0]


def tendon_paths(m, d, samples=12):
    """Current polyline of each visible tendon: MuJoCo's wrap points, with the arc between a wrapping geom's two tangent
    points filled in (a helix on a cylinder: angle and height linear; a great circle on a sphere), as mjv draws it."""
    paths = []
    for t in _visible_tendons(m):
        start, count = d.ten_wrapadr[t], d.ten_wrapnum[t]
        points, objects = d.wrap_xpos.reshape(-1, 3)[start:start + count], d.wrap_obj.ravel()[start:start + count]
        path = [points[0]]
        for i in range(1, count):
            g = int(objects[i])
            if g >= 0 and int(objects[i - 1]) == g:  # the arc on wrapping geom g between consecutive tangent points
                frame, center = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
                a, b = (frame.T @ (points[i - 1] - center)), (frame.T @ (points[i] - center))
                u = np.linspace(0, 1, samples)[1:-1, None]
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    start_angle, turn = np.arctan2(a[1], a[0]), (np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0]) + np.pi) % (2 * np.pi) - np.pi
                    angle, radius = start_angle + u * turn, m.geom_size[g, 0]
                    local = np.c_[radius * np.cos(angle), radius * np.sin(angle), a[2] + u * (b[2] - a[2])]
                else:  # sphere
                    local = (a + u * (b - a)) * (np.linalg.norm(a) / np.linalg.norm(a + u * (b - a), axis=1, keepdims=True))
                path.extend(center + local @ frame.T)
            path.append(points[i])
        paths.append(np.round(np.asarray(path), 5).tolist())
    return paths


def _visible_flexes(m):
    """Flexes MuJoCo would draw (e.g. a cloth, whose vertex bodies carry no geoms)."""
    flexes = [f for f in range(m.nflex) if m.flex_group[f] < 3 and m.flex_rgba[f, 3] > 0]
    assert all(m.flex_dim[f] in (2, 3) for f in flexes), "the inspector draws only 2-D and 3-D flexes"
    return flexes


def _flex_faces(m, f):
    """Triangles over the flex's own vertices: a 2-D flex's elements, a 3-D flex's boundary shell."""
    if m.flex_dim[f] == 2:
        start, count = m.flex_elemdataadr[f], m.flex_elemnum[f]
        return m.flex_elem[start:start + 3 * count].reshape(-1, 3)
    start, count = m.flex_shelldataadr[f], m.flex_shellnum[f]
    return m.flex_shell[start:start + 3 * count].reshape(-1, 3)


def flex_vertices(m, d):
    """Current vertex positions of each visible flex."""
    return [d.flexvert_xpos[m.flex_vertadr[f]:m.flex_vertadr[f] + m.flex_vertnum[f]] for f in _visible_flexes(m)]


def object_bodies(sim):
    """Bodies not welded to the world, not mocap (e.g. grasp targets) and outside the robot fixture's subtree: the object
    whose CoM the viewer marks."""
    m = sim._model
    robot = (int(m.body_rootid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ROBOTS[sim.robot][1])])
             if sim.robot else -1)
    return [b for b in range(1, m.nbody) if m.body_weldid[b] and m.body_mocapid[b] < 0 and m.body_rootid[b] != robot]


def _column(m, spec):
    """The live MjModel array (a writable view) behind one editable property."""
    values = getattr(m, spec["field"])
    return values[:, spec["column"]] if "column" in spec else values


def describe(sim, camera="fixed"):
    m = sim._model
    graspable = set(sim._constraints)
    geoms, meshes, vertices = [], {}, 0
    for g in range(m.ngeom):
        # Bake MuJoCo's material colour into geom_rgba (display only), so the opacity property reads the shown alpha.
        if m.geom_matid[g] >= 0 and np.allclose(m.geom_rgba[g], [.5, .5, .5, 1]):
            m.geom_rgba[g] = m.mat_rgba[m.geom_matid[g]]
        rgba = m.geom_rgba[g]
        if m.geom_group[g] >= 3 or rgba[3] <= 0:
            continue
        mesh = int(m.geom_dataid[g]) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else -1
        if mesh >= 0 and str(mesh) not in meshes:
            start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            face_start, face_count = m.mesh_faceadr[mesh], m.mesh_facenum[mesh]
            vertices += int(count)
            if vertices > MAX_VERTICES:
                raise ValueError("visible mesh vertices exceed inspector limit")
            meshes[str(mesh)] = {"vertices": _array(m.mesh_vert[start:start + count], "<f4"),
                                 "faces": _array(m.mesh_face[face_start:face_start + face_count], "<i4")}
        body = int(m.geom_bodyid[g])
        geoms.append({"id": g, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                      or f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body) or f'body {body}'} geom {g - m.body_geomadr[body]}",
                      "body": body, "type": int(m.geom_type[g]), "size": m.geom_size[g].tolist(),
                      "pos": m.geom_pos[g].tolist(), "quat": m.geom_quat[g].tolist(), "rgba": [float(x) for x in rgba],
                      "mesh": mesh})
    bodies = [{"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body {b}",
               "graspable": b in graspable} for b in range(m.nbody)]
    view = None
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam >= 0:
        frame = sim._data.cam_xmat[cam].reshape(3, 3)
        view = {"pos": sim._data.cam_xpos[cam].tolist(), "forward": (-frame[:, 2]).tolist(),
                "up": frame[:, 1].tolist(), "fovy": float(m.cam_fovy[cam])}
    tendons = [{"rgba": m.tendon_rgba[t].tolist(), "width": float(m.tendon_width[t])} for t in _visible_tendons(m)]
    flexes = [{"rgba": (m.mat_rgba[m.flex_matid[f]] if m.flex_matid[f] >= 0 else m.flex_rgba[f]).tolist(),
               "faces": _array(_flex_faces(m, f), "<i4"), "vertices": _array(xyz, "<f4")}
              for f, xyz in zip(_visible_flexes(m), flex_vertices(m, sim._data))]
    values = {key: _column(m, spec).tolist() for key, spec in PROPERTIES.items()}
    return {"type": "model", "bodies": bodies, "geoms": geoms, "meshes": meshes, "camera": view, "tendons": tendons,
            "flexes": flexes,
            "objects": object_bodies(sim), "values": values, "center": m.stat.center.tolist(), "extent": float(m.stat.extent), "timestep": float(m.opt.timestep),
            "robot": sim.robot}


class Session:
    def __init__(self, sim):
        self.sim, self.playing, self.speed, self.target = sim, False, 1.0, None
        self.debt, self.lagging = 0.0, False
        self.objects = object_bodies(sim)
        m = sim._model
        self.original = {field: getattr(m, field).copy() for spec in PROPERTIES.values() if spec.get("relative")
                         for field in (spec["field"], *spec.get("scales", ()))}

    def com(self):
        """Composite CoM of the object bodies and each one's own CoM, from the live (possibly edited) masses."""
        m, d = self.sim._model, self.sim._data
        mass = m.body_mass[self.objects]
        com = (mass @ d.xipos[self.objects] / mass.sum()).round(6).tolist() if mass.sum() > 0 else None
        return com, d.xipos[self.objects].round(6).tolist()

    def set_property(self, key, index, value):
        """Researcher-side edit of this worker's live MjModel; recorded archives, audits and scores are untouched."""
        spec = PROPERTIES[key]
        if "display" in spec:
            raise ValueError(f"{key} is a display property; the viewer applies it")
        m, d = self.sim._model, self.sim._data
        index, value, (low, high, _) = int(index), float(value), spec["range"]
        if not 0 <= index < (m.nbody if spec["kind"] == "body" else m.ngeom) or not low <= value <= high:
            raise ValueError(f"{key} edit out of range")
        if spec.get("relative"):
            for field in (spec["field"], *spec.get("scales", ())):
                getattr(m, field)[index] = self.original[field][index] * value
        else:
            _column(m, spec)[index] = value
        # Derived constants (subtree masses, inverse weights) follow the edit; mj_setConst works at qpos0, so the
        # current state is restored around it.
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(m, kind))
        mujoco.mj_getState(m, d, state, kind)
        mujoco.mj_setConst(m, d)
        mujoco.mj_setState(m, d, state, kind)
        mujoco.mj_forward(m, d)

    def grasp_state(self):
        if self.sim._grasp is None:
            return None
        m, d = self.sim._model, self.sim._data
        eq, mocap = self.sim._grasp
        body = int(m.eq_obj1id[eq])
        anchor = d.xpos[body] + d.xmat[body].reshape(3, 3) @ m.eq_data[eq, :3]
        return {"body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body), "anchor": anchor.tolist(),
                "handle": d.mocap_pos[mocap].tolist()}

    def state(self):
        d = self.sim._data
        com, coms = self.com()
        return {"com": com, "coms": coms, "type": "state", "time": float(d.time), "xpos": d.xpos.round(6).tolist(),
                "xquat": d.xquat.round(6).tolist(), "grasp": self.grasp_state(), "playing": self.playing,
                "speed": self.speed, "lagging": self.lagging,
                "unstable": int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number),
                "tendons": tendon_paths(self.sim._model, d),
                "flexes": [xyz.round(5).tolist() for xyz in flex_vertices(self.sim._model, d)]}

    def step(self, count):
        m, d = self.sim._model, self.sim._data
        limit = HANDLE_SPEED_M_S * m.opt.timestep
        for _ in range(count):
            if self.sim._grasp is not None and self.target is not None:
                handle = d.mocap_pos[self.sim._grasp[1]]
                delta = self.target - handle
                distance = float(np.linalg.norm(delta))
                handle += delta if distance <= limit else delta * (limit / distance)
            mujoco.mj_step(m, d)

    def command(self, message):
        op = message.get("op")
        if op in ("play", "pause"):
            self.playing, self.debt = op == "play", 0.0
        elif op == "speed":
            self.speed = float(np.clip(float(message["value"]), .05, 4))
        elif op == "reset":
            self.sim.action({"op": "reset"}, render=False)
            self.target, self.debt = None, 0.0
        elif op == "step":
            self.step(max(1, round(.1 / self.sim._model.opt.timestep)))
        elif op == "release":
            self.sim._grasp_controller.release()
            self.target = None
        elif op == "grasp":
            origin = np.asarray(message["origin"], dtype=float)
            direction = np.asarray(message["direction"], dtype=float)
            if origin.shape != (3,) or direction.shape != (3,) or not np.isfinite([*origin, *direction]).all() \
                    or np.linalg.norm(direction) == 0:
                raise ValueError("grasp ray must contain finite origin and nonzero direction")
            self.sim._grasp_controller.release()
            mujoco.mj_forward(self.sim._model, self.sim._data)
            hit = self.sim._ray_hit(origin, direction / np.linalg.norm(direction), VISIBLE_GROUPS)
            if hit is None:
                self.target = None
                reason = ("No graspable movable body under the click." if self.sim._constraints
                          else "This model has no named jointed non-robot bodies to grasp.")
                return {"type": "grasp", "ok": False, "reason": reason}
            body, point = hit
            self.sim._grasp_controller.attach(body, point)
            # Coupled grasps hold the world paused until the counterpart part is picked too.
            self.target, self.playing = point.copy(), bool(message.get("play", True))
            return {"type": "grasp", "ok": True, "point": point.tolist(),
                    "body": mujoco.mj_id2name(self.sim._model, mujoco.mjtObj.mjOBJ_BODY, body)}
        elif op == "set":
            self.set_property(message["property"], message["index"], message["value"])
        elif op == "target":
            position = np.asarray(message["position"], dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError("grasp target must contain three finite numbers")
            if self.sim._grasp is not None:
                m = self.sim._model
                reach = 10 * max(float(m.stat.extent), 1.0)
                self.target = m.stat.center + np.clip(position - m.stat.center, -reach, reach)
        else:
            raise ValueError(f"unknown inspector command: {op}")
        return None


def _emit(message):
    sys.stdout.write(json.dumps(message, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


def main(*, workspace):
    try:
        line = sys.stdin.buffer.readline(MAX_SETUP + 1)
        if len(line) > MAX_SETUP:
            raise ValueError("inspector setup exceeds byte limit")
        setup = json.loads(line)
        sim = load(workspace, setup)
        _emit(describe(sim, setup.get("camera", "fixed")))
    except Exception as exc:
        _emit({"type": "error", "error": type(exc).__name__ + ": " + str(exc)})
        return
    session, fd, pending = Session(sim), sys.stdin.fileno(), b""
    period, last = 1 / STATE_HZ, time.monotonic()
    while True:
        ready, _, _ = select.select([fd], [], [], max(0.0, last + period - time.monotonic()))
        if ready:
            chunk = os.read(fd, 65536)
            if not chunk:
                return
            pending += chunk
            if len(pending) > MAX_COMMAND and b"\n" not in pending:
                raise ValueError("inspector command exceeds byte limit")
            *lines, pending = pending.split(b"\n")
            for raw in lines:
                if not raw.strip():
                    continue
                try:
                    reply = session.command(json.loads(raw))
                except (ValueError, KeyError, TypeError) as exc:
                    reply = {"type": "notice", "text": str(exc)}
                if reply:
                    _emit(reply)
                _emit(session.state())
        now = time.monotonic()
        if now - last < period:
            continue
        if session.playing:
            dt = session.sim._model.opt.timestep
            session.debt += (now - last) * session.speed
            count = int(session.debt / dt)
            # Drop unpayable debt rather than spiraling; the viewer reports lag.
            budget = int(4 * period / dt) + 1
            session.lagging = count > budget
            count = min(count, budget)
            session.debt = 0.0 if session.lagging else session.debt - count * dt
            started = time.monotonic()
            session.step(count)
            if time.monotonic() - started > period:
                session.lagging = True
        last = now
        _emit(session.state())
