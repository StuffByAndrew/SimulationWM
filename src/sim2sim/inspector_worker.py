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

from .evaluation import _relative, extract_submission
from .runtime import decode_submission, extract_bundle
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
    """Jointed named bodies outside the top-level subtree that owns Panda joint1.

    Expanded includes can contribute several worldbody sections (MuJoCo merges
    them), e.g. a robot include placed before the scene's own worldbody.
    """
    worlds = _load_xml(scene).findall("worldbody")
    tops = [top for world in worlds for top in world.findall("body")]
    robot = {id(body) for top in tops if any(joint.get("name") == "joint1" for joint in top.iter("joint"))
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
    if names is None and robot == "panda":
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


def describe(sim, camera="fixed"):
    m = sim._model
    graspable = set(sim._constraints)
    geoms, meshes, vertices = [], {}, 0
    for g in range(m.ngeom):
        rgba = m.geom_rgba[g]
        if m.geom_matid[g] >= 0 and np.allclose(rgba, [.5, .5, .5, 1]):
            rgba = m.mat_rgba[m.geom_matid[g]]
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
        geoms.append({"body": int(m.geom_bodyid[g]), "type": int(m.geom_type[g]), "size": m.geom_size[g].tolist(),
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
    return {"type": "model", "bodies": bodies, "geoms": geoms, "meshes": meshes, "camera": view,
            "center": m.stat.center.tolist(), "extent": float(m.stat.extent), "timestep": float(m.opt.timestep),
            "robot": sim.robot}


class Session:
    def __init__(self, sim):
        self.sim, self.playing, self.speed, self.target = sim, False, 1.0, None
        self.debt, self.lagging = 0.0, False

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
        return {"type": "state", "time": float(d.time), "xpos": d.xpos.round(6).tolist(),
                "xquat": d.xquat.round(6).tolist(), "grasp": self.grasp_state(), "playing": self.playing,
                "speed": self.speed, "lagging": self.lagging,
                "unstable": int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)}

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
