# Parts adapted from RoboSimStudio (Apache-2.0, Copyright 2026 Kaiyuan Chen, https://github.com/robosimstudio/robosimstudio,
# commit d474a18b0b70c8f8fae3bbd8ce43a8456e57969b); see NOTICE and LICENSE. Rewritten for numpy and MuJoCo:
#   Panda.grasp, _basis, _score: robosimstudio/solvers/grasp/antipodal.py (antipodal pairs within a friction cone, an
#     approach fan perpendicular to the closing axis seeded from straight down, grasp frame from the hand's measured
#     approach/closing axes, balance/verticality/near scoring, dedupe) and robosimstudio_policy_agent/grasp_tools.py;
#     pad and approach clearance use MuJoCo collision of the whole robot instead of RoboSimStudio's boxes and rays.
#   Panda.linear: robosimstudio/solvers/ik/solver.py plan_line and robosimstudio/solvers/ik/arm_ik.py plan_cartesian
#     (a Cartesian line solved per waypoint, each seeded from the previous solution), with
#     robosimstudio_policy_agent/motion_tools.py plan_linear (lift/retreat use).
#   Panda.clearance: robosimstudio/core/probes/clearance.py (robot-to-scene minimum distance).
"""Panda motion-planning helpers for your own MuJoCo scene.xml, which must include the public robot/panda.xml.

CLI (prints JSON; a joint vector is the seven Panda joint angles in radians, as used by arm_target):
  python3 plan.py SCENE ik --target X Y Z [--quat W X Y Z] [--frame fingertips|hand] [--start Q1 .. Q7]
  python3 plan.py SCENE plan --start Q1 .. Q7 (--goal Q1 .. Q7 | --target X Y Z [--quat W X Y Z] [--frame ...])
  python3 plan.py SCENE interpolate --start Q1 .. Q7 --goal Q1 .. Q7
  python3 plan.py SCENE linear --start Q1 .. Q7 --target X Y Z [--quat W X Y Z] [--frame ...] [--cartesian-step M]
  python3 plan.py SCENE grasp (--body NAME | --near X Y Z) [--near X Y Z] [--start Q1 .. Q7] [--count N]
  python3 plan.py SCENE fk --start Q1 .. Q7
Options: --gripper-width W (finger opening assumed for collision checks, default 0.08), --ignore-body NAME (repeatable;
contacts with that body are allowed, e.g. an object to be grasped), --max-step RAD (largest joint change between returned
waypoints, default 0.1), --seed N. --start defaults to the reset pose; pass your current joint_positions_rad instead.
"fingertips" is the public panda_ee site between the finger pads; "hand" is the hand body origin. Without --quat only the
position is solved. A top-down approach with yaw psi is --quat 0 cos(psi/2) sin(psi/2) 0.
ik solves damped-least-squares inverse kinematics with random restarts and prefers collision-free solutions. plan checks
the straight joint-space line and otherwise runs RRT-connect with shortcutting; it returns waypoints after the start,
ending at the goal, to send one by one as arm_target joint_targets.
linear moves the frame along the straight world line from its pose at --start to --target (orientation turned evenly to
--quat, or held), solving IK every --cartesian-step metres (default 0.01) seeded from the previous waypoint. It fails
unless every waypoint converges and no joint changes more than --max-step between waypoints; collisions are reported
(collision_free, first_contact), not avoided. Use it to approach, retreat or lift.
grasp proposes parallel-jaw grasps on one body: the named one, or the one whose surface is nearest --near (--near also
restricts the search to within 0.096 m of that point and favours grasps near it). It pairs points of the body's collision
geoms whose surface normals oppose within a 20 degree friction cone and whose separation fits the jaws with 5 mm to spare
per side, then fans approach directions around the closing axis from straight down. A returned candidate converged and
was collision-free at the grasp with the fingers at open_width_m, at the pregrasp 0.1 m back along the approach, and
along the straight approach between them. To execute: plan to pregrasp_joint_targets; send approach_joint_targets with
gripper_width_m = open_width_m (the last one equals joint_targets); hold joint_targets with gripper_width_m =
close_width_m for a few hundred ticks; then lift with linear --start <joint_targets> --target <position + 0 0 0.1>
--quat <quat> --ignore-body BODY. close_width_m is 0: the gripper is a soft position servo whose squeeze grows with how far
the command lies inside the object. width_m is the measured contact separation; empty candidates come with rejection
counts (no pairs fit: the body is too wide for the 0.08 m jaws).
fk reports both frames' poses (quat W X Y Z), robot contacts, and clearance_m with clearance_geoms: the smallest distance
from a moving robot collision geom to a collidable geom of a non-ignored body, capped at 0.2 m.
Collision checks use your scene's initial object poses (qpos0) and count any robot contact deeper than 1 mm, including
self-contact. Results describe your model only.
Library: sys.path.insert(0, "/opt/public/tools/planning"); from plan import Panda; Panda("scene.xml").ik([x, y, z])
"""
import argparse
import json
import time

import mujoco
import numpy as np

HOME = np.array([0, 0, 0, -1.57079, 0, 1.57079, -.7853])
UP = np.array([0., 0., 1.])
CONE_DEG, SPARE, STANDOFF = 20, .005, .1  # friction cone half-angle; jaw margin per side [m]; pregrasp distance [m]


def _basis(approach, close):
    """Orthonormal frame whose columns are approach, close made perpendicular to it, and their cross product."""
    a = np.asarray(approach, float) / np.linalg.norm(approach)
    c = np.asarray(close, float) - (np.dot(close, a)) * a
    assert np.linalg.norm(c) > 1e-9, "closing axis is parallel to the approach"
    c = c / np.linalg.norm(c)
    return np.column_stack([a, c, np.cross(a, c)])


def _quat(matrix):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, float).ravel())
    return quat


def _matrix(quat):
    quat = np.asarray(quat, float)
    assert quat.shape == (4,) and np.linalg.norm(quat) > 0, "a quaternion is four numbers W X Y Z, not all zero"
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, quat / np.linalg.norm(quat))
    return matrix.reshape(3, 3)


def _surface(model, data, geom, count, rng):
    """count random points and outward unit normals on one collision geom, in world coordinates."""
    g, size = mujoco.mjtGeom, model.geom_size[geom]
    kind = g(int(model.geom_type[geom]))
    assert kind in (g.mjGEOM_SPHERE, g.mjGEOM_ELLIPSOID, g.mjGEOM_BOX, g.mjGEOM_CAPSULE, g.mjGEOM_CYLINDER, g.mjGEOM_MESH), \
        "cannot sample geom %d of type %d for grasping" % (geom, kind)
    u = rng.normal(size=(count, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    sign = rng.choice([-1., 1.], count)
    if kind in (g.mjGEOM_SPHERE, g.mjGEOM_ELLIPSOID):
        radii = size[[0, 0, 0]] if kind == g.mjGEOM_SPHERE else size
        points, normals = radii * u, u / radii
    elif kind == g.mjGEOM_BOX:
        area = np.roll(size, 1) * np.roll(size, 2)  # of the face pair normal to each axis
        rows, axis = np.arange(count), rng.choice(3, count, p=area / area.sum())
        points, normals = rng.uniform(-1, 1, (count, 3)) * size, np.zeros((count, 3))
        points[rows, axis], normals[rows, axis] = sign * size[axis], sign
    elif kind in (g.mjGEOM_CAPSULE, g.mjGEOM_CYLINDER):
        (r, h), capsule = size[:2], kind == g.mjGEOM_CAPSULE
        theta = rng.uniform(0, 2 * np.pi, count)
        ring = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(count)])
        side = (rng.random(count) < h / (h + (r if capsule else r / 2)))[:, None]  # side area 4 pi r h; caps 4 or 2 pi r^2
        cap = np.column_stack([u[:, :2], sign * np.abs(u[:, 2])]) if capsule else np.outer(sign, UP)
        disk = r * cap if capsule else r * np.sqrt(rng.random((count, 1))) * ring
        points = np.where(side, r * ring + np.outer(rng.uniform(-h, h, count), UP), disk + np.outer(sign * h, UP))
        normals = np.where(side, ring, cap)
    else:
        mesh = model.geom_dataid[geom]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        triangles = vertices[model.mesh_face[model.mesh_faceadr[mesh]:model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]]]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        area = np.linalg.norm(cross, axis=1)
        pick = rng.choice(len(triangles), count, p=area / area.sum())
        a, b = rng.random((2, count, 1))
        a, b = np.where(a + b > 1, 1 - a, a), np.where(a + b > 1, 1 - b, b)
        t = triangles[pick]
        points, normals = t[:, 0] + a * (t[:, 1] - t[:, 0]) + b * (t[:, 2] - t[:, 0]), cross[pick]
    rotation = data.geom_xmat[geom].reshape(3, 3)
    return data.geom_xpos[geom] + points @ rotation.T, (normals / np.linalg.norm(normals, axis=1, keepdims=True)) @ rotation.T


class Panda:
    def __init__(self, scene, gripper_width=.08, ignore_bodies=(), tolerance=1e-3):
        m = self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(m)

        def ident(kind, name):
            index = mujoco.mj_name2id(m, kind, name)
            assert index >= 0, "scene lacks %s %r of the public Panda fixture" % (kind.name, name)
            return index
        self.ident = ident
        joints = [ident(mujoco.mjtObj.mjOBJ_JOINT, "joint%d" % i) for i in range(1, 8)]
        self.qadr, self.dadr = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
        self.lower, self.upper = m.jnt_range[joints].T.copy()
        fingers = [ident(mujoco.mjtObj.mjOBJ_JOINT, name) for name in ("finger_joint1", "finger_joint2")]
        self.fadr, self.span = m.jnt_qposadr[fingers], 2 * m.jnt_range[fingers[0], 1]
        self.qpos0 = m.qpos0.copy()
        self.gripper(gripper_width)
        self.base = ident(mujoco.mjtObj.mjOBJ_BODY, "link0")
        self.robot = m.body_rootid[m.geom_bodyid] == self.base
        self.ignored = {ident(mujoco.mjtObj.mjOBJ_BODY, name) for name in ignore_bodies}
        hand, site, finger = ident(mujoco.mjtObj.mjOBJ_BODY, "hand"), ident(mujoco.mjtObj.mjOBJ_SITE, "panda_ee"), ident(mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        assert m.site_bodyid[site] == hand and m.body_parentid[finger] == hand, "panda_ee and left_finger must sit on the hand"
        self.frames = {"fingertips": (mujoco.mjtObj.mjOBJ_SITE, site), "hand": (mujoco.mjtObj.mjOBJ_BODY, hand)}
        # The hand's measured approach (towards the pads' site) and closing (left finger's slide) axes, in the hand frame.
        close = np.zeros(3)
        mujoco.mju_rotVecQuat(close, m.jnt_axis[fingers[0]], m.body_quat[finger])
        self.hand_axes, self.site_rotation = _basis(m.site_pos[site], close), _matrix(m.site_quat[site])
        self.tolerance, self.checks = tolerance, 0

    def gripper(self, width):
        """Set the finger opening [m] assumed by later collision checks."""
        assert 0 <= width <= self.span, "gripper width must be within 0..%g m" % self.span
        self.qpos0[self.fadr] = width / 2

    def _set(self, q):
        q = np.asarray(q, float)
        assert q.shape == (7,) and np.isfinite(q).all(), "a joint vector has seven finite radians"
        self.data.qpos[:] = self.qpos0
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def within_limits(self, q):
        return bool(np.all((self.lower - 1e-9 <= q) & (q <= self.upper + 1e-9)))

    def contacts(self, q):
        """Robot contacts deeper than the tolerance at q, as (geom, geom, depth_m) with names when authored."""
        self._set(q)
        mujoco.mj_collision(self.model, self.data)
        self.checks += 1
        m, found = self.model, []
        for (a, b), dist in zip(self.data.contact.geom, self.data.contact.dist):
            if dist < -self.tolerance and (self.robot[a] or self.robot[b]) and not {m.geom_bodyid[a], m.geom_bodyid[b]} & self.ignored:
                found.append([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "geom%d" % g for g in (a, b)] + [round(-float(dist), 5)])
        return found

    def clearance(self, q, limit=.2):
        """Smallest distance [m] from a moving robot collision geom to a collidable geom of a non-robot, non-ignored body
        (capped at limit), with that geom pair's names."""
        self._set(q)
        m, name = self.model, lambda g: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "geom%d" % g
        others = [b for b in range(m.ngeom) if not self.robot[b] and m.geom_bodyid[b] not in self.ignored]
        return min([(limit, None)] + [(round(mujoco.mj_geomDistance(m, self.data, a, b, limit, None), 5), [name(a), name(b)])
                                      for a in np.flatnonzero(self.robot & (m.geom_bodyid != self.base)) for b in others
                                      if m.geom_contype[a] & m.geom_conaffinity[b] or m.geom_contype[b] & m.geom_conaffinity[a]], key=lambda c: c[0])

    def pose(self, q, frame="fingertips"):
        """World position and 3x3 rotation of the frame at q."""
        kind, index = self.frames[frame]
        self._set(q)
        if kind == mujoco.mjtObj.mjOBJ_SITE:
            return self.data.site_xpos[index].copy(), self.data.site_xmat[index].reshape(3, 3).copy()
        return self.data.xpos[index].copy(), self.data.xmat[index].reshape(3, 3).copy()

    def fk(self, q):
        """Both frames' world poses (quat W X Y Z), robot contacts and clearance at q."""
        poses = {name: self.pose(q, name) for name in self.frames}
        return {**{name: {"position": p.tolist(), "quat": _quat(r).tolist()} for name, (p, r) in poses.items()},
                "contacts": self.contacts(q)[:10], **dict(zip(("clearance_m", "clearance_geoms"), self.clearance(q)))}

    def _error(self, q, target, rotation, frame):
        position, current = self.pose(q, frame)
        error = target - position
        if rotation is None:
            return error
        return np.concatenate([error, .5 * np.cross(current.T, rotation.T).sum(axis=0)])

    def ik(self, target, quat=None, frame="fingertips", start=HOME, iterations=300, damping=.03, restarts=30, seed=0, collision_free=True):
        """Damped-least-squares IK from start, then random restarts; the first converged (collision-free) solution wins."""
        target = np.asarray(target, float)
        assert target.shape == (3,) and np.isfinite(target).all(), "target is a finite world point [x, y, z]"
        rotation = None if quat is None else _matrix(quat)
        kind, index = self.frames[frame]
        rng, best = np.random.default_rng(seed), None
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for attempt in range(restarts + 1):
            q = np.clip(np.asarray(start, float) if attempt == 0 else rng.uniform(self.lower, self.upper), self.lower, self.upper)
            for _ in range(iterations):
                error = self._error(q, target, rotation, frame)
                if np.linalg.norm(error[:3]) < 1e-4 and np.linalg.norm(error[3:]) < 1e-3:
                    break
                (mujoco.mj_jacSite if kind == mujoco.mjtObj.mjOBJ_SITE else mujoco.mj_jacBody)(self.model, self.data, jacp, jacr, index)
                jacobian = (jacp if rotation is None else np.vstack([jacp, jacr]))[:, self.dadr]
                step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping ** 2 * np.eye(len(error)), error)
                q = np.clip(q + step * min(1, .2 / max(np.abs(step).max(), 1e-12)), self.lower, self.upper)
            error = self._error(q, target, rotation, frame)
            converged = bool(np.linalg.norm(error[:3]) < 1e-3 and np.linalg.norm(error[3:]) < 1e-2)
            contacts = self.contacts(q) if collision_free else []
            result = {"joint_targets": q.tolist(), "converged": converged, "collision_free": not contacts, "contacts": contacts[:10],
                      "position_error_m": float(np.linalg.norm(error[:3])), "rotation_error_rad": float(np.linalg.norm(error[3:])) if quat is not None else None,
                      "frame_position": self.pose(q, frame)[0].tolist(), "attempt": attempt}
            if converged and not contacts:
                return result
            if best is None or (converged, not contacts, -result["position_error_m"]) > (best["converged"], best["collision_free"], -best["position_error_m"]):
                best = result
        return best

    def linear(self, start, target, quat=None, frame="fingertips", step=.01, max_step=.1):
        """IK every step metres along the straight line from the frame's pose at start to target, the orientation turned
        evenly to quat (or held), each waypoint seeded from the previous solution. Stops at the first unconverged waypoint."""
        start, target = np.asarray(start, float), np.asarray(target, float)
        assert start.shape == (7,) and self.within_limits(start), "start is outside the Panda joint limits"
        assert target.shape == (3,) and np.isfinite(target).all() and step > 0, "target is a finite world point; step > 0"
        position, rotation = self.pose(start, frame)
        begin, turn = _quat(rotation), np.zeros(3)
        if quat is not None:
            mujoco.mju_subQuat(turn, _quat(_matrix(quat)), begin)
        count = max(1, int(np.ceil(max(np.linalg.norm(target - position) / step, np.linalg.norm(turn) / .05))))
        q, path, worst, contact = start, [], 0., None
        for k in range(1, count + 1):
            orientation = begin.copy()
            mujoco.mju_quatIntegrate(orientation, turn, k / count)
            result = self.ik(position + (target - position) * k / count, orientation, frame, q, restarts=0, collision_free=False)
            if not result["converged"]:
                break
            worst, q = max(worst, float(np.abs(np.subtract(result["joint_targets"], q)).max())), np.array(result["joint_targets"])
            path.append(np.round(q, 6).tolist())
            found = self.contacts(q) if contact is None else []
            if found:
                contact = {"waypoint": k, "contacts": found[:10]}
        return {"joint_targets": path, "reached": len(path) == count, "waypoints": count, "continuous": worst <= max_step + 1e-9,
                "max_joint_step_rad": round(worst, 6), "collision_free": contact is None, "first_contact": contact}

    def grasp(self, body=None, near=None, start=HOME, count=3, tries=40, points=300, seed=0):
        """Antipodal parallel-jaw grasps on the named body, or the body whose surface is nearest near, best first."""
        m, d, rng, began = self.model, self.data, np.random.default_rng(seed), time.perf_counter()
        assert body is not None or near is not None, "grasp needs a body name or a near point"
        near = None if near is None else np.asarray(near, float)
        self._set(HOME)
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and not self.robot[g] and (m.geom_contype[g] or m.geom_conaffinity[g])]
        if body is not None:
            target = self.ident(mujoco.mjtObj.mjOBJ_BODY, body)
            geoms = [g for g in geoms if m.geom_bodyid[g] == target]
        assert geoms, "no collidable non-robot body geoms to grasp"
        surface, normals = map(np.concatenate, zip(*(_surface(m, d, g, points, rng) for g in geoms)))
        owner = np.repeat(m.geom_bodyid[geoms], points)
        if body is None:
            target = owner[np.argmin(np.linalg.norm(surface - near, axis=1))]
        assert target not in self.ignored, "do not ignore the body being grasped"
        keep = owner == target
        if near is not None:
            keep &= np.linalg.norm(surface - near, axis=1) <= 1.2 * self.span
        keep = np.flatnonzero(keep)
        assert len(keep) >= 2, "no surface of the body within %g m of near" % (1.2 * self.span)
        keep = rng.choice(keep, min(points, len(keep)), replace=False)
        p, n = surface[keep], normals[keep]
        delta = p[None] - p[:, None]
        width = np.linalg.norm(delta, axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            u = delta / width[..., None]
            into, out = -np.einsum("ijk,ik->ij", u, n), np.einsum("ijk,jk->ij", u, n)
            ii, jj = np.nonzero(np.triu((width > 1e-3) & (width <= self.span - 2 * SPARE) & (np.minimum(into, out) >= np.cos(np.radians(CONE_DEG))), 1))
        cone = np.degrees(np.arccos(np.clip(np.minimum(into[ii, jj], out[ii, jj]), -1, 1)))
        pairs = []  # distinct grasps, best-opposed first
        for k in np.argsort(cone, kind="stable")[:2000]:
            centre, axis = (p[ii[k]] + p[jj[k]]) / 2, u[ii[k], jj[k]]
            if all(np.linalg.norm(centre - c) > .01 or abs(axis @ a) < np.cos(np.radians(20)) for c, a, *_ in pairs):
                pairs.append((centre, axis, float(width[ii[k], jj[k]]), float(cone[k]), [p[ii[k]].tolist(), p[jj[k]].tolist()]))
        candidates = []
        for index, (centre, axis, size, angle, _) in enumerate(pairs):
            down = -UP - (-UP @ axis) * axis  # the fan starts from the most downward approach perpendicular to the jaws
            down = down if np.linalg.norm(down) > 1e-6 else np.cross(axis, [1., 0, 0])
            down /= np.linalg.norm(down)
            for turn in np.arange(8) * np.pi / 4:
                approach = np.cos(turn) * down + np.sin(turn) * np.cross(axis, down)
                depth = np.max((p - centre) @ -approach)  # how far the body extends back from the grasp towards the palm
                score = _score(centre, approach, axis, size, angle, d.xipos[target], near, self.span, depth)
                candidates += [(score, index, approach, close) for close in (axis, -axis)]
        opened, rejected, found, used, evaluated = self.qpos0[self.fadr].sum(), dict.fromkeys(("grasp_ik", "grasp_collision", "pregrasp", "approach"), 0), [], set(), 0
        for score, index, approach, close in sorted(candidates, key=lambda c: -c[0]):
            if len(found) == count or evaluated == tries:
                break
            if index in used:
                continue
            evaluated += 1
            centre, axis, size, angle, contacts = pairs[index]
            self.gripper(min(self.span, size + 2 * SPARE))
            quat = _quat(_basis(approach, close) @ self.hand_axes.T @ self.site_rotation)
            grasp = self.ik(centre, quat, start=start, restarts=3, seed=seed)
            if not (grasp["converged"] and grasp["collision_free"]):
                rejected["grasp_ik" if not grasp["converged"] else "grasp_collision"] += 1
                continue
            pregrasp = self.ik(centre - STANDOFF * approach, quat, start=grasp["joint_targets"], restarts=3, seed=seed)
            if not (pregrasp["converged"] and pregrasp["collision_free"]):
                rejected["pregrasp"] += 1
                continue
            line = self.linear(pregrasp["joint_targets"], centre, quat)
            if not (line["reached"] and line["continuous"] and line["collision_free"]):
                rejected["approach"] += 1
                continue
            used.add(index)
            found.append({"joint_targets": line["joint_targets"][-1], "pregrasp_joint_targets": np.round(pregrasp["joint_targets"], 6).tolist(),
                          "approach_joint_targets": line["joint_targets"], "position": centre.tolist(), "quat": quat.tolist(),
                          "approach": approach.tolist(), "close_axis": close.tolist(), "contact_points": contacts, "width_m": size,
                          "open_width_m": min(self.span, size + 2 * SPARE), "close_width_m": 0.0, "cone_deg": angle,
                          "top_down": bool(approach @ UP < -.7), "score": score})
        self.gripper(opened)
        return {"body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, target), "candidates": found, "pairs": len(pairs),
                "evaluated": evaluated, "rejected": rejected, "seconds": round(time.perf_counter() - began, 3)}

    def segment_free(self, a, b, resolution=.02):
        count = max(1, int(np.ceil(np.abs(b - a).max() / resolution)))
        return all(not self.contacts(a + (b - a) * t) for t in np.linspace(0, 1, count + 1)[1:])

    def _extend(self, tree, sample, step):
        nodes, parents = tree
        nearest = int(np.argmin(np.linalg.norm(np.array(nodes) - sample, axis=1)))
        delta = sample - nodes[nearest]
        distance = np.linalg.norm(delta)
        new = sample if distance <= step else nodes[nearest] + delta * (step / distance)
        if not self.segment_free(nodes[nearest], new):
            return None
        nodes.append(new)
        parents.append(nearest)
        return len(nodes) - 1

    def plan(self, start, goal, max_step=.1, iterations=3000, seed=0, step=.3, shortcuts=200):
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        for name, q in (("start", start), ("goal", goal)):
            assert q.shape == (7,) and self.within_limits(q), name + " is outside the Panda joint limits"
            contacts = self.contacts(q)
            assert not contacts, "%s configuration is in collision: %s" % (name, contacts[:5])
        began, checks = time.perf_counter(), self.checks
        rng, method = np.random.default_rng(seed), "straight_line"
        path = [start, goal]
        if not self.segment_free(start, goal):
            method, path = "rrt_connect", self._rrt_connect(start, goal, rng, iterations, step)
            for _ in range(shortcuts):
                i, j = sorted(rng.choice(len(path), 2, replace=False))
                if j > i + 1 and self.segment_free(path[i], path[j]):
                    path = path[:i + 1] + path[j:]
        return {"joint_targets": interpolate(path, max_step)[1:], "method": method, "corners": len(path),
                "collision_checks": self.checks - checks, "seconds": round(time.perf_counter() - began, 3)}

    def _rrt_connect(self, start, goal, rng, iterations, step):
        trees = ([start], [-1]), ([goal], [-1])
        for iteration in range(iterations):
            grow, other = trees if iteration % 2 == 0 else trees[::-1]
            new = self._extend(grow, rng.uniform(self.lower, self.upper), step)
            if new is None:
                continue
            reached = self._extend(other, grow[0][new], step)
            while reached is not None and not np.allclose(other[0][reached], grow[0][new]):
                reached = self._extend(other, grow[0][new], step)
            if reached is not None:
                path = _trace(grow, new)[::-1] + _trace(other, reached)[1:]
                return path if grow is trees[0] else path[::-1]
        raise RuntimeError("RRT-connect found no collision-free path in %d iterations" % iterations)


def _score(centre, approach, axis, width, cone, com, near, span, depth):
    """Higher is better: opposed normals, little gravity moment about the jaw line, pads at most ~1 cm past the body's
    near side, from above, spare opening, near the requested point."""
    offset, heading = (centre - com) * [1, 1, 0], axis * [1, 1, 0]
    lever = 0. if np.linalg.norm(heading) < 1e-9 else np.linalg.norm(offset - (offset @ heading) / (heading @ heading) * heading)
    pointed = 1. if near is None else np.exp(-np.linalg.norm(centre - near) / .05)
    shallow = 1 - np.clip((depth - .01) / .035, 0, 1)
    return float(3 * (1 - cone / CONE_DEG) + 3 * np.exp(-lever / .015) + 1.5 * shallow + .5 * (1 - approach @ UP) + .5 * (span - width) / span + pointed)


def _trace(tree, index):
    nodes, parents, path = tree[0], tree[1], []
    while index >= 0:
        path.append(nodes[index])
        index = parents[index]
    return path


def interpolate(path, max_step=.1):
    """Joint-space waypoints along path (including its first point) with no joint changing more than max_step between them."""
    assert max_step > 0
    points = [np.asarray(path[0], float)]
    for a, b in zip(path, path[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        count = max(1, int(np.ceil(np.abs(b - a).max() / max_step)))
        points += [a + (b - a) * (k / count) for k in range(1, count + 1)]
    return [np.round(p, 6).tolist() for p in points]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog=__doc__.split("\n", 1)[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene")
    parser.add_argument("command", choices=("ik", "plan", "interpolate", "linear", "grasp", "fk"))
    parser.add_argument("--target", type=float, nargs=3)
    parser.add_argument("--quat", type=float, nargs=4)
    parser.add_argument("--frame", choices=("fingertips", "hand"), default="fingertips")
    parser.add_argument("--start", type=float, nargs=7, default=HOME.tolist())
    parser.add_argument("--goal", type=float, nargs=7)
    parser.add_argument("--gripper-width", type=float, default=.08)
    parser.add_argument("--ignore-body", action="append", default=[])
    parser.add_argument("--max-step", type=float, default=.1)
    parser.add_argument("--cartesian-step", type=float, default=.01)
    parser.add_argument("--body")
    parser.add_argument("--near", type=float, nargs=3)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    panda = Panda(args.scene, args.gripper_width, args.ignore_body)
    if args.command == "interpolate":
        assert args.goal is not None, "interpolate needs --goal"
        result = {"joint_targets": interpolate([args.start, args.goal], args.max_step)[1:]}
    elif args.command == "fk":
        result = panda.fk(args.start)
    elif args.command == "grasp":
        result = panda.grasp(args.body, args.near, args.start, args.count, seed=args.seed)
    elif args.command == "linear":
        assert args.target is not None, "linear needs --target"
        result = panda.linear(args.start, args.target, args.quat, args.frame, args.cartesian_step, args.max_step)
        assert result["reached"] and result["continuous"], "no continuous straight line: " + json.dumps({k: v for k, v in result.items() if k != "joint_targets"})
    elif args.command == "ik" or args.goal is None:
        assert args.target is not None, "ik, or plan without --goal, needs --target"
        result = panda.ik(args.target, args.quat, args.frame, args.start, seed=args.seed)
        if args.command == "plan":
            assert result["converged"] and result["collision_free"], "no converged collision-free IK goal: " + json.dumps(result)
            result = {"goal_ik": result, **panda.plan(args.start, result["joint_targets"], args.max_step, seed=args.seed)}
    else:
        assert args.target is None, "plan takes --goal or --target, not both"
        result = panda.plan(args.start, args.goal, args.max_step, seed=args.seed)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
