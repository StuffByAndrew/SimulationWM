"""Palm-down inverse kinematics for the panda_shadow fixture (researcher validation helper for touch-check.py).

The target is a point fixed on a hand body, by default the Shadow hand's `grasp_site` position (rh_palm-local (0, -0.035,
0.09): 35 mm in front of the palm at the finger bases). Palm down means the palmar normal (rh_palm -y) points along world -z;
the finger direction (rh_palm +z) is free unless `forward` is given. Damped least squares over the seven arm joints with the
hand joints held at `hand` (default its home pose) and joint-limit clipping; restarts from `start` (default the Panda home)
and seeded random poses; collision-free solutions are preferred."""
import mujoco
import numpy as np

from sim2sim.robots import ARM_HOME, ARM_JOINTS, HAND_HOME, HAND_JOINTS

UP = np.array([0., 0., 1.])
GRASP = (0., -.035, .09)


class IK:
    def __init__(self, model, hand=HAND_HOME):
        self.m, self.d = model, mujoco.MjData(model)
        arm = [model.joint(n).id for n in ARM_JOINTS]
        self.qpos, self.dofs = model.jnt_qposadr[arm], model.jnt_dofadr[arm]
        self.lo, self.hi = model.jnt_range[arm].T
        self.palm = model.body("rh_palm").id
        base = model.body("link0").id
        self.robot = {b for b in range(model.nbody) if model.body_rootid[b] == base}
        self.d.qpos[[model.jnt_qposadr[model.joint(n).id] for n in HAND_JOINTS]] = [hand.get(n, 0.) for n in HAND_JOINTS]

    def fk(self, q, body="rh_palm", point=GRASP):
        """World position of the body point and the palm rotation at arm joints q."""
        self.d.qpos[self.qpos] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        b = self.m.body(body).id
        return self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ point, self.d.xmat[self.palm].reshape(3, 3).copy()

    def collisions(self, q):
        """Names of the non-robot bodies the robot touches, and robot self-contacts, at q (objects at their initial pose)."""
        self.d.qpos[self.qpos] = q
        mujoco.mj_forward(self.m, self.d)
        found = set()
        for c in self.d.contact[:self.d.ncon]:
            a, b = (int(self.m.geom_bodyid[g]) for g in (c.geom1, c.geom2))
            if (a in self.robot) != (b in self.robot):
                found.add(self.m.body(b if a in self.robot else a).name or "world")
            elif a in self.robot and b in self.robot:
                found.add("self:" + self.m.body(a).name + "/" + self.m.body(b).name)
        return sorted(found)

    def _error(self, q, target, forward, body, point):
        pos, rot = self.fk(q, body, point)
        normal = rot[:, 1]
        e = [target - pos, np.cross(normal, UP)]
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        mujoco.mj_jac(self.m, self.d, jp, jr, pos, self.m.body(body).id)
        jp, jr = jp[:, self.dofs], jr[:, self.dofs]
        if forward is None:
            jr = (np.eye(3) - np.outer(normal, normal)) @ jr  # rotation about the palm normal (yaw) stays free
        else:
            e[1] = e[1] + np.cross(rot[:, 2], forward)
        return np.concatenate(e), np.vstack([jp, jr]), rot

    def solve(self, target, forward=None, body="rh_palm", point=GRASP, start=None, restarts=40, iterations=300, seed=0):
        target = np.asarray(target, float)
        forward = None if forward is None else np.asarray(forward, float)
        rng, results, best = np.random.default_rng(seed), [], None
        for k in range(restarts):
            q = np.asarray(ARM_HOME["panda"] if start is None else start, float) if k == 0 else rng.uniform(self.lo, self.hi)
            for _ in range(iterations):
                e, j, rot = self._error(q, target, forward, body, point)
                q = np.clip(q + j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(len(e)), e), self.lo, self.hi)
            e, _, rot = self._error(q, target, forward, body, point)
            tilt = float(np.degrees(np.arccos(np.clip(rot[:, 1] @ UP, -1, 1))))
            record = (np.linalg.norm(e[:3]), tilt, q)
            best = record if best is None or record[0] < best[0] else best
            if record[0] < 1e-3 and tilt < 1 and (forward is None or rot[:, 2] @ forward > np.cos(np.radians(1))):
                hit = self.collisions(q)
                results.append((bool(hit), record, hit))
                if not hit:
                    break
        collides, (error, tilt, q), hit = min(results, key=lambda r: r[0]) if results else (None, best, None)
        return {"ok": bool(results), "q": q.round(6).tolist(), "position_error_mm": round(1000 * float(error), 3),
                "palm_tilt_deg": round(tilt, 3), "collisions": hit, "forward": self.fk(q, body, point)[1][:, 2].round(6).tolist()}
