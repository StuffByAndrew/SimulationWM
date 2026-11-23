"""Seeded launcher: perturbed resets and named throwing protocols for one free body of a level (private manifest [launcher]).

Used by the real-time world (realtime.py: reset and the `throw` action) and by evaluations that need fresh launches of the
same distribution (die.py). The protocol names and nominal release states are public; the seed and perturbation sizes stay
private."""
from __future__ import annotations

import mujoco
import numpy as np


def _quat(rotvec):
    angle = float(np.linalg.norm(rotvec))
    out = np.array([1., 0, 0, 0])
    if angle > 0:
        mujoco.mju_axisAngle2Quat(out, np.asarray(rotvec, float) / angle, angle)
    return out


def _mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out / np.linalg.norm(out)


class Launcher:
    """Seeded launch states of one free body (private manifest [launcher] table).

    A reset puts the body at its perturbed rest pose; a throw releases it with a named protocol's perturbed state. Launch k of
    stream s always draws the same Gaussian perturbation (numpy default_rng([seed, s, k])) and first resets the whole
    simulation (zero velocities and solver warm start, robot home, grasp released), so every trajectory is fixed by the
    protocol, (s, k) and the actions that follow. Streams: 0 the agent session, 1 evaluation trials, 2 the empirical
    baseline, 3 the mechanics baseline's observation noise."""
    SESSION, TRIALS, EMPIRICAL, NOISE = range(4)

    def __init__(self, table):
        self.body, self.seed, self.rest, self.perturbation = table["body"], int(table["seed"]), table["rest"], table["perturbation"]
        self.protocols = {p["name"]: p for p in table["protocols"]}
        assert len(self.protocols) == len(table["protocols"]) > 0
        assert set(self.perturbation) == {"position_m", "angle_rad", "velocity_m_s", "angular_velocity_rad_s"}
        assert set(self.rest) == {"position", "quat", "position_m", "yaw_rad"}

    def public(self):
        """Protocol names and nominal release states: public. Seed and perturbation sizes stay private."""
        return {name: {"release_position_m": p["position"], "release_quat_wxyz": p["quat"], "release_velocity_m_s": p["velocity"],
                       "release_angular_velocity_rad_s": p["angular_velocity"]} for name, p in self.protocols.items()}

    def state(self, protocol, stream, index):
        """(position, wxyz quaternion, velocity, angular velocity) in world axes of launch index of stream; protocol None
        is the rest pose. Velocity is that of the body frame origin."""
        rng = np.random.default_rng([self.seed, stream, index])
        if protocol is None:
            rest = self.rest
            position = np.asarray(rest["position"], float) + np.append(rng.normal(0, rest["position_m"], 2), 0)
            return position, _mul(_quat([0, 0, rng.normal(0, rest["yaw_rad"])]), rest["quat"]), np.zeros(3), np.zeros(3)
        p, s = self.protocols[protocol], self.perturbation
        return (np.asarray(p["position"], float) + rng.normal(0, s["position_m"], 3),
                _mul(_quat(rng.normal(0, s["angle_rad"], 3)), p["quat"]),
                np.asarray(p["velocity"], float) + rng.normal(0, s["velocity_m_s"], 3),
                np.asarray(p["angular_velocity"], float) + rng.normal(0, s["angular_velocity_rad_s"], 3))

    def place(self, sim, position, quat, velocity, angular_velocity):
        """Reset sim, then set the body's free joint (MuJoCo free-joint angular velocity is in the body frame)."""
        sim.action({"op": "reset"}, render=False)
        model, data = sim._model, sim._data
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.body)
        joint = model.body_jntadr[body]
        assert body > 0 and model.body_jntnum[body] == 1 and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE, self.body
        q, v = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        quat = np.asarray(quat, float) / np.linalg.norm(quat)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, quat)
        data.qpos[q:q + 7] = [*position, *quat]
        data.qvel[v:v + 6] = [*velocity, *(rotation.reshape(3, 3).T @ np.asarray(angular_velocity, float))]
        mujoco.mj_forward(model, data)
