"""Persistent candidate-only simulation worker; execute exclusively inside OCI."""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import sys

from .robots import ROBOTS

MAX_LINE = 48 * 1024 * 1024
MAX_STATE = 4 * 1024 * 1024
PUBLIC_CONFIG = {"robot", "width", "height", "camera", "wrist_camera", "rgb", "depth", "calibration", "wrench", "tactile", "movable_camera",
                 "default_ticks", "timestep", "grasp_solref", "grasp_solimp", "recording_fps"}


# Public proprioception fields per robot; actuator_effort_nm is added unless simulation.effort = false.
PROPRIOCEPTION = {"panda": {"joint_positions_rad", "joint_velocities_rad_s", "finger_positions_m", "finger_velocities_m_s",
                            "gripper_width_m", "gripper_opening"},
                  "shadow_hand": {"joint_positions_rad", "joint_velocities_rad_s"}}


def snapshot_simulation(sim):
    """Private exact typed arrays plus grasp anchors; never send this to the agent."""
    state = {"schema_version": 1, "ticks": int(sim.ticks), "total_ticks": int(sim.total_ticks),
             "grasp": list(sim._grasp) if getattr(sim, "_grasp", None) is not None else None}
    if not hasattr(sim, "_data"):
        return state  # Small protocol test doubles have no native state.
    data = sim._data
    state["time"] = float(data.time)
    arrays = {}
    for key in ("qpos", "qvel", "act", "qacc_warmstart", "ctrl", "qfrc_applied", "xfrc_applied",
                "mocap_pos", "mocap_quat", "eq_active", "userdata", "plugin_state", "sensordata"):
        if hasattr(data, key):
            value = getattr(data, key)
            arrays[key] = {"dtype": value.dtype.str, "shape": list(value.shape),
                           "data_b64": base64.b64encode(value.tobytes(order="C")).decode()}
    value = sim._model.eq_data
    arrays["model.eq_data"] = {"dtype": value.dtype.str, "shape": list(value.shape),
                                "data_b64": base64.b64encode(value.tobytes(order="C")).decode()}
    state["arrays"] = arrays
    state["constraints"] = {str(body): list(ids) for body, ids in sim._constraints.items()}
    # Researcher-only projected evidence, never included in public observations.
    # Projection uses current native state without rendering or advancing physics.
    state["grasp_overlay"] = None
    if sim._grasp is not None:
        import numpy as np
        eq, mocap = sim._grasp
        body = int(sim._model.eq_obj1id[eq])
        anchor = data.xpos[body] + data.xmat[body].reshape(3, 3) @ sim._model.eq_data[eq, :3]
        calibration = sim.calibration()
        pose = np.asarray(calibration['world_from_camera'])
        intrinsic = np.asarray(calibration['intrinsics'])
        def project(point):
            camera = pose[:3, :3].T @ (point - pose[:3, 3])
            if not np.isfinite(camera).all() or camera[2] >= -1e-9:
                return None
            return [float(intrinsic[0, 2] + intrinsic[0, 0]*camera[0]/-camera[2]),
                    float(intrinsic[1, 2] - intrinsic[1, 1]*camera[1]/-camera[2])]
        state['grasp_overlay'] = {'anchor': project(anchor), 'handle': project(data.mocap_pos[mocap]),
                                  'width': sim.width, 'height': sim.height}
    return state


def full_observation(sim):
    return sim.observe(full=True) if hasattr(sim, "_model") else sim.observe()


def state_digest(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read():
    raw = sys.stdin.buffer.readline(MAX_LINE + 1)
    if not raw:
        raise EOFError("Host closed candidate protocol")
    if len(raw) > MAX_LINE or not raw.endswith(b"\n"):
        raise ValueError("Candidate message exceeds limit or is incomplete")
    message = json.loads(raw)
    if not isinstance(message, dict):
        raise ValueError("Candidate message must be an object")
    return message


def _emit(message):
    raw = json.dumps(message, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    if len(raw) > MAX_LINE:
        raise ValueError("Candidate result exceeds byte limit")
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()


def allowed_actions(config):
    """Physical actions a reconstruction world accepts (also used by the agent's local_controller tool)."""
    return ((("observe", "reset", ROBOTS[config["robot"]][2]) if config.get("robot") else ("observe", "grasp", "move", "release", "reset"))
            + (("camera",) if config.get("movable_camera") else ()))


def public_observation(complete, config):
    allowed = {"width", "height", "time", "ticks", "total_ticks", "frame", "episode", "grasped", "rgb", "rgb_format",
               "depth", "depth_format", "calibration", "wrench", "fixture"}
    if config.get("wrist_camera"):
        allowed.update({"wrist_rgb", "wrist_rgb_format", "wrist_depth", "wrist_depth_format", "wrist_calibration"})
    result = {key: value for key, value in complete.items() if key in allowed}
    for key, enabled in (("rgb", config.get("rgb", True)), ("depth", config.get("depth", False))):
        if not enabled:
            for prefix in ("", "wrist_"):
                result.pop(prefix + key, None)
                result.pop(prefix + key + "_format", None)
    if not config.get("calibration", True):
        result.pop("calibration", None)
        result.pop("wrist_calibration", None)
    if not config.get("wrench", False):
        result.pop("wrench", None)
    if config.get("robot") is not None:
        result.pop("grasped", None)
        result.pop("wrench", None)
        proprioception = complete.get("proprioception")
        if isinstance(proprioception, dict):
            keys = PROPRIOCEPTION[config["robot"]] | ({"actuator_effort_nm"} if config.get("effort", True) else set())
            result["proprioception"] = {key: value for key, value in proprioception.items() if key in keys}
        if config.get("tactile", False):
            result["tactile"] = complete["tactile"]
        if not any(config.get(key, default) for key, default in (("rgb", True), ("depth", False), ("calibration", True))):
            result.pop("width", None)
            result.pop("height", None)
    return result


def _checkpoint(sim, render=True):
    complete = full_observation(sim) if render else None
    state = snapshot_simulation(sim)
    if len(json.dumps(state)) > MAX_STATE:
        raise ValueError("Candidate state exceeds audit state limit")
    packet = {"state": state, "state_sha256": state_digest(state)}
    if complete is not None:
        packet.update(observation=public_observation(complete, sim.config), full_observation=complete)
    return packet


def main(*, project_dir=Path("/workspace/project")):
    from .simulation import Simulation
    from .evaluation import extract_submission
    from .runtime import decode_submission

    sim = None
    episode = 0
    version = None
    next_frame_time = 0.0
    try:
        while True:
            try:
                message = _read()
            except EOFError:
                break
            op = message.get("op")
            if op == "close":
                _emit({"type": "result", "ok": True})
                break
            before = sim.total_ticks if sim is not None else 0
            try:
                if op == "load":
                    if sim is not None:
                        raise ValueError("Use a new worker for each immutable candidate version")
                    project = decode_submission(message.get("project_b64"), 8 * 1024 * 1024)
                    scene = extract_submission(project, Path(project_dir), message.get("entrypoint", "scene.xml"))
                    supplied = message.get("config", {})
                    if not isinstance(supplied, dict) or set(supplied) - PUBLIC_CONFIG:
                        raise ValueError("Candidate configuration must contain public tool settings only")
                    config = dict(supplied)
                    config.setdefault("camera", "fixed")
                    sim = Simulation(scene, config)
                    version = hashlib.sha256(project).hexdigest()
                    next_frame_time = 1 / float(config.get("recording_fps", 30))
                elif op == "action":
                    if sim is None:
                        raise ValueError("Load a candidate before acting")
                    action = message["action"]
                    if not isinstance(action, dict) or action.get("op") not in allowed_actions(sim.config):
                        raise ValueError("Unsupported candidate action")
                    def callback(simulation):
                        nonlocal next_frame_time
                        now = float(simulation._data.time)
                        due = now + 1e-10 >= next_frame_time
                        checkpoint = _checkpoint(simulation, render=due)
                        if due:
                            while next_frame_time <= now + 1e-10:
                                next_frame_time += 1 / float(simulation.config.get("recording_fps", 30))
                        _emit({"type": "tick", "episode": episode, "version": version, **checkpoint})
                        acknowledgement = _read()
                        if acknowledgement.get("continue") is not True:
                            raise RuntimeError("Host interrupted candidate action")
                    sim._callback = callback
                    sim.action(action, render=False)
                    if action["op"] == "reset":
                        episode += 1
                        next_frame_time = 1 / float(sim.config.get("recording_fps", 30))
                else:
                    raise ValueError("Unsupported candidate operation")
                _emit({"type": "result", "ok": True, "version": version, "episode": episode,
                       "executed_ticks": sim.total_ticks - before, **_checkpoint(sim)})
            except Exception as exc:
                result = {"type": "result", "ok": False, "error_type": type(exc).__name__,
                          "error": "Candidate operation failed", "episode": episode, "version": version,
                          "executed_ticks": (sim.total_ticks - before) if sim is not None else 0}
                if sim is not None:
                    try:
                        result.update(_checkpoint(sim))
                    except Exception:
                        pass
                _emit(result)
    finally:
        if sim is not None:
            sim.close()


if __name__ == "__main__":
    main()
