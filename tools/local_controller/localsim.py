"""Local copy of the controller for your own project, and tactile sensor cell positions for any hand pose.

The local copy runs your project with the same simulation code and the same public settings (public_tool_config.json next
to this file) as the controller's reconstruction world. From the same fresh load, the same actions give the same
observations as load_reconstruction followed by those actions in the reconstruction world. It runs inside your sandbox;
it does not touch either controller world.

Python:
  import sys; sys.path.insert(0, "/opt/public/tools/local_controller"); import localsim
  world = localsim.LocalController("/workspace/project")  # optional: entrypoint="scene.xml", camera="fixed"
  world.observe()                                          # observation of the current state
  world.act({"op": "hand_target", "actuator_targets": [...], "ticks": 20})  # one action object, as in a sequence steps file
  world.run(steps)       # a list of action objects; returns per-step records like /workspace/sequences/NNNN.json
  world.act({"op": "reset"})
  localsim.taxels(joint_positions_rad)  # tactile cells for a hand pose: the 24 proprioception joint positions

taxels returns, per tactile pad (thumb, first, middle, ring, little, palm), the pad body's world position and rotation
(columns are the body x, y, z axes in world coordinates) and its grid of cells, rows distal to proximal, columns by
increasing body x, matching the tactile observation layout. A contact adds its normal force to a cell when its contact
point, in the pad body frame, lies within the cell's x_m and z_m ranges and has y <= y_max_m (the palmar side). Each cell
also gives surface_point_m and surface_normal: where a ray along the body +y axis through the cell's x/z center first
meets the pad's collision surface coming from the palmar side, and the outward surface normal there, in world
coordinates (null if that ray misses the pad). Positions are in meters in the controller's world frame. Only the public
robot fixture is used: no object is involved.

CLI (prints JSON):
  python3 localsim.py run PROJECT_DIR STEPS_FILE [--entrypoint scene.xml] [--camera fixed]
  python3 localsim.py taxels JOINT_POSITIONS_JSON_FILE   (a JSON list of 24 joint positions; omit for the reset pose)
"""
import argparse
import functools
import json
from pathlib import Path

import mujoco
import numpy as np

from sim2sim.candidate_worker import allowed_actions, public_observation
from sim2sim.robots import HAND_HOME, HAND_JOINTS
from sim2sim.simulation import Simulation, TACTILE

CONFIG_FILE = Path(__file__).with_name("public_tool_config.json")
FIXTURE = Path("/opt/public/robot")
IMAGE_KEYS = ("rgb", "depth", "wrist_rgb", "wrist_depth")


class LocalController:
    def __init__(self, project_dir="/workspace/project", entrypoint="scene.xml", camera="fixed", config=None):
        self.config = dict(json.loads(CONFIG_FILE.read_text()) if config is None else config, camera=camera)
        self.sim = Simulation(Path(project_dir) / entrypoint, self.config)

    def observe(self):
        return public_observation(self.sim.observe(), self.config)

    def act(self, action):
        if not isinstance(action, dict) or action.get("op") not in allowed_actions(self.config) or "world" in action:
            raise ValueError("an action is an object with op in %s and that op's fields, without world" % (allowed_actions(self.config),))
        return public_observation(self.sim.action(dict(action)), self.config)

    def run(self, steps):
        return [{"step": index, "op": step["op"], "observation": {k: v for k, v in self.act(step).items()
                                                                  if k not in IMAGE_KEYS and not k.endswith(("_format", "calibration"))}}
                for index, step in enumerate(steps)]

    def close(self):
        self.sim.close()


@functools.lru_cache(maxsize=4)
def _fixture(robot_dir):
    model = mujoco.MjModel.from_xml_string('<mujoco><compiler angle="radian" autolimits="true" meshdir="%s/assets"/>'
                                           '<include file="%s/shadow_hand.xml"/></mujoco>' % (robot_dir, robot_dir))
    model.geom_rgba[:, 3] = 1  # mj_ray skips fully transparent geoms; groups (set per pad below) select the pad's collision geoms
    pads = []
    for name, body, x_range, z_range, rows, columns, y_max in TACTILE["shadow_hand"][2]:
        body_id = model.body(body).id
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id and (model.geom_contype[g] or model.geom_conaffinity[g])]
        assert geoms, body
        pads.append((name, body, body_id, geoms, x_range, z_range, rows, columns, y_max))
    return model, mujoco.MjData(model), [model.joint(name).qposadr[0] for name in HAND_JOINTS], pads


def taxels(joint_positions=None, robot_dir=FIXTURE):
    model, data, qpos, pads = _fixture(str(robot_dir))
    joints = np.array([HAND_HOME.get(name, 0.0) for name in HAND_JOINTS] if joint_positions is None else joint_positions, dtype=float)
    assert joints.shape == (len(HAND_JOINTS),) and np.isfinite(joints).all(), "joint_positions needs %d finite values" % len(HAND_JOINTS)
    data.qpos[qpos] = joints
    mujoco.mj_kinematics(model, data)
    result, mask, hit, normal = {}, np.array([0, 1, 0, 0, 0, 0], np.uint8), np.zeros(1, np.int32), np.zeros(3)
    for name, body, body_id, geoms, (x0, x1), (z0, z1), rows, columns, y_max in pads:
        model.geom_group[:] = 0
        model.geom_group[geoms] = 1
        position, rotation = data.xpos[body_id].copy(), data.xmat[body_id].reshape(3, 3).copy()
        width, height = (x1 - x0) / columns, (z1 - z0) / rows
        cells = []
        for row in range(rows):
            cells.append([])
            for column in range(columns):
                x, z = x0 + (column + .5) * width, z1 - (row + .5) * height
                start = position + rotation @ np.array([x, -.1, z])
                distance = mujoco.mj_ray(model, data, start, rotation[:, 1].copy(), mask, 1, -1, hit, normal)
                assert distance < 0 or hit[0] in geoms
                cells[-1].append({"x_m": [x0 + column * width, x0 + (column + 1) * width], "z_m": [z1 - (row + 1) * height, z1 - row * height],
                                  "surface_point_m": None if distance < 0 else (start + distance * rotation[:, 1]).tolist(),
                                  "surface_normal": None if distance < 0 else normal.tolist()})
        result[name] = {"body": body, "body_position_m": position.tolist(), "body_rotation": rotation.tolist(), "y_max_m": y_max, "cells": cells}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a steps file on a fresh load of your project")
    run.add_argument("project_dir")
    run.add_argument("steps_file")
    run.add_argument("--entrypoint", default="scene.xml")
    run.add_argument("--camera", default="fixed")
    cells = commands.add_parser("taxels", help="tactile cell positions for a hand pose")
    cells.add_argument("joints_file", nargs="?")
    args = parser.parse_args(argv)
    if args.command == "run":
        world = LocalController(args.project_dir, args.entrypoint, args.camera)
        print(json.dumps(world.run(json.loads(Path(args.steps_file).read_text()))))
        world.close()
    else:
        print(json.dumps(taxels(None if args.joints_file is None else json.loads(Path(args.joints_file).read_text()))))


if __name__ == "__main__":
    main()
