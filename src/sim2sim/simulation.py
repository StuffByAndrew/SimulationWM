"""Private MuJoCo state and opt-in pixel, Panda arm or Shadow hand interaction boundaries.

Observations contain base64 PNG ``rgb`` (RGB8) and ``depth`` (uint16,
millimeters, saturated at 65535). Pixel coordinates start at the top left.
Robot paths expose only named joint/actuator proprioception and optional taxel
forces; ray hits, contact positions and other model state stay private.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

from .robots import ARM_ACTUATORS, ARM_HOME, ARM_JOINTS, GRIPPER, HAND_ACTUATORS, HAND_HOME, HAND_JOINTS, ROBOTS, actuators, joints


CAMERA_REACH_M = 4.0  # movable_camera bound on |pos| and |lookat| per axis (world frame)
_SPHERE_DIRECTIONS = np.array([[np.sqrt(1 - z * z) * np.cos(2.399963 * i), np.sqrt(1 - z * z) * np.sin(2.399963 * i), z]
                               for i, z in enumerate(np.linspace(1 - 1 / 512, -1 + 1 / 512, 512))])  # Fibonacci sphere: camera clearance rays


# Tactile pads: (name, body, body-frame x range, z range, rows, columns, largest body-frame y counted or None). Each pad
# bins contact normal force into rows (high z = distal first) by columns (increasing x); the total covers the whole body.
# Panda: the fixture's 17 x 17 mm fingertip_pad_collision_1 face. Shadow hand: the palmar (-y) half of each distal
# phalanx (f_distal_pst / th_distal_pst hull) and the palmar face of the palm.
_HAND_TIP = ((-.0095, .0095), (.004, .034), 3, 3, .001)
TACTILE = {"panda": ("finger_normal_force_n", "pad rows distal to proximal, columns across the finger",
                     [("left", "left_finger", (-.0085, .0085), (.036, .053), 3, 3, None),
                      ("right", "right_finger", (-.0085, .0085), (.036, .053), 3, 3, None)]),
           "shadow_hand": ("total_normal_force_n", "pad rows distal to proximal (decreasing body z), columns by increasing body x",
                           [(name, f"rh_{key}distal", *_HAND_TIP) for name, key in
                            (("thumb", "th"), ("first", "ff"), ("middle", "mf"), ("ring", "rf"), ("little", "lf"))]
                           + [("palm", "rh_palm", (-.042, .044), (-.012, .096), 4, 3, -.004)])}
TACTILE["panda_shadow"] = TACTILE["shadow_hand"]


def _load_xml(path, ancestors=()):
    """Expand includes before inspecting bodies, retaining nested relative paths."""
    path = Path(path).resolve()
    if path in ancestors:
        raise ValueError("cyclic MJCF include")
    root = ET.parse(path).getroot()
    for parent in list(root.iter()):
        for index, node in reversed(list(enumerate(parent))):
            if node.tag != "include":
                continue
            included = _load_xml(path.parent / node.attrib["file"], ancestors + (path,))
            parent.remove(node)
            for child in reversed(list(included)):
                parent.insert(index, child)
    return root


class PointGrasp:
    """Replaceable point constraint controller, without orientation actuation.

    Controllers implement attach(body, world_point), move(delta), and release().
    Simulation performs pixel raycasting and owns physics stepping.
    """
    def __init__(self, simulation):
        self.sim = simulation

    def attach(self, body, point):
        sim = self.sim
        eq, mocap = sim._constraints[body]
        sim._model.eq_obj1id[eq] = body  # flex vertex bodies share one connect equality
        sim._model.eq_data[eq, :3] = sim._data.xmat[body].reshape(3, 3).T @ (point - sim._data.xpos[body])
        sim._model.eq_data[eq, 3:6] = 0
        sim._data.mocap_pos[mocap] = point
        sim._data.eq_active[eq] = True
        sim._grasp = (eq, mocap)
        mujoco.mj_forward(sim._model, sim._data)

    def move(self, delta):
        if self.sim._grasp is not None:
            self.sim._data.mocap_pos[self.sim._grasp[1]] += delta

    def release(self):
        if self.sim._grasp is not None:
            self.sim._data.eq_active[self.sim._grasp[0]] = False
        self.sim._grasp = None


class Simulation:
    def __init__(self, scene_path, config=None):
        self.config = dict(config or {})
        self.robot = self.config.get("robot")
        if self.robot not in (None, *ROBOTS):
            raise ValueError("unsupported robot")
        self.width = int(self.config.get("width", 640))
        self.height = int(self.config.get("height", 480))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.config.get("tactile", False) is not False and (self.config["tactile"] is not True or self.robot is None):
            raise ValueError("tactile is a boolean and requires a robot")
        if self.config.get("movable_camera", False) not in (True, False):
            raise ValueError("movable_camera is a boolean")
        self.camera = self.config.get("camera", "fixed")
        self.default_ticks = int(self.config.get("default_ticks", 60))
        self.total_ticks = self.ticks = 0
        self._grasp = None
        self._callback = self.config.get("recording_callback")
        scene_path = Path(scene_path).resolve()
        root = _load_xml(scene_path)
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(root, "compiler")
        for key in ("meshdir", "texturedir"):
            compiler.set(key, str((scene_path.parent / compiler.get(key, compiler.get("assetdir", "."))).resolve()))
        compiler.set("assetdir", str((scene_path.parent / compiler.get("assetdir", ".")).resolve()))
        world = root.find("worldbody")
        if world is None:
            raise ValueError("scene needs a worldbody")
        if "scene_edit" in self.config:  # researcher-only probe fixtures edit the parsed MJCF
            self.config["scene_edit"](root)
        trusted_camera = self.config.get("trusted_camera")
        if trusted_camera is not None:
            for camera_node in list(root.iter("camera")):
                if camera_node.get("name") == self.camera:
                    for parent in root.iter():
                        if camera_node in list(parent):
                            parent.remove(camera_node)
                            break
            attributes = {"name": self.camera, "mode": "fixed"}
            for key in ("pos", "quat", "fovy"):
                value = trusted_camera[key]
                attributes[key] = str(value) if isinstance(value, (int, float, str)) else " ".join(map(str, value))
            ET.SubElement(world, "camera", **attributes)
        names = self.config.get("graspable_bodies")
        if names is None:
            # Every worldbody: an included robot file placed first contributes its own, and MuJoCo merges them.
            # Flex vertex bodies are grasped through the shared flex connect below, not one equality each.
            flex_bodies = {name for flex in root.iter("flex") for name in flex.get("body", "").split()}
            names = [b.get("name") for w in root.iter("worldbody") for b in w.iter("body")
                     if b.get("name") and b.get("name") not in flex_bodies and (b.find("joint") is not None or b.find("freejoint") is not None)]
        # Robot agents use robot actions only; the researcher inspector may opt in
        # to point grasps on explicitly listed non-robot bodies.
        point_grasps = self.robot is None or bool(self.config.get("panda_point_grasps"))
        self._body_names = list(names) if point_grasps else []
        # Cloth (2D flex): every moving flex vertex body is graspable through one connect whose body1 attach() retargets.
        flex_grasp = point_grasps and any(node.tag in ("flex", "flexcomp") for node in root.iter())
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        if flex_grasp:
            ET.SubElement(world, "body", name="_grasp_target_flex", mocap="true", pos="0 0 0")
            ET.SubElement(equality, "connect", name="_grasp_connect_flex", body1="world", body2="_grasp_target_flex", anchor="0 0 0",
                          active="false", solref=" ".join(map(str, self.config.get("grasp_solref", [0.01, 1]))),
                          solimp=" ".join(map(str, self.config.get("grasp_solimp", [0.9, 0.95, 0.001, 0.5, 2]))))
        for i, name in enumerate(self._body_names):
            ET.SubElement(world, "body", name=f"_grasp_target_{i}", mocap="true", pos="0 0 0")
            ET.SubElement(equality, "connect", name=f"_grasp_connect_{i}", body1=name,
                          body2=f"_grasp_target_{i}", anchor="0 0 0", active="false", solref=" ".join(map(str, self.config.get("grasp_solref", [0.01, 1]))),
                          solimp=" ".join(map(str, self.config.get("grasp_solimp", [0.9, 0.95, 0.001, 0.5, 2]))))
        visual = root.find("visual")
        if visual is None:
            visual = ET.SubElement(root, "visual")
        glob = visual.find("global")
        if glob is None:
            glob = ET.SubElement(visual, "global")
        glob.set("offwidth", str(max(self.width, int(glob.get("offwidth", 640)))))
        glob.set("offheight", str(max(self.height, int(glob.get("offheight", 480)))))
        self._model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self._data = mujoco.MjData(self._model)
        if "timestep" in self.config:
            self._model.opt.timestep = float(self.config["timestep"])
        self._constraints = {}
        for i, name in enumerate(self._body_names):
            body = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            eq = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_EQUALITY, f"_grasp_connect_{i}")
            target = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, f"_grasp_target_{i}")
            self._constraints[body] = (eq, int(self._model.body_mocapid[target]))
        if flex_grasp:
            eq = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_EQUALITY, "_grasp_connect_flex")
            mocap = int(self._model.body_mocapid[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "_grasp_target_flex")])
            for body in self._model.flex_vertbodyid:
                if self._model.body_weldid[body] != 0:
                    self._constraints.setdefault(int(body), (eq, mocap))
        if self.robot is not None:
            self._init_robot()
            self._reset_robot()
        self.fixture = None
        if mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_NUMERIC, "fixture_compression") >= 0:  # positioning-fixture scene
            from .fixture import Mechanism  # private carriage mechanism: evaluator image only, absent from the agent image
            self.fixture = Mechanism(self, self.config.get("fixture_seed"))
        mujoco.mj_forward(self._model, self._data)
        self._camera_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera)
        if self._camera_id < 0:
            raise ValueError(f"unknown fixed camera: {self.camera}")
        if int(self._model.cam_bodyid[self._camera_id]) != 0 or int(self._model.cam_mode[self._camera_id]) != 0:
            raise ValueError("observation camera must be fixed directly in worldbody")
        if hasattr(self._model, "cam_projection") and int(self._model.cam_projection[self._camera_id]) != 0:
            raise ValueError("only perspective observation cameras are supported")
        if np.any(self._model.cam_sensorsize[self._camera_id] != 0):
            raise ValueError("use fovy camera calibration; custom sensor/focal camera intrinsics are unsupported")
        self._wrist_camera_id = None
        wrist = self.config.get("wrist_camera")
        if wrist is not None:
            if self.robot != "panda" or not isinstance(wrist, str) or not wrist:
                raise ValueError("wrist_camera requires a named Panda hand camera")
            cam = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, wrist)
            hand = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "hand")
            if cam < 0 or hand < 1 or int(self._model.cam_bodyid[cam]) != hand or int(self._model.cam_mode[cam]) != 0:
                raise ValueError("wrist camera must be fixed directly on the Panda hand")
            if (hasattr(self._model, "cam_projection") and int(self._model.cam_projection[cam]) != 0) or np.any(self._model.cam_sensorsize[cam] != 0):
                raise ValueError("wrist camera requires perspective fovy calibration")
            self._wrist_camera_id = cam
        self._camera_data = mujoco.MjData(self._model) if self._wrist_camera_id is not None else None
        self._renderer = None
        self._grasp_controller = self.config.get("grasp_controller_factory", PointGrasp)(self)

    @staticmethod
    def _png(array):
        output = io.BytesIO()
        Image.fromarray(array).save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")

    def calibration(self, camera_id=None, data=None):
        """Public camera calibration; OpenGL camera axes: x right, y up, -z forward."""
        cam = self._camera_id if camera_id is None else camera_id
        data = self._data if data is None else data
        focal = self.height / (2 * np.tan(np.deg2rad(self._model.cam_fovy[cam]) / 2))
        pose = np.eye(4)
        pose[:3, :3] = data.cam_xmat[cam].reshape(3, 3)
        pose[:3, 3] = data.cam_xpos[cam]
        return {"intrinsics": [[float(focal), 0, self.width / 2 - .5],
                               [0, float(focal), self.height / 2 - .5], [0, 0, 1]],
                "world_from_camera": pose.tolist(), "axes": "x_right_y_up_minus_z_forward",
                "width": self.width, "height": self.height}

    def _wrench(self):
        """Object reaction on the ideal handle, expressed about its point anchor.

        Connect equations are world Cartesian anchor1-anchor2 coordinates.
        Body1 is the object, so the handle reaction is minus their multiplier.
        A point constraint transmits no couple; body-COM moments are not exposed.
        """
        force = np.zeros(3)
        active = self._grasp is not None and bool(self._data.eq_active[self._grasp[0]])
        if active:
            # The force the solver applied to the handle during the latest physics step (mj_step leaves its
            # constraint solution in efc_force; grasp attach and reset run mj_forward). Re-solving the integrated state
            # instead would read low by about m*v*dt/timeconst**2, a spurious velocity-proportional force (-10 N s/m on
            # a 0.4 kg stick at the default grasp softness). Reading the arrays changes no state, time or ticks.
            rows = np.flatnonzero((self._data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (self._data.efc_id == self._grasp[0]))
            if len(rows) != 3:
                raise RuntimeError("active point grasp must supply three constraint-force rows")
            force = -self._data.efc_force[rows]
        return {"force": force.tolist(), "torque": [0.0, 0.0, 0.0],
                "frame": "world", "reference": "grasp_point", "sign": "object_on_handle",
                "active": active, "force_unit": "N", "torque_unit": "N m"}

    def _ids(self, kind, names):
        result = np.array([mujoco.mj_name2id(self._model, kind, name) for name in names])
        if np.any(result < 0):
            raise ValueError(f"{self.robot} scene is missing required named joints, actuators or bodies")
        return result

    def _init_robot(self):
        self._proprioception_data = None
        actions = ROBOTS[self.robot][2]
        if self.config.get("tactile", False):
            self._pads = [(name, int(self._ids(mujoco.mjtObj.mjOBJ_BODY, [body])[0]), *rest)
                          for name, body, *rest in TACTILE[self.robot][2]]
        robot_joints = self._ids(mujoco.mjtObj.mjOBJ_JOINT, joints(self.robot))
        self._joint_qpos, self._joint_dofs = self._model.jnt_qposadr[robot_joints].copy(), self._model.jnt_dofadr[robot_joints].copy()
        self._effort_actuators = self._ids(mujoco.mjtObj.mjOBJ_ACTUATOR, actuators(self.robot))
        if "hand_target" in actions:
            hand = self._ids(mujoco.mjtObj.mjOBJ_JOINT, HAND_JOINTS)
            self._hand_qpos = self._model.jnt_qposadr[hand].copy()
            self._hand_actuators = self._ids(mujoco.mjtObj.mjOBJ_ACTUATOR, HAND_ACTUATORS)
            self._hand_home = np.array([HAND_HOME.get(name, 0.0) for name in HAND_JOINTS])
            if not np.all(self._model.actuator_ctrllimited[self._hand_actuators]):
                raise ValueError("Shadow hand actuators must have control ranges")
        if "arm_target" in actions:
            arm = self._ids(mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS)
            self._arm_actuators = self._ids(mujoco.mjtObj.mjOBJ_ACTUATOR, ARM_ACTUATORS)
            if np.any(self._model.jnt_type[arm] != mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError("Panda arm joints must be hinges")
            self._arm_qpos = self._model.jnt_qposadr[arm].copy()
            self._arm_ranges = self._model.jnt_range[arm].copy()
            self._arm_limited = self._model.jnt_limited[arm].astype(bool)
        if self.robot in GRIPPER:
            fingers = self._ids(mujoco.mjtObj.mjOBJ_JOINT, ["finger_joint1", "finger_joint2"])
            self._gripper_actuator = int(self._ids(mujoco.mjtObj.mjOBJ_ACTUATOR, ["actuator8"])[0])
            if np.any(self._model.jnt_type[fingers] != mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError("Panda finger joints must be slides")
            self._finger_qpos = self._model.jnt_qposadr[fingers].copy()
            self._finger_dofs = self._model.jnt_dofadr[fingers].copy()

    def _reset_robot(self):
        actions = ROBOTS[self.robot][2]
        if "arm_target" in actions:
            self._data.qpos[self._arm_qpos] = ARM_HOME[self.robot]
            self._data.ctrl[self._arm_actuators] = ARM_HOME[self.robot]
        if self.robot in GRIPPER:
            self._data.qpos[self._finger_qpos] = .04
            self._data.ctrl[self._gripper_actuator] = 255
        if "hand_target" in actions:
            # Position targets equal the home transmission lengths (joint angle, or J1 + J2 for coupled tendons).
            self._data.qpos[self._hand_qpos] = self._hand_home
            mujoco.mj_forward(self._model, self._data)
            self._data.ctrl[self._hand_actuators] = self._data.actuator_length[self._hand_actuators]

    def _proprioception(self):
        # mj_step forces precede integration. Solve a copied current state so
        # effort sensing cannot change warm starts, physics time or tick budgets.
        if self._proprioception_data is None:
            self._proprioception_data = mujoco.MjData(self._model)
        mujoco.mj_copyData(self._proprioception_data, self._model, self._data)
        mujoco.mj_forward(self._model, self._proprioception_data)
        result = {"joint_positions_rad": self._data.qpos[self._joint_qpos].tolist(),
                  "joint_velocities_rad_s": self._data.qvel[self._joint_dofs].tolist(),
                  "actuator_effort_nm": self._proprioception_data.actuator_force[self._effort_actuators].tolist()}
        if self.robot in GRIPPER:
            width = float(np.clip(np.sum(self._data.qpos[self._finger_qpos]), 0, .08))
            result.update(finger_positions_m=self._data.qpos[self._finger_qpos].tolist(),
                          finger_velocities_m_s=self._data.qvel[self._finger_dofs].tolist(), gripper_width_m=width, gripper_opening=width / .08)
        return result

    def _pad_contacts(self, data):
        """(pad, (row, column) or None outside the taxel grid, normal force, contact index) for every contact on a TACTILE
        pad body, binned by contact position in the pad body frame (researcher tools reuse it on recorded states)."""
        force = np.zeros(6)
        for name, body, (x0, x1), (z0, z1), rows, columns, y_max in self._pads:
            for i in range(data.ncon):
                contact = data.contact[i]
                if body not in (self._model.geom_bodyid[contact.geom1], self._model.geom_bodyid[contact.geom2]):
                    continue
                mujoco.mj_contactForce(self._model, data, i, force)
                x, y, z = data.xmat[body].reshape(3, 3).T @ (contact.pos - data.xpos[body])
                column, row = int((x - x0) // ((x1 - x0) / columns)), int((z1 - z) // ((z1 - z0) / rows))
                inside = 0 <= row < rows and 0 <= column < columns and (y_max is None or y <= y_max)
                yield name, (row, column) if inside else None, abs(force[0]), i

    def _tactile(self):
        """Normal contact force binned into each TACTILE pad's taxel grid by contact position in the pad body frame, plus
        the body's total. Uses the state _proprioception just forwarded; contacts with anything count, as on a real touch
        sensor, and their positions and identities stay private."""
        total_key, layout, _ = TACTILE[self.robot]
        out = {name: {"pad_normal_force_n": np.zeros((rows, columns)), total_key: 0.0} for name, _, _, _, rows, columns, _ in self._pads}
        for name, cell, force, _ in self._pad_contacts(self._proprioception_data):
            out[name][total_key] += force
            if cell is not None:
                out[name]["pad_normal_force_n"][cell] += force
        for pad in out.values():
            pad["pad_normal_force_n"] = pad["pad_normal_force_n"].tolist()
        return {**out, "unit": "N", "layout": layout}

    def observe(self, full=False):
        result = {"width": self.width, "height": self.height, "time": float(self._data.time),
                  "ticks": self.ticks, "total_ticks": self.total_ticks,
                  "grasped": self._grasp is not None}
        rgb_enabled = full or self.config.get("rgb", True)
        depth_enabled = full or self.config.get("depth", False)
        camera_data = self._data
        if self._camera_data is not None:
            # mj_step integrates qpos after its derived camera/geometry poses.
            # Refresh a copy so images and calibration match current joint state
            # without changing the live solver warm start or physics state.
            mujoco.mj_copyData(self._camera_data, self._model, self._data)
            mujoco.mj_forward(self._model, self._camera_data)
            camera_data = self._camera_data
        cameras = [("", self._camera_id)]
        if self._wrist_camera_id is not None:
            cameras.append(("wrist_", self._wrist_camera_id))
        # The near clip plane must never cut a surface, or a camera could see through opaque walls: wave-19 agents parked
        # the movable camera against a sealed box, then carried the box around a camera aimed at a clear spot. A surface
        # within the clearance covers the lens (black image, zero depth), as a real camera pressed against an object.
        covered = self._surface_within(camera_data.cam_xpos[self._camera_id], self._camera_clearance(), camera_data)
        for prefix, camera_id in cameras:
            if (rgb_enabled or depth_enabled) and prefix == "" and covered:
                if rgb_enabled:
                    result["rgb"], result["rgb_format"] = self._png(np.zeros((self.height, self.width, 3), np.uint8)), "png_rgb8"
                if depth_enabled:
                    result["depth"], result["depth_format"] = self._png(np.zeros((self.height, self.width), np.uint16)), "png_uint16_mm"
            elif rgb_enabled or depth_enabled:
                if self._renderer is None:
                    self._renderer = mujoco.Renderer(self._model, height=self.height, width=self.width)
                self._renderer.disable_depth_rendering()
                self._renderer.update_scene(camera_data, camera=camera_id)
                if rgb_enabled:
                    result[prefix + "rgb"] = self._png(self._renderer.render().copy())
                    result[prefix + "rgb_format"] = "png_rgb8"
                if depth_enabled:
                    self._renderer.enable_depth_rendering()
                    depth = self._renderer.render().copy()
                    self._renderer.disable_depth_rendering()
                    depth_mm = np.clip(np.nan_to_num(depth, nan=65.535, posinf=65.535) * 1000,
                                       0, 65535).astype(np.uint16)
                    result[prefix + "depth"] = self._png(depth_mm)
                    result[prefix + "depth_format"] = "png_uint16_mm"
            if full or self.config.get("calibration", True):
                result[prefix + "calibration"] = self.calibration(camera_id, camera_data)
        if self.config.get("wrench", False):
            result["wrench"] = self._wrench()
        if self.robot is not None:
            result["proprioception"] = self._proprioception()
            if self.config.get("tactile", False):
                result["tactile"] = self._tactile()
        if self.fixture is not None:
            result["fixture"] = self.fixture.readings()
        return result

    def _raycast(self, pixel):
        u, v = map(float, pixel)
        if not (np.isfinite([u, v]).all() and 0 <= u < self.width and 0 <= v < self.height):
            raise ValueError("pixel must be inside the observation")
        cam = self._camera_id
        scale = np.tan(np.deg2rad(self._model.cam_fovy[cam]) / 2)
        direction = np.array([(2 * (u + .5) / self.width - 1) * self.width / self.height * scale,
                              (1 - 2 * (v + .5) / self.height) * scale, -1.0])
        direction = self._data.cam_xmat[cam].reshape(3, 3) @ direction
        direction /= np.linalg.norm(direction)
        return self._ray_hit(self._data.cam_xpos[cam].copy(), direction)

    def _ray_hit(self, origin, direction, geomgroup=None):
        """Nearest graspable body and world hit point along a unit ray, or None. A flex (cloth) face hit selects the
        nearest vertex of the hit flex; the grasp then holds that vertex body at the hit point's offset."""
        geom, vertex = np.array([-1], dtype=np.int32), np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(self._model, self._data, origin, direction, geomgroup, 1, -1, geom)
        body = int(self._model.geom_bodyid[geom[0]]) if distance >= 0 else 0
        for flex in range(self._model.nflex):
            hit = mujoco.mj_rayFlex(self._model, self._data, 0, 0, 0, 1, 0, flex, origin, direction, vertex)
            if hit >= 0 and (distance < 0 or hit < distance):
                distance, body = hit, int(self._model.flex_vertbodyid[self._model.flex_vertadr[flex] + vertex[0]])
        if distance < 0:
            return None
        # Geometries may live in rigid children of the movable body.
        while body and body not in self._constraints:
            body = int(self._model.body_parentid[body])
        if body not in self._constraints:
            return None
        return body, origin + distance * direction

    def _push(self, ticks, displacement=None):
        """Validated (ticks, apply) that moves an active point grasp by displacement in equal per-tick steps."""
        if isinstance(ticks, bool) or int(ticks) != ticks or not 1 <= int(ticks) <= 100000:
            raise ValueError("ticks must be an integer between 1 and 100000")
        ticks = int(ticks)
        delta = np.zeros(3) if displacement is None else np.asarray(displacement, dtype=float)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError("displacement must contain three finite numbers")
        return ticks, lambda i: self._grasp_controller.move(delta / ticks)

    def _advance(self, ticks, displacement=None):
        self._run(*self._push(ticks, displacement))

    def _run(self, ticks, apply):
        """Step physics ticks times, calling apply(i) before step i."""
        for i in range(ticks):
            apply(i)
            mujoco.mj_step(self._model, self._data)
            self.ticks += 1
            self.total_ticks += 1
            if self._callback:
                self._callback(self)

    def motion(self, action):
        """A validated timed action (move, arm_target, hand_target) as (ticks, apply), where apply(i) sets its controls before
        physics step i. Lock-step actions run it at once; the real-time world (realtime.py) interleaves it with wall-clock
        stepping. Interpolated position targets start from the controls current when motion() is called."""
        op = action.get("op")
        if op == "move":
            if self._grasp is None:
                raise ValueError("move requires an active grasp")
            return self._push(action.get("ticks", self.default_ticks), action["displacement"])
        if op not in ("arm_target", "hand_target"):
            raise ValueError(f"{op} is not a timed action")
        ids, end, ticks = (self._arm_target if op == "arm_target" else self._hand_target)(action)
        start = self._data.ctrl[ids].copy()

        def apply(i):
            self._data.ctrl[ids] = start + (end - start) * ((i + 1) / ticks)
        return ticks, apply

    def _ticks(self, action):
        ticks = action.get("ticks", self.default_ticks)
        if isinstance(ticks, bool) or not isinstance(ticks, (int, np.integer)) or not 1 <= ticks <= 100000:
            raise ValueError("ticks must be an integer between 1 and 100000")
        return int(ticks)

    def _arm_target(self, action):
        if self.robot is None or "arm_target" not in ROBOTS[self.robot][2]:
            raise ValueError("arm_target requires a robot with an arm")
        ticks = self._ticks(action)
        targets = np.asarray(action["joint_targets"], dtype=float)
        if targets.shape != (7,) or not np.isfinite(targets).all():
            raise ValueError("joint_targets must contain seven finite radians")
        if np.any(self._arm_limited & ((targets < self._arm_ranges[:, 0]) | (targets > self._arm_ranges[:, 1]))):
            raise ValueError("joint_targets exceed Panda joint limits")
        if self.robot not in GRIPPER:
            if "gripper_width_m" in action:
                raise ValueError(f"robot={self.robot!r} has no gripper: arm_target takes no gripper_width_m")
            return self._arm_actuators, targets, ticks
        width = action["gripper_width_m"]
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not np.isfinite(width) or not 0 <= width <= .08:
            raise ValueError("gripper_width_m must be between 0 and 0.08")
        return np.append(self._arm_actuators, self._gripper_actuator), np.append(targets, width / .08 * 255), ticks

    def _hand_target(self, action):
        if self.robot is None or "hand_target" not in ROBOTS[self.robot][2]:
            raise ValueError("hand_target requires a robot with a hand")
        ticks = self._ticks(action)
        targets = np.asarray(action["actuator_targets"], dtype=float)
        if targets.shape != (len(HAND_ACTUATORS),) or not np.isfinite(targets).all():
            raise ValueError(f"actuator_targets must contain {len(HAND_ACTUATORS)} finite position targets")
        low, high = self._model.actuator_ctrlrange[self._hand_actuators].T
        if np.any((targets < low) | (targets > high)):
            raise ValueError("actuator_targets exceed Shadow hand control ranges")
        return self._hand_actuators, targets, ticks

    def step(self, op, args=None):
        return self.action(dict(args or {}, op=op))

    def action(self, action, *, render=True):
        op = action.get("op", action.get("action", "observe"))
        if self.robot is not None and op in ("grasp", "move", "release"):
            raise ValueError(f"point grasp actions are disabled for robot={self.robot!r}")
        if op in ("arm_target", "hand_target", "move"):
            self._run(*self.motion(dict(action, op=op)))
        elif op == "observe":
            if action.get("ticks", 0):
                self._advance(action["ticks"])
        elif op == "grasp":
            if self._grasp is not None:
                raise ValueError("release the current grasp first")
            hit = self._raycast(action["pixel"])
            if hit is not None:
                body, point = hit
                self._grasp_controller.attach(body, point)
        elif op == "release":
            self._grasp_controller.release()
            if action.get("ticks", 0):
                self._advance(action["ticks"])
        elif op == "reset":
            self._grasp_controller.release()
            mujoco.mj_resetData(self._model, self._data)
            self._grasp = None
            self.ticks = 0
            if self.robot is not None:
                self._reset_robot()
            if self.fixture is not None:
                self.fixture.restore()
            mujoco.mj_forward(self._model, self._data)
        elif op in ("wait", "advance"):
            self._advance(action.get("ticks", self.default_ticks))
        elif op == "camera":
            self._aim_camera(action["pos"], action["lookat"])
        elif op in ("reset_carriage", "release_carriage"):
            if self.fixture is None:
                raise ValueError(f"{op} requires a positioning-fixture scene")
            if op == "reset_carriage":
                assert self.fixture.rng is not None, "reset_carriage needs a private simulation.fixture_seed"
                self.fixture.reset_carriage(self.fixture.rng)
            else:
                self.fixture.release()
                if action.get("ticks", 0):
                    self._advance(action["ticks"])
        else:
            raise ValueError(f"unknown action: {op}")
        return self.observe() if render else None

    def _aim_camera(self, pos, lookat):
        """movable_camera: re-aim the observation camera (world frame, up = +z) without advancing physics time or ticks.
        The pose persists across resets; evaluation always builds fresh simulations with the trusted camera."""
        if self.config.get("movable_camera") is not True:
            raise ValueError("camera requires simulation.movable_camera = true")
        pos, lookat = np.asarray(pos, dtype=float), np.asarray(lookat, dtype=float)
        if pos.shape != (3,) or lookat.shape != (3,) or not (np.isfinite(pos).all() and np.isfinite(lookat).all()):
            raise ValueError("camera pos and lookat must be three finite numbers")
        if max(np.abs(pos).max(), np.abs(lookat).max()) > CAMERA_REACH_M or np.linalg.norm(lookat - pos) < .05:
            raise ValueError(f"camera pos and lookat must lie within {CAMERA_REACH_M} m of the origin and at least 0.05 m apart")
        clearance = self._camera_clearance()
        if self._surface_within(pos, clearance, self._data):
            raise ValueError(f"camera pos must stay at least {clearance:.3f} m from every surface")
        forward = (lookat - pos) / np.linalg.norm(lookat - pos)
        right = np.cross(forward, [0, 0, 1] if abs(forward[2]) < .999 else [0, 1, 0])
        right /= np.linalg.norm(right)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.column_stack([right, np.cross(right, forward), -forward]).ravel())
        self._model.cam_pos[self._camera_id], self._model.cam_quat[self._camera_id] = pos, quat
        mujoco.mj_camlight(self._model, self._data)

    def _camera_clearance(self):
        """Distance within which a surface could reach the observation camera's near clip plane (1.5 x frustum corner)."""
        tan = np.tan(np.deg2rad(self._model.cam_fovy[self._camera_id]) / 2)
        return 1.5 * self._model.vis.map.znear * self._model.stat.extent * np.sqrt(1 + tan * tan * (1 + (self.width / self.height) ** 2))

    def _surface_within(self, pos, distance, data):
        """Whether any geom surface (all groups, static included) lies within distance of pos, by rays in all directions; a
        ray from inside a geom reports its exit, so a point inside a geom or a small closed cavity counts too."""
        n = len(_SPHERE_DIRECTIONS)
        geom, dist = np.full(n, -1, dtype=np.int32), np.zeros(n)
        mujoco.mj_multiRay(self._model, data, np.asarray(pos, dtype=float), _SPHERE_DIRECTIONS.ravel(), None, 1, -1, geom, dist, None, n, distance)
        return bool(((geom >= 0) & (dist >= 0) & (dist < distance)).any())

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
