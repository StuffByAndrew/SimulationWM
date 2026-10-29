"""Private MuJoCo state and the pixel-only interaction boundary.

Observations contain base64 PNG ``rgb`` (RGB8) and ``depth`` (uint16,
millimeters, saturated at 65535). Pixel coordinates start at the top left.
Only rendered observations cross this boundary; ray hits and state stay private.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image


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
        self.width = int(self.config.get("width", 640))
        self.height = int(self.config.get("height", 480))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        self.camera = self.config.get("camera", "fixed")
        self.default_ticks = int(self.config.get("default_ticks", 60))
        self.total_ticks = self.ticks = 0
        self._grasp = None
        self._callback = self.config.get("recording_callback")
        scene_path = Path(scene_path).resolve()
        tree = ET.parse(scene_path)
        root = tree.getroot()
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(root, "compiler")
        for key in ("meshdir", "texturedir"):
            compiler.set(key, str((scene_path.parent / compiler.get(key, ".")).resolve()))
        for include in root.iter("include"):
            include.set("file", str((scene_path.parent / include.attrib["file"]).resolve()))
        world = root.find("worldbody")
        if world is None:
            raise ValueError("scene needs a worldbody")
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
            names = [b.get("name") for b in world.iter("body")
                     if b.get("name") and (b.find("joint") is not None or b.find("freejoint") is not None)]
        self._body_names = list(names)
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
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
        mujoco.mj_forward(self._model, self._data)
        self._camera_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera)
        if self._camera_id < 0:
            raise ValueError(f"unknown fixed camera: {self.camera}")
        self._renderer = mujoco.Renderer(self._model, height=self.height, width=self.width)
        self._grasp_controller = self.config.get("grasp_controller_factory", PointGrasp)(self)

    @staticmethod
    def _png(array):
        output = io.BytesIO()
        Image.fromarray(array).save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")

    def calibration(self):
        """Public camera calibration; OpenGL camera axes: x right, y up, -z forward."""
        cam = self._camera_id
        focal = self.height / (2 * np.tan(np.deg2rad(self._model.cam_fovy[cam]) / 2))
        pose = np.eye(4)
        pose[:3, :3] = self._data.cam_xmat[cam].reshape(3, 3)
        pose[:3, 3] = self._data.cam_xpos[cam]
        return {"intrinsics": [[float(focal), 0, self.width / 2 - .5],
                               [0, float(focal), self.height / 2 - .5], [0, 0, 1]],
                "world_from_camera": pose.tolist(), "axes": "x_right_y_up_minus_z_forward",
                "width": self.width, "height": self.height}

    def observe(self, full=False):
        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(self._data, camera=self.camera)
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        depth_mm = np.clip(np.nan_to_num(depth, nan=65.535, posinf=65.535) * 1000, 0, 65535).astype(np.uint16)
        result = {"rgb": self._png(rgb), "depth": self._png(depth_mm),
                "width": self.width, "height": self.height, "time": float(self._data.time),
                "ticks": self.ticks, "total_ticks": self.total_ticks,
                "grasped": self._grasp is not None,
                "depth_format": "png_uint16_mm", "rgb_format": "png_rgb8"}
        if not full and not self.config.get("rgb", True):
            result.pop("rgb")
            result.pop("rgb_format")
        if not full and not self.config.get("depth", True):
            result.pop("depth")
            result.pop("depth_format")
        if full or self.config.get("calibration", False):
            result["calibration"] = self.calibration()
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
        origin = self._data.cam_xpos[cam].copy()
        geom = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(self._model, self._data, origin, direction, None, 1, -1, geom)
        if distance < 0:
            return None
        body = int(self._model.geom_bodyid[geom[0]])
        # Geometries may live in rigid children of the movable body.
        while body and body not in self._constraints:
            body = int(self._model.body_parentid[body])
        if body not in self._constraints:
            return None
        return body, origin + distance * direction

    def _advance(self, ticks, displacement=None):
        if isinstance(ticks, bool) or int(ticks) != ticks or not 1 <= int(ticks) <= 100000:
            raise ValueError("ticks must be an integer between 1 and 100000")
        ticks = int(ticks)
        delta = np.zeros(3) if displacement is None else np.asarray(displacement, dtype=float)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError("displacement must contain three finite numbers")
        for _ in range(ticks):
            self._grasp_controller.move(delta / ticks)
            mujoco.mj_step(self._model, self._data)
            self.ticks += 1
            self.total_ticks += 1
            if self._callback:
                self._callback(self)

    def step(self, op, args=None):
        return self.action(dict(args or {}, op=op))

    def action(self, action):
        op = action.get("op", action.get("action", "observe"))
        if op == "observe":
            if action.get("ticks", 0):
                self._advance(action["ticks"])
        elif op == "grasp":
            if self._grasp is not None:
                raise ValueError("release the current grasp first")
            hit = self._raycast(action["pixel"])
            if hit is not None:
                body, point = hit
                self._grasp_controller.attach(body, point)
        elif op == "move":
            if self._grasp is None:
                raise ValueError("move requires an active grasp")
            self._advance(action.get("ticks", self.default_ticks), action["displacement"])
        elif op == "release":
            self._grasp_controller.release()
            if action.get("ticks", 0):
                self._advance(action["ticks"])
        elif op == "reset":
            self._grasp_controller.release()
            mujoco.mj_resetData(self._model, self._data)
            self._grasp = None
            self.ticks = 0
            mujoco.mj_forward(self._model, self._data)
        elif op in ("wait", "advance"):
            self._advance(action.get("ticks", self.default_ticks))
        else:
            raise ValueError(f"unknown action: {op}")
        return self.observe()

    def close(self):
        self._renderer.close()
