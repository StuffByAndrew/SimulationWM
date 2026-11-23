"""Write the biased-die levels (2026-11-23): environments/die_biased/ (point grasp) and environments/panda_small/die_biased/
(Franka, the unchanged shared fixture). A 40 mm regular dodecahedron (d12) with twelve colored faces rests in a bounded
throwing tray on the table. A tungsten slug sealed inside the resin die (invisible to RGB, depth and the grasp ray; no
contacts) moves its centre of mass 3 mm off centre, so the die is biased. Both variants share the tray, die, camera,
lighting and the private launcher (perturbed reset poses and four throwing protocols; src/sim2sim/realtime.py Launcher),
so only the actuator differs. Pure text generation, no MuJoCo: python experiments/validation/die-scenes.py"""
from pathlib import Path
import numpy as np

PHI = (1 + 5 ** .5) / 2
R_IN = .020                      # m die inradius: 40 mm across parallel faces; the Franka opens 80 mm, pads are 17 mm
SHELL_DENSITY = 1150.            # kg/m^3 cast resin (the visible die)
SLUG_R, SLUG_DENSITY = .006, 19300.  # tungsten sphere; its geom density is the excess over the resin it displaces
SLUG_OFFSET = .011               # m slug centre from the die centre (17 mm + 6 mm radius stays inside the 20 mm inradius)
TOP, TABLE = .40, (-.38, .38, -.42, .26)  # table top z and x/y extent, as in panda_small
TX, TY = 0., -.17                # tray centre on the table (0.48 m in front of the Franka base)
TRAY_IN, WALL_H, WALL_T, FLOOR_T = .15, .08, .01, .01  # inner half width, wall height above the floor (keeps bounces in), thicknesses
FLOOR = TOP + FLOOR_T            # tray floor top
DIE_FRICTION = "1.2 .01 .0001"   # grippy die for the weak fixture gripper (condim 4), as the sealed boxes
# Die-tray contact (explicit pairs, so the die can be both grippy and lively in the tray): felt-like sliding friction and a
# bouncy contact (damping ratio 0.3: restitution about 0.35). solref 0.004 needs >= 4 steps per impact: 1 ms timestep.
PAIR = 'condim="3" friction=".45 .45 .005 .0001 .0001" solref=".004 .3"'
EYE, TARGET, FOVY = np.array([.12, .22, .80]), np.array([TX, TY, .44]), 45  # fixed camera: tray in front, Franka behind
COLORS = [(.85, .10, .10), (1., .50, 0.), (.95, .85, .10), (.55, .85, .15), (.05, .45, .15), (.10, .80, .85),
          (.10, .25, .85), (.50, .15, .70), (.95, .35, .75), (.97, .97, .97), (.05, .05, .05), (.50, .30, .12)]
BODY_RGBA = ".55 .55 .55 1"      # neutral grey visible only along edges and corners between the face plates
# Private launcher (src/sim2sim/realtime.py): every reset and throw draws the next perturbation of a seeded sequence.
SEED = 20261123
PERTURB = {"position_m": .003, "angle_rad": .05, "velocity_m_s": .03, "angular_velocity_rad_s": .8}  # Gaussian std per axis
REST_PERTURB = {"position_m": .01, "yaw_rad": .3}
DEADLINE = .15                   # s after release: before the die settles (checked by die-check.py)
LEVELS = {"die_biased": False, "panda_small/die_biased": True}  # path: Franka


def fmt(v):
    return " ".join(f"{float(x):.6f}" for x in v)


def xyaxes(eye, target):
    forward = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(forward, [0, 0, 1]); x /= np.linalg.norm(x)
    return fmt([*x, *np.cross(x, forward)])


def rotation(a, b):
    """Rotation matrix taking unit a to unit b."""
    v, c = np.cross(a, b), float(a @ b)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def quat(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return [np.cos(angle / 2), *(np.sin(angle / 2) * axis)]


def die():
    """Body-frame vertices and outward face normals (face k at index k-1): face 1 points +z, faces 2-6 ring the top by
    increasing azimuth, and opposite faces sum to 13 as on a standard d12."""
    vertices = [np.array(v, float) for v in
                [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
                + [p for s in (-1, 1) for t in (-1, 1) for p in ((0, s / PHI, t * PHI), (s / PHI, t * PHI, 0), (t * PHI, 0, s / PHI))]]
    normals = [np.array(n, float) / np.linalg.norm(n) for s in (-1, 1) for t in (-1, 1) for n in ((0, s * PHI, t), (s * PHI, t, 0), (t, 0, s * PHI))]
    vertices, normals = np.array(vertices), np.array(normals)
    inradius = (vertices @ normals.T).max(axis=0)
    assert np.allclose(inradius, inradius[0]) and all(np.isclose(vertices @ n, inradius[0]).sum() == 5 for n in normals)
    turn = rotation(normals[0], np.array([0., 0, 1]))
    vertices, normals = (vertices @ turn.T) * R_IN / inradius[0], normals @ turn.T
    top = normals[normals[:, 2] > .99]
    ring = sorted((n for n in normals if 0 < n[2] < .99), key=lambda n: np.arctan2(n[1], n[0]))
    ordered = np.array([*top, *ring, *[-n for n in reversed(ring)], *(-top)])
    assert len(ordered) == 12 and all(np.allclose(ordered[k], -ordered[11 - k]) for k in range(12))
    return vertices, ordered


VERTICES, NORMALS = die()
SLUG = SLUG_OFFSET * (NORMALS[11] + .6 * NORMALS[7]) / np.linalg.norm(NORMALS[11] + .6 * NORMALS[7])  # toward faces 12 and 8
assert all(R_IN - n @ SLUG > SLUG_R + .002 for n in NORMALS)
EDGE = min(np.linalg.norm(a - b) for i, a in enumerate(VERTICES) for b in VERTICES[i + 1:])
PLATE_R = .85 * EDGE / (2 * np.tan(np.pi / 5))  # 85 % of the pentagon inradius
VOLUME = 3 * np.sqrt(25 + 10 * 5 ** .5) * EDGE ** 2 * R_IN / 3  # (1/3) inradius x surface area (12 pentagons)
SLUG_MASS = (SLUG_DENSITY - SHELL_DENSITY) * 4 / 3 * np.pi * SLUG_R ** 3
MASS = SHELL_DENSITY * VOLUME + SLUG_MASS
COM = SLUG_MASS * SLUG / MASS
REST = [TX, TY, FLOOR + R_IN + .0002]
# Public throwing protocols: nominal release state of the die's geometric centre (world frame, wxyz quaternion of the die
# body). Each throw adds the private perturbation.
PROTOCOLS = {
    "drop": ([TX, TY, FLOOR + .13], quat([1, 0, 0], .35), [0, 0, 0], [0, 0, 0]),       # tilted drop, no spin
    "roll": ([TX - .07, TY, FLOOR + .06], [1, 0, 0, 0], [.9, 0, 0], [0, 30, 0]),       # low forward roll along +x
    "toss": ([TX + .06, TY + .06, FLOOR + .12], quat([1, 1, 0], 1.2), [-.5, -.4, .6], [10, -14, 6]),  # lob with tumble
    "spin": ([TX, TY - .06, FLOOR + .10], quat([0, 1, 1], 2.2), [.3, .9, .4], [30, 10, 40]),         # fast spin into the walls
}


def scene(level, panda):
    plates = []
    for k, (n, color) in enumerate(zip(NORMALS, COLORS), 1):
        plates.append(f'<geom name="face_{k:02d}" type="cylinder" size="{PLATE_R:.6f} .0004" pos="{fmt((R_IN - .0003) * n)}" '
                      f'zaxis="{fmt(n)}" density="0" contype="0" conaffinity="0" rgba="{fmt(color)} 1"/>')
    walls = [f'<geom name="tray_wall_{side}" type="box" pos="{fmt(pos)}" size="{fmt(size)}" rgba=".30 .22 .16 1"/>'
             for side, pos, size in (("xlo", (-TRAY_IN - WALL_T / 2, 0, FLOOR_T + WALL_H / 2), (WALL_T / 2, TRAY_IN + WALL_T, WALL_H / 2)),
                                     ("xhi", (TRAY_IN + WALL_T / 2, 0, FLOOR_T + WALL_H / 2), (WALL_T / 2, TRAY_IN + WALL_T, WALL_H / 2)),
                                     ("ylo", (0, -TRAY_IN - WALL_T / 2, FLOOR_T + WALL_H / 2), (TRAY_IN, WALL_T / 2, WALL_H / 2)),
                                     ("yhi", (0, TRAY_IN + WALL_T / 2, FLOOR_T + WALL_H / 2), (TRAY_IN, WALL_T / 2, WALL_H / 2)))]
    tray = ["tray_floor", *(f"tray_wall_{side}" for side in ("xlo", "xhi", "ylo", "yhi"))]
    name = ("panda_small_" if panda else "") + level.split("/")[-1]
    return f'''<mujoco model="{name}">
  <!-- Researcher-only. Biased d12 die level: a {2000 * R_IN:g} mm regular dodecahedron of resin ({SHELL_DENSITY:g} kg/m^3) with a sealed
       {2000 * SLUG_R:g} mm tungsten slug ({SLUG_DENSITY:g} kg/m^3; geom density is the excess over the displaced resin) {1000 * SLUG_OFFSET:g} mm
       off centre toward faces 12 and 8: total {1000 * MASS:.1f} g, centre of mass {1000 * np.linalg.norm(COM):.2f} mm off the geometric
       centre. The slug has no contacts, sits in hidden geom group 3 and is enclosed by the opaque die. The tray ({2000 * TRAY_IN:g} mm square,
       {1000 * WALL_H:g} mm walls) contacts the die through explicit pairs: friction .45, solref .004 .3 (bouncy); hence the 1 ms step.
       Elliptic cones with impratio 10 for the weak fixture grip, as the sealed boxes. -->
  <compiler angle="radian" autolimits="true"{' meshdir="../robot/assets"' if panda else ''}/>
  <option timestep="0.001" integrator="implicitfast" cone="elliptic" impratio="10"><flag multiccd="enable"/></option>
  <visual><map znear="0.01" zfar="12"/></visual>
  <asset><mesh name="die_hull" vertex="{" ".join(fmt(v) for v in VERTICES)}"/></asset>
  <default><geom friction=".7 .01 .001"/></default>
  <worldbody>
    <light name="key" directional="true" pos="-2 -3 4" dir="1 1 -2" diffuse=".85 .82 .76"/>
    <light name="fill" directional="true" pos="2 1 3" dir="-1 -1 -2" diffuse=".3 .36 .45" castshadow="false"/>
    <camera name="fixed" pos="{fmt(EYE)}" xyaxes="{xyaxes(EYE, TARGET)}" fovy="{FOVY}"/>
    <geom name="ground" type="plane" size="3 3 .1" rgba=".70 .72 .73 1"/>
    <geom name="table" type="box" pos="{(TABLE[0] + TABLE[1]) / 2:g} {(TABLE[2] + TABLE[3]) / 2:g} {TOP - .05:g}" size="{(TABLE[1] - TABLE[0]) / 2:g} {(TABLE[3] - TABLE[2]) / 2:g} .05" rgba=".65 .52 .37 1"/>
{'    <geom name="panda_pedestal" type="box" pos="0 -.65 .2" size=".13 .13 .2" rgba=".25 .28 .32 1"/>' + chr(10) if panda else ''}    <body name="tray" pos="{TX:g} {TY:g} {TOP:g}">
      <geom name="tray_floor" type="box" pos="0 0 {FLOOR_T / 2:g}" size="{TRAY_IN + WALL_T:g} {TRAY_IN + WALL_T:g} {FLOOR_T / 2:g}" rgba=".16 .42 .26 1"/>
      {(chr(10) + "      ").join(walls)}
    </body>
    <body name="die" pos="{fmt(REST)}">
      <freejoint name="die_free"/>
      <geom name="die_shell" type="mesh" mesh="die_hull" density="{SHELL_DENSITY:g}" friction="{DIE_FRICTION}" condim="4" solref=".004 1" rgba="{BODY_RGBA}"/>
      <geom name="die_slug" type="sphere" size="{SLUG_R:g}" pos="{fmt(SLUG)}" density="{SLUG_DENSITY - SHELL_DENSITY:g}" contype="0" conaffinity="0" group="3" rgba=".3 .3 .32 1"/>
      {(chr(10) + "      ").join(plates)}
    </body>
  </worldbody>
  <contact>
    {(chr(10) + "    ").join(f'<pair geom1="die_shell" geom2="{geom}" {PAIR}/>' for geom in tray)}
  </contact>
{'  <include file="../robot/panda.xml"/>' + chr(10) if panda else ''}</mujoco>
'''


def manifest(level, panda):
    rows = [f'''[[launcher.protocols]]
name = "{name}"
position = [{", ".join(f"{x:.6f}" for x in pos)}]
quat = [{", ".join(f"{x:.6f}" for x in q)}]
velocity = [{", ".join(f"{x:g}" for x in v)}]
angular_velocity = [{", ".join(f"{x:g}" for x in w)}]''' for name, (pos, q, v, w) in PROTOCOLS.items()]
    return f'''name = "{("panda_small_" if panda else "") + level.split("/")[-1]}"
scene = "scene.xml"
camera = "fixed"
graspable_bodies = ["die"]
object_bodies = ["die"]
object_geoms = ["die_shell", "die_slug"]

# Private real-time launcher (src/sim2sim/realtime.py; generated by experiments/validation/die-scenes.py). Reset k and throw k of
# stream s use Gaussian perturbations drawn from numpy default_rng([seed, s, k]): stream 0 is the agent's session, 1 the
# evaluation trials, 2 the empirical-frequency baseline, 3 the mechanics baseline's observation noise. Protocol names and
# nominal release states are public; the seed and the perturbation sizes are not.
[launcher]
body = "die"
seed = {SEED}
rest = {{ position = [{", ".join(f"{x:.6f}" for x in REST)}], quat = [1, 0, 0, 0], position_m = {REST_PERTURB["position_m"]}, yaw_rad = {REST_PERTURB["yaw_rad"]} }}
perturbation = {{ {", ".join(f"{k} = {v}" for k, v in PERTURB.items())} }}

{(chr(10) + chr(10)).join(rows)}

# Private die evaluation (src/sim2sim/die.py): fresh protocol throws streamed to the agent's predictor in real time.
[die]
deadline_s = {DEADLINE}
trials_per_protocol = 30
empirical_throws = 200
mechanics_noise = {{ position_m = 0.001, angle_rad = 0.02, velocity_m_s = 0.02, angular_velocity_rad_s = 0.5 }}
'''


def outputs():
    return {Path("environments") / level / name: (scene if name == "scene.xml" else manifest)(level, panda)
            for level, panda in LEVELS.items() for name in ("scene.xml", "manifest.toml")}


if __name__ == "__main__":
    print(f"die: mass {MASS * 1000:.2f} g, COM offset {COM.round(5).tolist()} m ({np.linalg.norm(COM) * 1000:.2f} mm), edge {EDGE * 1000:.2f} mm")
    for target, text in outputs().items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        print("wrote", target)
