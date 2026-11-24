"""Write the touch-only tabletop levels (design docs/level-designs/touch-tabletop-20261124.md) and their robot fixture.

environments/panda_shadow/robot: the Franka arm of environments/panda_small/robot/panda.xml (links 0-7, gains raised so the
arm holds the hand) with the unchanged Menagerie Shadow hand subtree of environments/shadow_hand/robot/shadow_hand.xml mounted
on the flange in place of the Panda gripper. Levels (all free rigid objects on the panda_small table; touch only):
  table_shapes   L1: three abstract objects, each matched to one L2 object (bounding box, primitive count, mass, density layout)
  table_objects  L2: an unusually tall mug, a strongly curved banana and a short hammer at the same three poses
  bin_spiky      L3: a static open bin holding one spiky and three smooth balls of equal diameter and mass
Private: manifests (object bodies, labels, prior-defeating dimensions, bin) and table_objects/prior.xml (typical-size mug,
banana and hammer at the true poses: the prior-only baseline). Pure text generation, no MuJoCo:
    python experiments/validation/touch-scenes.py
"""
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sim2sim.primitive_metrics import volume  # noqa: E402
from sim2sim.tabletop import support  # noqa: E402
FAMILY = ROOT / "environments" / "panda_shadow"
PANDA, HAND = ROOT / "environments/panda_small/robot", ROOT / "environments/shadow_hand/robot"
# Arm position gains x ARM_GAIN (damping keeps kv = kp / 10): the 3.6 kg forearm and hand at ~0.3 m past the flange sag the
# stock gains by about a centimetre; x4 keeps the static palm error to a few mm over the workspace (touch-check.py).
ARM_GAIN = 4.0
MOUNT_QUAT = "0.9238795 0 0 -0.3826834"  # the Panda hand's flange mounting (-45 deg about the flange axis), forearm along it

TOP, TABLE = .40, (-.38, .38, -.42, .26)  # table top z and x/y extent, as in panda_small
EYE, TARGET, FOVY = np.array([.62, -.78, .98]), np.array([0., -.12, .45]), 45  # researcher camera (no agent images)
SURFACE = 'friction="0.8 0.005 0.0001" condim="4"'  # mu 0.8: a light touch does not slide the objects; condim 4 resists spin


def fmt(v):
    return " ".join("%.6f" % float(x) for x in np.ravel(v))


def xyaxes(eye, target):
    forward = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(forward, [0, 0, 1])
    x /= np.linalg.norm(x)
    return fmt([*x, *np.cross(x, forward)])


def fixture():
    """panda_shadow.xml text: Panda arm defaults/meshes/links 0-7/actuators 1-7, then the Shadow hand verbatim under link7."""
    panda, hand = ET.parse(PANDA / "panda.xml").getroot(), ET.parse(HAND / "shadow_hand.xml").getroot()
    out = ET.Element("mujocoinclude", model="panda_shadow")
    out.append(hand.find("option"))
    default = ET.SubElement(out, "default")
    arm_class = panda.find("default/default")
    for parent in list(arm_class.iter("default")):  # the gripper's finger joint and fingertip pad classes go with it
        for child in list(parent.findall("default")):
            if child.get("class") == "finger" or child.get("class").startswith("fingertip"):
                parent.remove(child)
    default.append(arm_class)
    default.extend(hand.find("default"))
    asset = ET.SubElement(out, "asset")
    for node in panda.find("asset"):
        if node.tag == "material" and node.get("name") == "off_white" or node.tag == "mesh" and node.get("name", "").startswith("link"):
            asset.append(node)
    asset.extend(hand.find("asset"))
    world = ET.SubElement(out, "worldbody")
    link0 = panda.find("worldbody/body")
    link7 = next(b for b in link0.iter("body") if b.get("name") == "link7")
    gripper = link7.find("body[@name='hand']")
    link7.remove(gripper)
    forearm = hand.find("worldbody/body[@name='rh_forearm']")
    forearm.set("pos", "0 0 0.107")
    forearm.set("quat", MOUNT_QUAT)
    link7.insert(list(link7).index(link7.find("geom")) + 1, forearm)
    world.append(link0)
    contact = ET.SubElement(out, "contact")
    contact.extend(panda.find("contact"))
    contact.extend(hand.find("contact"))
    out.append(hand.find("tendon"))
    actuator = ET.SubElement(out, "actuator")
    for node in panda.find("actuator"):
        if node.get("name") == "actuator8":
            continue
        kp = float(node.get("gainprm")) * ARM_GAIN
        node.set("gainprm", "%g" % kp)
        node.set("biasprm", "0 %g %g" % (-kp, -kp / 10))
        actuator.append(node)
    actuator.extend(hand.find("actuator"))
    ET.indent(out, "  ")
    return ET.tostring(out, encoding="unicode") + "\n"


def scene(level, comment, bodies, assets="", statics=""):
    """A level scene: panda_small table, pedestal and researcher camera, the level's objects, the panda_shadow fixture."""
    return f'''<mujoco model="panda_shadow_{level}">
  <!-- Researcher-only. {comment} -->
  <compiler angle="radian" autolimits="true" meshdir="../robot/assets"/>
  <option timestep="0.002" integrator="implicitfast"><flag multiccd="enable"/></option>
  <visual><map znear="0.01" zfar="12"/></visual>
  <asset>{assets}</asset>
  <worldbody>
    <light name="key" directional="true" pos="-2 -3 4" dir="1 1 -2" diffuse=".85 .82 .76"/>
    <light name="fill" directional="true" pos="2 1 3" dir="-1 -1 -2" diffuse=".3 .36 .45" castshadow="false"/>
    <camera name="fixed" pos="{fmt(EYE)}" xyaxes="{xyaxes(EYE, TARGET)}" fovy="{FOVY}"/>
    <geom name="ground" type="plane" size="3 3 .1" rgba=".70 .72 .73 1"/>
    <geom name="table" type="box" pos="{fmt([(TABLE[0] + TABLE[1]) / 2, (TABLE[2] + TABLE[3]) / 2, TOP - .05])}" size="{fmt([(TABLE[1] - TABLE[0]) / 2, (TABLE[3] - TABLE[2]) / 2, .05])}" {SURFACE} rgba=".65 .52 .37 1"/>
    <geom name="panda_pedestal" type="box" pos="0 -.65 .2" size=".13 .13 .2" rgba=".25 .28 .32 1"/>{statics}{bodies}
  </worldbody>
  <include file="../robot/panda_shadow.xml"/>
</mujoco>
'''


IMPORT = f"""# Franka arm with Shadow hand fixture

Composed from two Google DeepMind MuJoCo Menagerie imports, both under the Apache-2.0 license included as `LICENSE`:

- the compact Franka Emika Panda arm (`franka_emika_panda`, pinned commit `b0655057a9df2abe39ea7fc989914ba6f248301c`): links 0 to 7 with their inertials, joints, collision meshes and visual copies, mounted at
  world `(0, -0.65, 0.4)` facing positive world y. The Panda gripper (hand, fingers, their tendon and equality, actuator8) is
  removed. Arm position gains and damping are {ARM_GAIN:g} times the upstream values (force bounds unchanged), so the arm
  holds the heavier hand with a static error of a few millimetres.
- the Shadow Dexterous Hand E3M5 right hand (`shadow_hand`, pinned commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`): every body, inertial, joint, tendon, actuator, contact exclusion, geom and mesh unchanged,
  with its `<option cone="elliptic" impratio="10"/>`. Its forearm is attached to the arm's flange where the Panda hand was
  (`link7`-local `(0, 0, 0.107)`, rotated -45 degrees about the flange axis), the forearm axis along the flange axis.

Actions: `arm_target` drives actuators `actuator1` to `actuator7` (Panda joints 1 to 7); `hand_target` drives the 20 hand
actuators. A candidate project copies the public `robot/` directory unchanged, includes `robot/panda_shadow.xml` once from
its root `scene.xml`, and sets `<compiler angle="radian" autolimits="true" meshdir="robot/assets"/>`. Target-object meshes are
written inline (`<mesh name="..." vertex="x y z ..."/>`, collided as the convex hull of the vertices).

The sibling scenes in this repository hold private target objects and are not part of the public fixture.
"""


def write_fixture():
    robot = FAMILY / "robot"
    if robot.exists():
        shutil.rmtree(robot)
    (robot / "assets").mkdir(parents=True)
    (robot / "panda_shadow.xml").write_text(fixture())
    for source in [*sorted((PANDA / "assets").glob("link*")), *sorted((HAND / "assets").glob("*"))]:
        shutil.copyfile(source, robot / "assets" / source.name)
    shutil.copyfile(HAND / "LICENSE", robot / "LICENSE")
    (robot / "IMPORT.md").write_text(IMPORT)


# ---- Objects. Each is a list of geoms (name, kind, MJCF size, body-frame position, rotation matrix, density, mesh vertices);
# a body frame has its origin on the table under the bounding-box centre (recentred by place()).

def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def along(a, b):
    """Centre, rotation (z along a->b) and half length of a capsule/cylinder from a to b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    z = (b - a) / np.linalg.norm(b - a)
    x = np.cross([0, 1, 0] if abs(z[1]) < .9 else [1, 0, 0], z)
    x /= np.linalg.norm(x)
    return (a + b) / 2, np.column_stack([x, np.cross(z, x), z]), np.linalg.norm(b - a) / 2


def quat(m):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(m).as_quat()
    return np.array([w, x, y, z])


def rod(name, kind, radius, a, b, density):
    centre, rotation, half = along(a, b)
    return (name, kind, [radius, half], centre, rotation, density, None)


def box(name, half, pos, density, angle=0.):
    return (name, "box", list(half), np.asarray(pos, float), rot_z(angle), density, None)


def disc(name, radius, half, z, density, x=0.):
    return (name, "cylinder", [radius, half], np.array([x, 0, z]), np.eye(3), density, None)


def primitives(geoms):
    """primitive_metrics-style dicts (kind, size, pos, rotation) of body-frame geoms."""
    from scipy.spatial import ConvexHull
    return [dict(name=n, kind=k, size=ConvexHull(v) if k == "mesh" else np.pad(np.asarray(z, float), (0, 3 - len(z))), pos=p, rotation=r)
            for n, k, z, p, r, d, v in geoms]


def bounds(geoms):
    """Body-frame bounding box (low, high) of geoms."""
    prims, axes = primitives(geoms), np.eye(3)
    return np.array([-max(support(p, -a) for p in prims) for a in axes]), np.array([max(support(p, a) for p in prims) for a in axes])


def mass(geoms):
    return sum(d * volume(p["kind"], p["size"]) for p, (*_, d, _v) in zip(primitives(geoms), geoms))


def place(geoms, reference=None):
    """Shift geoms so the bounding box (of reference, default geoms) is centred on the body origin in x/y and rests on z = 0."""
    low, high = bounds(geoms if reference is None else reference)
    shift = -np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]])
    return [(n, k, z, p + shift, r, d, v) for n, k, z, p, r, d, v in geoms]


CERAMIC, BANANA, WOOD, STEEL = 2300., 950., 600., 7800.  # kg/m^3


def mug(height):
    """Hollow mug: a 12-sided wall of 4 mm boxes (75 mm across the flats), a 6 mm base disc and a three-capsule handle (+x)."""
    radius, wall, sides = .0375, .004, 12
    half_width = radius * np.tan(np.pi / sides)
    geoms = [disc("mug_base", radius - wall, .003, .003, CERAMIC)]
    for k in range(sides):
        angle = 2 * np.pi * k / sides
        geoms.append(box(f"mug_wall{k}", (wall / 2, half_width, height / 2),
                         [(radius - wall / 2) * np.cos(angle), (radius - wall / 2) * np.sin(angle), height / 2], CERAMIC, angle))
    top, low, out = .79 * height, .29 * height, radius + .0245  # handle attachments scale with the height
    for name, a, b in (("upper", (radius - .003, 0, top), (out, 0, top)), ("side", (out, 0, top), (out, 0, low)),
                       ("lower", (out, 0, low), (radius - .003, 0, low))):
        geoms.append(rod(f"mug_handle_{name}", "capsule", .005, a, b, CERAMIC))
    return geoms


def banana(curvature_radius, length=.190, radius=.0175, segments=6):
    """Six capsules along a circular arc (curvature radius, arc length) lying on the table, bowed toward +y."""
    phi = np.linspace(-length / curvature_radius / 2, length / curvature_radius / 2, segments + 1)
    points = np.column_stack([curvature_radius * np.sin(phi), curvature_radius * (1 - np.cos(phi)), np.full(len(phi), radius)])
    return [rod(f"banana_{k}", "capsule", radius, points[k], points[k + 1], BANANA) for k in range(segments)]


def hammer(handle):
    """Steel head 25 x 100 x 25 mm (long axis y) and a 22 mm wooden handle of the given exposed length along +x, lying flat."""
    return [box("hammer_head", (.0125, .05, .0125), (0, 0, .0125), STEEL),
            rod("hammer_handle", "cylinder", .011, (0, 0, .011), (.0125 + handle, 0, .011), WOOD)]


# L2 truth and the prior-only baseline: typical mug height, banana curvature and hammer handle length (all else equal), placed
# with the true object's shift, so the mug base, the banana's arc midpoint and the hammer head stay where the true ones are.
TRUE, TYPICAL = {"mug": mug(.120), "banana": banana(.090), "hammer": hammer(.160)}, {"mug": mug(.095), "banana": banana(.160), "hammer": hammer(.280)}
L2 = {name: place(geoms) for name, geoms in TRUE.items()}
PRIOR = {name: place(TYPICAL[name], TRUE[name]) for name in TRUE}


def tower(wall):
    """L1 match of the mug: three hollow pentagonal tubes stacked with x offsets (a stepped tower) on a base disc, capped."""
    rho, step, tube = .0394, .0156, .040  # vertex radius: 75 mm across in y; offsets widen x to the mug's 104 mm
    apothem = rho * np.cos(np.pi / 5)
    geoms = [disc("tower_base", apothem - wall, .003, .003, CERAMIC), disc("tower_cap", apothem - wall, .003, 3 * tube - .003, CERAMIC, 2 * step)]
    for level in range(3):
        for k in range(5):
            angle = np.pi / 5 + 2 * np.pi * k / 5  # wall normals between the vertices; a vertex points along +x
            geoms.append(box(f"tower_wall{level}{k}", (wall / 2, rho * np.sin(np.pi / 5), tube / 2),
                             [level * step + (apothem - wall / 2) * np.cos(angle), (apothem - wall / 2) * np.sin(angle), (level + .5) * tube],
                             CERAMIC, angle))
    return place(geoms)


def zigzag(density, segment=.036, bend=np.radians(39.4), radius=.0175):
    """L1 match of the banana: five capsules in a staircase with alternating bends (+39.4, -39.4, ... degrees) on the table."""
    heading, points = [0, bend, 0, bend, 0], [np.array([0., 0, radius])]
    for h in heading:
        points.append(points[-1] + segment * np.array([np.cos(h), np.sin(h), 0]))
    return place([rod(f"zigzag_{k}", "capsule", radius, points[k], points[k + 1], density) for k in range(5)])


def cross():
    """L1 match of the hammer: a light 22 mm square bar crossed at 60 degrees through its middle by a dense steel bar
    (the hammer head's size and density), a light sphere on one end of the light bar."""
    return place([box("cross_dense", (.0125, .05, .0125), (0, 0, .0125), STEEL, -np.pi / 6),
                  box("cross_bar", (.0825, .011, .011), (0, 0, .011), WOOD),
                  ("cross_ball", "sphere", [.0125], np.array([.0825, 0, .0125]), np.eye(3), WOOD, None)])


def tower_matching_mass():
    low, high = .002, .012
    for _ in range(50):
        wall = (low + high) / 2
        low, high = (wall, high) if mass(tower(wall)) < mass(L2["mug"]) else (low, wall)
    return tower(wall)


L1 = {"tower": tower_matching_mass(), "zigzag": zigzag(BANANA * mass(L2["banana"]) / mass(zigzag(BANANA))), "cross": cross()}
MATCH = {"tower": "mug", "zigzag": "banana", "cross": "hammer"}
# Shared poses (x, y on the table, yaw): >= 8 cm apart (touch-check.py), all within the palm-down workspace.
POSES = {"mug": (-.20, -.20, np.radians(120)), "banana": (0., .03, np.radians(15)), "hammer": (.19, -.22, np.radians(75))}
LABELS = {"mug": ["mug", "cup", "tankard", "stein"], "banana": ["banana", "plantain"], "hammer": ["hammer", "mallet"]}

# L3 (user decision 2026-11-24 after the discriminability check failed with 14 small cones): one spiky ball with six octahedral
# cones (20 mm proud, 29 mm base: at least one palm taxel cell, 86/3 mm) on a 22 mm core, and three smooth balls without a size or
# dynamics cue: equal top height (the spiky ball rests on three tips with three tips up, so its height is twice the tip-facet
# distance), equal mass and the spiky ball's compiled inertia (octahedral symmetry makes it isotropic) as explicit inertials.
BIN = np.array([0., -.12])
# Walls 30 mm (above the 27.5 mm ball radius): with the palm down the Shadow forearm hangs ~4 cm below the palm face, so
# 80 mm walls kept the palm 3-12 cm above every ball (jobs 63319921, 63320350); at <= 40 mm a pitched wrist reaches the tops.
BIN_HALF, BIN_WALL, BIN_HEIGHT, BIN_FLOOR = .12, .01, .03, .01  # inner half width, wall thickness, wall height above floor
FLOOR_Z = TOP + BIN_FLOOR
RIM_Z = FLOOR_Z + BIN_HEIGHT
CORE_R, BALL_MASS, SPIKE_BASE, SPIKE_HEIGHT = .022, .060, .0145, .020  # SPIKE_BASE: cone base radius on the core
SPIKE_DIRECTIONS = np.vstack([np.eye(3), -np.eye(3)])
SPIKES = len(SPIKE_DIRECTIONS)
TIP_R = CORE_R + SPIKE_HEIGHT
SMOOTH_R = TIP_R / np.sqrt(3)  # tip-facet distance of the octahedron: the spiky ball's resting centre height and half its height
assert SMOOTH_R > CORE_R + .002 and SPIKE_BASE < CORE_R
SEED = 20261124
RUBBER = 'friction="0.8 0.005 0.001" condim="6"'  # rolling friction stops a nudged smooth ball within a few cm


def spikes(rng):
    """Cone vertex sets (24-gon base on the core + apex) along the octahedral directions under a seeded random rotation."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    directions = SPIKE_DIRECTIONS @ rotation.T
    cones, angle = [], 2 * np.pi * np.arange(24) / 24
    for d in directions:
        u = np.cross(d, [1, 0, 0] if abs(d[0]) < .9 else [0, 1, 0])
        u /= np.linalg.norm(u)
        base = d * np.sqrt(CORE_R ** 2 - SPIKE_BASE ** 2)
        ring = base + SPIKE_BASE * (np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * np.cross(d, u))
        cones.append(np.vstack([ring, d * TIP_R]))
    return cones


def resting(cones):
    """Rotation that sets the spiky ball on three spike tips (an acute hull facet whose plane clears the sphere), and the
    centre height above the floor."""
    from scipy.spatial import ConvexHull
    tips = np.array([c[-1] for c in cones])
    hull = ConvexHull(tips)
    best = None
    for simplex, equation in zip(hull.simplices, hull.equations):
        a, b, c = tips[simplex]
        acute = all(np.dot(p - q, r - q) > 0 for q, p, r in ((a, b, c), (b, c, a), (c, a, b)))
        if acute and -equation[3] > CORE_R + .002 and (best is None or -equation[3] < best[1]):
            best = (equation[:3], -equation[3])
    normal, height = best
    assert abs(height - SMOOTH_R) < 1e-9, (height, SMOOTH_R)
    v, c = np.cross(normal, [0, 0, -1]), normal @ [0, 0, -1]
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c), height


def balls():
    """Seeded ball centres on the bin floor (>= 5 mm from the walls and each other, spike tips included) and the spiky index."""
    rng = np.random.default_rng(SEED)
    cones = spikes(rng)
    rotation, height = resting(cones)
    reach = TIP_R
    while True:  # rejection sampling of all four centres at once
        centres = BIN + rng.uniform(-(BIN_HALF - reach - .005), BIN_HALF - reach - .005, (4, 2))
        if min(np.linalg.norm(a - b) for i, a in enumerate(centres) for b in centres[:i]) > 2 * reach + .005:
            break
    return cones, rotation, height, centres, int(rng.integers(4))


def geoms_xml(geoms, surface=SURFACE, rgba=".78 .56 .28 1"):
    return "".join(f'\n      <geom name="{n}" type="{k}" size="{fmt(z)}" pos="{fmt(p)}" quat="{fmt(quat(r))}" density="{d:g}" {surface} rgba="{rgba}"/>'
                   if k != "mesh" else f'\n      <geom name="{n}" type="mesh" mesh="{n}" density="{d:g}" {surface} rgba="{rgba}"/>'
                   for n, k, z, p, r, d, v in geoms)


def body_xml(name, geoms, x, y, yaw, z=TOP, rotation=None, surface=SURFACE, rgba=".78 .56 .28 1", inertial=""):
    r = rot_z(yaw) if rotation is None else rotation
    return f'\n    <body name="{name}" pos="{fmt([x, y, z])}" quat="{fmt(quat(r))}">\n      <freejoint name="{name}"/>{inertial}{geoms_xml(geoms, surface, rgba)}\n    </body>'


def table_level(objects, poses):
    return "".join(body_xml(name, geoms, *poses[name]) for name, geoms in objects.items())


def compiled_inertia(geoms, assets):
    """Mass, body-frame COM and principal inertia MuJoCo compiles for one free body of these geoms."""
    import mujoco
    m = mujoco.MjModel.from_xml_string(f'<mujoco><asset>{assets}</asset><worldbody>{body_xml("b", geoms, 0, 0, 0, 0)}</worldbody></mujoco>')
    return float(m.body_mass[1]), m.body_ipos[1].copy(), m.body_inertia[1].copy()


def bin_level():
    cones, rotation, height, centres, spiky = balls()
    smooth_density = BALL_MASS / (4 / 3 * np.pi * SMOOTH_R ** 3)  # nominal; the explicit inertial sets the mass
    cone_geoms = [(f"spike{k}", "mesh", None, np.zeros(3), np.eye(3), 0., c) for k, c in enumerate(cones)]
    spiky_density = BALL_MASS / (volume("sphere", [CORE_R, 0, 0]) + sum(volume("mesh", p["size"]) for p in primitives(cone_geoms)))
    assets = "".join(f'\n    <mesh name="spike{k}" vertex="{fmt(c)}"/>' for k, c in enumerate(cones))
    spiky_parts = lambda name: [(name + "_core", "sphere", [CORE_R], np.zeros(3), np.eye(3), spiky_density, None)] + [  # noqa: E731
        (n, k, z, p, r, spiky_density, v) for n, k, z, p, r, d, v in cone_geoms]
    mass_, com, inertia = compiled_inertia(spiky_parts("b"), assets)
    assert abs(mass_ - BALL_MASS) < 1e-4 * BALL_MASS and np.linalg.norm(com) < 1e-6 and np.ptp(inertia) < 1e-3 * inertia.mean(), (mass_, com, inertia)
    inertial = f'\n      <inertial pos="0 0 0" mass="{mass_!r}" diaginertia="{fmt(np.full(3, inertia.mean()))}"/>'
    bodies = ""
    for i, (x, y) in enumerate(centres):
        name = f"ball{i + 1}"
        if i == spiky:
            bodies += body_xml(name, spiky_parts(name), x, y, 0, FLOOR_Z + height, rotation, RUBBER, ".55 .58 .64 1")
        else:
            bodies += body_xml(name, [(name + "_core", "sphere", [SMOOTH_R], np.zeros(3), np.eye(3), smooth_density, None)], x, y, 0,
                               FLOOR_Z + SMOOTH_R, None, RUBBER, ".55 .58 .64 1", inertial)
    walls = [("bin_floor", (BIN_HALF + BIN_WALL, BIN_HALF + BIN_WALL, BIN_FLOOR / 2), (0, 0, TOP + BIN_FLOOR / 2))]
    for name, (sx, sy) in (("bin_wall_xp", (1, 0)), ("bin_wall_xn", (-1, 0)), ("bin_wall_yp", (0, 1)), ("bin_wall_yn", (0, -1))):
        half = (BIN_WALL / 2, BIN_HALF + BIN_WALL, BIN_HEIGHT / 2) if sx else (BIN_HALF + BIN_WALL, BIN_WALL / 2, BIN_HEIGHT / 2)
        walls.append((name, half, ((sx or 0) * (BIN_HALF + BIN_WALL / 2), (sy or 0) * (BIN_HALF + BIN_WALL / 2), FLOOR_Z + BIN_HEIGHT / 2)))
    statics = "".join(f'\n    <geom name="{n}" type="box" size="{fmt(h)}" pos="{fmt(np.add(p, [*BIN, 0]))}" {SURFACE} rgba=".30 .45 .38 1"/>'
                      for n, h, p in walls)
    return bodies, assets, statics, centres, spiky, cones


def manifest(level, bodies, geoms, extra):
    return f'''name = "panda_shadow_{level}"
scene = "scene.xml"
camera = "fixed"
graspable_bodies = []

# Private metric selection (generated by experiments/validation/touch-scenes.py); the robot fixture is excluded.
object_bodies = [{", ".join(f'"{b}"' for b in bodies)}]
object_geoms = [{", ".join(f'"{g}"' for g in geoms)}]

[reconstruction]
enabled = true
samples = 2048
seed = 20261124
{extra}'''


def write_levels():
    for level, objects, poses, extra in (
            ("table_objects", L2, POSES, "\n[tabletop.labels]\n" + "".join(f"{b} = [{', '.join(repr(s).replace(chr(39), chr(34)) for s in v)}]\n" for b, v in LABELS.items())
             + "".join(f'\n[[tabletop.prior_dimensions]]\nname = "{n}"\nbody = "{b}"\naxis = [{fmt(a).replace(" ", ", ")}]\n' for n, b, a in (
                 ("mug_height", "mug", [0, 0, 1]), ("banana_bow_depth", "banana", rot_z(POSES["banana"][2]) @ [0, 1, 0]),
                 ("hammer_length", "hammer", rot_z(POSES["hammer"][2]) @ [1, 0, 0])))),
            ("table_shapes", L1, {name: POSES[MATCH[name]] for name in L1}, "")):
        directory = FAMILY / level
        directory.mkdir(parents=True, exist_ok=True)
        what = ", ".join(f"{n} ({mass(g) * 1000:.0f} g)" for n, g in objects.items())
        (directory / "scene.xml").write_text(scene(level, f"Touch-only tabletop level {level}: {what} on the table.", table_level(objects, poses)))
        (directory / "manifest.toml").write_text(manifest(level, objects, [g[0] for geoms in objects.values() for g in geoms], extra))
    (FAMILY / "table_objects" / "prior.xml").write_text(scene("table_objects_prior", "Prior-only baseline candidate: typical-size mug "
                                                             "(95 mm tall), banana (160 mm curvature radius) and hammer (280 mm handle) at the true poses.",
                                                             table_level(PRIOR, POSES)))
    bodies, assets, statics, centres, spiky, cones = bin_level()
    names = [f"ball{i + 1}" for i in range(4)]
    geoms = [f"{n}_core" for n in names] + [f"spike{k}" for k in range(SPIKES)]
    (FAMILY / "bin_spiky").mkdir(exist_ok=True)
    (FAMILY / "bin_spiky" / "scene.xml").write_text(scene("bin_spiky", f"Touch-only bin level: a static open bin (inner {2000 * BIN_HALF:g} mm square, "
                                                          f"walls {1000 * BIN_HEIGHT:g} mm) with four {1000 * BALL_MASS:g} g balls of equal height and inertia: "
                                                          f"{names[spiky]} is a {2000 * CORE_R:g} mm core with {SPIKES} cones ({1000 * SPIKE_HEIGHT:g} mm "
                                                          f"proud, {2000 * SPIKE_BASE:g} mm base), the others {2000 * SMOOTH_R:.1f} mm smooth spheres.",
                                                          bodies, assets, statics))
    (FAMILY / "bin_spiky" / "manifest.toml").write_text(manifest("bin_spiky", names, geoms, f'''
[tabletop]
spiky = "{names[spiky]}"
bin = {{ center = [{BIN[0]:g}, {BIN[1]:g}], inner_half = {BIN_HALF:g}, floor_z = {FLOOR_Z:g}, rim_z = {RIM_Z:g} }}
'''))


if __name__ == "__main__":
    write_fixture()
    write_levels()
    for name, geoms in {**L2, **L1}.items():
        low, high = bounds(geoms)
        print(f"{name}: {len(geoms)} primitives, {1000 * mass(geoms):.1f} g, bbox {np.round(1000 * (high - low), 1).tolist()} mm")
    for name, match in MATCH.items():
        size, true = (np.subtract(*bounds(L1[name])[::-1]), np.subtract(*bounds(L2[match])[::-1]))
        ratio = [mass(L1[name]) / mass(L2[match]), *(size / true)]
        assert abs(len(L1[name]) - len(L2[match])) <= 1 and np.all(np.abs(np.array(ratio) - 1) <= .10), (name, ratio)
