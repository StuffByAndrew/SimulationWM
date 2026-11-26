"""Write the childproof bottle levels (2026-11-25, docs/level-designs/childproof-bottle-20261125.md):
environments/bottle_{empty,contents}/{scene.xml, manifest.toml}. Deterministic; run from the repository root:
.venv/bin/python experiments/validation/bottle-scenes.py

A free neutral-grey bottle (hollow: base, wall, shoulder, neck; a collar under the cap hides the gap) stands on the table.
Its cap is a push-and-turn bayonet made only of plain-MJCF primitive contacts and joint springs, all inside the opaque
cap shell:
- `cap` (free, hidden): the inner cap. Two hooks inside its skirt sit under two lugs on the neck. A spring washer (`liner`,
  child slide joint, preload LINER_PRELOAD_N) on the neck rim pushes the cap up, so at rest each hook sits in a notch
  (NOTCH_M deep) in its lug's underside: the cap cannot turn until it is pushed down out of the notch (a detent; without it
  a fast spin of the bottle would turn the light inner cap by inertia).
- `cap_shell` (graspable, the visible cap): child of the inner cap on a frictionless hinge and a slide joint whose spring
  holds it up with preload SHELL_PRELOAD_N. Pushing it down past the preload makes its hidden teeth mesh with teeth on the
  inner cap's top (explicit contact pairs: parent-child contacts are otherwise filtered); turning the engaged shell turns the
  inner cap. Unengaged, the shell spins freely and turns nothing.
- After the hooks leave the lugs (turned by OPEN_DEG_APPROX or more, a quarter turn opens it) the cap lifts off.
The contents variant adds three loose objects of distinct shapes and colours inside the bottle (CONTENTS, ~14 g); bottle and
cap are byte-identical across variants. Graspable: bottle and cap shell only."""
from pathlib import Path

import numpy as np

ROOT = Path("environments")
VARIANTS = {"bottle_empty": False, "bottle_contents": True}
TIMESTEP = .001                  # s: stiff contacts and stops on gram-scale parts need timeconst 2 ms = 2 steps
Z0 = .40                         # table top, m; the bottle frame's origin is its base centre on the table
GREY_BOTTLE, GREY_CAP = ".62 .63 .64 1", ".50 .51 .52 1"  # neutral look (user: no medicine colours or labels)
HIDDEN = ".35 .36 .37 1"         # hidden parts: never visible while closed (the check proves it)
RHO_PLASTIC, RHO_STEEL = 1100., 7800.   # kg/m^3: bottle and cap parts; the spring washer
FRICTION = ".5 .005 .0001"
STIFF = 'solref=".002 1" solimp=".95 .99 .001 .5 2"'  # every contact, pair and stop: the default 20 ms time constant lets a 5 g part sink ~10 mm under a few N
# Bottle (bottle-frame z, m): base, wall, shoulder, neck, collar (radii in, out)
BASE = (.0225, .003)
WALL = (.0205, .0225, BASE[1], .072)
SHOULDER = (.012, .0225, .072, .076)
NECK = (.012, .0145, .076, .094)            # neck rim top at .094; opening diameter 24 mm
COLLAR = (.0195, .0225, .076, .087)         # hides the gap under the shell skirt
LUG_R = (.0140, .0167)                      # lugs at 0 and 180 deg, overlapping the neck wall
LUG_DEG, NOTCH_DEG = 36., 19.               # lug half width; notch half width (the hook sits in it at rest)
LUG_Z, NOTCH_M = (.086, .090), .001         # lug underside in the notch, top; the lug outside the notch reaches NOTCH_M lower
# Inner cap (bottle-frame z at rest)
CAP_DISC = (.0187, .096, .099)
CAP_SKIRT = (.0172, .0187, .083, .096)
PRELOAD_OVERLAP = 3e-5                      # m: preloaded contacts and the shell's top stop start this far engaged, so the
                                            # springs push on active constraints from the first step (exact touching is inactive)
HOOK_R, HOOK_DEG = (.0150, .0175), 14.
HOOK_Z = (.084 + PRELOAD_OVERLAP, .086 + PRELOAD_OVERLAP)   # hook top in the notch underside at rest
TEETH_R, TOOTH_DEG, TEETH = (.008, .014), 7., 6   # 6 inner teeth at k*60 deg, 6 shell teeth at 30 + k*60 deg
INNER_TOOTH_Z, TOOTH_GAP, SHELL_TOOTH_M = (.099, .101), .001, .0025
LINER = (.0140, .001)                       # spring washer radius and thickness, resting on the rim
LINER_K, LINER_PRELOAD_N, LINER_TRAVEL = 500., 4., .0018   # N/m; preload at contact; cap descent until its stop
SHELL_K, SHELL_PRELOAD_N = 400., 7.6        # N/m; preload at the top stop (engage at ~8.0 N, full stop ~8.7 N)
SHELL_TRAVEL = TOOTH_GAP + .0018            # shell stop: teeth overlap 1.8 mm
SHELL_R, SHELL_SKIRT_BOTTOM = (.0235, .0255), .083
SHELL_DISC_Z = INNER_TOOTH_Z[1] + TOOTH_GAP + SHELL_TOOTH_M   # shell disc bottom; disc 2 mm thick
LINER_DAMPING, SHELL_DAMPING = .5, .2       # N s/m on the slide joints (settles the springs)
OPEN_DEG_APPROX = LUG_DEG + HOOK_DEG        # the hooks clear the lugs past about this turn
CONTENTS = [  # name, type, size, bottle-frame position, density kg/m^3, rgba
    ("cube", "box", (.006, .006, .006), (.007, .007, BASE[1] + .006), 2700., ".80 .18 .15 1"),
    ("rod", "cylinder", (.004, .010), (-.009, 0., BASE[1] + .010), RHO_STEEL, ".18 .32 .78 1"),
    ("ball", "sphere", (.005,), (.008, -.010, BASE[1] + .005), 2500., ".90 .78 .15 1"),
]
EYE, LOOK, FOVY = np.array([.17, -.24, .60]), np.array([0., 0., Z0 + .05]), 45
# Private [bottle] metric table (src/sim2sim/bottle.py): push depths of the cap grasp (m; 0 = twist only, then below and above
# the engage threshold: the check reports the push force at each), turns (deg), lift (m), source ticks (TIMESTEP).
PROBES = {"push_m": [0., .002, .004, .005, .006, .008], "one_grasp_push_m": .008, "turn_deg": 90.,
          "angles_deg": [30., 40., 50., 60., 70., 80., 90., 120.], "lift_m": .02, "push_ticks": 300, "turn_ticks": 500,
          "lift_ticks": 300, "settle_ticks": 300, "open_lift_m": .01, "open_separation_m": .01}


def f(v):
    return " ".join(f"{float(x):.6f}" for x in np.atleast_1d(v))


def box(name, center, half, yaw, attrs):
    return (f'<geom name="{name}" type="box" pos="{f(center)}" size="{f(half)}" euler="0 0 {yaw:.9f}" {attrs}/>')


def arc(name, r, z, a0, a1, n, attrs, dz=0.):
    """n boxes covering the annular sector r=(in, out), z=(low, high) between angles a0..a1 (rad); dz shifts z (body frame)."""
    step = (a1 - a0) / n
    half = [(r[1] - r[0]) / 2, r[1] * np.tan(step / 2) * 1.02, (z[1] - z[0]) / 2]
    rc = (r[0] + r[1]) / 2
    out = []
    for i in range(n):
        a = a0 + (i + .5) * step
        out.append(box(f"{name}_{i}", [rc * np.cos(a), rc * np.sin(a), (z[0] + z[1]) / 2 - dz], half, a, attrs))
    return out


def ring(name, r_in, r_out, z0, z1, n, attrs, dz=0.):
    return arc(name, (r_in, r_out), (z0, z1), 0., 2 * np.pi, n, attrs, dz)


def xyaxes(eye, target):
    forward = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(forward, [0, 0, 1]); x /= np.linalg.norm(x)
    return f([*x, *np.cross(x, forward)])


def hidden_parts():
    """Bottle-body geom names that only the open bottle shows (neck and lugs); the inner cap, liner and teeth are hidden too."""
    return ([f"neck_{i}" for i in range(24)] + [f"lug{k}_{p}_{i}" for k in range(2) for p, n in (("a", 2), ("n", 2), ("b", 2))
                                                 for i in range(n)])


def scene(contents):
    plastic = f'density="{RHO_PLASTIC:g}" friction="{FRICTION}"'
    bottle = [f'<geom name="base" type="cylinder" pos="0 0 {BASE[1] / 2:.6f}" size="{BASE[0]:.6f} {BASE[1] / 2:.6f}" {plastic} rgba="{GREY_BOTTLE}"/>']
    bottle += ring("wall", *WALL, 32, f'{plastic} rgba="{GREY_BOTTLE}"')
    bottle += ring("shoulder", *SHOULDER, 32, f'{plastic} rgba="{GREY_BOTTLE}"')
    bottle += ring("collar", *COLLAR, 32, f'{plastic} rgba="{GREY_BOTTLE}"')
    bottle += ring("neck", *NECK, 24, f'{plastic} rgba="{HIDDEN}"')
    lug, notch = np.deg2rad(LUG_DEG), np.deg2rad(NOTCH_DEG)
    for k in range(2):  # each lug: outer parts reach NOTCH_M lower than its middle (the notch the hook rests in)
        c = k * np.pi
        low = (LUG_Z[0] - NOTCH_M, LUG_Z[1])
        bottle += arc(f"lug{k}_a", LUG_R, low, c - lug, c - notch, 2, f'{plastic} rgba="{HIDDEN}"')
        bottle += arc(f"lug{k}_n", LUG_R, LUG_Z, c - notch, c + notch, 2, f'{plastic} rgba="{HIDDEN}"')
        bottle += arc(f"lug{k}_b", LUG_R, low, c + notch, c + lug, 2, f'{plastic} rgba="{HIDDEN}"')
    rim = NECK[3]
    cap_z = rim  # the inner cap's frame origin: on the axis at the neck rim
    hook = np.deg2rad(HOOK_DEG)
    cap = [f'<geom name="cap_disc" type="cylinder" pos="0 0 {(CAP_DISC[1] + CAP_DISC[2]) / 2 - cap_z:.6f}" '
           f'size="{CAP_DISC[0]:.6f} {(CAP_DISC[2] - CAP_DISC[1]) / 2:.6f}" {plastic} rgba="{HIDDEN}"/>']
    cap += ring("cap_skirt", *CAP_SKIRT, 24, f'{plastic} rgba="{HIDDEN}"', cap_z)
    for k in range(2):
        cap += arc(f"hook{k}", HOOK_R, HOOK_Z, k * np.pi - hook, k * np.pi + hook, 2, f'{plastic} rgba="{HIDDEN}"', cap_z)
    tooth = np.deg2rad(TOOTH_DEG)
    teeth = 'contype="0" conaffinity="0" ' + plastic + f' rgba="{HIDDEN}"'
    for k in range(TEETH):
        cap += arc(f"cap_tooth{k}", TEETH_R, INNER_TOOTH_Z, k * np.pi / 3 - tooth, k * np.pi / 3 + tooth, 1, teeth, cap_z)
    shell = [f'<geom name="shell_top" type="cylinder" pos="0 0 {SHELL_DISC_Z + .001 - cap_z:.6f}" size="{SHELL_R[1]:.6f} .001" '
             f'{plastic} rgba="{GREY_CAP}"/>']
    shell += ring("shell_skirt", SHELL_R[0], SHELL_R[1], SHELL_SKIRT_BOTTOM, SHELL_DISC_Z, 32, f'{plastic} rgba="{GREY_CAP}"', cap_z)
    for k in range(TEETH):
        a = np.pi / 6 + k * np.pi / 3
        shell += arc(f"shell_tooth{k}", TEETH_R, (SHELL_DISC_Z - SHELL_TOOTH_M, SHELL_DISC_Z), a - tooth, a + tooth, 1, teeth, cap_z)
    liner_z = rim + LINER[1] / 2 - PRELOAD_OVERLAP - cap_z
    shell_ref = SHELL_PRELOAD_N / SHELL_K          # slide q = 0 at the top stop; the spring pushes up with the preload there
    liner_ref = -LINER_PRELOAD_N / LINER_K         # liner q = 0 touching the rim; the spring pushes it down with the preload
    pairs = "\n".join(f'    <pair geom1="shell_tooth{i}_0" geom2="cap_tooth{j}_0" condim="3" {STIFF} friction="{FRICTION.split()[0]} '
                      f'{FRICTION.split()[0]} .005 .0001 .0001"/>' for i in range(TEETH) for j in range(TEETH))
    loose = "".join(
        f'    <body name="{name}" pos="{f(np.add(pos, [0, 0, Z0]))}">\n      <freejoint name="{name}_free"/>\n'
        f'      <geom name="{name}" type="{kind}" size="{f(size)}" density="{rho:g}" friction="{FRICTION}" rgba="{rgba}"'
        f'/>\n    </body>\n'
        for name, kind, size, pos, rho, rgba in CONTENTS) if contents else ""
    indent = "\n      "
    return f"""<mujoco model="{'bottle_contents' if contents else 'bottle_empty'}">
  <!-- Researcher-only. Childproof bottle level (experiments/validation/bottle-scenes.py): a free grey bottle (diameter 45 mm,
       87 mm to the collar) with a push-and-turn bayonet cap (diameter 51 mm): the visible shell spins freely on the hidden inner cap
       and is held up by a {SHELL_K:g} N/m spring preloaded to {SHELL_PRELOAD_N:g} N; pushed down past the preload its hidden teeth mesh with the
       inner cap's, so turning it turns the inner cap. The inner cap's two hooks sit in notches under two lugs on the neck, pressed
       up by a {LINER_PRELOAD_N:g} N spring washer; pushed down and turned by about {OPEN_DEG_APPROX:g} degrees or more, the cap lifts off.
       {'Contents: a 12 mm aluminium cube, an 8 x 20 mm steel rod and a 10 mm glass ball, loose inside.' if contents else 'Empty.'} -->
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{TIMESTEP}" integrator="implicitfast"><flag multiccd="enable"/></option>
  <visual><map znear="0.01" zfar="12"/></visual>
  <default><geom {STIFF}/><joint solreflimit=".002 1" solimplimit=".95 .99 .001 .5 2"/></default>
  <worldbody>
    <light name="key" directional="true" pos="-2 -3 4" dir="1 1 -2" diffuse=".85 .82 .76"/>
    <light name="fill" directional="true" pos="2 1 3" dir="-1 -1 -2" diffuse=".3 .36 .45" castshadow="false"/>
    <camera name="fixed" pos="{f(EYE)}" xyaxes="{xyaxes(EYE, LOOK)}" fovy="{FOVY}"/>
    <geom name="ground" type="plane" size="3 3 .1" rgba=".70 .72 .73 1"/>
    <geom name="table" type="box" pos="0 0 {Z0 - .05:.6f}" size=".3 .3 .05" friction=".8 .005 .0001" rgba=".65 .52 .37 1"/>
    <body name="bottle" pos="0 0 {Z0:.6f}">
      <freejoint name="bottle_free"/>
      {indent.join(bottle)}
    </body>
    <body name="cap" pos="0 0 {Z0 + cap_z:.6f}">
      <freejoint name="cap_free"/>
      {indent.join(cap)}
      <body name="liner" pos="0 0 {liner_z:.6f}">
        <joint name="liner" type="slide" axis="0 0 1" range="-.0005 {LINER_TRAVEL:.6f}" stiffness="{LINER_K:g}" springref="{liner_ref:.6f}" damping="{LINER_DAMPING:g}"/>
        <geom name="liner" type="cylinder" size="{LINER[0]:.6f} {LINER[1] / 2:.6f}" density="{RHO_STEEL:g}" friction="{FRICTION}" rgba="{HIDDEN}"/>
      </body>
      <body name="cap_shell">
        <joint name="shell_turn" type="hinge" axis="0 0 1"/>
        <joint name="shell_push" type="slide" axis="0 0 1" range="{-SHELL_TRAVEL:.6f} {-PRELOAD_OVERLAP:.6f}" stiffness="{SHELL_K:g}" springref="{shell_ref:.6f}" damping="{SHELL_DAMPING:g}"/>
        {(indent + '  ').join(shell)}
      </body>
    </body>
{loose}  </worldbody>
  <contact>
{pairs}
  </contact>
</mujoco>
"""


def geoms(text):
    import re
    return re.findall(r'<geom name="([^"]+)"', text)


def manifest(name, contents, text):
    """Private manifest: object bodies and geoms (all of them: the metrics cover hidden parts and contents), graspable
    bodies, reconstruction sampling and the [bottle] metric table (src/sim2sim/bottle.py)."""
    names = [g for g in geoms(text) if g not in ("ground", "table")]
    bodies = ["bottle", "cap"] + [c[0] for c in CONTENTS if contents]
    q = lambda items: "[" + ", ".join(f'"{i}"' for i in items) + "]"  # noqa: E731
    return f"""name = "{name}"
scene = "scene.xml"
camera = "fixed"
graspable_bodies = ["bottle", "cap_shell"]

# Top-level objects (the cap object holds the hidden inner cap, its spring washer and the visible shell).
object_bodies = {q(bodies)}
object_geoms = {q(names)}

[reconstruction]
enabled = true
samples = 2048
seed = 20261125

# Mechanism probes, engage force and opening angle (src/sim2sim/bottle.py): probe pixels at the shell's top centre and the
# bottle wall point facing the fixed camera; interior = [radius, top] of the bottle's inside in its frame (a content
# outside it is tipped out); contents = the content bodies.
[bottle]
shell_point = {f6(shell_point())}
bottle_point = {f6(bottle_point())}
{chr(10).join(f"{k} = {f6(v) if isinstance(v, list) else v}" for k, v in PROBES.items())}
interior = {f6([WALL[0], NECK[3]])}
contents = {q(bodies[2:])}
"""


def f6(values):
    return "[" + ", ".join(f"{float(v):.6f}" for v in values) + "]"


def shell_point():
    """The shell's top centre in its body frame (the inner cap's frame, on the axis at the neck rim)."""
    return [0., 0., SHELL_DISC_Z + .002 - NECK[3]]


def bottle_point():
    """The bottle wall point facing the fixed camera, 40 mm above the base, in the bottle frame."""
    return [*(WALL[1] * EYE[:2] / np.linalg.norm(EYE[:2])), .04]


def outputs():
    out = {}
    for name, contents in VARIANTS.items():
        text = scene(contents)
        out[ROOT / name / "scene.xml"] = text
        out[ROOT / name / "manifest.toml"] = manifest(name, contents, text)
    return out


if __name__ == "__main__":
    for path, text in outputs().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(path)
