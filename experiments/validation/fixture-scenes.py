"""Write the spring-powered positioning-fixture Franka levels (2026-11-23): environments/panda_small/fixture_<pad>/.

A small carriage rides a rail (frictionless slide joint plus light bearing drag). A latched spring plunger behind it
launches it; a brake pad in a closed caliper housing presses on a thin blade that trails behind the carriage, so the pad
acts over the whole travel. Two dials on tracks in front of the rail, each a sliding block with a vertical peg the open
gripper straddles (form closure: no grip force needed), set the spring compression and the pad pressure; each reads 0-10
on a tick scale. A ruler beside the rail (mm, zero at the carriage index when the spring is relaxed) carries two
coloured target bands. Levels differ only in the pad:

- fixture_rigid: a rigid brake shoe. The pad's reset lift settles to the same equilibrium, so releases repeat.
- fixture_rubber: a hinged rubber flap pre-curved backward; it buckles under the holder, its folded tail drags and
  bunches against the moving blade (stick-slip), and how it lands during the reset settle sets its contact state.
- fixture_bimodal: a straight flap with a grippy rubber face on one side and a slick fabric backing on the other.
  Whichever way the small reset tilt buckles it decides which face lies on the blade: two stopping ranges.

Pressure: a fixed tendon spring L = pad_z + s * pad_preload (stiffness k_pad) loads the pad holder. pad_preload is a hidden
nut that the cocking cycle (reset_carriage) moves to the pressure dial's position and latches (equality latch_preload), so
the dials carry no load: MuJoCo's dry friction is a soft constraint and creeps under any sustained load.
Run from the repository root (no physics): .venv/bin/python experiments/validation/fixture-scenes.py"""
from pathlib import Path
import numpy as np

TOP = .40                                  # table top z (panda_small)
YR, X0 = -.15, .02                         # rail centre line y; world x of the ruler zero (carriage index, spring relaxed)
CAR = (.02, .015, .012)                    # carriage block half sizes (40 x 30 x 24 mm)
RAIL_TOP = TOP + .016                      # 16 mm tall rail
ZC = RAIL_TOP + CAR[2] + .0005             # carriage centre (0.5 mm above the rail: the slide joint carries it)
BLADE = (.19, .008, .0025)                 # blade half sizes (5 mm thick: a seating pad cannot tunnel through it)
ZB = ZC + CAR[2] + BLADE[2]                # blade centre z
BLADE_TOP = ZB + BLADE[2]
XP = -.05                                  # pad centre x: under the blade for every carriage position -0.045 .. 0.245 m
PAD = (.015, .007)                         # pad footprint half sizes (30 x 14 mm)
TRAVEL = .245                              # end-bumper joint limit: 245 mm on the ruler
C_MIN, C_MAX = .008, .040                  # spring compression at dial 0 and 10, m
K_SPRING = 300.                            # plunger spring, N/m
M_CAR, M_BLADE, M_PLUNGER = .08, .02, .02  # kg
DIALS = {"compression_dial": -.20, "pressure_dial": -.02}  # dial track start x (reading 0); 0.1 m of travel to reading 10
YD = -.30                                  # dial track centre y (0.35 m in front of the Franka base)
DIAL_FRICTION = 1.5                        # N dry friction on each dial (the robot pushes it; nothing else loads it)
TARGETS = {"target": (70., 100., ".2 .7 .3 1"), "narrow_target": (18., 22., ".95 .45 .1 1")}  # ruler mm, public
TIMESTEP = .002
EYE, LOOK, FOVY = np.array([.55, -.85, .95]), np.array([-.03, -.17, .42]), 45

# Pad variants. shoe: rigid brake shoe friction. flap: a hinged strip (segments, segment length m, thickness m, hinge
# stiffness N m/rad, hinge damping N m s/rad) hanging from the holder's rear (-x) bottom edge, or its front edge with
# front_root (the moving blade then pushes a tucked strip into its root, so it buckles and sticks and slips); springref:
# root hinge rest angle (negative turns the tip forward); mu: sliding friction of its (+x, -x) faces (a strip tucked
# forward lays its -x face on the blade, one tucked backward its +x face); holder_mu: the holder's bare bottom; rock: optional holder pitch hinge
# (stiffness N m/rad, damping). k_pad: pad tendon stiffness N/m; reset: (lift lo m, lift hi m, hinge noise rad) added to
# the written pose; n_range: pad force at dial 0 and 10 at the working height. Blade friction 0.1 is lower than every pad,
# so the pad surface sets each contact's coefficient.
LEVELS = {
    "fixture_rigid": {"shoe": .5, "k_pad": 400., "reset": (.0005, .002, 0.), "n_range": (1.0, 5.0)},
    "fixture_rubber": {"flap": (3, .007, .002, .004, 1e-4), "front_root": True, "springref": .3, "mu": (.9, .9), "holder_mu": .25,
                       "rock": (.05, .002), "k_pad": 400., "reset": (0., .002, .1), "n_range": (.8, 3.2)},
    "fixture_bimodal": {"flap": (3, .010, .002, .004, 1e-4), "springref": .03, "mu": (.25, 1.1), "holder_mu": .3,
                        "rock": None, "k_pad": 250., "reset": (0., .002, .08), "n_range": (1.0, 3.5)},
}
SETTLE_TICKS = 200
RUBBER_SOLREF = ".01 1"
GRID_SEED, EVAL_SEED = 2026112301, 2026112302


def fmt(v):
    return " ".join(f"{float(x):.6f}" for x in v)


def xyaxes(eye, target):
    forward = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(forward, [0, 0, 1]); x /= np.linalg.norm(x)
    return fmt([*x, *np.cross(x, forward)])


def ruler():
    """Ruler bar beside the rail: ticks every 10 mm from -50 to 250 mm (major every 50, red zero), and the target bands."""
    y, top = YR - .03, TOP + .02
    out = [f'<geom name="ruler" type="box" pos="{fmt([X0 + .1, y, top - .01])}" size=".165 .012 .01" rgba=".93 .93 .9 1"/>']
    for mm in range(-50, 251, 10):
        major = mm % 50 == 0
        rgba = ".8 .1 .1 1" if mm == 0 else ".05 .05 .05 1"
        out.append(f'<geom name="tick_{mm}" type="box" pos="{fmt([X0 + mm / 1000, y - .012 + (.006 if major else .003), top + .0003])}" '
                   f'size=".0006 {.006 if major else .003} .0003" rgba="{rgba}" contype="0" conaffinity="0"/>')
    for name, (lo, hi, rgba) in TARGETS.items():
        out.append(f'<geom name="{name}_band" type="box" pos="{fmt([X0 + (lo + hi) / 2000, y + .004, top + .0002])}" '
                   f'size="{(hi - lo) / 2000:.6f} .006 .0002" rgba="{rgba}" contype="0" conaffinity="0"/>')
    return out


def dial(name, x0):
    """A dial track (world) and its slider: reading = joint travel / 10 mm; it starts at reading 5."""
    ticks = [f'<geom name="{name}_tick_{i}" type="box" pos="{fmt([x0 + i / 100, YD - .016, TOP + .0083])}" '
             f'size=".0006 {.004 if i % 5 == 0 else .0025} .0003" rgba=".05 .05 .05 1" contype="0" conaffinity="0"/>' for i in range(11)]
    track = (f'<geom name="{name}_track" type="box" pos="{fmt([x0 + .05, YD, TOP + .004])}" size=".075 .022 .004" rgba=".35 .37 .4 1"/>')
    rgba = ".85 .75 .15 1" if name == "compression_dial" else ".2 .45 .85 1"
    body = f'''<body name="{name}" pos="{fmt([x0 + .05, YD, TOP + .0135])}">
      <joint name="{name}" type="slide" axis="1 0 0" range="0 .1" ref=".05" frictionloss="{DIAL_FRICTION}" damping="3"/>
      <geom name="{name}_base" type="box" size=".012 .012 .0045" mass=".03" rgba="{rgba}"/>
      <geom name="{name}_peg" type="cylinder" pos="0 0 .03" size=".008 .025" mass=".01" rgba="{rgba}"/>
      <geom name="{name}_pointer" type="box" pos="0 -.014 .0047" size=".0012 .004 .0002" rgba=".9 .1 .1 1" contype="0" conaffinity="0" mass="0"/>
    </body>'''
    return [track, *ticks], body


def pad(level):
    """Pad holder (slide z, optional pitch, inside the housing) and the level's pad: (body xml, written holder height above
    the blade top, working height above the blade top)."""
    spec = LEVELS[level]
    contact = 'contype="4" conaffinity="4" condim="3"'
    rock = spec.get("rock")
    joints = ('<joint name="pad_z" type="slide" axis="0 0 1" range="-.04 .03" damping="20"/>'
              + (f'<joint name="pad_pitch" type="hinge" axis="0 1 0" stiffness="{rock[0]}" damping="{rock[1]}" range="-.3 .3"/>' if rock else ""))
    holder = (f'<geom name="pad_holder" type="box" size="{PAD[0]} {PAD[1]} .005" mass=".03" friction="{spec.get("holder_mu", .3)} .005 .0001" '
              f'{contact} rgba=".4 .4 .42 1"/>')
    if "shoe" in spec:
        shoe = (f'<geom name="pad_shoe" type="box" pos="0 0 -.007" size="{PAD[0]} {PAD[1]} .002" mass=".01" friction="{spec["shoe"]} .005 .0001" '
                f'{contact} rgba=".15 .15 .15 1"/>')
        return joints + holder + shoe, .009, .009
    count, h, t, stiffness, damping = spec["flap"]
    mass = PAD[1] * 2 * h * t * 1100  # rubber 1100 kg/m^3
    body = ""
    for i in reversed(range(count)):
        faces = "".join(f'<geom name="pad_{i}_{side}" type="box" pos="{sign * t / 4} 0 {-h / 2}" size="{t / 4} {PAD[1]} {h / 2}" mass="{mass / 2:.7f}" '
                        f'friction="{mu} .005 .0001" solref="{RUBBER_SOLREF}" {contact} rgba="{".1 .1 .1 1" if mu >= .5 else ".2 .3 .6 1"}"/>'
                        for side, sign, mu in (("front", 1, spec["mu"][0]), ("rear", -1, spec["mu"][1])))
        body = (f'<body name="pad_seg_{i}" pos="{f"{(1 if spec.get("front_root") else -1) * (PAD[0] + t):.4f} 0 -.005" if i == 0 else f"0 0 {-h}"}">'
                f'<joint name="pad_hinge_{i}" type="hinge" axis="0 1 0" stiffness="{stiffness}" springref="{spec["springref"] if i == 0 else 0}" '
                f'damping="{damping}" armature="2e-6"/>{faces}{body}</body>')
    return joints + holder + body, .005 + count * h + .001, .005 + t


def scene(level):
    spec = LEVELS[level]
    pad_xml, written, working = pad(level)
    # The holder is written with the shoe touching the blade, or the strip hanging straight with its tip 1 mm above it.
    zh, held = BLADE_TOP + written, written - working
    n0, n1 = spec["n_range"]
    flap = spec.get("flap")
    mass_pad = .03 + (.01 if flap is None else flap[0] * PAD[1] * 2 * flap[1] * flap[2] * 1100)
    # Down force at pad_z = q, dial travel d: k_pad (q + s d - L0); at the working height (q = -held) the pad force is
    # n0 + (n1 - n0) d / 0.1 including the pad's own weight. The dial feels s times the pad force.
    k_pad = spec["k_pad"]
    s_pad = (n1 - n0) / (k_pad * .1)
    l0 = -held - (n0 - mass_pad * 9.81) / k_pad
    tracks, bodies = zip(*(dial(name, x0) for name, x0 in DIALS.items()))
    # Opaque caliper housing around the pad (world geometry; the robot can touch it). It is long enough to cover a strip
    # folded back behind the holder, its side walls come down to just above the carriage, and the blade runs through
    # slots in its end walls (the walls reach 3 mm below the blade top), so no camera pose sees the pad.
    half, top, low = .055, TOP + .104, BLADE_TOP - .0045
    housing = [
        f'<geom name="caliper_top" type="box" pos="{fmt([XP, YR, top + .003])}" size="{half} .022 .003" rgba=".55 .2 .2 1"/>',
        *(f'<geom name="caliper_side_{s}" type="box" pos="{fmt([XP, YR + sgn * .0155, (low + top) / 2])}" '
          f'size="{half} .0065 {(top - low) / 2:.6f}" rgba=".55 .2 .2 1"/>' for s, sgn in (("l", -1), ("r", 1))),
        *(f'<geom name="caliper_{s}" type="box" pos="{fmt([XP + sgn * (half - .0015), YR, (BLADE_TOP - .003 + top) / 2])}" '
          f'size=".0015 .009 {(top - BLADE_TOP + .003) / 2:.6f}" rgba=".55 .2 .2 1"/>' for s, sgn in (("back", -1), ("front", 1))),
    ]
    inner = 'contype="2" conaffinity="2"'
    return f'''<mujoco model="panda_small_{level}">
  <!-- Researcher-only. Positioning fixture "{level}" (experiments/validation/fixture-scenes.py). Spring {K_SPRING:g} N/m,
       compression {C_MIN * 1000:g}-{C_MAX * 1000:g} mm over the compression dial; pad tendon {k_pad:g} N/m, pad force about
       {n0:g}-{n1:g} N over the pressure dial at the working height; pad {spec}. -->
  <compiler angle="radian" autolimits="true" meshdir="../robot/assets"/>
  <option timestep="{TIMESTEP}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual><map znear="0.01" zfar="12"/></visual>
  <default><geom friction=".7 .01 .001"/></default>
  <custom>
    <numeric name="fixture_compression" data="{C_MIN} {(C_MAX - C_MIN) / .1:g}"/>
    <numeric name="fixture_reset" data="{SETTLE_TICKS} {spec["reset"][0]} {spec["reset"][1]} {spec["reset"][2]}"/>
  </custom>
  <worldbody>
    <light name="key" directional="true" pos="-2 -3 4" dir="1 1 -2" diffuse=".85 .82 .76"/>
    <light name="fill" directional="true" pos="2 1 3" dir="-1 -1 -2" diffuse=".3 .36 .45" castshadow="false"/>
    <camera name="fixed" pos="{fmt(EYE)}" xyaxes="{xyaxes(EYE, LOOK)}" fovy="{FOVY}"/>
    <geom name="ground" type="plane" size="3 3 .1" rgba=".70 .72 .73 1"/>
    <geom name="table" type="box" pos="0 -0.08 0.35" size="0.38 0.34 .05" rgba=".65 .52 .37 1"/>
    <geom name="panda_pedestal" type="box" pos="0 -.65 .2" size=".13 .13 .2" rgba=".25 .28 .32 1"/>
    <geom name="rail" type="box" pos="{fmt([.095, YR, TOP + .008])}" size=".215 .01 .008" rgba=".6 .62 .65 1"/>
    <geom name="launcher" type="box" pos="{fmt([-.155, YR, TOP + .018])}" size=".045 .02 .018" rgba=".3 .3 .33 1"/>
    {(chr(10) + "    ").join(ruler() + housing + [g for track in tracks for g in track])}
    <body name="carriage" pos="{fmt([X0, YR, ZC])}">
      <joint name="carriage_x" type="slide" axis="1 0 0" range="-.05 {TRAVEL}" damping=".05" frictionloss=".03" solreflimit=".01 2"/>
      <geom name="carriage" type="box" size="{fmt(CAR)}" mass="{M_CAR}" {inner} rgba=".85 .85 .88 1"/>
      <geom name="blade" type="box" pos="{fmt([-.17, 0, CAR[2] + BLADE[2]])}" size="{fmt(BLADE)}" mass="{M_BLADE}" friction=".1 .005 .0001" contype="4" conaffinity="4" condim="3" rgba=".75 .77 .8 1"/>
      <geom name="carriage_index" type="box" pos="0 {-CAR[1] - .0003} 0" size=".0008 .0003 {CAR[2]}" rgba=".9 .1 .1 1" contype="0" conaffinity="0" mass="0"/>
    </body>
    <body name="plunger" pos="{fmt([X0 - CAR[0] - .03, YR, ZC])}">
      <joint name="plunger_x" type="slide" axis="1 0 0" range="-.05 0" stiffness="{K_SPRING}" springref="0" damping=".02" solreflimit=".005 1"/>
      <geom name="plunger_rod" type="cylinder" euler="0 1.5707963 0" size=".006 .028" mass="{M_PLUNGER * .7}" {inner} rgba=".5 .5 .52 1"/>
      <geom name="plunger_plate" type="box" pos=".028 0 0" size=".002 .008 .008" mass="{M_PLUNGER * .3}" {inner} rgba=".5 .5 .52 1"/>
    </body>
    <body name="pad_preload" pos="{fmt([XP, YR, TOP + .09])}">
      <joint name="pad_preload" type="slide" axis="1 0 0" range="0 .1" ref=".05" damping="1"/>
      <inertial pos="0 0 0" mass=".001" diaginertia="1e-8 1e-8 1e-8"/>
    </body>
    <body name="pad_holder" pos="{fmt([XP, YR, zh])}">
      {pad_xml}
    </body>
    {(chr(10) + "    ").join(bodies)}
  </worldbody>
  <equality>
    <joint name="latch_plunger" joint1="plunger_x" polycoef="0 0 0 0 0" active="false" solref=".004 1"/>
    <joint name="latch_carriage" joint1="carriage_x" polycoef="0 0 0 0 0" active="false" solref=".004 1"/>
    <joint name="latch_preload" joint1="pad_preload" polycoef="0 0 0 0 0" solref=".004 1"/>
  </equality>
  <tendon>
    <fixed name="pad_spring" stiffness="{k_pad}" springlength="{l0:.6f} {l0:.6f}">
      <joint joint="pad_z" coef="1"/>
      <joint joint="pad_preload" coef="{s_pad:.6f}"/>
    </fixed>
  </tendon>
  <include file="../robot/panda.xml"/>
</mujoco>
'''


def grid():
    """Held-out evaluation settings: one seeded point per cell of a 5 x 5 grid over both dials (0.5-9.5)."""
    rng = np.random.default_rng(GRID_SEED)
    return [(round(float(1 + 2 * i + rng.uniform(-.8, .8)), 2), round(float(1 + 2 * j + rng.uniform(-.8, .8)), 2)) for i in range(5) for j in range(5)]


def manifest(level):
    settings = ", ".join(f"[{u}, {v}]" for u, v in grid())
    targets = "".join(f'\n[[fixture.targets]]\nname = "{name}"\nlo_mm = {lo}\nhi_mm = {hi}\n' for name, (lo, hi, _) in TARGETS.items())
    return f'''name = "panda_small_{level}"
scene = "scene.xml"
camera = "fixed"
graspable_bodies = []
object_bodies = ["carriage"]
object_geoms = ["carriage"]

# Private fixture evaluation (src/sim2sim/fixture.py; generated by experiments/validation/fixture-scenes.py). Held-out
# settings [compression, pressure] in dial divisions, each released `repeats` times; each chosen setting `chosen_repeats`
# times; the predictor gives `samples` stopping positions per query. Trial perturbations are seeded from `seed`.
[fixture]
seed = {EVAL_SEED}
settings = [{settings}]
repeats = 16
chosen_repeats = 40
samples = 200
max_release_ticks = 2500
trajectory_every = 5
{targets}'''


def outputs():
    root = Path("environments/panda_small")
    return {**{root / level / "scene.xml": scene(level) for level in LEVELS}, **{root / level / "manifest.toml": manifest(level) for level in LEVELS}}


if __name__ == "__main__":
    for target, text in outputs().items():
        target.parent.mkdir(exist_ok=True)
        target.write_text(text)
        print("wrote", target)
