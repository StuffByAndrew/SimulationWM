"""Write the explanation-depth gear-train level (2026-11-24, docs/level-designs/gear-train-depth-20261124.md):
environments/gear_train/{scene.xml, shallow.xml, manifest.toml}.

A vertical panel carries a crank (hinge, handle knob at 40 mm radius) on the left, a dial hand (hinge) on the right and
between them a train of 20 gears (idlers and two compound pairs on two layers; overall ratio hand:crank = -7.5). Gears are
thin cylinders of one material with massless, non-colliding tooth visuals, axles at their centres; crank and hand carry
counterweights, so nothing has a gravity torque. Couplings are joint equalities (ideal gears); every axle has the same
viscous damping. No moving geom collides with anything, and the manifest makes only crank and hand graspable.

scene.xml is the deep source (every gear its own body and hinge). shallow.xml, the private shallow reference, keeps crank
and hand, couples them directly with the overall ratio and lumps the gears' inertia (sum I_i r_i^2) and damping
(sum b r_i^2, r_i = gear speed / crank speed) into the crank joint's armature and damping. The manifest's G-cheap gate
(`cheap_gate_rad`) is the worst probe error (src/sim2sim/gear_train.py) of the shallow reference with armature and damping
each scaled by 1 - P or 1 + P (four combinations), against the source.

Physics (the gate): run on a Slurm test node from the repository root:
PYTHONPATH=src .venv/bin/python experiments/validation/gear-scenes.py"""
from itertools import product
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path("environments/gear_train")
TIMESTEP = .0005                  # s: the near-ideal couplings need a short step (user, 2026-11-24)
RHO = 2700.                       # kg/m^3, every gear (one material)
RHO_CRANK = 2700.                 # kg/m^3, crank and hand parts
Z0 = .45                          # axle height, m
PANEL = (.46, .01, .2)            # panel half sizes (front face at y = 0)
T = .003                          # wheel and front-part half thickness, m
LAYER_Y = (-.007, -.015)          # wheel layer centres (L0 behind L1); crank arm and hand pointer at FRONT_Y
FRONT_Y = -.024
TOOTH_PITCH, TOOTH = .008, (.0015, .0018)  # tooth spacing along the pitch circle; tooth radial and tangential half sizes, m
B = 1.0e-4                        # viscous damping of every axle, N m s/rad: crank coast time constant about 0.12 s
SOLREF, SOLIMP = ".001 1", ".9999 .9999 .001 .5 2"  # near-ideal gear couplings (time constant at the 2-step minimum)
SLIP_M = .002                     # public [simulation] grasp_slip_m: the grip lets go when stretched farther (m)
SHALLOW_TIMESTEPS = (.0005, .001, .002, .004, .005, .008, .01)  # s: shallow.xml takes the largest that passes the gate
SOLVER = ' tolerance="1e-12"'     # <option> extra: Newton runs to convergence (the default 1e-8 stops a stiff train early)
PHI = np.deg2rad(50)              # mesh directions alternate +PHI, -PHI from +x: a zigzag train
CRANK_R, HANDLE_R, KNOB = .045, .040, (.008, .012)  # crank wheel radius, handle radius; knob radius and half length (along -y)
HAND_R, POINTER, HAND_CW = .024, (.005, .035), .03  # hand pinion radius; pointer half width and half length; counterweight offset
# Gears in train order: (in layer, in radius, out layer, out radius); an idler's in and out wheel are the same wheel.
IDLER = lambda layer, r: (layer, r, layer, r)  # noqa: E731
GEARS = ([IDLER(0, r) for r in (.026, .022, .030, .024, .028, .023)] + [(0, .016, 1, .032)]
         + [IDLER(1, r) for r in (.022, .027, .024, .030, .023, .026)] + [(1, .016, 0, .032)]
         + [IDLER(0, r) for r in (.025, .022, .028, .024, .027, .023)])
REMOVALS = ("gear_03", "gear_10", "gear_14")  # G-deep: a crank-side idler, a mid-train idler, a compound pair
GATE_P = .1                                   # G-cheap gate perturbation of the lumped armature and damping
EYE, LOOK, FOVY = np.array([0., -.95, .70]), np.array([0., 0., Z0]), 45
PROBES = [  # private, within the handle's reach: the grasped point follows its circle about the axle in CHORD_DEG chords;
    # (name, body, local point, radius m, start angle deg in the panel's x-z plane, sweep deg, grasped seconds, settle seconds)
    ("crank_turn_slow", "crank", (0, -.051, -HANDLE_R), HANDLE_R, -90, 120, 3.0, .8),
    ("crank_turn_fast", "crank", (0, -.051, -HANDLE_R), HANDLE_R, -90, 120, 1.8, .8),
    ("crank_turn_back", "crank", (0, -.051, -HANDLE_R), HANDLE_R, -90, -60, .9, .8),
    ("hand_push", "hand", (0, FRONT_Y - T, .06), .06, 90, 30, .8, .8),
    ("hand_push_fast", "hand", (0, FRONT_Y - T, .06), .06, 90, -40, .8, .8),
]
CHORD_DEG = 5
OVERPULLS = [  # check only: straight pulls ending 1 cm (crank) or 2 cm (hand) beyond reach, which the slip must end first;
    # (name, body, local point, displacement m, grasped seconds, settle seconds)
    ("crank_overpull_slow", "crank", (0, -.051, -HANDLE_R), (.05, 0, 0), 2.0, .5),
    ("crank_overpull_fast", "crank", (0, -.051, -HANDLE_R), (.05, 0, 0), .32, .8),
    ("hand_overpull", "hand", (0, FRONT_Y - T, .06), (.08, 0, 0), .4, .8),
]
WORLD = 'contype="0" conaffinity="0"'


def fmt(v):
    return " ".join(f"{float(x):.6f}" for x in v)


def xyaxes(eye, target):
    forward = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(forward, [0, 0, 1]); x /= np.linalg.norm(x)
    return fmt([*x, *np.cross(x, forward)])


def train():
    """Bodies in train order: name, axle x, z, wheels {layer: radius}, speed ratio to the crank, coupling to the previous."""
    stages = [("crank", (0, CRANK_R, 0, CRANK_R))] + [(f"gear_{i:02d}", g) for i, g in enumerate(GEARS, 1)] + [("hand", IDLER(0, HAND_R))]
    bodies, x, z, speed = [], 0., 0., 1.
    for k, (name, (lin, rin, lout, rout)) in enumerate(stages):
        if k:
            prev = stages[k - 1][1]
            assert prev[2] == lin, f"{name} meshes on another layer than its driver"
            d = prev[3] + rin
            phi = PHI if k % 2 else -PHI
            x, z, ratio = x + d * np.cos(phi), z + d * np.sin(phi), -prev[3] / rin
            speed *= ratio
        bodies.append({"name": name, "x": x, "z": z, "wheels": {lin: rin, lout: rout}, "speed": speed,
                       "coupling": ratio if k else None})
    shift = -(bodies[0]["x"] + bodies[-1]["x"]) / 2
    for b in bodies:
        b["x"] += shift
        b["z"] += Z0
    assert abs(bodies[-1]["speed"] + 7.5) < 1e-12, bodies[-1]["speed"]
    # Clearance: wheels on one layer that do not mesh stay apart by tooth height plus 3 mm; all lie on the panel.
    wheels = [(i, layer, r) for i, b in enumerate(bodies) for layer, r in b["wheels"].items()]
    for (i, la, ra), (j, lb, rb) in product(wheels, wheels):
        if i < j and la == lb and j - i > 1:
            gap = np.hypot(bodies[i]["x"] - bodies[j]["x"], bodies[i]["z"] - bodies[j]["z"]) - ra - rb
            assert gap > 2 * (TOOTH[0] + .0005) + .003, (bodies[i]["name"], bodies[j]["name"], gap)
    for b in bodies:
        r = max(b["wheels"].values()) + .07 * (b["name"] == "hand")
        assert abs(b["x"]) + r < PANEL[0] - .01 and abs(b["z"] - Z0) + r < PANEL[2] - .01, b["name"]
    return bodies


def inertia(r):
    """Axial inertia of one wheel (cylinder of radius r, thickness 2T, density RHO), kg m^2."""
    return RHO * np.pi * r * r * 2 * T * r * r / 2


def lumped():
    """Shallow armature (sum I_i r_i^2) and crank damping (b (1 + sum r_i^2)) over the gears between crank and hand."""
    gears = train()[1:-1]
    return (sum(sum(inertia(r) for r in g["wheels"].values()) * g["speed"] ** 2 for g in gears),
            B * (1 + sum(g["speed"] ** 2 for g in gears)))


def wheel(name, layer, r, rgba, rho):
    y = LAYER_Y[layer]
    n = round(2 * np.pi * r / TOOTH_PITCH)
    teeth = "".join(f'\n        <geom name="{name}_l{layer}_t{t:02d}" type="box" size="{TOOTH[0]} {T} {TOOTH[1]}" mass="0" {WORLD} rgba="{rgba}" '
                    f'pos="{fmt([(r + .0005) * np.cos(a), y, (r + .0005) * np.sin(a)])}" euler="0 {-np.rad2deg(a):.4f} 0"/>'
                    for t, a in enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False)))
    return (f'\n        <geom name="{name}_l{layer}" type="cylinder" size="{r} {T}" pos="0 {y} 0" euler="90 0 0" density="{rho:g}" '
            f'{WORLD} rgba="{rgba}"/>' + teeth)


def hub(name, front_y):
    return f'\n        <geom name="{name}_hub" type="cylinder" fromto="0 {LAYER_Y[0] + T:.6f} 0 0 {front_y:.6f} 0" size=".006" mass="0" {WORLD} rgba=".25 .25 .27 1"/>'


def crank_parts():
    knob_v = np.pi * KNOB[0] ** 2 * 2 * KNOB[1]
    cw = np.sqrt(knob_v / (np.pi * 2 * T))  # same volume on the front layer at the opposite radius: balanced
    ky = FRONT_Y - T - KNOB[1]
    return (f'\n        <geom name="crank_arm" type="box" size=".006 {T} {HANDLE_R}" pos="0 {FRONT_Y} 0" density="{RHO_CRANK:g}" {WORLD} rgba=".45 .47 .5 1"/>'
            f'\n        <geom name="crank_counterweight" type="cylinder" size="{cw:.6f} {T}" pos="0 {FRONT_Y} {HANDLE_R}" euler="90 0 0" density="{RHO_CRANK:g}" {WORLD} rgba=".3 .31 .33 1"/>'
            f'\n        <geom name="crank_knob" type="cylinder" size="{KNOB[0]} {KNOB[1]}" pos="0 {ky:.6f} {-HANDLE_R}" euler="90 0 0" density="{RHO_CRANK:g}" {WORLD} rgba=".8 .15 .12 1"/>'
            + hub("crank", FRONT_Y - T))


def hand_parts():
    moment = 2 * POINTER[0] * 2 * T * 2 * POINTER[1] * POINTER[1]  # pointer volume times its centroid radius
    cw = np.sqrt(moment / HAND_CW / (np.pi * 2 * T))  # counterweight disc on the front layer balancing the pointer
    return (f'\n        <geom name="hand_pointer" type="box" size="{POINTER[0]} {T} {POINTER[1]}" pos="0 {FRONT_Y} {POINTER[1]}" density="{RHO_CRANK:g}" {WORLD} rgba=".08 .08 .1 1"/>'
            f'\n        <geom name="hand_counterweight" type="cylinder" size="{cw:.6f} {T}" pos="0 {FRONT_Y} {-HAND_CW}" euler="90 0 0" density="{RHO_CRANK:g}" {WORLD} rgba=".08 .08 .1 1"/>'
            + hub("hand", FRONT_Y - T))


def scene(shallow=False, armature_scale=1., damping_scale=1., timestep=TIMESTEP):
    bodies = train()
    armature, damping = lumped()
    out = []
    for b in bodies:
        if shallow and b["name"].startswith("gear_"):
            continue
        extra = ""
        if b["name"] == "crank":
            parts, rgba = crank_parts(), ".62 .64 .68 1"
            if shallow:
                extra = f' armature="{armature * armature_scale:.9g}"'
        elif b["name"] == "hand":
            parts, rgba = hand_parts(), ".62 .64 .68 1"
        else:
            parts, rgba = hub(b["name"], LAYER_Y[max(b["wheels"])] - T), (".80 .62 .25 1", ".72 .5 .2 1")[max(b["wheels"])]
        joint = f'{b["name"]}_hinge' if b["name"] in ("crank", "hand") else b["name"]
        d = damping * damping_scale if shallow and b["name"] == "crank" else B
        out.append(f'''    <body name="{b["name"]}" pos="{b["x"]:.6f} 0 {b["z"]:.6f}">
      <joint name="{joint}" type="hinge" axis="0 1 0" damping="{d:.9g}"{extra}/>'''
                   + "".join(wheel(b["name"], layer, r, rgba, RHO if b["name"].startswith("gear_") else RHO_CRANK) for layer, r in sorted(b["wheels"].items())).replace("\n        ", "\n      ")
                   + parts.replace("\n        ", "\n      ") + "\n    </body>")
    if shallow:
        couplings = [("hand_hinge", "crank_hinge", bodies[-1]["speed"])]
    else:
        names = ["crank_hinge"] + [b["name"] for b in bodies[1:-1]] + ["hand_hinge"]
        couplings = [(names[k], names[k - 1], bodies[k]["coupling"]) for k in range(1, len(bodies))]
    equality = "".join(f'\n    <joint joint1="{a}" joint2="{b}" polycoef="0 {c:.12g} 0 0 0" solref="{SOLREF}" solimp="{SOLIMP}"/>' for a, b, c in couplings)
    hand = bodies[-1]
    ticks = "".join(f'\n    <geom name="dial_tick_{i:02d}" type="box" size=".002 .001 .008" pos="{fmt([hand["x"] + .078 * np.sin(a), -.0025, hand["z"] + .078 * np.cos(a)])}" '
                    f'euler="0 {np.rad2deg(a):.4f} 0" {WORLD} rgba=".1 .1 .1 1"/>' for i, a in enumerate(np.linspace(0, 2 * np.pi, 12, endpoint=False)))
    return f'''<mujoco model="gear_train">
  <compiler angle="degree"/>
  <option timestep="{timestep:g}"{SOLVER}/>
  <visual><map znear="0.01" zfar="12"/></visual>
  <worldbody>
    <light name="key" directional="true" pos="-1 -3 3" dir="1 3 -2" diffuse=".8 .78 .72"/>
    <light name="fill" directional="true" pos="2 -1 2" dir="-2 1 -1" diffuse=".35 .38 .45" castshadow="false"/>
    <camera name="fixed" pos="{fmt(EYE)}" xyaxes="{xyaxes(EYE, LOOK)}" fovy="{FOVY}"/>
    <geom name="ground" type="plane" size="3 3 .1" rgba=".70 .72 .73 1"/>
    <geom name="stand" type="box" pos="0 .03 {(Z0 - PANEL[2]) / 2:.6f}" size=".14 .06 {(Z0 - PANEL[2]) / 2:.6f}" rgba=".3 .32 .35 1"/>
    <geom name="panel" type="box" pos="0 {PANEL[1]} {Z0}" size="{fmt(PANEL)}" rgba=".86 .87 .84 1"/>
    <geom name="dial_face" type="cylinder" pos="{hand["x"]:.6f} -.001 {hand["z"]:.6f}" size=".09 .001" euler="90 0 0" {WORLD} rgba=".97 .97 .94 1"/>{ticks}
{chr(10).join(out)}
  </worldbody>
  <equality>{equality}
  </equality>
</mujoco>
'''


def calibrate():
    """(gate, shallow timestep, {timestep: probe error}): the gate is the worst probe error (rad) of the shallow reference with
    armature and damping scaled by 1 -/+ GATE_P at TIMESTEP; the shallow timestep is the largest of SHALLOW_TIMESTEPS up to
    which every unperturbed shallow error stays within the gate."""
    sys.path.insert(0, "src")
    from sim2sim.gear_train import compile_plain, panel_normal, probe_errors, reference_runs
    from sim2sim.simulation import Simulation
    with tempfile.TemporaryDirectory() as temp:
        (Path(temp) / "scene.xml").write_text(scene())
        normal = panel_normal(*compile_plain(Path(temp) / "scene.xml"))
        runs = reference_runs(Simulation(Path(temp) / "scene.xml", SOURCE_CONFIG), probe_tables(), normal)

        def error(*args):
            (Path(temp) / "shallow.xml").write_text(scene(True, *args))
            return probe_errors(runs, Simulation(Path(temp) / "shallow.xml", CANDIDATE_CONFIG), normal)["probe_error_rad"]
        gate_rad = float(f"{max(error(a, d) for a, d in product((1 - GATE_P, 1 + GATE_P), repeat=2)):.4g}")
        errors = {dt: error(1., 1., dt) for dt in SHALLOW_TIMESTEPS}
    passing = [dt for dt in SHALLOW_TIMESTEPS if all(errors[t] <= gate_rad for t in SHALLOW_TIMESTEPS if t <= dt)]
    return gate_rad, max(passing), errors


SOURCE_CONFIG = {"width": 320, "height": 240, "camera": "fixed", "graspable_bodies": ["crank", "hand"], "grasp_slip_m": SLIP_M}
CANDIDATE_CONFIG = {"width": 320, "height": 240, "camera": "fixed", "grasp_slip_m": SLIP_M}  # own timestep, graspable bodies inferred


def probe_tables():
    """Manifest probes: chords along each probe's circle, equal time each."""
    out = []
    for name, body, point, radius, start, sweep, seconds, settle in PROBES:
        n = round(abs(sweep) / CHORD_DEG)
        angles = np.deg2rad(start + sweep * np.arange(n + 1) / n)
        path = radius * np.column_stack([np.cos(angles), np.zeros(n + 1), np.sin(angles)])
        out.append({"name": name, "body": body, "local_point": list(point), "settle_ticks": round(settle / TIMESTEP),
                    "moves": [[*np.round(d, 6).tolist(), round(seconds / n / TIMESTEP)] for d in np.diff(path, axis=0)]})
    return out


def overpull_tables():
    return [{"name": n, "body": b, "local_point": list(p), "settle_ticks": round(s / TIMESTEP),
             "moves": [[*(np.asarray(d) / 4).tolist(), round(t / 4 / TIMESTEP)] for _ in range(4)]} for n, b, p, d, t, s in OVERPULLS]


def axle_tolerance():
    """40 % of the smallest distance between two gear axles, rounded down to 1 mm."""
    gears = train()[1:-1]
    least = min(np.hypot(a["x"] - b["x"], a["z"] - b["z"]) for a, b in product(gears, gears) if a is not b)
    return np.floor(.4 * least * 1000) / 1000


def manifest(gate_rad, shallow_dt, errors):
    probes = "".join(f'''
[[probes]]
name = "{p["name"]}"
body = "{p["body"]}"
local_point = [{", ".join(f"{v:g}" for v in p["local_point"])}]
moves = [{", ".join("[" + ", ".join(f"{v:g}" for v in m) + "]" for m in p["moves"])}]
settle_ticks = {p["settle_ticks"]}
''' for p in probe_tables())
    return f'''name = "gear_train"
scene = "scene.xml"
camera = "fixed"
graspable_bodies = ["crank", "hand"]
{probes}
# Private gear-train evaluation (src/sim2sim/gear_train.py; generated by experiments/validation/gear-scenes.py).
# cheap_gate_rad: worst probe error of shallow.xml with its lumped armature and damping each scaled by {1 - GATE_P:g} or {1 + GATE_P:g},
# at the source timestep {TIMESTEP:g} s. shallow.xml runs at {shallow_dt:g} s, the largest step whose probe error stays within the gate
# (probe error by timestep: {", ".join(f"{dt:g} s {e:.3g} rad" for dt, e in errors.items())}).
[gear_train]
shallow_reference = "shallow.xml"
removals = [{", ".join(f'"{r}"' for r in REMOVALS)}]
axle_tolerance_m = {axle_tolerance():g}
cheap_gate_rad = {gate_rad:.4g}
cost_runs = 5
cost_seconds = 1.0
cost_torque_nm = 0.02
'''


def outputs():
    gate_rad, shallow_dt, errors = calibrate()
    return {ROOT / "scene.xml": scene(), ROOT / "shallow.xml": scene(True, timestep=shallow_dt),
            ROOT / "manifest.toml": manifest(gate_rad, shallow_dt, errors)}


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    for target, text in outputs().items():
        target.write_text(text)
        print("wrote", target)
