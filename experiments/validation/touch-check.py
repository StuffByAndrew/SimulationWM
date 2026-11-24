"""Check the touch-only tabletop levels (touch-scenes.py) on a Slurm test node; writes OUTDIR/check.json and previews.

Checks: generated files equal the generator output; public fixture allowlist; matched L1/L2 constraints from compiled models;
object gaps; every object rests (2 s, robot at home); the robot home pose is palm down; palm-down IK over every object, the
whole bin interior and 15 cm above it, with the static palm error (sag) of held poses; the mandatory L3 touch
discriminability check (scripted palm and fingertip presses on the spiky and a smooth ball); a scripted grab of the spiky
ball; and the private metrics on reference candidates (identity, prior-only baseline, wrong shapes, L3 parameters files).
    sbatch -p test -c 2 --mem 8G -t 120 -o experiments/validation/touch-check-%j.log \
        --wrap "MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python experiments/validation/touch-check.py experiments/validation/touch-check"
"""
import base64
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/validation"))
from touch_ik import IK  # noqa: E402
from sim2sim.evaluation import evaluate_submission  # noqa: E402
from sim2sim.primitive_metrics import _primitives, exposed_surface  # noqa: E402
from sim2sim.robots import ARM_HOME, HAND_ACTUATORS, HAND_HOME  # noqa: E402
from sim2sim.runtime import public_robot_files  # noqa: E402
from sim2sim.simulation import Simulation  # noqa: E402
from sim2sim.tabletop import _objects, extent, grab, support  # noqa: E402

spec = importlib.util.spec_from_file_location("scenes", ROOT / "experiments/validation/touch-scenes.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
FAMILY = S.FAMILY
LEVELS = ("table_shapes", "table_objects", "bin_spiky")
CONFIG = {"robot": "panda_shadow", "rgb": False, "depth": False, "calibration": False, "tactile": True, "width": 320, "height": 240}
HOME_TARGET = (0., -.10, S.TOP + .25)
PALM = (0.001, -.012, .042)  # rh_palm-local centre of the palmar taxel face
GRIP = (0.005, -.012, .075)  # rh_palm-local point under the finger bases: the ball sits here when the fingers close
FF_PAD = (0., -.007, .018)   # rh_ffdistal-local centre of the fingertip pad
CURL = {**HAND_HOME, "rh_FFJ3": 1.0, "rh_FFJ2": .45, "rh_FFJ1": .45}  # first finger bent down to press with its pad
PITCHED = {**HAND_HOME, "rh_WRJ1": -.6}  # wrist flexed: lifts the forearm, which otherwise hangs ~4 cm below the palm face
FLAT = {**PITCHED, **{f"rh_{f}J{j}": 0. for f in ("FF", "MF", "RF", "LF") for j in (4, 3, 2, 1)}}  # fingers straight
WRAP = {**{f"rh_{f}J{j}": q for f in ("FF", "MF", "RF", "LF") for j, q in ((3, 1.4), (2, 1.2), (1, 1.0))},
        "rh_THJ5": .3, "rh_THJ4": 1.2, "rh_THJ3": 0., "rh_THJ2": .3, "rh_THJ1": .5}  # closing pose of the pinned L3 grab


def manifest(level):
    import tomllib
    return tomllib.loads((FAMILY / level / "manifest.toml").read_text())


def generated_files_match():
    with tempfile.TemporaryDirectory() as temp:
        S.FAMILY = Path(temp)
        S.write_fixture()
        S.write_levels()
        S.FAMILY = FAMILY
        made = {p.relative_to(temp).as_posix(): p.read_bytes() for p in Path(temp).rglob("*") if p.is_file()}
    have = {p.relative_to(FAMILY).as_posix(): p.read_bytes() for p in FAMILY.rglob("*") if p.is_file()}
    return made == have


def hand_targets(joints):
    """hand_target actuator targets holding a joint pose (a coupled *J0 target is J1 + J2)."""
    return [joints.get(n.replace("_A_", "_")[:-1] + "1", 0) + joints.get(n.replace("_A_", "_")[:-1] + "2", 0) if n.endswith("J0")
            else joints.get(n.replace("_A_", "_"), 0) for n in HAND_ACTUATORS]


def object_geometry(sim, m):
    """Per object: compiled mass, primitive count, body-frame bounding box size and exposed surface samples."""
    prims = _objects(sim, _primitives(sim, geom_names=m["object_geoms"]))
    out = {}
    for name, group in prims.items():
        body = sim._model.body(name).id
        rotation, origin = sim._data.xmat[body].reshape(3, 3), sim._data.xpos[body]
        local = [dict(p, pos=rotation.T @ (p["pos"] - origin), rotation=rotation.T @ p["rotation"]) for p in group]
        size = [extent(local, a) for a in np.eye(3)]
        out[name] = {"mass_kg": float(sim._model.body_subtreemass[body]), "primitives": len(group), "bbox_m": np.round(size, 5).tolist(),
                     "surface": exposed_surface(group, 2048, 0), "top_z": max(support(p, [0, 0, 1]) for p in group)}
    return out


def settle(sim, names):
    """Object displacement and final speed after 2 s with the robot holding its home pose."""
    start = {n: sim._data.xpos[sim._model.body(n).id].copy() for n in names}
    sim.action({"op": "observe", "ticks": 1000}, render=False)
    return {n: {"moved_mm": round(1000 * float(np.linalg.norm(sim._data.xpos[sim._model.body(n).id] - start[n])), 4),
                "speed_m_s": float(np.linalg.norm(sim._data.cvel[sim._model.body(n).id][3:]))} for n in names}


def hold(sim, ik, q, ticks=750, hand=HAND_HOME):
    """Static errors holding arm pose q with the hand at pose `hand`: flange (arm sag), grasp point (arm + wrist) and joint 5-7 efforts."""
    sim.action({"op": "reset"}, render=False)
    sim.action({"op": "hand_target", "actuator_targets": hand_targets(hand), "ticks": 1}, render=False)
    sim._data.qpos[sim._arm_qpos] = sim._data.ctrl[sim._arm_actuators] = q
    mujoco.mj_forward(sim._model, sim._data)
    flange = sim._data.xpos[sim._model.body("link7").id].copy()
    sim.action({"op": "observe", "ticks": ticks}, render=False)
    grasp, _ = ik.fk(q)
    site = sim._data.site_xpos[sim._model.site("grasp_site").id]
    return {"flange_mm": round(1000 * float(np.linalg.norm(sim._data.xpos[sim._model.body("link7").id] - flange)), 3),
            "grasp_point_mm": round(1000 * float(np.linalg.norm(site - grasp)), 3),
            "max_wrist_effort_nm": round(float(np.abs(sim._data.actuator_force[sim._arm_actuators[4:]]).max()), 3)}


def reach(sim, ik, targets):
    results = [ik.solve(t) for t in targets]
    return {"targets": len(targets), "ik_ok": sum(r["ok"] for r in results), "collision_free": sum(r["ok"] and not r["collisions"] for r in results),
            "collisions": sorted({c for r in results for c in r["collisions"] or []}), "failures": [list(t) for t, r in zip(targets, results) if not r["ok"]],
            "max_position_error_mm": max(r["position_error_mm"] for r in results if r["ok"]),
            "max_palm_tilt_deg": max(r["palm_tilt_deg"] for r in results if r["ok"]), "solutions": results}


def taxels(observation):
    return {pad: np.asarray(v["pad_normal_force_n"]) for pad, v in observation["tactile"].items() if isinstance(v, dict)}


def press(sim, ball, offset, mode):
    """Lower the palm centre (mode palm, hand at home) or the curled first finger's pad (mode finger) onto a ball top at a
    horizontal offset, 2 mm per step, until the pads carry 1.5 N; returns the taxel pattern and contact geometry at that step."""
    joints, body, point = {"palm": (PITCHED, "rh_palm", PALM), "flat": (FLAT, "rh_palm", PALM), "finger": (CURL, "rh_ffdistal", FF_PAD)}[mode]
    ik = IK(sim._model, joints)
    sim.action({"op": "reset"}, render=False)
    centre = sim._data.xpos[sim._model.body(ball).id].copy()
    top = centre[2] + S.SMOOTH_R  # every ball is 2 * SMOOTH_R tall
    x, y = centre[:2] + offset
    first = ik.solve([x, y, top + .02], body=body, point=point)
    assert first["ok"], (ball, offset, mode, first)
    sim.action({"op": "hand_target", "actuator_targets": hand_targets(joints), "ticks": 200}, render=False)
    sim.action({"op": "arm_target", "joint_targets": first["q"], "ticks": 500}, render=False)
    q, steps = first["q"], []
    for depth in np.arange(.018, -.03, -.002):
        r = ik.solve([x, y, top + depth], forward=first["forward"], body=body, point=point, start=q, restarts=1)
        q = r["q"]
        obs = sim.action({"op": "arm_target", "joint_targets": q, "ticks": 50})
        grids = taxels(obs)
        total = float(sum(g.sum() for g in grids.values()))
        steps.append(total)
        if total >= 1.5:
            break
    cells = np.concatenate([g.ravel() for g in grids.values()])
    active = cells > .02 * cells.sum()
    hits = [c for c in sim._data.contact[:sim._data.ncon] if sim._model.body(int(sim._model.geom_bodyid[c.geom1])).name == ball
            or sim._model.body(int(sim._model.geom_bodyid[c.geom2])).name == ball]
    robot = {sim._model.body(n).id for n in ("rh_palm", "rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal", "rh_thdistal")}
    hand_hits = [c.pos for c in hits if {int(sim._model.geom_bodyid[c.geom1]), int(sim._model.geom_bodyid[c.geom2])} & robot]
    return {"offset_mm": (1000 * np.asarray(offset)).round(1).tolist(), "steps": len(steps), "total_n": round(steps[-1], 3),
            "contact_depth_mm": round(1000 * (next((d for d, t in zip(np.arange(.018, -.03, -.002), steps) if t > .05), np.nan)), 1),
            "active_taxels": int(active.sum()), "max_taxel_share": round(float(cells.max() / cells.sum()), 3) if cells.sum() else None,
            "pads": sorted(p for p, g in grids.items() if g.sum() > .02 * cells.sum()),
            "hand_contact_points": len(hand_hits), "ball_moved_mm": round(1000 * float(np.linalg.norm(sim._data.xpos[sim._model.body(ball).id] - centre)), 2),
            "pattern": {p: g.round(3).tolist() for p, g in grids.items() if g.sum()},
            "palm_active_taxels": int((grids["palm"] > .05 * grids["palm"].sum()).sum()) if grids["palm"].sum() else 0,
            "body_totals_n": {p: round(v["total_normal_force_n"], 3) for p, v in obs["tactile"].items() if isinstance(v, dict) and v["total_normal_force_n"]}}


def identify(sim, names, rng, mode, rule, presses=3):
    """One trial of a scripted touch identification (3 palm presses per ball at random +-15 mm offsets, wrist flexed, fingers
    curled (mode palm) or straight (flat)). Rule `taxels`: the first ball whose press lights two or more palm taxels is named.
    Rule `height`: the ball with the most presses whose first contact comes >= 14 mm above the ball top is named (a tie names
    none): the spiky ball's side tips meet the fingers early."""
    if rule == "taxels":
        for ball in rng.permutation(names):
            for _ in range(presses):
                if press(sim, ball, rng.uniform(-.015, .015, 2), mode)["palm_active_taxels"] >= 2:
                    return str(ball)
        return None
    early = {ball: sum(press(sim, ball, rng.uniform(-.015, .015, 2), mode)["contact_depth_mm"] >= 14 for _ in range(presses)) for ball in names}
    best = max(early.values())
    return next(b for b, n in early.items() if n == best) if list(early.values()).count(best) == 1 else None


def scripted_grab(sim, ball, dz=.025, closing=None, forward=None):
    """Open hand palm down (wrist flexed, PITCHED) over the ball, lower around it, close every finger and the thumb, lift 15 cm."""
    sim.action({"op": "reset"}, render=False)
    centre = sim._data.xpos[sim._model.body(ball).id].copy()
    open_hand = {**PITCHED, **{f"rh_{f}J{j}": 0. for f in ("FF", "MF", "RF", "LF") for j in (3, 2, 1)}, "rh_THJ5": .6, "rh_THJ4": 1.1}
    ik = IK(sim._model, open_hand)
    above = ik.solve([*centre[:2], centre[2] + .10], forward=forward, point=GRIP)
    closed = {**open_hand, **(closing or {**{f"rh_{f}J{j}": q for f in ("FF", "MF", "RF", "LF") for j, q in ((3, 1.2), (2, 1.0), (1, 1.0))},
                                          "rh_THJ5": .8, "rh_THJ4": 1.2, "rh_THJ2": .5, "rh_THJ1": .8})}
    sim.action({"op": "hand_target", "actuator_targets": hand_targets(open_hand), "ticks": 200}, render=False)
    sim.action({"op": "arm_target", "joint_targets": above["q"], "ticks": 600}, render=False)
    down = ik.solve([*centre[:2], centre[2] + dz], forward=above["forward"], point=GRIP, start=above["q"], restarts=1)
    sim.action({"op": "arm_target", "joint_targets": down["q"], "ticks": 500}, render=False)
    sim.action({"op": "hand_target", "actuator_targets": hand_targets(closed), "ticks": 400}, render=False)
    lift = ik.solve([*centre[:2], centre[2] + .20], forward=above["forward"], point=GRIP, start=down["q"], restarts=1)
    sim.action({"op": "arm_target", "joint_targets": lift["q"], "ticks": 800}, render=False)
    sim.action({"op": "observe", "ticks": 250}, render=False)
    return grab(sim, manifest("bin_spiky"))


def candidate(level, edit=None, scene_file="scene.xml", parameters=None):
    """Candidate ZIP: a level scene (optionally edited) at the root, the unchanged public fixture, optional parameters.json."""
    root = ET.fromstring((FAMILY / level / scene_file).read_text())
    root.find("compiler").set("meshdir", "robot/assets")
    root.find("include").set("file", "robot/panda_shadow.xml")
    if edit:
        edit(root)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("scene.xml", ET.tostring(root, encoding="unicode"))
        if parameters is not None:
            archive.writestr("parameters.json", parameters)
        for path in sorted((FAMILY / "robot").rglob("*")):
            if path.is_file():
                archive.writestr("robot/" + path.relative_to(FAMILY / "robot").as_posix(), path.read_bytes())
    return buffer.getvalue()


def one_box(root):
    """Edit: every free object replaced by one 30 cm x 30 cm x 5 cm box named after nothing in particular."""
    world = root.find("worldbody")
    for body in world.findall("body"):
        world.remove(body)
    body = ET.SubElement(world, "body", name="thing", pos=f"0 -0.1 {S.TOP + .025}")
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "geom", type="box", size=".15 .15 .025", density="500")


def evaluate(level, archive, metrics):
    with tempfile.TemporaryDirectory() as temp:
        report = evaluate_submission(archive, dict(source_dir=str(FAMILY), source_entrypoint=f"{level}/scene.xml", manifest=f"{level}/manifest.toml",
                                                   output_dir=temp, metrics=metrics, source_config=CONFIG, camera="fixed", width=64, height=48,
                                                   reconstruction={"samples": 2048, "seed": 20261029}))
    assert report["status"] == "ok", report.get("error")
    scores = report["scores"]
    summary = {k: v for k, v in scores["objects"].items() if k not in ("per_object", "surface_samples", "surface_seed")} if "objects" in scores else {}
    summary["per_object"] = {r["name"]: {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()
                                         if k in ("candidate_body", "chamfer_m", "centroid_error_m", "found", "label_correct")}
                             | {d["name"]: {k: round(v, 5) for k, v in d.items() if isinstance(v, float)} for d in r["dimensions"]}
                             for r in scores.get("objects", {}).get("per_object", [])}
    for key in ("identification",):
        if key in scores:
            summary[key] = scores[key]
    if "reconstruction" in scores:
        summary["chamfer_m"] = scores["reconstruction"]["chamfer_distance_m"]
        summary["center_of_mass_error_m"] = scores["reconstruction"]["center_of_mass_error_m"]
    return summary


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    results = {"generated_files_match": generated_files_match(), "public_fixture_files": sorted(public_robot_files(FAMILY / "robot", "panda_shadow"))}
    assert results["generated_files_match"]
    sims = {level: Simulation(FAMILY / level / "scene.xml", CONFIG) for level in LEVELS}
    # Home pose: robots.ARM_HOME is the palm-down IK solution 25 cm above the table centre.
    sim = sims["table_objects"]
    ik = IK(sim._model)
    solved = ik.solve(HOME_TARGET)
    position, rotation = ik.fk(np.array(ARM_HOME["panda_shadow"]))
    results["home"] = {"ik_solution": solved["q"], "configured": list(ARM_HOME["panda_shadow"]),
                       "configured_grasp_error_mm": round(1000 * float(np.linalg.norm(position - HOME_TARGET)), 3),
                       "configured_palm_tilt_deg": round(float(np.degrees(np.arccos(np.clip(rotation[2, 1], -1, 1)))), 3),
                       "contacts_at_home": {level: IK(s._model).collisions(np.array(ARM_HOME["panda_shadow"])) for level, s in sims.items()}}
    print(json.dumps({"home": results["home"]}), flush=True)
    geometry = {}
    for level, sim in sims.items():
        m = manifest(level)
        geometry[level] = object_geometry(sim, m)
        names = list(geometry[level])
        gaps = {f"{a}/{b}": round(float(np.min(np.linalg.norm(geometry[level][a]["surface"][:, None] - geometry[level][b]["surface"][None], axis=2))), 4)
                for i, a in enumerate(names) for b in names[:i]}
        results[level] = {"objects": {n: {k: v for k, v in g.items() if k != "surface"} for n, g in geometry[level].items()},
                          "min_gap_m": gaps, "rest": settle(sim, names)}
        full = Simulation(FAMILY / level / "scene.xml", {**CONFIG, "rgb": True, "calibration": True}).observe(full=True)
        (out / f"{level}.png").write_bytes(base64.b64decode(full["rgb"]))
        print(json.dumps({level: results[level]}), flush=True)
    for name, match in S.MATCH.items():
        a, b = geometry["table_shapes"][name], geometry["table_objects"][match]
        results.setdefault("matching", {})[name] = {"match": match, "mass_ratio": round(a["mass_kg"] / b["mass_kg"], 4),
                                                   "bbox_ratio": np.round(np.divide(a["bbox_m"], b["bbox_m"]), 4).tolist(),
                                                   "primitive_difference": a["primitives"] - b["primitives"]}
    # Reach: palm down over every object (2 cm and 15 cm above its top) and over the bin interior (5 x 5 grid) at the floor,
    # the rim and 15 cm above the rim; static errors at the high targets.
    sim = sims["table_objects"]
    for level in ("table_shapes", "table_objects"):
        targets = [(*g["surface"].mean(axis=0)[:2], g["top_z"] + h) for g in geometry[level].values() for h in (.02, .15)]
        r = reach(sims[level], IK(sims[level]._model), targets)
        r["hold"] = [hold(sims[level], IK(sims[level]._model), s["q"]) for s in r["solutions"][1::2]]
        results[level]["reach"] = {k: v for k, v in r.items() if k != "solutions"}
    grid = np.linspace(-(S.BIN_HALF - .02), S.BIN_HALF - .02, 5)
    targets = [(S.BIN[0] + gx, S.BIN[1] + gy, z) for z in (S.FLOOR_Z + .04, S.RIM_Z, S.RIM_Z + .15) for gx in grid for gy in grid]
    ik = IK(sims["bin_spiky"]._model, PITCHED)  # the bin needs the flexed wrist (PITCHED)
    r = reach(sims["bin_spiky"], ik, targets)
    r["hold"] = [hold(sims["bin_spiky"], ik, s["q"], hand=PITCHED) for s in r["solutions"][50:]]
    results["bin_spiky"]["reach"] = {k: v for k, v in r.items() if k != "solutions"}
    print(json.dumps({"reach": {level: results[level]["reach"] for level in LEVELS}}), flush=True)
    # No size or dynamics cue (user, 2026-11-24): per ball, compiled mass and principal inertia, top height and horizontal width
    # every 15 degrees of yaw.
    m = manifest("bin_spiky")
    sim = sims["bin_spiky"]
    sim.action({"op": "reset"}, render=False)
    groups = _objects(sim, _primitives(sim, geom_names=m["object_geoms"]))
    results["bin_spiky"]["cues"] = {name: {
        "mass_kg": float(sim._model.body_mass[sim._model.body(name).id]), "inertia_kg_m2": sim._model.body_inertia[sim._model.body(name).id].tolist(),
        "top_z_m": max(support(p, [0, 0, 1]) for p in g),
        "width_m": [extent(g, [np.cos(a), np.sin(a), 0]) for a in np.radians(np.arange(0, 180, 15))]} for name, g in groups.items()}
    print(json.dumps({"cues": results["bin_spiky"]["cues"]}), flush=True)
    # Touch discriminability (mandatory): presses on the spiky ball and on a smooth ball.
    spiky, smooth = m["tabletop"]["spiky"], next(n for n in m["object_bodies"] if n != m["tabletop"]["spiky"])
    offsets = [np.array([dx, dy]) for dx in (-.012, 0, .012) for dy in (-.012, 0, .012)]
    presses = {f"{mode}/{kind}": [press(sims["bin_spiky"], ball, o, mode) for o in offsets]
               for mode in ("palm", "finger") for kind, ball in (("spiky", spiky), ("smooth", smooth))}
    results["discriminability"] = {"spiky": spiky, "smooth": smooth, "presses": presses, "summary": {
        key: {stat: [p[stat] for p in rows] for stat in ("active_taxels", "max_taxel_share", "hand_contact_points", "contact_depth_mm", "ball_moved_mm")}
        for key, rows in presses.items()}}
    print(json.dumps({"discriminability": results["discriminability"]["summary"]}), flush=True)
    # Reference floors, not gates (user, 2026-11-24: L3 is an explicitly hard identification task): scripted procedures with 3
    # presses per ball, 40 seeded trials each, stored with the level for reports.
    floors = {}
    for mode, rule in (("palm", "taxels"), ("flat", "taxels"), ("palm", "height")):
        rng = np.random.default_rng(20261124)
        named = [identify(sims["bin_spiky"], m["object_bodies"], rng, mode, rule) for _ in range(40)]
        floors[f"{rule}_{'curled' if mode == 'palm' else 'straight'}_fingers"] = {
            "trials": len(named), "rate": sum(n == spiky for n in named) / len(named), "named": named}
    results["identification_floors"] = floors
    print(json.dumps({"identification_floors": {k: v["rate"] for k, v in floors.items()}}), flush=True)
    # Existence proof: with 40 mm walls an open-loop sweep (12 hand yaws x 3 grip heights x 3 closing poses, job 63458855) lifted the
    # spiky ball in 1 of 108 tries (this one); it is a hard grasp that needs the fingers aligned with the tips.
    results["scripted_grab"] = scripted_grab(sims["bin_spiky"], spiky, .012, WRAP, [np.cos(np.radians(270)), np.sin(np.radians(270)), 0.])
    print(json.dumps({"scripted_grab": results["scripted_grab"]}), flush=True)
    # Private metrics on reference candidates.
    table_metrics = ["objects", "chamfer", "center_of_mass", "density"]
    results["evaluation"] = {
        "table_objects/identity": evaluate("table_objects", candidate("table_objects"), table_metrics),
        "table_objects/prior": evaluate("table_objects", candidate("table_objects", scene_file="prior.xml"), table_metrics),
        "table_objects/one_box": evaluate("table_objects", candidate("table_objects", one_box), table_metrics),
        "table_shapes/identity": evaluate("table_shapes", candidate("table_shapes"), table_metrics),
        "table_shapes/l2_objects": evaluate("table_shapes", candidate("table_objects"), table_metrics)}
    ball = lambda name: sims["bin_spiky"]._model.body(name).id  # noqa: E731
    sims["bin_spiky"].action({"op": "reset"}, render=False)
    xyz = {n: sims["bin_spiky"]._data.xpos[ball(n)].round(4).tolist() for n in m["object_bodies"]}
    for name, parameters in (("correct", json.dumps({"spiky_ball_xyz": xyz[spiky]})), ("wrong_ball", json.dumps({"spiky_ball_xyz": xyz[smooth]})),
                             ("missing", None), ("malformed", '{"spiky_ball_xyz": [0, 1]}'), ("not_json", "spiky")):
        results["evaluation"][f"bin_spiky/{name}"] = evaluate("bin_spiky", candidate("bin_spiky", parameters=parameters), ["objects", "identification"])
    print(json.dumps({"evaluation": results["evaluation"]}), flush=True)
    (out / "check.json").write_text(json.dumps(results, indent=1, default=float))
    (FAMILY / "bin_spiky" / "reference_floors.json").write_text(json.dumps(
        {"source": "experiments/validation/touch-check.py", "procedures": {k: {"trials": v["trials"], "rate": v["rate"]} for k, v in floors.items()}},
        indent=1) + "\n")
    assert results["scripted_grab"]["success"], results["scripted_grab"]


if __name__ == "__main__":
    main(Path(sys.argv[1]))
