"""Check the childproof bottle levels and prove the design's premises (Slurm test node only), from the repository root:
PYTHONPATH=src .venv/bin/python experiments/validation/bottle-check.py OUTDIR

Files equal the generator output; bottle and cap compile identically in both variants; both scenes rest. Hidden: from
VIEWS viewpoints around the closed bottle, upright and lying on its side, no pixel of the inner cap, spring washer, teeth,
neck, lugs or contents shows (segmentation, hence neither RGB nor depth), no grasp ray reaches them, and the empty and
contents scenes render bit-identical RGB and depth (the contents are not inferable from any view of the closed bottle).
Mechanism, through the harness's own weld grasps (two simultaneous, as the probes use them): twist only and an
under-threshold push leave it closed; a push above the preload, a quarter turn and a lift open it, deterministically and
stably (max weld violation and speed reported); the push-force and turn-angle sweeps; whether one grasp opens it.
Metric `bottle` and the contents breakdown through the real evaluator on reference candidates: identity (both variants);
no teeth (never opens); no lugs (opens without a push); wrong contents; missing contents (gaps, not zeros).
Writes OUTDIR/{check.json, *.png, eval-*}."""
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import sys
import tomllib
import zipfile

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, "src")
from sim2sim import bottle as B  # noqa: E402
from sim2sim.evaluation import evaluate_submission  # noqa: E402
from sim2sim.simulation import Simulation  # noqa: E402

spec = importlib.util.spec_from_file_location("scenes", "experiments/validation/bottle-scenes.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
VIEWS = 64                  # viewpoints per pose (Fibonacci sphere, radius VIEW_R about the bottle centre)
VIEW_R = .25
REST_M, REST_SPEED = 5e-4, 1e-2   # after REST_S: every body within this of its start, every speed below (m, rad; per s)
REST_S = 2.
MAX_VIOLATION_M, MAX_SPEED = .02, 30.  # stability of the scripted probes: weld translation residual (m); bottle and cap
# |qvel| (m/s, rad/s): a cap popping off the neck spins at ~20 rad/s, a blow-up is orders of magnitude faster
CONFIG = {"width": 320, "height": 240, "camera": "fixed", "rgb": True, "depth": True, **B.PROBE_CONFIG}
result = {}
for path, text in S.outputs().items():
    assert path.read_text() == text, f"{path} differs from the generator output"
scenes = {name: S.ROOT / name / "scene.xml" for name in S.VARIANTS}
tables = {name: tomllib.loads((S.ROOT / name / "manifest.toml").read_text()) for name in S.VARIANTS}
CONTENT_BODIES = [c[0] for c in S.CONTENTS]

# Bottle and cap compile identically across variants.
models = {name: mujoco.MjModel.from_xml_path(str(path)) for name, path in scenes.items()}
shared = ("bottle", "cap", "liner", "cap_shell")


def signature(m):
    rows = []
    for b in shared:
        i = m.body(b).id
        rows.append([m.body_mass[i], *m.body_inertia[i], *m.body_ipos[i], *m.body_pos[i]])
        for g in np.flatnonzero(m.geom_bodyid == i):
            rows.append([m.geom_type[g], *m.geom_size[g], *m.geom_pos[g], *m.geom_quat[g], *m.geom_rgba[g], *m.geom_friction[g]])
    return rows


assert all(np.array_equal(a, b) for a, b in zip(*(signature(m) for m in models.values()))), "bottle or cap differ between variants"
result["masses_g"] = {name: {m.body(i).name: round(1000 * float(m.body_mass[i]), 3) for i in range(1, m.nbody)} for name, m in models.items()}

# Rest.
result["rest"] = {}
for name, path in scenes.items():
    sim = Simulation(path, CONFIG)
    start = sim._data.xpos.copy()
    sim.action({"op": "wait", "ticks": round(REST_S / sim._model.opt.timestep)}, render=False)
    moved, speed = float(np.abs(sim._data.xpos - start).max()), float(np.abs(sim._data.qvel).max())
    result["rest"][name] = {"max_displacement_m": moved, "max_speed": speed}
    assert moved < REST_M and speed < REST_SPEED, (name, result["rest"][name])
    sim.close()

# Hidden parts: segmentation from VIEWS viewpoints, upright and lying; grasp rays; bit-identical renders across variants.
def hidden_geoms(m):
    names = {"cap", "liner", *CONTENT_BODIES}
    bodies = {i for i in range(m.nbody) if m.body(i).name in names}
    return {g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies or re.match(r"(neck|lug|shell_tooth)", m.geom(g).name)}


def fibonacci(n):
    k = np.arange(n) + .5
    z, phi = 1 - 2 * k / n, np.pi * (1 + 5 ** .5) * k
    return np.column_stack([np.sqrt(1 - z ** 2) * np.cos(phi), np.sqrt(1 - z ** 2) * np.sin(phi), z])


def lay_down(sim):
    """Pose every free body rotated 90 deg about the world x axis about the bottle's base edge, lying on the table."""
    m, d = sim._model, sim._data
    turn = np.zeros(4)
    mujoco.mju_axisAngle2Quat(turn, [1, 0, 0], np.pi / 2)
    pivot = np.array([0, 0, S.Z0 + S.SHELL_R[1]])  # the cap (the widest part) rests on the table
    for j in np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
        a = m.jnt_qposadr[j]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, turn)
        d.qpos[a:a + 3] = rotation.reshape(3, 3) @ (d.qpos[a:a + 3] - [0, 0, S.Z0]) + pivot
        q = d.qpos[a + 3:a + 7].copy()
        mujoco.mju_mulQuat(d.qpos[a + 3:a + 7], turn, q)
    mujoco.mj_forward(m, d)
    return pivot + [0, -.05, 0]


renders, result["hidden"] = {}, {}
for name, path in scenes.items():
    sim = Simulation(path, {**CONFIG, "movable_camera": True})
    m, d = sim._model, sim._data
    hidden = hidden_geoms(m)
    renderer = mujoco.Renderer(m, height=240, width=320)
    record = {"hidden_geoms": len(hidden), "views": 0, "hidden_pixels": 0, "ray_hits_hidden": 0, "rays": 0}
    for pose in ("upright", "lying"):
        sim.action({"op": "reset"}, render=False)
        centre = np.array([0, 0, S.Z0 + .055]) if pose == "upright" else lay_down(sim)
        for i, direction in enumerate(fibonacci(VIEWS)):
            eye = centre + VIEW_R * direction
            if eye[2] < S.Z0 + .03:  # below or at the table top (the camera keeps 2 cm from every surface)
                continue
            sim._aim_camera(eye, centre)
            renderer.update_scene(d, camera=sim._camera_id)
            renderer.enable_segmentation_rendering()
            seg = renderer.render().copy()
            renderer.disable_segmentation_rendering()
            geoms = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
            record["hidden_pixels"] += int(np.isin(geoms, list(hidden)).sum())
            record["views"] += 1
            renderer.update_scene(d, camera=sim._camera_id)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            renders.setdefault(name, []).append((rgb, depth))
            if i % 8 == 0:  # grasp rays through every 4th pixel
                for v in range(0, 240, 4):
                    for u in range(0, 320, 4):
                        cam = sim._camera_id
                        scale = np.tan(np.deg2rad(m.cam_fovy[cam]) / 2)
                        ray = d.cam_xmat[cam].reshape(3, 3) @ [(2 * (u + .5) / 320 - 1) * 320 / 240 * scale, (1 - 2 * (v + .5) / 240) * scale, -1.]
                        geom = np.array([-1], dtype=np.int32)
                        mujoco.mj_ray(m, d, d.cam_xpos[cam], ray / np.linalg.norm(ray), None, 1, -1, geom)
                        record["rays"] += 1
                        record["ray_hits_hidden"] += int(geom[0] in hidden)
        if pose == "upright":
            sim._aim_camera(np.array([.12, -.16, S.Z0 + .14]), centre)
            renderer.update_scene(d, camera=sim._camera_id)
            Image.fromarray(renderer.render()).save(OUT / f"{name}-{pose}.png")
    renderer.close()
    sim.close()
    result["hidden"][name] = record
    assert record["hidden_pixels"] == 0 and record["ray_hits_hidden"] == 0, (name, record)
result["identical_renders"] = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(*renders.values()))
assert result["identical_renders"] and len(renders["bottle_empty"]) == len(renders["bottle_contents"]), "closed bottles render differently"

# Mechanism, through the harness weld grasps (bottle.py's own probe actions).
def weld_rows(sim):
    eq = [sim._grasps[i][0] for i in range(2) if sim._grasps[i] is not None]
    rows = np.flatnonzero((sim._data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & np.isin(sim._data.efc_id, eq))
    return rows


def procedure(path, push, turn, two=True, table=None):
    """Outcome, push force, samples, and the max weld translation residual and speed over the whole probe."""
    table = table or tables["bottle_contents"]["bottle"]
    sim = Simulation(path, CONFIG)
    shell_px, bottle_px = B.pixels(sim, table)
    aim = B.targets(sim, shell_px, bottle_px)
    assert aim is not None
    peak = {"violation_m": 0., "violation_rad": 0., "speed": 0., "bottle_cap_speed": 0.}
    mechanism = np.flatnonzero(np.isin(sim._model.dof_bodyid, [sim._model.body(n).id for n in shared]))

    def watch(current):
        rows = weld_rows(current)
        for key, part in (("violation_m", [r for k, r in enumerate(rows) if k % 6 < 3]),
                          ("violation_rad", [r for k, r in enumerate(rows) if k % 6 >= 3])):
            if part:
                peak[key] = max(peak[key], float(np.abs(current._data.efc_pos[part]).max()))
        peak["speed"] = max(peak["speed"], float(np.abs(current._data.qvel).max()))
        peak["bottle_cap_speed"] = max(peak["bottle_cap_speed"], float(np.abs(current._data.qvel[mechanism]).max()))
    sim._callback = watch
    times, samples, opened, force, held = B.run(sim, B.actions(table, shell_px, bottle_px, push, turn, two), aim, table["open_lift_m"])
    sim.close()
    return {"opened": opened, "push_force_n": force, "grasps_held": held, **peak,
            "max_cap_above_bottle_m": float(np.max(samples[:, 2] - samples[:, 3])),
            "cap_turn_deg": float(np.degrees(samples[-1, 0])), "bottle_turn_deg": float(np.degrees(samples[-1, 1]))}, samples


table = tables["bottle_contents"]["bottle"]
path = scenes["bottle_contents"]
mech = result["mechanism"] = {}
# Force gate (two welds, push, quarter turn, lift): twist only (0 mm) and every push whose force stays below the shell's
# preload leave it closed; every push past the preload plus the tooth stroke opens it.
sweep = {d: procedure(path, d, table["turn_deg"]) for d in table["push_m"]}
mech["force_sweep"] = {f"{1000 * d:.1f}mm": r for d, (r, _) in sweep.items()}
(OUT / "check.json").write_text(json.dumps(result, indent=2))
below = [r for r in mech["force_sweep"].values() if r["push_force_n"] < S.SHELL_PRELOAD_N]
above = [r for r in mech["force_sweep"].values() if r["push_force_n"] > S.SHELL_PRELOAD_N + S.SHELL_K * S.SHELL_TRAVEL]
assert not mech["force_sweep"]["0.0mm"]["opened"], "twist only opened it"
assert len(below) >= 2 and above and not any(r["opened"] for r in below) and all(r["opened"] for r in above), mech["force_sweep"]
deepest = table["push_m"][-1]
mech["repeat_deepest"], second = procedure(path, deepest, table["turn_deg"])
mech["deterministic"] = bool(np.array_equal(sweep[deepest][1], second))
assert mech["deterministic"], "the scripted opening is not deterministic"
for key, r in mech["force_sweep"].items():  # stable: grasps hold, small weld residuals, no blow-up of bottle or cap
    assert r["grasps_held"] and r["violation_m"] < MAX_VIOLATION_M and r["bottle_cap_speed"] < MAX_SPEED, (key, r)
mech["one_grasp"] = {f"{1000 * d:.1f}mm": procedure(path, d, table["turn_deg"], two=False)[0]
                     for d in sorted({*table["push_m"], table["one_grasp_push_m"]})}
mech["angle_sweep"] = {f"{a}deg": procedure(path, table["push_m"][-1], a)[0]["opened"] for a in table["angles_deg"]}
assert mech["angle_sweep"][f"{table['turn_deg']}deg"], "the quarter turn does not open at the deepest probe push"
(OUT / "check.json").write_text(json.dumps(result, indent=2))


# Metric on reference candidates.
def zipped(text):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("scene.xml", text)
    return buffer.getvalue()


def evaluate(level, label, text):
    output = OUT / f"eval-{level}-{label}"
    shutil.rmtree(output, ignore_errors=True)
    report = evaluate_submission(zipped(text), {"source_dir": str(S.ROOT / level), "output_dir": str(output),
                                                "metrics": ["bottle", "objects", "chamfer", "center_of_mass"], "camera": "fixed",
                                                "width": 320, "height": 240, "reconstruction": {"samples": 2048, "seed": 20261029},
                                                "source_config": {"grasp": "weld"}})
    assert report["status"] == "ok", report.get("error")
    scores = report["scores"]
    bottle, objects = scores["bottle"], scores["objects"]
    return {"outcome_agreement": bottle["outcome_agreement"], "probes": {p["name"]: [p["source_opened"], p["candidate_opened"]] for p in bottle["probes"]},
            "cap_lift_rms_m": bottle["mean_cap_lift_rms_m"], "cap_rotation_rms_rad": bottle["mean_cap_rotation_rms_rad"],
            "engage": bottle["engage"], "angle": bottle["angle"], "chamfer_m": scores["reconstruction"].get("chamfer_distance_m"),
            "com_error_m": scores["reconstruction"].get("center_of_mass_error_m"), "found_fraction": objects.get("found_fraction"),
            "objects": {r["name"]: {"candidate": r["candidate_body"], "chamfer_m": r["chamfer_m"], "centroid_error_m": r["centroid_error_m"],
                                    "found": r["found"], "reason": r.get("unassigned_reason")} for r in objects["per_object"]}}


text = {name: path.read_text() for name, path in scenes.items()}
no_teeth = re.sub(r"  <contact>.*</contact>\n", "", text["bottle_contents"], flags=re.S)
no_lugs = re.sub(r'\s*<geom name="lug[^>]*/>', "", text["bottle_contents"])
wrong = text["bottle_contents"].replace('type="box" size="0.006000 0.006000 0.006000"', 'type="sphere" size="0.006000"')
assert no_teeth != text["bottle_contents"] and no_lugs != text["bottle_contents"] and wrong != text["bottle_contents"]
metric = result["metric"] = {}
metric["empty_identity"] = evaluate("bottle_empty", "identity", text["bottle_empty"])
metric["contents_identity"] = evaluate("bottle_contents", "identity", text["bottle_contents"])
metric["no_teeth"] = evaluate("bottle_contents", "no-teeth", no_teeth)
metric["no_lugs"] = evaluate("bottle_contents", "no-lugs", no_lugs)
metric["wrong_contents"] = evaluate("bottle_contents", "wrong-contents", wrong)
metric["missing_contents"] = evaluate("bottle_contents", "missing-contents", text["bottle_empty"])
(OUT / "check.json").write_text(json.dumps(result, indent=2))
for key in ("empty_identity", "contents_identity"):
    r = metric[key]
    assert r["outcome_agreement"] == 1 and r["cap_lift_rms_m"] < 1e-9 and r["engage"]["engage_force_error_n"] == 0, (key, r)
    assert r["angle"]["opening_angle_error_deg"] == 0 and r["chamfer_m"] < 1e-9 and r["com_error_m"] < 1e-9 and r["found_fraction"] == 1, (key, r)
assert metric["no_teeth"]["outcome_agreement"] < 1 and metric["no_teeth"]["engage"]["reason"] == "never_opens", metric["no_teeth"]
assert metric["no_teeth"]["engage"]["engage_force_error_n"] is None and metric["no_teeth"]["angle"]["opening_angle_error_deg"] is None
assert metric["no_lugs"]["outcome_agreement"] < 1, metric["no_lugs"]
assert metric["wrong_contents"]["objects"]["cube"]["chamfer_m"] > 1e-4, metric["wrong_contents"]
missing = metric["missing_contents"]["objects"]
assert all(missing[c]["candidate"] is None and missing[c]["chamfer_m"] is None and missing[c]["reason"] for c in CONTENT_BODIES), missing
assert metric["missing_contents"]["com_error_m"] > 1e-4, metric["missing_contents"]
print(json.dumps(result, indent=2))
print("BOTTLE CHECK PASSED")
