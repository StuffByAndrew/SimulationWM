"""Check the gear-train level and prove its premise (Slurm test node only), from the repository root:
PYTHONPATH=src .venv/bin/python experiments/validation/gear-check.py OUTDIR

Files equal the generator output (including the physics-computed gate); nothing has a gravity torque and the scene rests;
no moving geom can collide; the shallow reference's lumped crank inertia equals the deep train's effective inertia; source
probes are deterministic; every gear renders in RGB and depth from the fixed camera and refuses the grasp while crank and hand
accept it; the wave 28 configs carry the generator's public grasp softness. Through the real evaluator (evaluate_submission,
metric gear_train, the level's grasp): the deep reference (a copy of the source) and the shallow reference at the source
timestep score within noise on every probe, hard crank pulls and springback included (each probe's crank and hand RMS error
< GATE_SHARE of the gate); shallow.xml at its own larger timestep passes the gate, and the deep reference costs >= 10x it per
simulated second; the classes are deep / shallow, G-cheap passes both, removals answer for deep and are cannot_answer for
shallow; a renamed deep copy with flipped crank and hand hinge axes scores like the source; and the unmodified shallow
model's prediction differs from every removed-gear source. Writes OUTDIR/{check.json, *.png, eval-*}."""
import base64
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, "src")
from sim2sim.evaluation import evaluate_submission  # noqa: E402
from sim2sim.gear_train import compile_plain, panel_normal, probe_errors, reference_runs, remove_body  # noqa: E402
from sim2sim.simulation import Simulation  # noqa: E402

spec = importlib.util.spec_from_file_location("scenes", "experiments/validation/gear-scenes.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
GATE_SHARE = .1      # "within noise": the shallow reference's probe error is below this share of the G-cheap gate
COST_RATIO = 10.     # the deep reference costs at least this many times the shallow one
MIN_PIXELS = 15      # each gear's visible pixels from the fixed camera (320 x 240)
result = {}
for path, text in S.outputs().items():
    assert path.read_text() == text, f"{path} differs from the generator output"
SCENE, SHALLOW = S.ROOT / "scene.xml", S.ROOT / "shallow.xml"
table = __import__("tomllib").loads((S.ROOT / "manifest.toml").read_text())
gears = [b["name"] for b in S.train()[1:-1]]
GRASP = {"grasp_solref": S.GRASP_SOLREF, "grasp_solimp": S.GRASP_SOLIMP}
for path in sorted(Path("experiments/configs/wave28-20261124").glob("*.toml")):
    assert {k: v for k, v in __import__("tomllib").loads(path.read_text())["simulation"].items() if k in GRASP} == GRASP, path

# Statics: balanced bodies, rest, no collisions.
model, data = compile_plain(SCENE)
for j in range(model.njnt):
    b = model.jnt_bodyid[j]
    assert np.linalg.norm(np.cross(data.xipos[b] - data.xanchor[j], data.xaxis[j])) < 1e-6, f"{model.body(b).name} has a gravity torque"
moving = model.geom_bodyid != 0
assert not model.geom_contype[moving].any() and not model.geom_conaffinity[moving].any(), "a moving geom can collide"
mujoco.mj_step(model, data, nstep=1000)
result["rest_max_speed_rad_s"] = float(np.abs(data.qvel).max())
assert result["rest_max_speed_rad_s"] < 1e-9 and data.ncon == 0

# Lumped inertia: shallow crank armature + crank + hand r^2 equals the deep train's effective inertia v^T M v at the crank.
speeds = np.array([b["speed"] for b in S.train()])
M = np.zeros((model.nv, model.nv))
mujoco.mj_fullM(model, data, M)
shallow_model, shallow_data = compile_plain(SHALLOW)
Ms = np.zeros((2, 2))
mujoco.mj_fullM(shallow_model, shallow_data, Ms)
deep_inertia, shallow_inertia = float(speeds @ M @ speeds), float(np.array([1, speeds[-1]]) @ Ms @ np.array([1, speeds[-1]]))
result["effective_crank_inertia_kg_m2"] = {"deep": deep_inertia, "shallow": shallow_inertia}
assert abs(deep_inertia - shallow_inertia) < 1e-6 * deep_inertia
result["lumped"] = dict(zip(("armature_kg_m2", "damping_n_m_s"), S.lumped()))

# Renders: every gear visible in RGB and nearer than the panel in depth; grasps refuse gears.
source = Simulation(SCENE, {**S.SOURCE_CONFIG, "rgb": True, "depth": True, "movable_camera": True})
bare = Simulation(SHALLOW, {**S.CANDIDATE_CONFIG, "rgb": True, "depth": True})
obs, obs_bare = source.observe(full=True), bare.observe(full=True)
for name, o in (("rgb", obs), ("depth", obs), ("rgb_shallow", obs_bare)):
    Image.open(io.BytesIO(base64.b64decode(o[name.split("_")[0]]))).save(OUT / f"{name}.png")
source._renderer.enable_segmentation_rendering()
source._renderer.update_scene(source._data, camera=source.camera)
seg = source._renderer.render().copy()
source._renderer.disable_segmentation_rendering()
depth = np.asarray(Image.open(io.BytesIO(base64.b64decode(obs["depth"])))).astype(float)
depth_bare = np.asarray(Image.open(io.BytesIO(base64.b64decode(obs_bare["depth"])))).astype(float)
is_geom = (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[:, :, 0] >= 0)
body_of = np.where(is_geom, source._model.geom_bodyid[np.maximum(seg[:, :, 0], 0)], -1)
result["gears"] = {}
for name in gears:
    bid = mujoco.mj_name2id(source._model, mujoco.mjtObj.mjOBJ_BODY, name)
    ys, xs = np.nonzero(body_of == bid)
    nearer = float(np.median(depth_bare[ys, xs] - depth[ys, xs])) if len(xs) else None
    k = np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
    source.action({"op": "grasp", "pixel": [float(xs[k]), float(ys[k])]}, render=False)
    refused = source._grasp is None
    source.action({"op": "release"}, render=False)
    result["gears"][name] = {"pixels": int(len(xs)), "median_nearer_than_panel_mm": nearer, "grasp_refused": refused}
    assert len(xs) >= MIN_PIXELS and nearer > 2 and refused, (name, result["gears"][name])
source._aim_camera([0., -.42, S.Z0 + .12], [0., 0., S.Z0])
Image.open(io.BytesIO(base64.b64decode(source.observe(full=True)["rgb"]))).save(OUT / "rgb_close.png")
source.close(), bare.close()

# Determinism of the source probes.
normal = panel_normal(*compile_plain(SCENE))
probes = table["probes"]
runs = [reference_runs(Simulation(SCENE, S.SOURCE_CONFIG), probes, normal) for _ in range(2)]
assert all(np.array_equal(a[4], b[4]) for a, b in zip(*runs)), "source probes are not deterministic"
result["probe_ranges_rad"] = {run[0]: [float(np.ptp(run[4][:, 0])), float(np.ptp(run[4][:, 1]))] for run in runs[0]}


def zipped(text):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("scene.xml", text)
    return buffer.getvalue()


def evaluate(name, text):
    output = OUT / f"eval-{name}"
    shutil.rmtree(output, ignore_errors=True)
    report = evaluate_submission(zipped(text), {"source_dir": str(S.ROOT), "output_dir": str(output), "metrics": ["gear_train"],
                                                "camera": "fixed", "width": 320, "height": 240, **GRASP})
    assert report["status"] == "ok", report.get("error")
    return report["scores"]["gear_train"]


flipped = SCENE.read_text().replace('name="crank_hinge" type="hinge" axis="0 1 0"', 'name="crank_hinge" type="hinge" axis="0 -1 0"')
flipped = flipped.replace('name="hand_hinge" type="hinge" axis="0 1 0"', 'name="hand_hinge" type="hinge" axis="0 -1 0"')
for line in [l for l in flipped.splitlines() if 'joint2="crank_hinge"' in l or 'joint1="hand_hinge"' in l]:
    c = line.split('polycoef="0 ')[1].split()[0]
    flipped = flipped.replace(line, line.replace(f'polycoef="0 {c} ', f'polycoef="0 {-float(c):.12g} '))
flipped = flipped.replace("gear_", "wheel")
scores = {name: evaluate(name, text) for name, text in (("deep", SCENE.read_text()), ("shallow", S.scene(True)), ("shallow_own_step", SHALLOW.read_text()),
                                                       ("flipped", flipped))}
gate = table["gear_train"]["cheap_gate_rad"]
for name, s in scores.items():
    result[name] = {"probe_error_rad": s["accuracy"]["probe_error_rad"], "probes": s["accuracy"]["probes"],
                    "depth_class": s["structure"]["depth_class"], "matched": s["structure"]["matched"],
                    "timestep_s": s["accuracy"]["candidate_timestep_s"],
                    "seconds_per_simulated_second": s["sim_cost"]["candidate"]["seconds_per_simulated_second"],
                    "cost_relative_to_shallow": s["sim_cost"]["candidate"]["relative_to_shallow"], "counts": s["sim_cost"]["candidate"],
                    "cheap": s["cheap"], "removal": s["removal"]}
result["gate_rad"] = gate
result["cost_ratio_deep_over_shallow"] = [s["sim_cost"]["deep_reference"]["relative_to_shallow"] for s in scores.values()]
print(json.dumps(result, indent=1))
(OUT / "check.json").write_text(json.dumps(result, indent=1))
deep, shallow, own, flip = scores["deep"], scores["shallow"], scores["shallow_own_step"], scores["flipped"]
assert deep["accuracy"]["probe_error_rad"] < 1e-9 and flip["accuracy"]["probe_error_rad"] < 1e-6, "copies must score exactly"
assert max(max(p["crank_rms_rad"], p["hand_rms_rad"]) for p in shallow["accuracy"]["probes"]) < GATE_SHARE * gate, "shallow not within noise"
assert min(result["cost_ratio_deep_over_shallow"]) >= COST_RATIO, "deep is not >= 10x the shallow cost"
assert (deep["structure"]["depth_class"], shallow["structure"]["depth_class"], flip["structure"]["depth_class"]) == ("deep", "shallow", "deep")
assert deep["structure"]["matched"] == len(gears) == flip["structure"]["matched"]
assert deep["cheap"]["status"] == shallow["cheap"]["status"] == own["cheap"]["status"] == "ok" and own["structure"]["depth_class"] == "shallow"
assert own["accuracy"]["candidate_timestep_s"] > S.TIMESTEP and abs(own["cost_relative_to_shallow"] - 1) < .5  # same model, timed twice
assert deep["removal"]["answered"] == len(table["gear_train"]["removals"]) and deep["removal"]["mean_probe_error_rad"] < 1e-9
assert flip["removal"]["answered"] == len(table["gear_train"]["removals"]) and flip["removal"]["mean_probe_error_rad"] < 1e-6
assert shallow["removal"]["cannot_answer"] == len(table["gear_train"]["removals"])

# Removal changes the answer: the unmodified shallow model against each removed-gear source.
plain = compile_plain(SCENE)
result["shallow_vs_removed_source_rad"] = {}
for name in table["gear_train"]["removals"]:
    removed = Simulation(SCENE, {**S.SOURCE_CONFIG, "scene_edit": remove_body(plain[0], mujoco.mj_name2id(plain[0], mujoco.mjtObj.mjOBJ_BODY, name))})
    error = probe_errors(reference_runs(removed, probes, normal), Simulation(SHALLOW, S.CANDIDATE_CONFIG), normal)["probe_error_rad"]
    result["shallow_vs_removed_source_rad"][name] = error
    assert error > 10 * gate, (name, error)
(OUT / "check.json").write_text(json.dumps(result, indent=1))
print(json.dumps({k: result[k] for k in ("gate_rad", "cost_ratio_deep_over_shallow", "shallow_vs_removed_source_rad")}, indent=1))
print("gear-check passed")
