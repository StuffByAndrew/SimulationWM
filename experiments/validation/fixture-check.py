"""Check the positioning-fixture levels (Slurm test node only): .venv/bin/python experiments/validation/fixture-check.py OUTDIR [LEVEL ...]
Files equal the generator output; each scene compiles and holds a cocked carriage at rest; public observations carry only the
dial readings; the pad is invisible from every camera pose around the caliper; a scripted Franka sets both dials with IK; the
rigid level repeats, the rubber level spreads with the settled pad state and its strip buckles and snaps (stick-slip), the bimodal level splits
into two stopping ranges by fold direction; and the metric ranks an empirical predictor above a mean-only one, with the
mean-only choice failing the narrow target on the bimodal level. Writes OUTDIR/<level>/{check.json,*.png,<predictor>/view.html}."""
import importlib.util
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from PIL import Image

sys.path[:0] = ["src", "tools/planning"]
from plan import Panda  # noqa: E402
from sim2sim.candidate_worker import public_observation  # noqa: E402
from sim2sim.fixture import DIAL_M, collect_predictions, evaluate, release_trial  # noqa: E402
from sim2sim.simulation import Simulation  # noqa: E402

spec = importlib.util.spec_from_file_location("scenes", "experiments/validation/fixture-scenes.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
OUT = Path(sys.argv[1])
LEVELS = sys.argv[2:] or list(S.LEVELS)
PUBLIC = {"robot": "panda", "width": 320, "height": 240, "rgb": True, "depth": True, "calibration": True, "movable_camera": True}
DOWN = [0, np.cos(np.pi / 4), np.sin(np.pi / 4), 0]  # fingertips down, fingers closing along world x
for path, text in S.outputs().items():
    assert path.read_text() == text, f"{path} differs from the generator output"


def png(sim, name, folder):
    Image.open(__import__("io").BytesIO(__import__("base64").b64decode(sim.observe(full=True)["rgb"]))).save(folder / name)


def pad_pixels(sim):
    """Pad holder/pad geometry visible from camera poses all around the caliper (segmentation pixels per view)."""
    model = sim._model
    bodies = {b for b in range(model.nbody) if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) in ("pad_holder",)
              or (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("pad_seg")}
    renderer = mujoco.Renderer(model, 240, 320)
    renderer.enable_segmentation_rendering()
    centre, counts = np.array([S.XP, S.YR, S.BLADE_TOP + .01]), []
    for height in (-.012, .01, .05, .15):
        for azimuth in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            for radius in (.06, .15):
                pos = centre + [radius * np.cos(azimuth), radius * np.sin(azimuth), height]
                try:
                    sim.action({"op": "camera", "pos": pos.tolist(), "lookat": centre.tolist()}, render=False)
                except ValueError:
                    continue  # inside or against geometry: the controller refuses that pose
                renderer.update_scene(sim._data, camera=sim._camera_id)
                ids = renderer.render()
                geoms = ids[:, :, 0][(ids[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids[:, :, 0] >= 0)]
                counts.append(int(np.isin(model.geom_bodyid[geoms], list(bodies)).sum()))
    renderer.close()
    return counts


def drive(sim, q, width, ticks):
    sim.action({"op": "arm_target", "joint_targets": list(map(float, q)), "gripper_width_m": width, "ticks": ticks}, render=False)


def set_dial(sim, panda, name, reading, tolerance=.15):
    """Franka, closed loop on the public dial reading: straddle the peg with the fully open gripper, close on it (the coupled
    fingers centre it: form closure along the slide), slide it by the remaining error, open and lift; repeat until within
    tolerance. Returns the readings after each stroke."""
    strokes = []
    for _ in range(5):
        current = sim.fixture.readings()[name]
        if abs(current - reading) < tolerance:
            return strokes
        x, y, z = S.DIALS[name] + current * DIAL_M, S.YD, S.TOP + .055
        above = panda.ik([x, y, z + .1], DOWN, start=np.array(sim._data.qpos[sim._arm_qpos]))
        assert above["converged"], above
        drive(sim, above["joint_targets"], .08, 400)
        shift = (reading - current) * DIAL_M
        for target, width, step_ticks in (([x, y, z], .08, 12), ([x + shift, y, z], 0., 25), ([x + shift, y, z + .1], .08, 12)):
            q = np.array(sim._data.qpos[sim._arm_qpos])
            if width == 0.:
                drive(sim, q, 0., 150)  # close on the peg before sliding
            path = panda.linear(q, target, DOWN, step=.005)
            assert path["reached"], path
            for waypoint in path["joint_targets"]:
                drive(sim, waypoint, width, step_ticks)
            drive(sim, path["joint_targets"][-1], width, 300)  # let the arm finish tracking before letting go
            drive(sim, path["joint_targets"][-1], .08, 150)
        strokes.append(sim.fixture.readings()[name])
    raise AssertionError(f"{name} did not reach {reading}: {strokes}")


def pad_dynamics(sim, u, v, rng):
    """One release with private per-tick pad data while the carriage slides (index past 2 mm, faster than 20 mm/s): the
    holder's height range, buckle-and-snap events (holder height peaks more than 1 mm above the lowest height within
    +-10 ticks) and the brake force fluctuation (coefficient of variation of the 5-tick mean of -m dv/dt)."""
    sim.action({"op": "reset"}, render=False)
    sim.fixture.set_dials(u, v)
    sim.fixture.reset_carriage(rng)
    sim.fixture.release()
    mass = float(sim._model.body_subtreemass[mujoco.mj_name2id(sim._model, mujoco.mjtObj.mjOBJ_BODY, "carriage")])
    heights, forces = [], []
    for _ in range(1500):
        before = sim._data.qvel[sim.fixture.dof["carriage_x"]]
        sim._advance(1)
        after = sim._data.qvel[sim.fixture.dof["carriage_x"]]
        if sim.fixture.position_mm() > 2 and after > .02:
            heights.append(1000 * sim._data.qpos[sim.fixture.qpos["pad_z"]])
            forces.append(-mass * (after - before) / sim._model.opt.timestep)
    z, f = np.array(heights), np.convolve(forces, np.ones(5) / 5, mode="valid")
    snaps = sum(z[i] == z[max(0, i - 10):i + 11].max() and z[i] - z[max(0, i - 10):i + 11].min() > 1 for i in range(len(z)))
    return {"pad_height_range_mm": float(np.ptp(z)), "snaps": int(snaps), "brake_force_cv": float(f.std() / f.mean()), "slide_ticks": len(z)}


def settled(sim, u, v, rng):
    sim.action({"op": "reset"}, render=False)
    sim.fixture.set_dials(u, v)
    sim.fixture.reset_carriage(rng)
    root = mujoco.mj_name2id(sim._model, mujoco.mjtObj.mjOBJ_JOINT, "pad_hinge_0")
    return {"pad_z_mm": 1000 * float(sim._data.qpos[sim.fixture.qpos["pad_z"]]),
            "fold": None if root < 0 else ("tucked" if sim._data.qpos[sim.fixture.hinge_qpos].sum() < 0 else "back")}  # net fold of the strip


for level in LEVELS:
    folder = OUT / level
    folder.mkdir(parents=True, exist_ok=True)
    scene = Path("environments/panda_small") / level / "scene.xml"
    table = __import__("tomllib").loads((scene.parent / "manifest.toml").read_text())["fixture"]
    targets = {t["name"]: t for t in table["targets"]}
    sim = Simulation(scene, {**PUBLIC, "fixture_seed": 1})
    report = {"level": level}
    started = time.time()
    sim.action({"op": "reset_carriage"}, render=False)
    report["step_us"] = (time.time() - started) / sim.fixture.settle_ticks * 1e6
    before = sim.fixture.position_mm()
    sim._advance(500)
    report["cocked_rest_drift_mm"] = abs(sim.fixture.position_mm() - before)
    assert report["cocked_rest_drift_mm"] < .05, report
    public = public_observation(sim.observe(full=True), PUBLIC)
    report["public_keys"] = sorted(public)
    assert set(public["fixture"]) == {"compression_dial", "pressure_dial", "carriage_cocked"} and "pad" not in json.dumps(
        {k: v for k, v in public.items() if k not in ("rgb", "depth")})
    png(sim, "fixed.png", folder)
    sim.action({"op": "camera", "pos": [.12, -.34, .56], "lookat": [.06, -.17, .42]}, render=False)
    png(sim, "ruler.png", folder)
    counts = pad_pixels(sim)
    report["pad_visibility"] = {"views": len(counts), "max_pad_pixels": max(counts), "views_with_pad_pixels": sum(c > 0 for c in counts)}
    assert len(counts) > 50 and max(counts) == 0, report["pad_visibility"]

    # Scripted Franka sets both dials, then the release from the robot-set dials matches the evaluator's direct setting.
    sim = Simulation(scene, {**PUBLIC, "fixture_seed": 1})
    panda = Panda(scene, ignore_bodies=list(S.DIALS))
    report["franka_strokes"] = {name: set_dial(sim, panda, name, value) for name, value in (("compression_dial", 8.0), ("pressure_dial", 3.0))}
    drive(sim, [0, 0, 0, -1.57079, 0, 1.57079, -.7853], .08, 500)
    readings = report["franka_dials"] = {name: sim.fixture.readings()[name] for name in S.DIALS}
    sim.fixture.rng = np.random.default_rng(1)  # the same perturbation draw as the direct trial below
    sim.action({"op": "reset_carriage"}, render=False)
    sim.action({"op": "release_carriage", "ticks": 1500}, render=False)
    robot_stop = sim.fixture.position_mm()
    direct = release_trial(Simulation(scene, PUBLIC), readings["compression_dial"], readings["pressure_dial"], np.random.default_rng(1), 1500, 5)
    report["franka_release_mm"], report["direct_release_mm"] = robot_stop, direct["stop_mm"]
    png(sim, "after_franka_release.png", folder)

    # Distributions over the held-out grid (eval-identical trials) with the settled pad state of each trial.
    sim = Simulation(scene, PUBLIC)
    grid = []
    for index, (u, v) in enumerate(table["settings"]):
        trials = []
        for r in range(int(table["repeats"])):
            state = settled(sim, u, v, np.random.default_rng([int(table["seed"]), index, r]))
            stop = release_trial(sim, u, v, np.random.default_rng([int(table["seed"]), index, r]), int(table["max_release_ticks"]), 50)["stop_mm"]
            trials.append({**state, "stop_mm": stop})
        stops = np.array([t["stop_mm"] for t in trials])
        grid.append({"setting": [u, v], "mean_mm": float(stops.mean()), "sd_mm": float(stops.std()), "trials": trials})
    free = [g for g in grid if max(t["stop_mm"] for t in g["trials"]) < S.TRAVEL * 1000 - 1]  # settings not reaching the end bumper
    report["grid_sd_mm"] = {"max": max(g["sd_mm"] for g in free), "median": float(np.median([g["sd_mm"] for g in free])), "settings": len(free)}
    dynamics = [pad_dynamics(sim, u, v, np.random.default_rng([3, i])) for i, (u, v) in enumerate([(5, 2), (7, 4), (10, 6)] * 4)]
    report["pad_dynamics"] = {"per_release": dynamics, **{f"median_{k}": float(np.median([d[k] for d in dynamics]))
                                                         for k in ("pad_height_range_mm", "snaps", "brake_force_cv")}}
    if level == "fixture_rigid":
        assert report["grid_sd_mm"]["max"] < .5, report["grid_sd_mm"]
    if level == "fixture_rubber":
        assert report["grid_sd_mm"]["median"] > 3, report["grid_sd_mm"]
        r = [np.corrcoef([t["pad_z_mm"] for t in g["trials"]], [t["stop_mm"] for t in g["trials"]])[0, 1] for g in free if g["sd_mm"] > 1]
        report["settle_height_stop_correlation"] = {"median": float(np.median(r)), "per_setting": [round(float(x), 3) for x in r]}
        assert report["pad_dynamics"]["median_snaps"] >= 1, report["pad_dynamics"]
    if level == "fixture_bimodal":
        folds = [t["fold"] for g in grid for t in g["trials"]]
        separated = [max(t["stop_mm"] for t in g["trials"] if t["fold"] == "tucked") < min(t["stop_mm"] for t in g["trials"] if t["fold"] == "back")
                     for g in grid if {t["fold"] for t in g["trials"]} == {"tucked", "back"}]
        report["bimodal"] = {"tucked_fraction": folds.count("tucked") / len(folds), "settings_with_both_folds": len(separated),
                             "fold_separates_ranges": sum(separated)}
        assert .2 < report["bimodal"]["tucked_fraction"] < .9 and sum(separated) >= .8 * len(separated) >= 10, report["bimodal"]
    (folder / "grid.json").write_text(json.dumps(grid))

    # Metric sanity: an empirical predictor (independent releases) against a mean-only one, each with its own chosen settings.
    search = {(u, v): np.array([release_trial(sim, u, v, np.random.default_rng([99, int(10 * u), int(10 * v), r]), 2500, 100)["stop_mm"]
                                for r in range(12)]) for u in np.linspace(0, 10, 11) for v in np.linspace(0, 10, 11)}
    search.update({tuple(setting): np.array([release_trial(sim, *setting, np.random.default_rng([98, i, r]), 2500, 100)["stop_mm"] for r in range(16)])
                   for i, setting in enumerate(table["settings"])})
    hit = lambda y, t: float(np.mean((y >= t["lo_mm"]) & (y <= t["hi_mm"])))
    centre = lambda t: (t["lo_mm"] + t["hi_mm"]) / 2
    choosers = {"empirical": lambda t: max(search, key=lambda s: (hit(search[s], t), -abs(np.median(search[s]) - centre(t)))),
                "mean_only": lambda t: min(search, key=lambda s: abs(search[s].mean() - centre(t)))}
    samplers = {"empirical": lambda y, n: np.random.default_rng(0).choice(y, n).tolist(), "mean_only": lambda y, n: [float(y.mean())] * n}
    report["metric_sanity"] = {}
    for name in choosers:
        chosen = {t: list(choosers[name](targets[t])) for t in targets}
        def call(function, **kwargs):
            if function == "settings":
                return {"ok": True, "value": {t: {"compression": u, "pressure": v} for t, (u, v) in chosen.items()}}
            return {"ok": True, "value": samplers[name](search[(kwargs["compression"], kwargs["pressure"])], kwargs["n"])}
        scores = evaluate(sim, table, collect_predictions(table, call), folder / name / "fixture")
        report["metric_sanity"][name] = {"chosen": chosen, "prediction": scores["prediction"],
                                         "targets": {t: {k: s[k] for k in ("measured_hit_fraction", "predicted_hit_fraction")} for t, s in scores["targets"].items()}}
    sanity = report["metric_sanity"]
    assert sanity["empirical"]["prediction"]["crps_mm"] <= sanity["mean_only"]["prediction"]["crps_mm"] + .05, sanity
    if level == "fixture_bimodal":
        assert sanity["mean_only"]["targets"]["narrow_target"]["measured_hit_fraction"] < .1 < sanity["empirical"]["targets"]["narrow_target"]["measured_hit_fraction"], sanity
    (folder / "check.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "pad_dynamics"}, indent=1), flush=True)
    print(level, "pad_dynamics", json.dumps({k: v for k, v in report["pad_dynamics"].items() if k != "per_release"}), flush=True)
print("fixture check passed:", LEVELS)
