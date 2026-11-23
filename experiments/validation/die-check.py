"""Check the biased-die levels on a Slurm test node (never the login node), from the repository root:
PYTHONPATH=src .venv/bin/python experiments/validation/die-check.py OUTDIR [--throws N]
Files equal die-scenes.py output; both scenes compile and the die rests on face 1; the slug is invisible to RGB, depth and
the grasp ray; launches are deterministic; protocols give protocol-dependent face distributions; the bias is measurable
against a fair die; the deadline precedes rest; the die stays in the tray and clear of the Franka at home; a scripted Franka
grips, lifts and throws the die; a point grasp selects, moves and releases it; and the die metric orders uniform, empirical
and oracle predictors correctly through the real-time predictor stream."""
from contextlib import contextmanager
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import tomllib
import zipfile

import mujoco
import numpy as np
from scipy import stats

from sim2sim import die
from sim2sim.launcher import Launcher
from sim2sim.simulation import Simulation

OUT = Path(sys.argv[1])
THROWS = int(sys.argv[sys.argv.index("--throws") + 1]) if "--throws" in sys.argv else 150
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("die_scenes", "experiments/validation/die-scenes.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
summary = {"throws_per_protocol": THROWS}
assert all(path.read_text() == text for path, text in gen.outputs().items()), "environment files differ from die-scenes.py output"
FRAME_RATE, DEADLINE = 25, gen.DEADLINE


def config(panda, **extra):
    return {**({"robot": "panda"} if panda else {}), "width": 320, "height": 240, "depth": True, "camera": "fixed", **extra}


def save(png_b64, name):
    (OUT / name).write_bytes(__import__("base64").b64decode(png_b64))


def without_slug(root):
    for parent in root.iter():
        for geom in list(parent):
            if geom.tag == "geom" and geom.get("name") == "die_slug":
                parent.remove(geom)


def ids(model, launcher):
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, launcher.body)
    return body, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "die_slug")


def outcomes(sim, launcher, protocol, stream, count, robot_bodies=None):
    """Outcome faces of count launches, plus robot-contact and out-of-tray counts."""
    faces, touched, outside = [], 0, 0
    body = ids(sim._model, launcher)[0]
    for k in range(count):
        record = die.throw(sim, launcher, launcher.state(protocol, stream, k), FRAME_RATE, DEADLINE, sim.config, frames=False)
        faces.append(record["outcome"])
        x, y, _ = record["position_rest"]
        outside += abs(x - gen.TX) > gen.TRAY_IN or abs(y - gen.TY) > gen.TRAY_IN
        if robot_bodies is not None and k < 20:  # replay with contact inspection
            launcher.place(sim, *launcher.state(protocol, stream, k))
            for _ in range(600):
                mujoco.mj_step(sim._model, sim._data, nstep=5)
                pairs = {(sim._model.geom_bodyid[c.geom1], sim._model.geom_bodyid[c.geom2]) for c in sim._data.contact[:sim._data.ncon]}
                if any(body in pair and set(pair) & robot_bodies for pair in pairs):
                    touched += 1
                    break
    return np.bincount(np.array(faces) - 1, minlength=12), touched, outside


for level, panda in gen.LEVELS.items():
    if "--level" in sys.argv and level != sys.argv[sys.argv.index("--level") + 1]:
        continue
    began = time.time()
    scene = Path("environments") / level / "scene.xml"
    manifest = tomllib.loads((scene.parent / "manifest.toml").read_text())
    launcher, report = Launcher(manifest["launcher"]), {}
    sim = Simulation(scene, config(panda))
    model, data = sim._model, sim._data
    body, slug = ids(model, launcher)
    dof = model.jnt_dofadr[model.body_jntadr[body]]
    report["mass_kg"], report["com_die_frame_m"] = float(model.body_subtreemass[body]), model.body_ipos[body].round(6).tolist()
    assert model.geom_group[slug] == 3 and model.geom_contype[slug] == model.geom_conaffinity[slug] == 0
    # Rest: the perturbed reset pose settles on face 1 within 1 s.
    rests = []
    for k in range(5):
        launcher.place(sim, *launcher.state(None, 0, k))
        sim.action({"op": "observe", "ticks": 1000}, render=False)
        rests.append((die.top_face(data.xquat[body], die.geometry(model, body)[0]), float(np.linalg.norm(data.qvel[dof:dof + 6]))))
    assert all(face == 1 and cos > .999 for (face, cos), _ in rests), rests
    report["rest_speed"] = max(speed for _, speed in rests)
    rest_obs = sim.observe(full=True)
    save(rest_obs["rgb"], f"{level.replace('/', '_')}-rest.png")
    # Hidden slug: identical RGB and depth with the slug removed at the same states, and the grasp ray never hits it.
    fair = Simulation(scene, config(panda, scene_edit=without_slug))
    mismatched, slug_hits, rays = 0, 0, 0
    for protocol in [None, *launcher.protocols]:
        launcher.place(sim, *launcher.state(protocol, 0, 0))
        sim.action({"op": "observe", "ticks": 60}, render=False)
        fair._data.qpos[:] = data.qpos
        mujoco.mj_forward(fair._model, fair._data)
        mujoco.mj_forward(model, data)
        a, b = sim.observe(full=True), fair.observe(full=True)
        mismatched += a["rgb"] != b["rgb"] or a["depth"] != b["depth"]
        if protocol == "drop":
            save(a["rgb"], f"{level.replace('/', '_')}-drop-60ms.png")
        calibration = sim.calibration()
        for u in range(0, 320, 2):
            for v in range(0, 240, 2):
                cam = sim._camera_id
                scale = np.tan(np.deg2rad(model.cam_fovy[cam]) / 2)
                direction = data.cam_xmat[cam].reshape(3, 3) @ np.array([(2 * (u + .5) / 320 - 1) * 320 / 240 * scale, (1 - 2 * (v + .5) / 240) * scale, -1.])
                geom = np.array([-1], np.int32)
                if mujoco.mj_ray(model, data, data.cam_xpos[cam], direction / np.linalg.norm(direction), None, 1, -1, geom) >= 0 and model.geom_bodyid[geom[0]] == body:
                    rays += 1
                    slug_hits += geom[0] == slug
    assert mismatched == 0 and slug_hits == 0 and rays > 0, (mismatched, slug_hits, rays)
    report["slug"] = {"image_mismatches": mismatched, "die_rays": rays, "slug_ray_hits": slug_hits}
    # Determinism: the same launch in two separately built simulations gives bit-identical trajectories.
    other = Simulation(scene, config(panda))
    for protocol in launcher.protocols:
        a = die.throw(sim, launcher, launcher.state(protocol, 1, 7), FRAME_RATE, DEADLINE, sim.config, frames=False)
        b = die.throw(other, launcher, launcher.state(protocol, 1, 7), FRAME_RATE, DEADLINE, other.config, frames=False)
        assert a["trajectory"] == b["trajectory"] and a["outcome"] == b["outcome"], protocol
    report["deterministic"] = True
    # Protocol distributions, bias against a fair die, deadline before rest, tray and robot clearance.
    robot = {b for b in range(model.nbody) if model.body_rootid[b] == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")} if panda else None
    counts, fair_counts, protocols = {}, {}, {}
    for protocol in launcher.protocols:
        counts[protocol], touched, outside = outcomes(sim, launcher, protocol, 2, THROWS, robot)
        fair_counts[protocol] = outcomes(fair, launcher, protocol, 2, THROWS)[0]
        timing = [die.throw(sim, launcher, launcher.state(protocol, 1, k), FRAME_RATE, DEADLINE, sim.config, frames=False) for k in range(30)]
        table = np.array([counts[protocol], fair_counts[protocol]])
        table = table[:, table.sum(0) > 0]
        protocols[protocol] = {"biased": counts[protocol].tolist(), "fair": fair_counts[protocol].tolist(),
                               "bias_chi2_p": float(stats.chi2_contingency(table)[1]) if table.shape[1] > 1 else None,
                               "robot_contacts_of_20": touched, "outside_tray": int(outside),
                               "rest_time_s": [round(min(t["rest_time_s"] for t in timing), 3), round(float(np.median([t["rest_time_s"] for t in timing])), 3)],
                               "unsettled": sum(not t["settled"] for t in timing), "cocked_on_wall": sum(t["cocked"] for t in timing),
                               "moving_at_deadline": sum(t["at_deadline"]["speed_m_s"] > .05 or t["at_deadline"]["spin_rad_s"] > 1 for t in timing),
                               "deadline_face_matches_outcome": sum(t["at_deadline"]["face"] == t["outcome"] for t in timing)}
        print(level, protocol, json.dumps(protocols[protocol]), flush=True)
    table = np.array(list(counts.values()))
    report["protocols"] = protocols
    assert all(p["outside_tray"] == p["robot_contacts_of_20"] == p["unsettled"] == 0 and p["cocked_on_wall"] <= 2 and p["moving_at_deadline"] >= 28 for p in protocols.values()), protocols
    report["protocol_homogeneity_chi2_p"] = float(stats.chi2_contingency(table[:, table.sum(0) > 0])[1])
    pooled = np.array([table.sum(0), np.array(list(fair_counts.values())).sum(0)])
    report["pooled_bias_chi2_p"] = float(stats.chi2_contingency(pooled[:, pooled.sum(0) > 0])[1])
    assert report["protocol_homogeneity_chi2_p"] < 1e-6 and report["pooled_bias_chi2_p"] < 1e-3, report
    # Actuation: point grasp select/move/release, or a scripted Franka grip, lift and throw (planning tool IK).
    launcher.place(sim, *launcher.state(None, 0, 0))
    start = data.xpos[body].copy()
    if not panda:
        cal = sim.calibration()
        pose, k = np.array(cal["world_from_camera"]), np.array(cal["intrinsics"])
        c = pose[:3, :3].T @ (start - pose[:3, 3])
        pixel = [float(k[0, 2] + k[0, 0] * c[0] / -c[2]), float(k[1, 2] - k[1, 1] * c[1] / -c[2])]
        assert sim.action({"op": "grasp", "pixel": pixel}, render=False) is None and sim._grasp is not None
        sim.action({"op": "move", "displacement": [0, 0, .12], "ticks": 400}, render=False)
        lifted = float(data.xpos[body][2] - start[2])
        sim.action({"op": "move", "displacement": [-.04, 0, 0], "ticks": 80}, render=False)
        speed = float(np.linalg.norm(data.qvel[dof:dof + 3]))
        sim.action({"op": "release", "ticks": 2000}, render=False)
        report["point_grasp"] = {"lift_m": lifted, "release_speed_m_s": speed, "rest_face": die.top_face(data.xquat[body], die.geometry(model, body)[0])[0],
                                 "rest_height_m": float(data.xpos[body][2])}
        assert lifted > .1 and speed > .3 and abs(report["point_grasp"]["rest_height_m"] - start[2]) < .01, report["point_grasp"]
    else:
        sys.path.insert(0, "tools/planning")
        from plan import Panda
        planner = Panda(str(scene))
        die_q = model.jnt_qposadr[model.body_jntadr[body]]
        planner.qpos0[die_q:die_q + 7] = data.qpos[die_q:die_q + 7]  # plan on the perturbed rest pose
        # Fingertips down, jaws closing horizontally along world x, 8 mm above the centre: the pads press on the upward-tilted
        # upper faces, which pushes the die into the floor, and friction (1.2 > tan 26.6 deg) carries it when lifting.
        down, centre = [0, np.cos(np.pi / 4), np.sin(np.pi / 4), 0], data.xpos[body] + [0, 0, .008]

        def ik(point, seed):
            result = planner.ik(point, down, start=seed, restarts=5)
            assert result["converged"] and result["collision_free"], result
            return result["joint_targets"]
        pre = ik(centre + [0, 0, .1], [0, 0, 0, -1.57079, 0, 1.57079, -.7853])
        grip = ik(centre, pre)
        lift = ik(centre + [0, 0, .15], grip)
        for q, width, ticks in ((pre, .08, 1500), (grip, .08, 800), (grip, 0., 1200), (lift, 0., 1000)):
            sim.action({"op": "arm_target", "joint_targets": q, "gripper_width_m": width, "ticks": ticks}, render=False)
        lifted = float(data.xpos[body][2] - start[2])
        swing = np.array(lift) + [.2, 0, 0, 0, 0, 0, 0]  # base-yaw swing, gripper opened halfway: a short toss within the tray
        sim.action({"op": "arm_target", "joint_targets": list(np.array(lift) + .5 * (swing - lift)), "gripper_width_m": 0., "ticks": 120}, render=False)
        sim.action({"op": "arm_target", "joint_targets": list(swing), "gripper_width_m": .08, "ticks": 60}, render=False)
        speed = float(np.linalg.norm(data.qvel[dof:dof + 3]))
        sim.action({"op": "observe", "ticks": 2500}, render=False)
        x, y, z = data.xpos[body]
        report["franka"] = {"lift_m": lifted, "release_speed_m_s": speed, "rest_position": [float(x), float(y), float(z)],
                            "in_tray": bool(abs(x - gen.TX) < gen.TRAY_IN and abs(y - gen.TY) < gen.TRAY_IN)}
        save(sim.observe(full=True)["rgb"], "panda_small_die_biased-after-throw.png")
        assert lifted > .1 and speed > .2 and report["franka"]["in_tray"], report["franka"]
    report["seconds"] = round(time.time() - began, 1)
    summary[level] = report
    print(level, json.dumps(report, default=int), flush=True)
    for s in (sim, fair, other):
        s.close()

# Metric sanity through the real-time stream (a local process stands in for the isolated container): uniform,
# per-protocol empirical frequencies learned from session-stream throws, and a predictor that always answers late.
scene = Path("environments/die_biased/scene.xml")
manifest = tomllib.loads((scene.parent / "manifest.toml").read_text())
launcher = Launcher(manifest["launcher"])
sim = Simulation(scene, config(False))
learned = {p: (np.bincount([die.throw(sim, launcher, launcher.state(p, Launcher.SESSION, k), FRAME_RATE, DEADLINE, sim.config, frames=False)["outcome"] - 1
                            for k in range(THROWS)], minlength=12) / THROWS).tolist() for p in launcher.protocols}
sim.close()
PREDICT = '''import json, sys, time
table = TABLE
for line in sys.stdin:
    m = json.loads(line)
    if m["type"] == "start":
        print(json.dumps({"type": "ready"}), flush=True)
    elif m["type"] == "trial":
        time.sleep(DELAY)
        p = table[m["protocol"]] if table else [1 / 12] * 12
        print(json.dumps({"type": "prediction", "trial": m["trial"], "probabilities": p, "estimates": {"com_die_frame_m": [0, 0, -.002]}}), flush=True)
    elif m["type"] == "end":
        break
'''
run = Path(tempfile.mkdtemp(dir=OUT))
sim_config = {**config(False), "realtime": True, "frame_rate": FRAME_RATE}
evaluation = {"entrypoint": "predict.py", "die": {"trials_per_protocol": 10, "empirical_throws": THROWS}}
results = {}
for index, (name, table, delay) in enumerate((("uniform", None, 0), ("empirical", learned, 0), ("late", learned, .3))):
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("predict.py", PREDICT.replace("TABLE", repr(table)).replace("DELAY", str(delay)))

    @contextmanager
    def predictor():
        process = subprocess.Popen([sys.executable, "-m", "sim2sim.predictor", "stream", str(run / ("predictor-%d" % index))], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            yield process
        finally:
            process.kill()
            process.wait()
    began = time.time()
    report = die.evaluate(bundle.getvalue(), run, run / ("version-%03d" % (index + 1)), scene, sim_config, manifest, evaluation, predictor, run, 3600. * index)
    results[name] = {key: report["scores"]["die"][key] for key in ("brier", "accuracy", "latency_median_s", "on_time", "late", "malformed", "missing", "baselines")}
    results[name]["seconds"] = round(time.time() - began, 1)
    print(name, json.dumps(results[name]), flush=True)
u, e, late = results["uniform"], results["empirical"], results["late"]
assert u["on_time"] == e["on_time"] == 40 and late["late"] + late["missing"] == 40 and abs(late["brier"] - 11 / 12) < 1e-9, results
assert abs(u["brier"] - 11 / 12) < 1e-9 and e["brier"] < u["brier"] and u["baselines"]["oracle"]["brier"] == 0, results
assert u["baselines"]["empirical"]["brier"] < u["baselines"]["uniform"]["brier"], results
summary["metric_sanity"] = results
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=int))
(OUT / "die-view.html").write_bytes((run / "die-view.html").read_bytes())
print("PASS", OUT)
