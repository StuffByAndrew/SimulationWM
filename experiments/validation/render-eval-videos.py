"""Researcher-only side-by-side videos of the private evaluator's tests on a run's final scored submission: hidden scene (left)
against the agent's model (right), for the results report.

usage (Slurm test node; parses candidate MJCF): MUJOCO_GL=egl .venv/bin/python experiments/validation/render-eval-videos.py OUT_DIR RUN_DIR...

- Box levels (`arm_probes`): each scripted arm throw, replayed exactly as `evaluation._panda_probe` runs it, filmed by a camera
  that follows the hidden scene's object (the same camera path is used for the model); the largest free object body (the box
  shell) is drawn semi-transparent so the contents show.
- Tunnel levels (`force_profile`): the force probe exactly as `force_profile.measure` runs it (wave 16-17 runs: the rescored probe
  recorded in force-profile-rescore.json), filmed from the trusted camera every sample, with the force the handle applies drawn
  as an arrow at the grasp point, over a plot of resistance against travel along the tunnel that fills in as the probe runs.
- Cloth levels: the evaluator's own bar-drop side-by-side videos of the final version are copied, and two stills are made: the
  trusted-camera RGB at rest, and each cloth's bending-stiffness field (per flex, the median magnitude of MuJoCo's compiled
  per-edge bending coefficients, the same computation for truth and model; relative units, one log colour scale).
Writes OUT_DIR/<run>/<clip>.mp4 (and .png stills) and OUT_DIR/<run>/clips.json. Re-simulation of the scored tests, not a recording of the agent's session.
"""
import json
import re
import shutil
import sys
import tomllib
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from sim2sim.evaluation import ROBOTS, _free_object_bodies, _trusted_camera
from sim2sim.force_profile import _at, _spec
from sim2sim.simulation import Simulation

W, H = 640, 480
MANIFEST_KEYS = ('probes', 'correspondence', 'camera', 'graspable_bodies', 'reconstruction', 'hinge_correspondence', 'hinge_matching',
                 'object_bodies', 'object_geoms', 'arm_probes', 'expert_trajectories', 'transfer_tasks', 'force_profile', 'cloth')


def final_version(run):
    """The last scored version (a rejected submission has no scores): its evaluation report and archived project directory."""
    version = [v for v in sorted(run.glob("version-*")) if json.loads((v / "evaluation.json").read_text())["status"] == "ok"][-1]
    report = json.loads((version / "evaluation.json").read_text())
    return version, report, version / "evaluation" / report["project_path"]


def setup(run):
    """Evaluation config, source and candidate simulations, exactly as evaluation.evaluate_submission builds them."""
    private = json.loads((run / "config-private.json").read_text())
    config = dict(private["evaluation"], source_dir=str(run / "source"))
    config["source_config"] = json.loads((run / "manifest.json").read_text())["resolved_source_config"]
    manifest = Path(config["source_dir"]) / config.get("manifest", "manifest.toml")
    metadata = tomllib.loads(manifest.read_text()) if manifest.is_file() else {}
    for key in MANIFEST_KEYS:
        if key not in config and key in metadata:
            config[key] = metadata[key]
    simconfig = dict(config["source_config"])
    for key in ("camera", "width", "height", "timestep", "grasp_solref", "grasp_solimp", "default_ticks", "graspable_bodies"):
        if key in config:
            simconfig[key] = config[key]
    version, report, project = final_version(run)
    entry = project / config.get("entrypoint", "scene.xml")
    for xml in project.rglob("*.xml"):  # researcher-side native parse: no plugins or extensions, as report-extract-models.py checks
        assert not re.search(r"<\s*(plugin|extension)\b", xml.read_text()), xml
    source_path = Path(config["source_dir"]) / config.get("source_entrypoint", config.get("entrypoint", "scene.xml"))
    source = Simulation(source_path, simconfig)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, source._data.cam_xmat[source._camera_id])
    candidate_config = dict(simconfig, timestep=float(source._model.opt.timestep),
                            trusted_camera={"pos": source._data.cam_xpos[source._camera_id].tolist(), "quat": quat.tolist(),
                                            "fovy": float(source._model.cam_fovy[source._camera_id])})
    candidate_config.pop("graspable_bodies", None)
    return config, simconfig, candidate_config, source_path, entry, source, version, report


def renderer(sim):
    sim._model.vis.global_.offwidth, sim._model.vis.global_.offheight = max(W, sim._model.vis.global_.offwidth), max(H, sim._model.vis.global_.offheight)
    return mujoco.Renderer(sim._model, H, W)


def label(img, text):
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((8, 6), text, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(im)


def write(path, frames, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=7, macro_block_size=8)


# ---------------- box levels: scripted arm throws ----------------
def see_through(sim):
    """Draw the largest free object body (the container shell) at 25% opacity; inner bodies stay opaque."""
    m = sim._model
    size = lambda b: sum(np.prod(2 * m.geom_size[g][:3] + 1e-9) for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_rgba[g][3] > 0)
    shell = max(_free_object_bodies(sim), key=size)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == shell and m.geom_rgba[g][3] > 0:
            m.geom_rgba[g][3] = 0.25
            if m.geom_matid[g] >= 0:
                m.mat_rgba[m.geom_matid[g]][3] = 0.25


def film_probe(sim, probe, path=None, every=10):
    """evaluation._panda_probe's step loop, with a frame every `every` ticks. Without `path` the camera follows this world's
    free-object COM (smoothed) and the path is returned, so the other world can be filmed along exactly the same path."""
    sim.action({"op": "reset"}, render=False)
    r, frames, follow = renderer(sim), [], path is None
    path = [] if follow else list(path)
    free = _free_object_bodies(sim)
    mass = sim._model.body_mass[free]
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 0.55, 150.0, -20.0

    def grab(current):
        if current.ticks % every == 0:
            if follow:
                com = mass @ current._data.xipos[free] / mass.sum()
                path.append(com if not path else 0.85 * path[-1] + 0.15 * com)
            camera.lookat[:] = path[len(frames)]
            r.update_scene(current._data, camera=camera)
            frames.append(r.render().copy())
    old, sim._callback = sim._callback, grab
    try:
        for step in probe["steps"]:
            sim.action({**step, "op": ROBOTS[sim.robot][2][0]}, render=False)
    finally:
        sim._callback = old
        r.close()
    assert len(frames) == len(path), "the shared camera path must cover every frame"
    return frames, path


def box(run, out):
    config, simconfig, cconfig, _, entry, source, version, report = setup(run)
    candidate = Simulation(entry, cconfig)
    _trusted_camera(source, candidate)
    see_through(source), see_through(candidate)
    clips = []
    for probe in config["arm_probes"]:
        a, path = film_probe(source, probe)
        b, _ = film_probe(candidate, probe, path)
        assert len(a) == len(b), (run.name, probe["name"])
        frames = [np.hstack([label(x, "hidden scene (truth)"), label(y, "agent's model")]) for x, y in zip(a, b)]
        path = out / run.name / f"throw-{probe['name']}.mp4"
        write(path, frames, fps=round(1 / (10 * float(source._model.opt.timestep))))
        clips.append(dict(run=run.name, family="box", clip=probe["name"], file=str(path.relative_to(out)), version=version.name))
    source.close(), candidate.close()
    return clips


# ---------------- tunnel levels: force probe ----------------
def project(sim, point):
    """Pixel of a world point in the 2W x 2H render of the observation camera (zoomed view, cropped later)."""
    cam = sim._camera_id
    p = sim._data.cam_xmat[cam].reshape(3, 3).T @ (point - sim._data.cam_xpos[cam])
    scale = np.tan(np.deg2rad(sim._model.cam_fovy[cam]) / 2)
    return np.array([(p[0] / (-p[2] * scale) * H / W + 1) * W, (1 - p[1] / (-p[2] * scale)) * H])


def film_force(scene, simconfig, spec):
    """force_profile.measure, filmed: per sample a frame, the grasp point and the force the handle applies (= -wrench)."""
    path, arc, strokes, every, settle = _spec(spec)
    config = {**simconfig, "rgb": False, "depth": False, "calibration": False, "wrench": True, "recording_callback": None}
    if config.get("robot") == "panda":
        config["panda_point_grasps"] = True
    sim = Simulation(scene, config)
    try:
        u, v, rad = *spec["grasp_pixel"], int(spec["grasp_radius_px"])
        rings = sorted(((du, dv) for du in range(-rad, rad + 1) for dv in range(-rad, rad + 1) if du * du + dv * dv <= rad * rad), key=lambda d: d[0] ** 2 + d[1] ** 2)
        hit = next((h for du, dv in rings if (h := sim._raycast([u + du, v + dv])) is not None), None)
        if hit is None:
            return None
        sim._grasp_controller.attach(*hit)
        if settle:
            sim._advance(settle)
        sim._model.vis.global_.offwidth, sim._model.vis.global_.offheight = max(2 * W, sim._model.vis.global_.offwidth), max(2 * H, sim._model.vis.global_.offheight)
        r, dt, s, rows = mujoco.Renderer(sim._model, 2 * H, 2 * W), float(sim._model.opt.timestep), 0., []
        eq = sim._grasp[0]
        for index, (target, speed) in enumerate(strokes):
            ticks, done = max(1, round(abs(target - s) / speed / dt)), 0
            while done < ticks:
                n = min(every, ticks - done)
                nxt = s + (target - s) * n / (ticks - done)
                sim._advance(n, _at(path, arc, nxt)[0] - _at(path, arc, s)[0])
                done, s = done + n, nxt
                force = np.asarray(sim._wrench()["force"])
                body = int(sim._model.eq_obj1id[eq])
                point = sim._data.xpos[body] + sim._data.xmat[body].reshape(3, 3) @ sim._model.eq_data[eq, :3]
                r.update_scene(sim._data, camera=sim._camera_id)
                rows.append(dict(stroke=index, s=s, resistance=float(-force @ _at(path, arc, s)[1]), frame=r.render().copy(),
                                 pixel=project(sim, point), tip=project(sim, point - 0.004 * force)))  # 4 mm per N of applied force
        r.close()
        return rows
    finally:
        sim.close()


def arrow(img, a, b):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.line([tuple(a), tuple(b)], fill=(255, 220, 0), width=4)
    v = b - a
    if np.linalg.norm(v) > 4:
        u = v / np.linalg.norm(v)
        n = np.array([-u[1], u[0]])
        d.polygon([tuple(b), tuple(b - 12 * u + 6 * n), tuple(b - 12 * u - 6 * n)], fill=(255, 220, 0))
    d.ellipse([a[0] - 4, a[1] - 4, a[0] + 4, a[1] + 4], fill=(255, 60, 60))
    return np.asarray(im)


def tunnel(run, out):
    config, simconfig, cconfig, source_path, entry, source, version, report = setup(run)
    source.close()
    rescore = run / "force-profile-rescore.json"
    spec = json.loads(rescore.read_text())["probe"] if rescore.is_file() else config["force_profile"]  # the scored probe
    a, b = film_force(source_path, simconfig, spec), film_force(entry, cconfig, spec)
    assert a is not None, run.name
    fig, ax = plt.subplots(figsize=(2 * W / 100, 3.44), dpi=100)
    S = lambda rows: np.array([1000 * x["s"] for x in rows])
    F = lambda rows: np.array([x["resistance"] for x in rows])
    ax.plot(S(a), F(a), color="k", alpha=0.18, lw=1)
    lo, hi = min(F(a).min(), F(b).min() if b else 0), max(F(a).max(), F(b).max() if b else 0)
    if b:
        ax.plot(S(b), F(b), color="#eb6834", alpha=0.18, lw=1)
    ax.set_xlabel("handle travel along the tunnel (mm)"), ax.set_ylabel("resistance against the push (N)")
    ax.set_ylim(lo - 0.05 * (hi - lo) - 0.5, hi + 0.05 * (hi - lo) + 0.5)
    ax.grid(alpha=0.3)
    la, = ax.plot([], [], color="k", lw=2, label="hidden scene (truth)")
    lb, = ax.plot([], [], color="#eb6834", lw=2, label="agent's model" if b else "agent's model: probe missed the handle")
    da, = ax.plot([], [], "o", color="k")
    db, = ax.plot([], [], "o", color="#eb6834")
    ax.legend(loc="upper left")
    fig.tight_layout()
    frames = []
    blank = np.zeros((H, W, 3), np.uint8)
    c = np.clip(np.round(a[0]["pixel"] - [W / 2, H / 2]).astype(int), 0, [W, H])  # zoom window: centred on the hidden scene's grasp point
    zoom = lambda img: img[c[1]:c[1] + H, c[0]:c[0] + W]
    for i, x in enumerate(a):
        y = b[i] if b else None
        la.set_data(S(a[:i + 1]), F(a[:i + 1])), da.set_data([S(a)[i]], [F(a)[i]])
        if b:
            lb.set_data(S(b[:i + 1]), F(b[:i + 1])), db.set_data([S(b)[i]], [F(b)[i]])
        prev = a[i - 1]["s"] if i else 0.
        ax.set_title(f"stroke {x['stroke'] + 1}: {'push in' if x['s'] >= prev else 'pull back'}", fontsize=10)
        fig.canvas.draw()
        plot = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        top = np.hstack([label(zoom(arrow(x["frame"], x["pixel"], x["tip"])), "hidden scene (truth)"),
                         label(zoom(arrow(y["frame"], y["pixel"], y["tip"])), "agent's model") if y else label(blank, "agent's model: probe missed the handle")])
        frames.append(np.vstack([top, np.asarray(Image.fromarray(plot).resize((2 * W, plot.shape[0])))]))
    plt.close(fig)
    path = out / run.name / "force-probe.mp4"
    write(path, frames, fps=30)
    return [dict(run=run.name, family="tunnel", clip="force probe", file=str(path.relative_to(out)), version=version.name)]


# ---------------- cloth levels: the evaluator's own drop videos ----------------
def bending_field(sim):
    """Per flex: (vertex xy, triangles, median |compiled per-edge bending coefficients|) at the current state."""
    m, d, out = sim._model, sim._data, []
    for f in range(m.nflex):
        if m.flex_dim[f] != 2:
            continue
        v0, nv, e0, ne = m.flex_vertadr[f], m.flex_vertnum[f], m.flex_elemadr[f], m.flex_elemnum[f]
        tri = m.flex_elem[m.flex_elemdataadr[f]:m.flex_elemdataadr[f] + 3 * ne].reshape(ne, 3)
        edges = m.flex_bending.reshape(m.nflexedge, -1)[m.flex_edgeadr[f]:m.flex_edgeadr[f] + m.flex_edgenum[f]]  # one row of coefficients per edge
        out.append((d.flexvert_xpos[v0:v0 + nv, :2].copy(), tri, float(np.median(np.abs(edges).sum(axis=1)))))
    assert out, "no 2D flex"
    return out


def cloth_stills(run, out):
    config, simconfig, cconfig, _, entry, source, version, report = setup(run)
    candidate = Simulation(entry, cconfig)
    _trusted_camera(source, candidate)
    images, fields = [], []
    for sim in (source, candidate):
        sim.action({"op": "reset"}, render=False)
        r = renderer(sim)
        r.update_scene(sim._data, camera=sim._camera_id)
        images.append(r.render().copy())
        r.close()
        fields.append(bending_field(sim))
    rgb = out / run.name / "rest-rgb.png"
    rgb.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.hstack([label(images[0], "hidden scene (truth)"), label(images[1], "agent's model")])).save(rgb)
    values = [v for field in fields for _, _, v in field if v > 0]
    norm = matplotlib.colors.LogNorm(vmin=min(values), vmax=max(values)) if values else None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for ax, field, title in zip(axes, fields, ("hidden scene (truth)", "agent's model")):
        for xy, tri, v in field:
            ax.tripcolor(xy[:, 0], xy[:, 1], tri, facecolors=np.full(len(tri), max(v, norm.vmin)), norm=norm, cmap="viridis", edgecolors="none")
        ax.set(title=title, aspect="equal", xlabel="x (m)", ylabel="y (m)")
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes, label="bending stiffness (compiled, relative)", shrink=0.85)
    field = out / run.name / "stiffness-field.png"
    fig.savefig(field, dpi=100)
    plt.close(fig)
    lift = next((version / "evaluation" / "artifacts").glob("*/cloth/stiffness_map.png"))
    shutil.copy(lift, out / run.name / "lift-force-map.png")
    source.close(), candidate.close()
    return [dict(run=run.name, family="cloth-still", clip=name, file=str((out / run.name / f).relative_to(out)), version=version.name)
            for name, f in (("RGB at rest", "rest-rgb.png"), ("bending-stiffness field", "stiffness-field.png"), ("lift force at 16 points", "lift-force-map.png"))]


def cloth(run, out):
    version, report, _ = final_version(run)
    drops = report["scores"]["cloth"]["drops"]
    art = next((version / "evaluation" / "artifacts").glob("*/cloth"))
    clips = []
    for i, d in enumerate(drops):
        if d["obstacle"] != "bar":
            continue
        path = out / run.name / f"drop-{i:03d}-bar.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(art / f"drop_{i:03d}" / "side_by_side.mp4", path)
        clips.append(dict(run=run.name, family="cloth", clip=f"drop {i + 1} onto a bar (yaw {d['yaw_deg']:.0f}°, tilt {d['tilt_deg']:.0f}°)",
                          file=str(path.relative_to(out)), version=version.name))
    return clips + cloth_stills(run, out)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    for run in map(Path, sys.argv[2:]):  # one OUT/<run>/clips.json per run, so parallel invocations never share a file
        evaluation = json.loads(final_version(run)[0].joinpath("evaluation.json").read_text())["scores"]
        family = box if "arm_probes" in evaluation and "-hand-" not in run.name else tunnel if "force_profile" in evaluation and "-msd-" not in run.name \
            else cloth if "cloth" in evaluation else None
        assert family is not None, run.name
        clips = family(run, out)
        (out / run.name / "clips.json").write_text(json.dumps(clips, indent=1))
        print(run.name, "ok", flush=True)
