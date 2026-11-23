"""Biased-die prediction evaluation (researcher-only; metric selector `die`).

The agent submits a predictor program (a ZIP with predict.py). On the first submission of a run the host simulates fresh
throws of every public protocol from the trusted source scene (launcher stream 1, the same perturbation distribution as the
agent's session), plus the baselines; the results are cached under RUN/die/. Each submission's predictor then runs in its own
isolated container and receives every trial's camera frames in real time (realtime.stream_predictor). Scores: multiclass
Brier score (sum over faces of (p - outcome)^2; 0 is perfect, the uniform allocation scores 1 - 1/faces), top-face accuracy
(ties share credit) and answer latency, per protocol and overall. A late, malformed or missing answer is scored as the
uniform allocation. Baselines on the same trials: uniform; empirical per-protocol face frequencies from separate source throws
(stream 2); the deterministic mechanics model, i.e. the true source rigid-body model simulated once from the release state as
observed with Gaussian measurement error ([die] mechanics_noise, stream 3) and predicting that outcome with certainty; and
the oracle, the same model from the exact release state in a separately built simulation (Brier 0 exactly when the source is
deterministic). RUN/die-view.html is the self-contained researcher view.
"""
from __future__ import annotations

import base64
import hashlib
import io
import itertools
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from .launcher import Launcher, _mul, _quat
from .realtime import stream_predictor

FACE_GEOM, HULL_GEOM = "face_%02d", "die_shell"   # level contract: face plates face_01.. and the convex hull geom
SAMPLE_S = .02                                    # trajectory and rest-detection sample period
REST = (.01, .2, .2)                              # at rest: below 10 mm/s and 0.2 rad/s for 0.2 s (the bouncy tray contact leaves
                                                  # an invisible ~0.1 rad/s chatter on a resting die)
COCKED = .995                                     # a resting die whose top cosine is lower leans on a wall; its outcome is still
                                                  # the most upward face
MAX_SECONDS = 6.
GAP_S = .25                                       # pause between evaluation trials; answers in it count as late
PREDICTOR = (
    "Evaluation runs `python3 predict.py` from your unpacked ZIP in a fresh sandbox like yours (same packages and resource limits, "
    "no network, no controller, no files from your session). It reads JSON lines on stdin and writes JSON lines on stdout; flush "
    "after every line and send diagnostics to stderr. First {\"type\":\"start\",\"faces\":F,\"frame_rate_hz\":R,\"deadline_s\":D,"
    "\"protocols\":{...},\"trials\":N}: reply {\"type\":\"ready\"} within 120 s. Then for each trial i, at the moment the "
    "launcher releases the die (exactly as your throw action does, robot at its home pose), {\"type\":\"trial\",\"trial\":i,"
    "\"protocol\":name}, immediately followed by frame 0, and then {\"type\":\"frame\",\"trial\":i,\"index\":k,\"time\":k/R, ...} "
    "at k/R seconds after release in real time, for every k/R < D; the remaining fields are those of your observations "
    "(same modalities and encodings), from the fixed camera at its initial pose. Then {\"type\":\"trial_end\",\"trial\":i} "
    "at D, and {\"type\":\"end\"} after the last trial. Answer each trial once, before D seconds after its release: "
    "{\"type\":\"prediction\",\"trial\":i,\"probabilities\":[F numbers, face 1 first, nonnegative, summing to 1]}, optionally "
    "with \"estimates\":{\"com_die_frame_m\":[x,y,z],\"release_velocity_m_s\":[vx,vy,vz],\"release_angular_velocity_rad_s\":"
    "[wx,wy,wz]} (your reconstructed centre of mass in the die frame and release velocities in world axes; recorded, not "
    "scored). Only the first answer per trial counts. Each trial is scored with the multiclass Brier score (the sum over faces "
    "of (probability - outcome)^2, outcome 1 for the face that ends up uppermost and 0 otherwise); top-face accuracy and "
    "answer latency are recorded. A late, malformed or missing answer counts as the uniform allocation.")


def geometry(model, body):
    """Face normals (row k-1 = face k), face RGB and hull vertices, in the die body frame."""
    normals, colors = [], []
    for k in itertools.count(1):
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FACE_GEOM % k)
        if geom < 0:
            break
        assert model.geom_bodyid[geom] == body
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        normals.append(rotation.reshape(3, 3)[:, 2])
        colors.append(model.geom_rgba[geom, :3])
    hull = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, HULL_GEOM)
    mesh = model.geom_dataid[hull]
    assert len(normals) >= 2 and model.geom_bodyid[hull] == body and mesh >= 0
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[hull])
    vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]] @ rotation.reshape(3, 3).T + model.geom_pos[hull]
    return np.array(normals), np.array(colors), vertices


def top_face(quat, normals):
    """(face number, cosine of its normal with +z) for a body orientation."""
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quat, float))
    dots = normals @ rotation.reshape(3, 3)[2]
    return int(np.argmax(dots)) + 1, float(dots.max())


def public_contract(sim, launcher, table, frame_rate):
    """The task's die contract: faces, outcome rule, protocols, deadline and the predictor protocol. No private values."""
    body = mujoco.mj_name2id(sim._model, mujoco.mjtObj.mjOBJ_BODY, launcher.body)
    normals, colors, _ = geometry(sim._model, body)
    return {"faces": [{"face": k, "rgb": colors[k - 1].round(3).tolist(), "normal_die_frame": normals[k - 1].round(6).tolist()}
                      for k in range(1, len(normals) + 1)],
            "die_frame": "origin at the die's geometric centre; a release quaternion (wxyz) rotates die-frame vectors into world axes",
            "outcome": "the face whose outward normal points most nearly straight up (+z) once the die has come to rest",
            "protocols": "task launcher_protocols", "deadline_s": table["deadline_s"], "frame_rate_hz": frame_rate, "predictor": PREDICTOR}


def throw(sim, launcher, release, frame_rate, deadline, public_config, frames):
    """Simulate one launch from release until the die rests: outcome, trajectory samples, state at the deadline and (if
    frames) the public observations at k / frame_rate before the deadline."""
    from .candidate_worker import public_observation
    launcher.place(sim, *release)
    model, data = sim._model, sim._data
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, launcher.body)
    dof = model.jnt_dofadr[model.body_jntadr[body]]
    normals = geometry(model, body)[0]
    dt = float(model.opt.timestep)
    frame_ticks, sample_ticks = round(1 / (frame_rate * dt)), round(SAMPLE_S / dt)
    chunk = math.gcd(frame_ticks, sample_ticks)
    still, tick, observed, trajectory, at_deadline = 0, 0, [], [], None
    com = data.xipos[body].tolist()
    while True:
        t = tick * dt
        if frames and tick % frame_ticks == 0 and t < deadline - 1e-9:
            mujoco.mj_forward(model, data)  # poses of the current state, as real-time frames (mj_step leaves them one step behind)
            observation = public_observation(sim.observe(full=True), public_config)
            observed.append({key: value for key, value in observation.items() if key not in ("ticks", "total_ticks", "grasped")})
        if tick % sample_ticks == 0:
            speed, spin = float(np.linalg.norm(data.qvel[dof:dof + 3])), float(np.linalg.norm(data.qvel[dof + 3:dof + 6]))
            trajectory.append([round(t, 4), *data.xpos[body].round(6).tolist(), *data.xquat[body].round(6).tolist()])
            if at_deadline is None and t >= deadline - 1e-9:
                at_deadline = {"face": top_face(data.xquat[body], normals)[0], "speed_m_s": speed, "spin_rad_s": spin}
            still = still + 1 if speed < REST[0] and spin < REST[1] else 0
            if still * SAMPLE_S >= REST[2] or t >= MAX_SECONDS:
                break
        mujoco.mj_step(model, data, nstep=chunk)
        tick += chunk
    face, dot = top_face(data.xquat[body], normals)
    return {"outcome": face, "top_cosine": dot, "cocked": dot < COCKED, "settled": still * SAMPLE_S >= REST[2], "rest_time_s": t, "at_deadline": at_deadline,
            "com_world_release": com, "position_rest": data.xpos[body].tolist(), "trajectory": trajectory, "frames": observed}


def observed_release(launcher, release, noise, index):
    """The release state as measured with the mechanics baseline's Gaussian error (launcher stream 3, trial index)."""
    rng = np.random.default_rng([launcher.seed, Launcher.NOISE, index])
    position, quat, velocity, angular = release
    return (position + rng.normal(0, noise["position_m"], 3), _mul(_quat(rng.normal(0, noise["angle_rad"], 3)), quat),
            velocity + rng.normal(0, noise["velocity_m_s"], 3), angular + rng.normal(0, noise["angular_velocity_rad_s"], 3))


def generate(directory, scene, sim_config, launcher, table, frame_rate):
    """Simulate the evaluation trials and baselines once per run into directory (researcher-only)."""
    from .simulation import Simulation
    directory.mkdir()
    (directory / "frames").mkdir()
    sim, fresh = Simulation(scene, sim_config), Simulation(scene, sim_config)
    try:
        body = mujoco.mj_name2id(sim._model, mujoco.mjtObj.mjOBJ_BODY, launcher.body)
        normals, colors, vertices = geometry(sim._model, body)
        names, count, deadline = list(launcher.protocols), int(table["trials_per_protocol"]), float(table["deadline_s"])
        trials = []
        for i in range(count * len(names)):
            protocol = names[i % len(names)]
            release = launcher.state(protocol, Launcher.TRIALS, i)
            record = throw(sim, launcher, release, frame_rate, deadline, sim_config, frames=True)
            final = sim.observe(full=True)["rgb"]
            (directory / "frames" / ("%04d.json" % i)).write_text(json.dumps({"frames": record.pop("frames"), "final_rgb": final}))
            mechanics = throw(sim, launcher, observed_release(launcher, release, table["mechanics_noise"], i), frame_rate, deadline, sim_config, frames=False)
            oracle = throw(fresh, launcher, release, frame_rate, deadline, sim_config, frames=False)
            trials.append({"trial": i, "protocol": protocol, "release": dict(zip(("position", "quat", "velocity", "angular_velocity"),
                                                                                  (np.asarray(x).tolist() for x in release))),
                           **record, "mechanics": mechanics["outcome"], "oracle": oracle["outcome"]})
        empirical = {}
        for p, protocol in enumerate(names):
            throws = int(table["empirical_throws"])
            counts = np.bincount([throw(sim, launcher, launcher.state(protocol, Launcher.EMPIRICAL, p * throws + j), frame_rate, deadline,
                                        sim_config, frames=False)["outcome"] - 1 for j in range(throws)], minlength=len(normals))
            empirical[protocol] = {"counts": counts.tolist(), "probabilities": (counts / throws).tolist()}
        edge = min(np.linalg.norm(a - b) for i, a in enumerate(vertices) for b in vertices[i + 1:])
        meta = {"faces": len(normals), "normals": normals.tolist(), "colors": colors.tolist(), "vertices": vertices.tolist(),
                "edges": [[i, j] for i in range(len(vertices)) for j in range(i + 1, len(vertices)) if np.linalg.norm(vertices[i] - vertices[j]) < 1.01 * edge],
                "com_die_frame": sim._model.body_ipos[body].tolist(), "calibration": sim.calibration(), "protocols": launcher.public(),
                "deadline_s": deadline, "frame_rate_hz": frame_rate, "table": table}
        (directory / "trials.json").write_text(json.dumps({"meta": meta, "trials": trials, "empirical": empirical}))
    finally:
        sim.close()
        fresh.close()


def _brier(p, face):
    target = np.zeros(len(p))
    target[face - 1] = 1
    return float(((np.asarray(p) - target) ** 2).sum())


def _hit(p, face):
    p = np.asarray(p)
    best = np.flatnonzero(p >= p.max() - 1e-12)
    return float(face - 1 in best) / len(best)


def score(data, results):
    """Per-trial rows and per-protocol/overall summaries for the predictor and every baseline."""
    faces, rows = data["meta"]["faces"], []
    uniform = [1 / faces] * faces
    for trial, result in zip(data["trials"], results):
        assert trial["trial"] == result["trial"]
        allocations = {"predictor": result["probabilities"] if result["status"] == "on_time" else uniform, "uniform": uniform,
                       "empirical": data["empirical"][trial["protocol"]]["probabilities"],
                       **{name: np.eye(faces)[trial[name] - 1].tolist() for name in ("mechanics", "oracle")}}
        rows.append({**result, "protocol": trial["protocol"], "outcome": trial["outcome"],
                     "brier": {name: _brier(p, trial["outcome"]) for name, p in allocations.items()},
                     "accuracy": {name: _hit(p, trial["outcome"]) for name, p in allocations.items()}})

    def summary(selected):
        latencies = [row["latency_s"] for row in selected if row["status"] == "on_time"]
        return {"brier": float(np.mean([row["brier"]["predictor"] for row in selected])),
                "accuracy": float(np.mean([row["accuracy"]["predictor"] for row in selected])),
                "latency_median_s": float(np.median(latencies)) if latencies else None, "trials": len(selected),
                **{status: sum(row["status"] == status for row in selected) for status in ("on_time", "late", "malformed", "missing")},
                "baselines": {name: {"brier": float(np.mean([row["brier"][name] for row in selected])),
                                     "accuracy": float(np.mean([row["accuracy"][name] for row in selected]))}
                              for name in ("uniform", "empirical", "mechanics", "oracle")}}
    return {**summary(rows), "protocols": {name: summary([row for row in rows if row["protocol"] == name]) for name in data["empirical"]}}, rows


def evaluate(submission, output, version_output, scene, sim_config, manifest, evaluation, predictor, wall_elapsed):
    """Score one predictor submission; predictor() is a context manager yielding its launched, isolated process."""
    version_output.mkdir(parents=True, exist_ok=False)
    table = {**manifest["die"], **evaluation.get("die", {})}
    launcher, frame_rate, directory = Launcher(manifest["launcher"]), sim_config["frame_rate"], output / "die"
    if not (directory / "trials.json").exists():
        generate(directory, scene, sim_config, launcher, table, frame_rate)
    data = json.loads((directory / "trials.json").read_text())
    start = {"type": "start", "faces": data["meta"]["faces"], "frame_rate_hz": frame_rate, "deadline_s": data["meta"]["deadline_s"],
             "protocols": launcher.public(), "trials": len(data["trials"])}
    trials = (({"type": "trial", "trial": t["trial"], "protocol": t["protocol"]},
               json.loads((directory / "frames" / ("%04d.json" % t["trial"])).read_text())["frames"]) for t in data["trials"])
    with predictor() as proc:
        ready, results = stream_predictor(proc, submission, start, trials, frame_rate, data["meta"]["deadline_s"],
                                          version_output / "predictor-output.jsonl", GAP_S)
    summary, rows = score(data, results)
    (version_output / "die-predictions.json").write_text(json.dumps({"wall_elapsed": wall_elapsed, "ready": ready, "trials": rows}))
    report = {"status": "ok", "version": hashlib.sha256(submission).hexdigest(), "scores": {"die": summary}, "predictor_ready": ready,
              "wall_elapsed": wall_elapsed}
    (version_output / "evaluation.json").write_text(json.dumps(report, indent=2))
    write_view(output)
    return report


def _jpeg(png_b64):
    from PIL import Image
    buffer = io.BytesIO()
    Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGB").save(buffer, "JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode()


def write_view(output):
    """RUN/die-view.html: one self-contained researcher page over every die checkpoint of the run."""
    directory = Path(output) / "die"
    data = json.loads((directory / "trials.json").read_text())
    for trial in data["trials"]:
        images = json.loads((directory / "frames" / ("%04d.json" % trial["trial"])).read_text())
        trial["image_release"], trial["image_final"] = (_jpeg(images["frames"][0]["rgb"]) if "rgb" in images["frames"][0] else None,
                                                        _jpeg(images["final_rgb"]))
    checkpoints = []
    for path in sorted(Path(output).glob("version-*/die-predictions.json")):
        record = json.loads(path.read_text())
        report = json.loads((path.parent / "evaluation.json").read_text())
        checkpoints.append({"name": path.parent.name, "wall_elapsed": record["wall_elapsed"], "ready": record["ready"],
                            "scores": report["scores"]["die"], "trials": record["trials"]})
    payload = json.dumps({"run": Path(output).name, **data, "checkpoints": checkpoints}, separators=(",", ":")).replace("</", "<\\/")
    (Path(output) / "die-view.html").write_text(VIEW.replace("__DATA__", payload))


VIEW = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Die evaluation</title><style>
:root{--bg:#fbfbfa;--card:#fff;--fg:#1f2329;--mute:#6a717c;--line:#e2e4e8;--pred:#2f6bd0;--obs:#30343a;--emp:#d0842f;--mech:#8b52c4;--uni:#9aa0a8;--com:#d23b3b;--vel:#1f9d6a;--ang:#c43d9a}
@media (prefers-color-scheme:dark){:root{--bg:#121418;--card:#1a1d22;--fg:#e4e6e9;--mute:#99a0aa;--line:#2d3138;--obs:#e4e6e9;--uni:#6f7680}}
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 8px}h2{font-size:14px;margin:0 0 8px}.bar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}
svg{width:100%;height:auto;display:block}svg text{fill:var(--fg);font-size:11px}.axis{stroke:var(--line)}.mute{color:var(--mute)}
canvas{width:100%;height:auto;border-radius:4px;background:#000}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
td,th{padding:2px 6px;border-bottom:1px solid var(--line);text-align:right}td:first-child,th:first-child{text-align:left}.key span{display:inline-block;width:10px;height:10px;margin:0 4px 0 10px;border-radius:2px}
select,input{max-width:100%}</style></head><body>
<h1>Die evaluation · <span id="run"></span></h1>
<div class="bar"><label>Protocol <select id="protocol"></select></label><label>Checkpoint <select id="checkpoint"></select></label></div>
<div class="grid">
<div class="card"><h2>Predicted vs observed face frequencies</h2><div class="key"><span style="background:var(--obs)"></span>observed<span style="background:var(--pred)"></span>predictor (mean allocation)<span style="background:var(--emp)"></span>empirical baseline</div><svg id="freq" viewBox="0 0 440 220"></svg></div>
<div class="card"><h2>Calibration (all faces, binned allocations)</h2><svg id="cal" viewBox="0 0 440 260"></svg></div>
<div class="card"><h2>Brier score vs experiment time</h2><div class="key"><span style="background:var(--pred)"></span>predictor<span style="background:var(--uni)"></span>uniform<span style="background:var(--emp)"></span>empirical<span style="background:var(--mech)"></span>mechanics</div><svg id="budget" viewBox="0 0 440 220"></svg><table id="table"></table></div>
<div class="card"><h2>Trial replay (evaluator-only overlays)</h2><div class="bar"><select id="trial"></select><label><input type="checkbox" id="final"> final frame</label></div>
<canvas id="replay" width="320" height="240"></canvas><input type="range" id="time" min="0" value="0" style="width:100%">
<div class="key"><span style="background:var(--com)"></span>true COM<span style="background:var(--vel)"></span>release velocity<span style="background:var(--ang)"></span>angular velocity (dashed: agent estimates)</div><div id="info" class="mute"></div></div>
</div><script>
const D=__DATA__,F=D.meta.faces,$=id=>document.getElementById(id),css=n=>getComputedStyle(document.body).getPropertyValue(n).trim();
$("run").textContent=D.run;const P=Object.keys(D.empirical);
for(const p of ["all",...P])$("protocol").add(new Option(p,p));
D.checkpoints.forEach((c,i)=>$("checkpoint").add(new Option(`${c.name} · ${(c.wall_elapsed/3600).toFixed(2)} h`,i)));$("checkpoint").value=D.checkpoints.length-1;
const sel=()=>$("protocol").value,cp=()=>D.checkpoints[+$("checkpoint").value],inP=t=>sel()=="all"||t.protocol==sel();
const uni=Array(F).fill(1/F),alloc=r=>r.status=="on_time"?r.probabilities:uni;
function svg(el,parts){el.innerHTML=parts.join("")}
function axes(w,h,m,ymax,ylab,xt){const o=[`<line class="axis" x1="${m}" y1="${h-m}" x2="${w-8}" y2="${h-m}"/><line class="axis" x1="${m}" y1="8" x2="${m}" y2="${h-m}"/>`];
 for(let i=0;i<=4;i++){const y=h-m-(h-m-8)*i/4;o.push(`<line class="axis" x1="${m}" x2="${w-8}" y1="${y}" y2="${y}" opacity=".5"/><text x="${m-4}" y="${y+4}" text-anchor="end">${(ymax*i/4).toFixed(2)}</text>`)}
 o.push(`<text x="12" y="${(h-m)/2}" transform="rotate(-90 12 ${(h-m)/2})" text-anchor="middle">${ylab}</text>`);if(xt)o.push(`<text x="${(w+m)/2}" y="${h-4}" text-anchor="middle">${xt}</text>`);return o}
function freq(){const T=D.trials.filter(inP),c=cp(),rows=c?c.trials.filter(inP):[],n=T.length||1,obs=Array(F).fill(0),pred=Array(F).fill(0),emp=Array(F).fill(0);
 T.forEach(t=>{obs[t.outcome-1]+=1/n;D.empirical[t.protocol].probabilities.forEach((p,k)=>emp[k]+=p/n)});rows.forEach(r=>alloc(r).forEach((p,k)=>pred[k]+=p/(rows.length||1)));
 const w=440,h=220,m=36,ymax=Math.max(.2,...obs,...pred,...emp)*1.1,bw=(w-m-8)/F,o=axes(w,h,m,ymax,"frequency","face");
 [[obs,"--obs"],[c?pred:null,"--pred"],[emp,"--emp"]].forEach(([s,col],j)=>s&&s.forEach((v,k)=>{const x=m+k*bw+2+j*(bw-4)/3,y=h-m-(h-m-8)*v/ymax;o.push(`<rect x="${x}" y="${y}" width="${(bw-4)/3-1}" height="${h-m-y}" fill="${css(col)}"><title>face ${k+1}: ${v.toFixed(3)}</title></rect>`)}));
 for(let k=0;k<F;k++)o.push(`<rect x="${m+k*bw+bw/2-5}" y="${h-m+3}" width="10" height="6" fill="rgb(${D.meta.colors[k].map(x=>x*255|0)})" stroke="${css("--line")}"/><text x="${m+k*bw+bw/2}" y="${h-m+20}" text-anchor="middle">${k+1}</text>`);svg($("freq"),o)}
function cal(){const c=cp(),o=axes(440,260,36,1,"observed frequency","mean allocated probability");if(!c){svg($("cal"),o);return}
 const B=10,sum=Array(B).fill(0),hit=Array(B).fill(0),cnt=Array(B).fill(0);c.trials.filter(inP).forEach(r=>alloc(r).forEach((p,k)=>{const b=Math.min(B-1,Math.floor(p*B));sum[b]+=p;hit[b]+=(k+1==r.outcome);cnt[b]++}));
 const X=v=>36+(440-44)*v,Y=v=>260-36-(260-44)*v;o.push(`<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="${css("--uni")}" stroke-dasharray="4 3"/>`);
 for(let i=0;i<=4;i++)o.push(`<text x="${X(i/4)}" y="${Y(0)+14}" text-anchor="middle">${i/4}</text>`);
 const pts=[];for(let b=0;b<B;b++)if(cnt[b]){pts.push([X(sum[b]/cnt[b]),Y(hit[b]/cnt[b])]);o.push(`<circle cx="${X(sum[b]/cnt[b])}" cy="${Y(hit[b]/cnt[b])}" r="${2+Math.sqrt(cnt[b])/3}" fill="${css("--pred")}" opacity=".8"><title>${cnt[b]} allocations, mean ${(sum[b]/cnt[b]).toFixed(3)}, observed ${(hit[b]/cnt[b]).toFixed(3)}</title></circle>`)}
 o.push(`<polyline fill="none" stroke="${css("--pred")}" points="${pts.map(p=>p.join(",")).join(" ")}"/>`);svg($("cal"),o)}
function stat(s){return sel()=="all"?s:s.protocols[sel()]}
function budget(){const C=D.checkpoints,w=440,h=220,m=36;if(!C.length){svg($("budget"),axes(w,h,m,1,"Brier","")); $("table").innerHTML="<tr><td>No predictor checkpoints yet.</td></tr>";return}
 const S=C.map(c=>stat(c.scores)),base=S[0].baselines,ymax=Math.max(...S.map(s=>s.brier),...Object.values(base).map(b=>b.brier))*1.1,tmax=Math.max(...C.map(c=>c.wall_elapsed),1)/3600,
 X=t=>m+(w-m-8)*t/tmax,Y=v=>h-m-(h-m-8)*v/ymax,o=axes(w,h,m,ymax,"Brier score","experiment time (h)");
 for(let i=0;i<=4;i++)o.push(`<text x="${X(tmax*i/4)}" y="${h-m+14}" text-anchor="middle">${(tmax*i/4).toFixed(1)}</text>`);
 [["uniform","--uni"],["empirical","--emp"],["mechanics","--mech"]].forEach(([b,col])=>o.push(`<line x1="${m}" x2="${w-8}" y1="${Y(base[b].brier)}" y2="${Y(base[b].brier)}" stroke="${css(col)}" stroke-dasharray="5 3"><title>${b}: ${base[b].brier.toFixed(3)}</title></line>`));
 o.push(`<polyline fill="none" stroke="${css("--pred")}" stroke-width="2" points="${S.map((s,i)=>X(C[i].wall_elapsed/3600)+","+Y(s.brier)).join(" ")}"/>`);
 S.forEach((s,i)=>o.push(`<circle cx="${X(C[i].wall_elapsed/3600)}" cy="${Y(s.brier)}" r="4" fill="${css("--pred")}"><title>${C[i].name}: Brier ${s.brier.toFixed(3)}, accuracy ${s.accuracy.toFixed(3)}</title></circle>`));svg($("budget"),o);
 const s=stat(cp().scores),f=v=>v==null?"–":v.toFixed(3);$("table").innerHTML=`<tr><th></th><th>Brier</th><th>accuracy</th></tr><tr><td>predictor (${cp().name})</td><td>${f(s.brier)}</td><td>${f(s.accuracy)}</td></tr>`+
 Object.entries(s.baselines).map(([k,b])=>`<tr><td>${k}</td><td>${f(b.brier)}</td><td>${f(b.accuracy)}</td></tr>`).join("")+`<tr><td colspan="3" class="mute">median latency ${f(s.latency_median_s)} s · on time ${s.on_time}/${s.trials} · late ${s.late} · malformed ${s.malformed} · missing ${s.missing}${cp().ready?"":" · predictor never became ready"}</td></tr>`}
function rot(q,v){const[w,x,y,z]=q,[a,b,c]=v,ix=w*a+y*c-z*b,iy=w*b+z*a-x*c,iz=w*c+x*b-y*a,iw=-x*a-y*b-z*c;return[ix*w+iw*-x+iy*-z-iz*-y,iy*w+iw*-y+iz*-x-ix*-z,iz*w+iw*-z+ix*-y-iy*-x]}
function project(p){const K=D.meta.calibration.intrinsics,M=D.meta.calibration.world_from_camera,d=[0,1,2].map(i=>p[i]-M[i][3]),c=[0,1,2].map(j=>M[0][j]*d[0]+M[1][j]*d[1]+M[2][j]*d[2]);
 return c[2]>=0?null:[K[0][2]+K[0][0]*c[0]/-c[2],K[1][2]-K[1][1]*c[1]/-c[2]]}
const img=new Image();let current=null;
function trials(){const T=D.trials.filter(inP);$("trial").innerHTML="";T.forEach(t=>$("trial").add(new Option(`#${t.trial} ${t.protocol} → face ${t.outcome}`,t.trial)));pick()}
function pick(){current=D.trials.find(t=>t.trial==+$("trial").value);if(!current)return;$("time").max=current.trajectory.length-1;
 $("time").value=$("final").checked?current.trajectory.length-1:0;img.src="data:image/jpeg;base64,"+($("final").checked?current.image_final:current.image_release||current.image_final)}
function line(ctx,a,b,col,dash){if(!a||!b)return;ctx.strokeStyle=col;ctx.setLineDash(dash?[4,3]:[]);ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.beginPath();ctx.arc(...b,2.5,0,7);ctx.fillStyle=col;ctx.fill()}
function draw(){const t=current;if(!t)return;const cv=$("replay"),ctx=cv.getContext("2d");cv.width=D.meta.calibration.width;cv.height=D.meta.calibration.height;
 if(img.complete)ctx.drawImage(img,0,0,cv.width,cv.height);const s=t.trajectory[+$("time").value],pos=s.slice(1,4),q=s.slice(4,8),W=v=>rot(q,v).map((x,i)=>x+pos[i]);
 ctx.setLineDash([]);ctx.strokeStyle="#ffffff";ctx.lineWidth=1;for(const[i,j]of D.meta.edges){const a=project(W(D.meta.vertices[i])),b=project(W(D.meta.vertices[j]));if(a&&b){ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}}
 const com=W(D.meta.com_die_frame),c0=project(com);if(c0){ctx.fillStyle=css("--com");ctx.beginPath();ctx.arc(...c0,4,0,7);ctx.fill()}
 const R=t.release,rp=R.position,r=c=>project(c),row=cp()&&cp().trials.find(x=>x.trial==t.trial),E=row&&row.estimates||{};
 line(ctx,r(rp),r(rp.map((x,i)=>x+.1*R.velocity[i])),css("--vel"));line(ctx,r(rp),r(rp.map((x,i)=>x+.002*R.angular_velocity[i])),css("--ang"));
 if(Array.isArray(E.release_velocity_m_s))line(ctx,r(rp),r(rp.map((x,i)=>x+.1*E.release_velocity_m_s[i])),css("--vel"),1);
 if(Array.isArray(E.release_angular_velocity_rad_s))line(ctx,r(rp),r(rp.map((x,i)=>x+.002*E.release_angular_velocity_rad_s[i])),css("--ang"),1);
 if(Array.isArray(E.com_die_frame_m)){const e=project(W(E.com_die_frame_m));if(e){ctx.strokeStyle=css("--com");ctx.setLineDash([2,2]);ctx.beginPath();ctx.arc(...e,6,0,7);ctx.stroke()}}
 const top=row?alloc(row).map((p,k)=>[p,k+1]).sort((a,b)=>b[0]-a[0]).slice(0,3).map(([p,k])=>`${k}:${p.toFixed(2)}`).join(" "):"–";
 $("info").textContent=`t = ${s[0].toFixed(2)} s · outcome face ${t.outcome} (rest ${t.rest_time_s.toFixed(2)} s) · mechanics ${t.mechanics} · oracle ${t.oracle} · face up at deadline ${t.at_deadline?t.at_deadline.face:"–"} · predictor ${row?row.status:"–"}${row&&row.latency_s!=null?" "+row.latency_s.toFixed(3)+" s":""} · top allocation ${top} · |v| ${Math.hypot(...R.velocity).toFixed(2)} m/s · |ω| ${Math.hypot(...R.angular_velocity).toFixed(1)} rad/s · COM offset ${(1000*Math.hypot(...D.meta.com_die_frame)).toFixed(2)} mm`}
img.onload=draw;$("time").oninput=draw;$("trial").onchange=pick;$("final").onchange=pick;
function all(){freq();cal();budget();trials()}$("protocol").onchange=all;$("checkpoint").onchange=()=>{freq();cal();budget();draw()};all();
</script></body></html>
"""


if __name__ == "__main__":
    import sys
    write_view(Path(sys.argv[1]))
