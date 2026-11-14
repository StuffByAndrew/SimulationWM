"""Report spec: sim2sim results of 2026-11-06 to 2026-11-14 (waves 11-19 and the spring-level camera repeats): every finished, valid strict
run not in report-spec-20261104-plain.py, plus the two geometry baselines of that report as references. Written for a reader new to the project.
Inputs (compute node, job script in docs/experiment-log.md 2026-11-14): report-work-20261114/ (run summaries, compiled models and checkpoint
indexes of waves 16-19, traces and chat pages of every new run); waves 11-15 models and checkpoints from rerun-analysis/*-{H..M}.json.
Build: .venv/bin/python experiments/validation/build-results-report.py data experiments/validation/report-spec-20261114.py \\
           experiments/reports/report-work-20261114 experiments/reports/results-report-20261114
       then `text` (seconds) and `zip`. Every number in the text comes from ctx (the built data) or the run directories."""
import json, math, re
from pathlib import Path

import numpy as np

VAL = Path(__file__).resolve().parent
MAIN = Path("/n/home00/aejxie/sim2sim/experiments")  # ignored researcher data lives in the main checkout
WORK = MAIN / "reports" / "report-work-20261114"
PLAIN = MAIN / "reports" / "report-work-plain"
RA = MAIN / "validation" / "rerun-analysis"
NETRUNS = Path("/n/netscratch/qianqian_lab/Lab/aejxie/sim2sim-experiments/runs")
REF = {"boat": "w10-boat-base-20261105-001", "assembly": "w10-assembly-base-20261105-001"}  # baselines from the previous report
SUMMARIES = (MAIN / "validation" / "results-w11-18.json", MAIN / "validation" / "results-w19.json", (MAIN / "validation" / "results-wave10.json", set(REF.values())))
INVALID = set()  # invalid attempts (wave 19 -001, marble-equal -002) are not in NAMES; Appendix A1 describes them
MODELS = (RA / "models-H.json", RA / "models-I.json", RA / "models-J.json", RA / "models-K.json", RA / "models-L.json", RA / "models-M.json",
          MAIN / "validation" / "rerun-analysis-plain" / "models-F.json", MAIN / "validation" / "rerun-analysis-plain" / "models-G.json", WORK / "models-w16-19.json")
CHECKPOINTS = (RA / "checkpoints-H.json", RA / "checkpoints-I.json", RA / "checkpoints-J.json", RA / "checkpoints-K.json", RA / "checkpoints-L.json",
               RA / "checkpoints-M.json", MAIN / "validation" / "rerun-analysis-plain" / "checkpoints-F.json",
               MAIN / "validation" / "rerun-analysis-plain" / "checkpoints-G.json", WORK / "checkpoints-w16-19.json")
TRACES = (WORK / "traces", PLAIN / "traces")
CHAT = (WORK / "chat", PLAIN / "chat")
EDITS = WORK / "edits.json"
VENDOR = Path("/n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-experiments/report-work-20261103/vendor")
CONTINUATIONS = {}
TOOLS = {"sequence": r"sequence"}
NOTE_FLAGS = {}
EXAMPLES = {}  # no verbatim strategy examples in this report
L = lambda s: s  # readability marker for long display strings

# ---------------- runs ----------------
SPRING_RUNS = {("k", "movable"): ["w11-tetra-spring-k-20261106-001", "w11-tetra-spring-k-r2-20261107-001", "w11-tetra-spring-k-r3-20261107-001"],
               ("k", "fixed"): ["w13-tetra-spring-k-fixedcam-20261106-001", "w13-tetra-spring-k-fixedcam-r2-20261107-001", "w13-tetra-spring-k-fixedcam-r3-20261107-001"],
               ("block", "movable"): ["w11-tetra-spring-block-20261106-001", "w11-tetra-spring-block-r2-20261107-001", "w11-tetra-spring-block-r3-20261107-001"],
               ("block", "fixed"): ["w13-tetra-spring-block-fixedcam-20261106-001", "w13-tetra-spring-block-fixedcam-r2-20261107-001",
                                    "w13-tetra-spring-block-fixedcam-r3-20261107-001"]}
SPRING_VARIANTS = {"square": "w11-tetra-spring-k-square-20261106-001", "plug": "w11-tetra-spring-k-plug-20261106-001"}
GEO = {"boat": {"base": REF["boat"], "movecam": "w14-boat-movecam-20261107-001", "sam_offered": "w12-boat-sam3d-20261106-001",
                "sam_directed": "w12-boat-sam3d-directed-20261106-001", "sam_required": "w15-boat-sam3d-required-20261107-001"},
       "assembly": {"base": REF["assembly"], "movecam": "w14-assembly-movecam-20261107-001", "sam_directed": "w12-assembly-sam3d-directed-20261106-001",
                    "sam_required": "w15-assembly-sam3d-required-20261107-001"}}
LEVELS = ["spring", "damper", "nonlinear", "latch", "curved"]
SETUPS = [("grasp", "point grasp"), ("grasp-seq", "point grasp + sequence tool"), ("franka", "Franka arm"), ("franka-seq", "Franka arm + sequence tool")]
TUNNEL = {(lv, su): f"w16-{lv}-{su}-20261109-001" for lv in LEVELS for su, _ in SETUPS}
HINTED = {("curved", "grasp"): "w17-curved-grasp-hint-20261110-001", ("curved", "grasp-seq"): "w17-curved-grasp-seq-hint-20261110-001",
          ("damper", "grasp"): "w17-damper-grasp-hint-20261110-001", ("damper", "grasp-seq"): "w17-damper-grasp-seq-hint-20261110-001",
          ("latch", "grasp-seq"): "w17-latch-grasp-seq-hint-20261110-001", ("nonlinear", "grasp"): "w17-nonlinear-grasp-hint-20261110-002",
          ("nonlinear", "grasp-seq"): "w17-nonlinear-grasp-seq-hint-20261110-001", ("spring", "grasp"): "w17-spring-grasp-hint-20261110-001"}
CUT_SHORT = "w17-nonlinear-grasp-hint-20261110-001"  # orchestrator error: its hold was never reviewed; rerun as -002
BOX = {("heavy", "no hint"): "w18-marble-heavy-20261113-001", ("equal", "no hint"): "w18-marble-equal-20261113-001",
       ("cylinder", "no hint"): "w18-cylinder-20261113-001", ("heavy", "hint"): "w19-marble-heavy-hint-20261114-002"}
BOX_LEVEL = {"heavy": "box_marble_heavy", "equal": "box_marble_equal", "cylinder": "box_cylinder"}
BOX_NAME = {"heavy": "Sealed box, heavy ball", "equal": "Sealed box, light ball", "cylinder": "Sealed box, cylinder"}
PROBES = ["throw_forward", "throw_side", "toss_up", "tilt_shake"]
PROBE_NAME = {"throw_forward": "throw forward", "throw_side": "throw sideways", "toss_up": "toss up", "tilt_shake": "tilt and shake"}
LEVEL_NAME = {"spring": "spring", "damper": "spring + damper", "nonlinear": "nonlinear spring", "latch": "push-push latch", "curved": "curved tunnel, spring"}

rep_n = lambda r: int(m.group(1)) if (m := re.search(r"-r(\d)-\d{8}-\d{3}$", r)) else 1
NAMES = {}
for (lv, cam), runs in SPRING_RUNS.items():
    for r in runs:
        NAMES[r] = ("Spring level, " + ("stiffness stated" if lv == "k" else "block mass stated"), f"{cam} camera, run {rep_n(r)}")
NAMES[SPRING_VARIANTS["square"]] = ("Spring level, square plates, stiffness stated", "movable camera")
NAMES[SPRING_VARIANTS["plug"]] = ("Spring level with plug, stiffness stated", "movable camera")
for scene, d in GEO.items():
    label = dict(base="baseline (previous report)", movecam="movable camera", sam_offered="SAM-3D offered", sam_directed="SAM-3D offered + when to use it",
                 sam_required="SAM-3D use required")
    for k, r in d.items():
        NAMES[r] = ("Boat" if scene == "boat" else "Random assembly", label[k])
for (lv, su), r in TUNNEL.items():
    NAMES[r] = (f"Tunnel: {LEVEL_NAME[lv]}", dict(SETUPS)[su])
for (lv, su), r in HINTED.items():
    NAMES[r] = (f"Tunnel: {LEVEL_NAME[lv]}", dict(SETUPS)[su] + ", hint")
NAMES[CUT_SHORT] = ("Tunnel: nonlinear spring", "point grasp, hint (cut short)")
for (lv, h), r in BOX.items():
    NAMES[r] = (BOX_NAME[lv], h)
ORDER = list(NAMES)
VIEWER = {"spring-k": SPRING_RUNS[("k", "movable")] + SPRING_RUNS[("k", "fixed")],
          "spring-block": SPRING_RUNS[("block", "movable")] + SPRING_RUNS[("block", "fixed")],
          "box": list(BOX.values())}
COMPARISONS = {}
FIGURES = {}
for lv in LEVELS:  # final force profiles (plot-force-profiles.py: every submission of a run)
    for su, _ in SETUPS:
        FIGURES[f"fp-{TUNNEL[(lv, su)]}.png"] = str(MAIN / "validation" / "wave16-analysis" / "force-profiles" / f"{TUNNEL[(lv, su)]}-force-profile.png")
for r in list(HINTED.values()) + [CUT_SHORT]:
    FIGURES[f"fp-{r}.png"] = str(MAIN / "validation" / "wave17-analysis" / "force-profiles" / f"{r}-force-profile.png")


def final_version_dir(run):
    return sorted((NETRUNS / run).glob("version-*"))[-1]


for (lv, h), r in BOX.items():  # composite-CoM trajectory of the throw-forward probe: final submission, and a solid box with the same exterior
    FIGURES[f"com-{r}.png"] = str(next(final_version_dir(r).glob("evaluation/artifacts/*/arm_probe_000/com.png")))
for lv, level in BOX_LEVEL.items():
    FIGURES[f"com-solid-{level}.png"] = str(next((MAIN / "validation" / "wave18-analysis" / "baseline" / level).glob("artifacts/*/arm_probe_000/com.png")))
BASELINE = json.loads((MAIN / "validation" / "wave18-analysis" / "baseline" / "box-baseline.json").read_text())  # solid uniform box, same exterior and mass

# ---------------- metrics ----------------
SPRING_KEYS = ["r:mass:cluster", "r:mass:spring_base", "r:mass:spring_top", "r:stiffness"]
METRICS = {"r:mass:cluster": ("sphere cluster mass", "ratio"), "r:mass:spring_base": ("bottom plate mass", "ratio"), "r:mass:spring_top": ("top plate mass", "ratio"),
           "r:mass:block": ("block mass", "ratio"), "r:mass:plug": ("plug mass", "ratio"), "r:stiffness": ("spring stiffness", "ratio"),
           "mech_rel": ("mechanism force error (÷ true force)", "err"), "fp_push_n": ("slow-push force error (N)", "err"),
           "fp_retract_n": ("slow-retract force error (N)", "err"), "fp_fast_n": ("fast-cycle force error (N)", "err"),
           "traj_mm": ("composite COM path error, mean of 4 throws (mm)", "err"),
           **{f"traj:{p}_mm": (f"COM path error, {PROBE_NAME[p]} (mm)", "err") for p in PROBES}}


def estimates(run, source, m, default):
    """Spring levels: mass on each source body (every positive-mass candidate body goes to the nearest source body by COM, so a split model
    still sums correctly) and the stiffest slide joint. Tunnel and box levels: none (their evaluator metrics are in extra_metrics)."""
    names = [b["name"] for b in source["bodies"]]
    if run.startswith(("w16", "w17", "w18", "w19")):
        return {}
    if "spring_base" not in names:
        return default(source, m)
    out = {f"mass:{n}": 0.0 for n in names}
    for b in m["bodies"]:
        if b["mass"] > 0:
            near = min(source["bodies"], key=lambda s: math.dist(s["xipos"], b["xipos"]))
            out[f"mass:{near['name']}"] += b["mass"]
    springs = [j["stiffness"] for j in m["joints"] if j["type"] == 2 and j["stiffness"] > 0]
    out["stiffness"] = max(springs) if springs else 0.0
    return out


def _fp_errors(fp):
    """RMS resistance error over the mechanism's working range (arc 30-59 mm, before the hard-stop spike), as tunnel-summary.py computes it."""
    src, cand = fp["source"], fp["candidate"]
    stroke, s, a = (np.asarray(src[k]) for k in ("stroke", "arc_length_m", "resistance_n"))
    window, out = (s >= .03) & (s <= .059), {}
    c = np.asarray(cand["resistance_n"]) if cand["status"] == "ok" else None
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    for label, strokes in (("push", (0, 2)), ("retract", (1, 3)), ("fast", (4, 5))):
        sel = window & np.isin(stroke, strokes)
        out[f"fp_{label}_n"] = None if c is None else rms(c[sel] - a[sel])
    out["mech_rel"] = None if c is None else rms(c[window] - a[window]) / rms(a[window])
    return out


_FP = {}


def extra_metrics(run, i, hist):
    if run.startswith(("w16", "w17")):
        if run not in _FP:
            _FP[run] = json.loads((NETRUNS / run / "force-profile-rescore.json").read_text())["versions"]
        assert len(_FP[run]) >= i + 1, run
        return _fp_errors(_FP[run][f"version-{i + 1:03d}"])
    if run.startswith(("w18", "w19")):
        ev = json.loads((NETRUNS / run / f"version-{i + 1:03d}" / "evaluation.json").read_text())
        assert ev["version"] == hist["version"], (run, i)
        probes = {p["name"]: p["object_com"] for p in ev["scores"]["arm_probes"] if p["status"] == "ok"}
        assert set(probes) == set(PROBES), (run, i, list(probes))
        return {"traj_mm": 1000 * float(np.mean([probes[p]["trajectory_mean_error_m"] for p in PROBES])),
                "r:mass": probes["throw_forward"]["candidate_mass_kg"] / probes["throw_forward"]["source_mass_kg"],
                **{f"traj:{p}_mm": 1000 * probes[p]["trajectory_mean_error_m"] for p in PROBES}}
    return {}


def engagement():
    out = {}
    for w in ("wave16-analysis", "wave17-analysis"):
        for x in map(json.loads, (MAIN / "validation" / w / "engagement.jsonl").read_text().splitlines()):
            out[x["run"]] = x
    return out


def quote(run, text):
    """A verbatim model progress note (exactly one note must contain the text)."""
    hits = [x for x in map(json.loads, (NETRUNS / run / "progress.jsonl").read_text().splitlines()) if x.get("origin") == "model" and text in x["text"]]
    assert len(hits) == 1, (run, text, len(hits))
    return dict(text=hits[0]["text"], run=run, name=" · ".join(NAMES[run]), minute=round(hits[0]["wall_elapsed"] / 60))


META = dict(title="Hidden mechanisms, hidden contents and a movable camera: sim2sim results, 6–14 November 2026",
            subtitle=L("Runs finished 2026-11-06 to 2026-11-14 · {n} runs (two are baselines repeated from the previous report) · each run is one "
                       "<code>gpt-6-astra</code> agent (medium reasoning) working alone in an isolated container, up to 4 h (spring and geometry levels) or 6 h "
                       "(tunnel and box levels). <b>Mostly one run per condition</b>; the spring-level camera comparison has three runs per condition."),
            intro=L(
                "<h2 style='border:0;margin-top:20px'>The experiment</h2>"
                "<p>Each run hides a MuJoCo scene and asks an LLM agent to rebuild it as a MuJoCo scene file whose geometry and physics match. The agent "
                "never sees the hidden file. It can look through a camera, act on the hidden scene, load its own candidates into a second simulator, and "
                "submit candidates; a private evaluator scores each submission. The previous report (scenes with free objects, hinged levers and springs) "
                "found that geometry comes from the camera, and that absolute mass needs a force reading or one known mass. This report covers the "
                "next questions: <b>can an agent identify physics it cannot see</b> (a mechanism at the closed end of a tunnel, a body sealed inside a "
                "box), and <b>does controlling the camera, or a 3D shape-recovery tool, help</b>?</p>"
                "<h3>Levels</h3><div class='card scroll'><table class='intro'>"
                "<tr><th>level</th><th>what it is</th><th>what is scored</th></tr>"
                "<tr><td>Spring level</td><td>A Franka Panda arm, a table with a round hole (13.6 cm wide, 9 cm deep) and a spring lying on its side: two "
                "plates (12 g and 8 g, frictionless against the table) joined by an ideal linear spring (21.7 N/m, slight damping), next to a cluster of four "
                "fused spheres (57.2 g; masses 1:2:4:8). The task states either the stiffness or, in the block variant, the mass of an extra 23.5 g block. "
                "Variants: square plates; a 31.7 g plug of unstated mass in the hole. <b>No force sensing</b>: the Franka's joint torques are hidden</td>"
                "<td>shape (Chamfer), COM, each part's mass, stiffness</td></tr>"
                "<tr><td>Boat, random assembly</td><td>The free-standing geometry scenes of the previous report (fixed camera, point grasp with force "
                "reading)</td><td>shape, COM, mass</td></tr>"
                "<tr><td>Tunnel</td><td>A stick (0.38 kg, with a knob) in a frictionless bore. 30 mm past the stick's end, out of view at the tunnel's closed "
                "far end, sits a plunger with 30 mm of travel to a hard stop. Five mechanisms behind it: a linear <b>spring</b> (200 N/m); "
                "<b>spring + damper</b> (+20 N·s/m); a <b>nonlinear spring</b> (linear to 10 mm, a 2 N plateau to 20 mm, then a quadratic rise); a "
                "<b>push-push latch</b> that holds the plunger in after one push and releases it on the next; and the spring in a <b>curved</b> tunnel "
                "(radius 0.25 m). The task never mentions a tunnel, spring or mechanism</td>"
                "<td>resistance against the stick along the push path (force profile), shape</td></tr>"
                "<tr><td>Sealed box</td><td>An 80 g, 57 × 82 × 42 mm closed box the Franka can pick up. Inside, behind 4 mm walls: a 19 mm ball that "
                "rolls (<b>heavy ball</b>: 68 g ball in a 12 g shell; <b>light ball</b>: 40 g in a 40 g shell), or a 55 g cylinder that rolls along one "
                "axis. The exterior is identical in all three</td>"
                "<td>path of the composite COM (box + contents) during four scripted throws by the arm, against a solid box with the same exterior and "
                "mass</td></tr></table></div>"
                "<h3>Conditions</h3><div class='card scroll'><table class='intro'>"
                "<tr><th>condition</th><th>what it means</th></tr>"
                "<tr><td>point grasp</td><td>The agent pins any visible point of an object to an ideal handle, moves it, and reads the force on the handle</td></tr>"
                "<tr><td>Franka arm</td><td>The agent drives a simulated Franka Panda: joint targets, joint angles and (except on spring levels) joint torques</td></tr>"
                "<tr><td>movable camera</td><td>The agent may re-aim the observation camera anywhere (an action that costs no simulation time). Otherwise one "
                "fixed camera</td></tr>"
                "<tr><td>sequence tool</td><td>The agent can submit a list of up to 2000 physical actions written by its own code, run in one call, and get "
                "every step's readings back</td></tr>"
                "<tr><td>hint</td><td>One added task sentence says that the level hides something (the tunnel: \"a mechanism at its far end, out of view\"; "
                "the box: \"hidden contents that can move inside it\") without saying what</td></tr>"
                "<tr><td>SAM-3D</td><td>An action that sends an image and a mask of one part to a 3D shape-recovery model (SAM-3D on a GPU) and returns a mesh. "
                "Offered; offered with a sentence on when it helps; or required before modeling each part</td></tr>"
                "</table><p class='note'>All runs use the stop-when-done prompt of the previous report (finish when nothing about the reconstruction differs "
                "from the hidden scene). Each run has an API cost limit the agent never sees; tunnel and box runs pause at that limit for review and were "
                "either resumed (if the agent was still making progress on the hidden mechanism) or stopped (Methods).</p></div>"))
GLOSSARY = [
    ("Estimate ÷ true (×k)", "The agent's value divided by the true value, read from its compiled scene file. ×1 = exact; \"0 (dropped)\" = left out."),
    ("Chamfer, COM error", "Symmetric surface distance between true and reconstructed shapes (mm), and distance between their centers of mass (mm)."),
    ("Mechanism force error", "Tunnel levels: the evaluator pushes the stick in and out (two slow cycles, one fast) in the hidden scene and in the "
     "submission, and compares the resistance along the path. Reported as the RMS difference over the mechanism's working range (30–59 mm of travel, "
     "before the hard stop), divided by the RMS true resistance there: 0 = exact, 1 = as wrong as modeling no mechanism at all."),
    ("COM path error", "Box levels: the evaluator's arm picks the box up and throws or shakes it four ways; the error is the mean distance between "
     "the true and reconstructed composite COM (box + contents) over each throw. Compared with a solid uniform box of the same exterior and mass."),
    ("Run-to-run range", "Spread of three runs of the same condition (shaded bands in the time courses). A single-run difference smaller than this is not evidence."),
    ("Held / resumed / stopped", "At its API cost limit a tunnel or box run paused. A researcher read its notes and resumed it with a higher limit if it was "
     "still working on the hidden mechanism, or stopped it. Paused time is not counted in its wall clock."),
]


def strategy(ctx):
    from collections import Counter
    kinds = Counter()
    for v in ctx.runs.values():
        kinds.update(v["taxonomy"]["script_counts"])
    share = lambda *ks: round(100 * sum(kinds[k] for k in ks) / sum(kinds.values()))
    return dict(answer=L(f"As before: systematic fitting, then checking. {share('fit_sim', 'fit_data')}% of {sum(kinds.values())} bash calls ran an optimizer "
                         f"or parameter sweep, {share('simulate')}% a single local simulation or render, {share('analytic')}% closed-form physics."))


def methods(ctx):
    return [
        "<b>Every run is a strict run</b>: isolated container, no access to scene files, model called through the API. The builder asserts this for every run.",
        "<b>Mostly one run per condition.</b> Only the spring-level camera comparison has three runs per condition; elsewhere read differences as trends "
        "unless they are large (the tunnel hint and the box contents).",
        "<b>Physical parameters</b> are read by compiling each hidden scene and submission with MuJoCo. Spring levels: every positive-mass body of a "
        "submission is assigned to the nearest true body by COM and summed; the spring stiffness is the stiffest slide joint. Tunnel and box scores "
        "are the private evaluator's (force profile, composite COM paths).",
        "<b>Tunnel force profiles were rescored</b> after the runs (2026-11-10/11): the first probe used a fixed world ray and missed candidates built "
        "in shifted frames. The rescored probe grasps through a trusted camera pixel; every submission of waves 16–17 scores.",
        "<b>Point-grasp force reading changed on 2026-11-09</b> (tunnel runs onward): it now reports the force applied in the last step. The old reading "
        "was low by about 10 N·s/m × velocity (phantom damping); damping estimates from earlier point-grasp runs are not comparable.",
        "<b>Franka \"no force sensing\" is incomplete.</b> Hiding the joint torques does not remove load sensing: the position controller's steady-state "
        "joint deflection grows with the payload, and two spring-level agents used it as a scale.",
        "<b>Reviews at the cost limit</b> (tunnel, box): resumed if the agent had engaged the hidden mechanism and its model of it was incomplete; "
        "stopped otherwise. One hinted tunnel run was never reviewed (orchestrator error) and stopped at 3.1 h; it was rerun.",
    ]


def sections(ctx):
    R, f = ctx.runs, ctx.final
    blk = lambda **kw: kw
    conv = lambda series, keys, **kw: blk(type="convergence", keys=keys, **kw,
                                          series=[dict(**({"runs": s[0]} if isinstance(s[0], list) else {"run": s[0]}), label=s[1], **(s[2] if len(s) > 2 else {})) for s in series])
    lnk = lambda r, t=None: f"<a href='run.html?run={r}'>{t or R[r]['condition']}</a>"
    x = lambda v, d=3: "—" if v is None else ctx.x(v, d)
    mm = lambda v, d=3: "—" if v is None else ctx.mm(v, d)
    pct = lambda v: "—" if v is None else "exact" if v < 1e-3 else f"{100 * v:.0f}%"
    hl = lambda txt, good: f"<span class='hl {'good' if good else 'bad'}'>{txt}</span>"
    counters = {}

    def cap(kind, text, appendix=False):
        counters[(kind, appendix)] = counters.get((kind, appendix), 0) + 1
        return blk(type="html", html=f"<p class='caption'><b>{kind} {'A' if appendix else ''}{counters[(kind, appendix)]}.</b> {text}</p>")

    lead = lambda text: blk(type="html", html=f"<p>{text}</p>")
    first_le = lambda r, k, v: next((u["t"] for u in R[r]["subs"] if u["m"].get(k) is not None and u["m"][k] <= v), None)
    minute = lambda t: "never" if t is None else f"min {t:.0f}"
    med = lambda vals: float(np.median([v for v in vals if v is not None]))
    sec = []

    # ---------------- 1 · spring levels and the movable camera ----------------
    within = lambda r, k, tol=0.05: f(r, k) is not None and abs(f(r, k) - 1) <= tol
    plates_exact = lambda r: within(r, "r:mass:spring_base", 0.01) and within(r, "r:mass:spring_top", 0.01)
    rows = []
    for lv in ("k", "block"):
        for cam in ("movable", "fixed"):
            for r in SPRING_RUNS[(lv, cam)]:
                rows.append([("stiffness stated" if lv == "k" else "block mass stated"), f"{cam} camera", lnk(r, f"run {rep_n(r)}"),
                             hl(mm(f(r, "chamfer_mm")), f(r, "chamfer_mm") < 1.5), mm(f(r, "com_mm")), hl(x(f(r, "r:mass:cluster")), within(r, "r:mass:cluster")),
                             hl(f"{x(f(r, 'r:mass:spring_base'))} / {x(f(r, 'r:mass:spring_top'))}", plates_exact(r)),
                             "stated" if lv == "k" else hl(x(f(r, "r:stiffness")), within(r, "r:stiffness", 0.2))])
    pairs = [(lv, i) for lv in ("k", "block") for i in range(3)]
    better = sum(f(SPRING_RUNS[(lv, "movable")][i], "chamfer_mm") < f(SPRING_RUNS[(lv, "fixed")][i], "chamfer_mm") for lv, i in pairs)
    plates = {cam: sum(plates_exact(r) for lv in ("k", "block") for r in SPRING_RUNS[(lv, cam)]) for cam in ("movable", "fixed")}
    medc = {(lv, cam): med([f(r, "chamfer_mm") for r in SPRING_RUNS[(lv, cam)]]) for lv in ("k", "block") for cam in ("movable", "fixed")}
    var_rows = [[lnk(r, NAMES[r][0]), mm(f(r, "chamfer_mm")), mm(f(r, "com_mm")), x(f(r, "r:mass:cluster")),
                 f"{x(f(r, 'r:mass:spring_base'))} / {x(f(r, 'r:mass:spring_top'))}", x(R[r]["subs"][-1]["m"].get("r:mass:plug"))]
                for r in SPRING_VARIANTS.values()]
    sec.append(dict(id="camera", nav="Movable camera", tag="three runs per condition on the spring levels; one on boat and assembly",
        title="1 · Spring levels: a movable camera improves shape and makes the spring identifiable",
        q=None, answer=L(f"On the spring levels the movable camera gave better shape in {better} of {len(pairs)} paired runs (median Chamfer "
                         f"{mm(medc[('k', 'movable')])} vs {mm(medc[('k', 'fixed')])} and {mm(medc[('block', 'movable')])} vs {mm(medc[('block', 'fixed')])}). With the block "
                         f"mass stated, the stiffness came out {', '.join(x(f(r, 'r:stiffness')) for r in SPRING_RUNS[('block', 'movable')])} with the movable camera and "
                         f"{', '.join(x(f(r, 'r:stiffness')) for r in SPRING_RUNS[('block', 'fixed')])} with the fixed one, and the movable camera got both "
                         f"plate masses exact in {plates['movable']}/6 runs against {plates['fixed']}/6. On the boat and random assembly it changed nothing "
                         "beyond run-to-run noise (section 2)."),
        blocks=[lead(L("The spring levels have no force sensing, so masses must come from motion: how far the stated-stiffness spring compresses under a "
                       "weight, or how fast the two plates oscillate. Those motions are small (millimetres) and seen face-on only from some viewpoints. With "
                       "the movable camera agents used 37–51 camera moves per run for face-on plate measurements and orthogonal views.")),
                cap("Table", L("<b>Spring levels, movable vs fixed camera, three runs each</b> (final submission; green = Chamfer under 1.5 mm, mass within 5%, "
                               "plates within 1%, stiffness within 20%). Masses are estimate ÷ true; plates are bottom / top. ×0.833 / ×1.25 means two equal plates (10 g each).")),
                blk(type="table", head=["level", "camera", "run", "Chamfer", "COM error", "cluster mass", "plate masses", "stiffness"], rows=rows),
                cap("Figure", L("<b>Time course: the movable camera reaches a close shape during the run; the fixed camera mostly never does.</b> Mean (line) "
                                "and range (shading) of the three runs per condition at every submission. First submission with Chamfer under 1.5 mm, per run: " +
                                "; ".join(f"{'stiffness stated' if lv == 'k' else 'block mass stated'}, {cam} camera: " +
                                          ", ".join(minute(first_le(r, 'chamfer_mm', 1.5)) for r in SPRING_RUNS[(lv, cam)])
                                          for lv in ("k", "block") for cam in ("movable", "fixed")) + ".")),
                conv([[SPRING_RUNS[("k", "movable")], "movable camera"], [SPRING_RUNS[("k", "fixed")], "fixed camera"]], ["chamfer_mm", "com_mm"] + SPRING_KEYS[:3],
                     title="Spring level, stiffness stated"),
                conv([[SPRING_RUNS[("block", "movable")], "movable camera"], [SPRING_RUNS[("block", "fixed")], "fixed camera"]], ["chamfer_mm", "com_mm"] + SPRING_KEYS,
                     title="Spring level, block mass stated"),
                cap("Figure", "<b>Spring-level reconstructions in 3D</b> (solid = truth, wireframe = reconstruction; slider = submission)."),
                blk(type="viewer", group="spring-k", title="Stiffness stated, all six runs"),
                blk(type="viewer", group="spring-block", title="Block mass stated, all six runs"),
                cap("Table", L("<b>Two variants, one run each (movable camera).</b> Square plates that fit the hole at any angle, and a plug of unstated mass "
                               "standing in the hole. No agent put the spring into the hole, and none resolved the 1:2:4:8 sphere masses of the cluster (all "
                               "modeled equal spheres, which is the remaining COM error).")),
                blk(type="table", head=["level", "Chamfer", "COM error", "cluster mass", "plate masses", "plug mass"], rows=var_rows)]))
    sam_calls = {r: sum(n for k, n in R[r]["actions"].items() if k.startswith("sam3d|")) for d in GEO.values() for k, r in d.items() if k.startswith("sam")}
    sec.append(dict(id="sam3d", nav="SAM-3D", tag="one run per condition",
        title="2 · Geometry scenes: neither a movable camera nor a 3D shape-recovery tool changes shape accuracy; the tool is used only when required",
        q=None, answer=L(f"Offered SAM-3D, agents never called it ({', '.join(str(sam_calls[GEO[s][k]]) for s, k in (('boat', 'sam_offered'), ('boat', 'sam_directed'), ('assembly', 'sam_directed')))} "
                         f"calls), even when the task said when it helps. Required, they called it {sam_calls[GEO['boat']['sam_required']]} and "
                         f"{sam_calls[GEO['assembly']['sam_required']]} times, mostly in the first 15 minutes, and still built primitive unions; the meshes "
                         "served as hints and Chamfer did not improve. The movable camera did not change shape accuracy either."),
        blocks=[lead(L("The tool sat in every model request of these runs. The agents' notes never mention it unless the task required it; when required, "
                       "the assembly agent used one returned sphere as a hypothesis and then checked it against depth measurements.")),
                cap("Table", L("<b>Boat and random assembly: movable camera and SAM-3D against the baseline of the previous report</b> (one run each, final "
                               "submission). The previous report's three assembly baselines spread 13.7–18.8 mm in Chamfer, larger than any difference here.")),
                blk(type="table", head=["scene", "condition", "Chamfer", "COM error", "mass ÷ true", "SAM-3D calls"],
                    rows=[[R[r]["scene"], lnk(r), mm(f(r, "chamfer_mm")), mm(f(r, "com_mm")), x(f(r, "r:mass")),
                           str(sum(n for k, n in R[r]["actions"].items() if k.startswith("sam3d|"))) if "sam" in k else "—"]
                          for scene, d in GEO.items() for k, r in d.items()]),
                conv([[GEO["boat"]["base"], "baseline"], [GEO["boat"]["movecam"], "movable camera"], [GEO["boat"]["sam_required"], "SAM-3D required", {"color": "--third"}]],
                     ["chamfer_mm", "com_mm"], title="Boat"),
                conv([[GEO["assembly"]["base"], "baseline"], [GEO["assembly"]["movecam"], "movable camera"], [GEO["assembly"]["sam_required"], "SAM-3D required", {"color": "--third"}]],
                     ["chamfer_mm", "com_mm"], title="Random assembly")]))

    # ---------------- 3 · tunnel ----------------
    eng = engagement()
    mech = lambda r: f(r, "mech_rel")
    cell = lambda r: (f"{lnk(r, pct_err(mech(r)))}" + ("" if eng[r]["max_plunger_compression"] > 0 else " <span class='muted'>(never touched it)</span>"))
    pct_err = lambda v: "—" if v is None else f"{v:.2f}"
    grid = [[LEVEL_NAME[lv]] + [cell(TUNNEL[(lv, su)]) for su, _ in SETUPS] + [cell(HINTED[(lv, "grasp")]) if (lv, "grasp") in HINTED else "—",
                                                                               cell(HINTED[(lv, "grasp-seq")]) if (lv, "grasp-seq") in HINTED else "—"] for lv in LEVELS]
    good = lambda r: mech(r) is not None and mech(r) < 0.3
    unh_good = [r for r in TUNNEL.values() if good(r)]
    hint_good = [r for r in HINTED.values() if good(r)]
    never = [r for r in TUNNEL.values() if eng[r]["max_plunger_compression"] == 0]
    fr_eng = all(eng[r]["max_plunger_compression"] > 0 for (lv, su), r in TUNNEL.items() if su.startswith("franka"))
    sec.append(dict(id="tunnel", nav="Tunnel", tag="20 runs without hint, 8 with · one run per condition",
        title="3 · A mechanism out of view: found when the task says one is there, rarely otherwise",
        q=None, answer=L(f"Without a hint, {len(unh_good)} of {len(TUNNEL)} runs ended with a mechanism model within 30% of the true resistance; "
                         f"{len(never)} point-grasp runs never pushed the stick far enough to touch the plunger. With one sentence saying a mechanism is "
                         f"there, every hinted run modeled a mechanism and {len(hint_good)} of {len(HINTED)} got within 30%, on conditions where the unhinted runs had "
                         "nothing. The misses are the curved tunnel (right mechanism, wrong force curve) and the latch (it reproduces the latching but "
                         "its final version has one spurious force spike)."),
        blocks=[lead(L("The stick first slides freely for 30 mm, so an agent that only pulls the stick out, or pushes it a little, sees a plain sliding "
                       "rod. Finding the mechanism needs a push past the free travel and attention to the resistance that follows." +
                       (" Every Franka run pushed the plunger; with the point grasp, agents often pulled the stick out and fitted its shape instead." if fr_eng else ""))),
                cap("Table", L("<b>Final mechanism force error, every tunnel run</b> (0 = exact, 1 ≈ no mechanism; values above 1 are worse than modeling "
                               "nothing). Hinted runs (last two columns) are reruns of the unhinted conditions in which no mechanism was found; \"never touched "
                               "it\" = the plunger never moved in the hidden scene (from the recorded states).")),
                blk(type="table", head=["mechanism", "point grasp", "point grasp + sequence", "Franka", "Franka + sequence", "point grasp, hint", "point grasp + sequence, hint"],
                    rows=grid),
                cap("Figure", L("<b>Force profiles, without and with the hint</b> (researcher plots: the true resistance in black, each submission colored "
                                "from early to late; left two panels slow push/retract cycles, right the fast cycle).")),
                blk(type="images", items=[dict(src=f"fp-{r}.png", label=" · ".join(NAMES[r]), run=r) for lv, su in (("damper", "grasp"), ("nonlinear", "grasp-seq"))
                                          for r in (TUNNEL[(lv, su)], HINTED[(lv, su)])]),
                cap("Figure", L("<b>Time course of the mechanism error, without and with the hint.</b> One point per submission.")),
                conv([[TUNNEL[(lv, "grasp")], f"{LEVEL_NAME[lv]}, no hint"] for lv in ("damper", "spring")] +
                     [[HINTED[(lv, "grasp")], f"{LEVEL_NAME[lv]}, hint", {"color": c}] for lv, c in (("damper", "--third"), ("spring", "--fourth"))],
                     ["mech_rel"], title="Point grasp", xmax=360)]))

    # ---------------- 4 · box ----------------
    base = {lv: 1000 * float(np.mean([BASELINE[level][p]["trajectory_mean_error_m"] for p in PROBES])) for lv, level in BOX_LEVEL.items()}
    brow = []
    for (lv, h), r in BOX.items():
        brow.append([lnk(r, BOX_NAME[lv]), h] + [hl(mm(f(r, f"traj:{p}_mm")), f(r, f"traj:{p}_mm") < 1000 * BASELINE[BOX_LEVEL[lv]][p]["trajectory_mean_error_m"])
                                                for p in PROBES] + [f"<b>{mm(f(r, 'traj_mm'))}</b>", mm(base[lv]), x(f(r, "r:mass"))])
    sec.append(dict(id="box", nav="Sealed box", tag="one run per condition · a fourth run is still in progress",
        title="4 · Contents sealed in a box: inferred only from how the load shifts, and only when the contents slide or the task says they exist",
        q=None, answer=L("Without a hint only the cylinder agent found the contents (a sliding body, from load-centre shifts in tilted grasps). Both ball agents "
                         "fitted contact and friction around a rigid box instead, and their models move worse than a plain solid box. With the hint, the "
                         "heavy-ball agent modeled a free sphere inside a hollow shell and became the best ball model."),
        blocks=[cap("Table", L("<b>Composite COM path error per throw</b> (mm, final submission; green = better than a solid uniform box with the same "
                               "exterior and mass, the \"nothing inside\" model). Total mass was exact in every final model.")),
                blk(type="table", head=["level", "task", *[PROBE_NAME[p] for p in PROBES], "mean", "solid box", "mass ÷ true"], rows=brow),
                cap("Figure", L("<b>Throw forward: true (blue) and reconstructed (dashed) composite COM</b> for each run's final submission, and for the solid "
                                "box. Bottom panels: distance between them.")),
                blk(type="images", items=[dict(src=f"com-{r}.png", label=" · ".join(NAMES[r]), run=r) for r in BOX.values()] +
                                         [dict(src=f"com-solid-{BOX_LEVEL['heavy']}.png", label="Solid uniform box, heavy-ball level")]),
                blk(type="quotes", title="What the agents said", items=[
                    quote(BOX[("cylinder", "no hint")], "A sliding internal ballast would explain"),
                    quote(BOX[("cylinder", "no hint")], "about 19 times lower error"),
                    quote(BOX[("equal", "no hint")], "The lift indicates an 80 g mass and a likely off-center mass distribution"),
                    quote(BOX[("heavy", "hint")], "hollow six-wall box with a freely moving internal sphere")]),
                cap("Figure", "<b>Time course of the mean COM path error</b> (the solid box scores " + ", ".join(f"{base[lv]:.0f} mm ({BOX_NAME[lv].split(', ')[1]})" for lv in BOX_LEVEL) + ")."),
                conv([[r, " · ".join(NAMES[r])] for r in BOX.values()], ["traj_mm"], title="Composite COM path error", xmax=380),
                cap("Figure", "<b>Box reconstructions in 3D</b>: the hidden body is visible here (solid = truth, wireframe = reconstruction)."),
                blk(type="viewer", group="box", title="Sealed boxes")]))

    # ---------------- 5 · strategies ----------------
    sec.append(dict(id="strategy", nav="Strategies", tag=None, title="5 · How the agents work",
        q=None, answer=None, blocks=[lead("Every bash command and progress note, classified with the keyword rules of the previous report."), blk(type="strategy")]))

    # ---------------- appendix ----------------
    sec.append(dict(id="invalid", nav="A1", appendix=True, tag=None, title="A1 · Invalid runs and the fixes they prompted", q=None, answer=None,
        blocks=[lead(L("Three hinted box runs were invalid and are not in this report. In the first two (2026-11-14) the agents parked the movable camera "
                       "against the box, and its near clip plane cut through the 4 mm wall: \"The cavity view reveals a spherical internal weight\". The "
                       "camera now refuses positions closer than its near-plane reach to any surface. The relaunched light-ball run then carried the box "
                       "until the camera sat inside the cavity; now any surface inside the camera's clearance blacks out the image. Its third attempt "
                       "was still running when this report was built. No unhinted box run did either.")),
                lead(L(f"The hinted nonlinear tunnel run {lnk(CUT_SHORT, 'with the point grasp')} paused at its cost limit and was never reviewed "
                       f"(orchestrator error), so it stopped at {R[CUT_SHORT]['wall_min']:.0f} min; its rerun is the one in the tunnel table (section 3)."))]))
    sec.append(dict(id="effort", nav="A2", appendix=True, tag=None, title="A2 · Time and cost", q=None, answer=None,
        blocks=[blk(type="bars", title="Wall clock used (minutes)", unit="min", rows=[dict(run=r, label=" · ".join(NAMES[r]), v=R[r]["wall_min"],
                                                                                         color="--abl" if R[r]["reason"] == "budget_suspended" else "--base") for r in ORDER]),
                blk(type="bars", title="API cost (USD, gateway estimate)", unit="$", rows=[dict(run=r, label=" · ".join(NAMES[r]), v=R[r]["cost_usd"], color="--faint") for r in ORDER])]))
    sec.append(dict(id="runs", nav="A3", appendix=True, tag=None, title="A3 · All runs", q=None, answer=None,
                    blocks=[blk(type="html", html="<p>Click a name for its page with the full conversation and the images the model received.</p>"), blk(type="runs")]))
    return sec
