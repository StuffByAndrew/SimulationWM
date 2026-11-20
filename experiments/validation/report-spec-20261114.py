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
LABREPORTS = Path("/n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-experiments/reports")  # superseded work folders moved here 2026-11-20
WORK = LABREPORTS / "report-work-20261114"
PLAIN = LABREPORTS / "report-work-plain"
RA = MAIN / "validation" / "rerun-analysis"
NETRUNS = Path("/n/netscratch/qianqian_lab/Lab/aejxie/sim2sim-experiments/runs")
REF = {"boat": "w10-boat-base-20261105-001", "assembly": "w10-assembly-base-20261105-001"}  # baselines from the previous report
WORK2 = MAIN / "reports" / "report-work-20261118"  # runs finished 2026-11-14..18 (waves 19-22 and replicates)
WORK3 = MAIN / "reports" / "report-work-20261120"  # Gemini runs (2026-11-19/20) and astra replicates finished 2026-11-18
WITH_MEDIA = True  # evaluation videos, cloth stills and hand overlays (render-eval-videos.py, re-extracted models)
WITH_SCALE = True  # spring-as-a-scale tables (spring-scale-usage.py)
SUMMARIES = (MAIN / "validation" / "results-w11-18.json", MAIN / "validation" / "results-w19.json", MAIN / "validation" / "results-w20-22.json",
             *(MAIN / "validation" / f"results-{n}.json" for n in ("gemini-neon", "gemini-hydrogen", "gemini-neon-set1", "gemini-hydrogen-set1", "rep-1117-1118")),
             MAIN / "validation" / "results-w19-003.json", (MAIN / "validation" / "results-wave10.json", set(REF.values())))
INVALID = set()  # invalid attempts (wave 19 -001, marble-equal -002) are not in NAMES; Appendix A1 describes them
MODELS = (RA / "models-H.json", RA / "models-I.json", RA / "models-J.json", RA / "models-K.json", RA / "models-L.json", RA / "models-M.json",
          MAIN / "validation" / "rerun-analysis-plain" / "models-F.json", MAIN / "validation" / "rerun-analysis-plain" / "models-G.json", WORK / "models-w16-19.json", WORK2 / "models.json", WORK3 / "models.json")
CHECKPOINTS = (RA / "checkpoints-H.json", RA / "checkpoints-I.json", RA / "checkpoints-J.json", RA / "checkpoints-K.json", RA / "checkpoints-L.json",
               RA / "checkpoints-M.json", MAIN / "validation" / "rerun-analysis-plain" / "checkpoints-F.json",
               MAIN / "validation" / "rerun-analysis-plain" / "checkpoints-G.json", WORK / "checkpoints-w16-19.json", WORK2 / "checkpoints.json", WORK3 / "checkpoints.json")
TRACES = (WORK / "traces", WORK2 / "traces", WORK3 / "traces", PLAIN / "traces")
CHAT = (WORK / "chat", WORK2 / "chat", WORK3 / "chat", PLAIN / "chat")
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
       ("cylinder", "no hint"): "w18-cylinder-20261113-001", ("heavy", "hint"): "w19-marble-heavy-hint-20261114-002",
       ("equal", "hint"): "w19-marble-equal-hint-20261114-003", ("heavy", "hint, run 2"): "w19-marble-heavy-hint-r2-20261118-001",
       ("equal", "hint, run 2"): "w19-marble-equal-hint-r2-20261118-001"}
BOX_LEVEL = {"heavy": "box_marble_heavy", "equal": "box_marble_equal", "cylinder": "box_cylinder"}
BOX_NAME = {"heavy": "Sealed box, heavy ball", "equal": "Sealed box, light ball", "cylinder": "Sealed box, cylinder"}
PROBES = ["throw_forward", "throw_side", "toss_up", "tilt_shake"]
PROBE_NAME = {"throw_forward": "throw forward", "throw_side": "throw sideways", "toss_up": "toss up", "tilt_shake": "tilt and shake"}
LEVEL_NAME = {"spring": "spring", "damper": "spring + damper", "nonlinear": "nonlinear spring", "latch": "push-push latch", "curved": "curved tunnel, spring"}

W20 = lambda n: f"w20-{n}-20261116-001"
OUT = {"spring": W20("spring-out"), "latch": W20("latch-out")}  # tunnels with the stick starting outside
MSD = {("spring", "visible"): "w20-msd-spring-visible-20261116-002", ("spring", "hidden"): W20("msd-spring-hidden"),
       ("damper", "visible"): W20("msd-damper-visible"), ("damper", "hidden"): W20("msd-damper-hidden")}
POINT = {("tetra", "k"): W20("tetra-spring"), ("tetra", "block"): W20("tetra-spring-block"), ("cube", "k"): W20("cube-spring"), ("cube", "block"): W20("cube-spring-block")}
LIGHT = {  # family: ([runs without the true lighting], [runs with it]); first runs were extended to 6 h, replicates stopped at their cost limit
    "spring-out": ([OUT["spring"], "w20-spring-out-r2-20261118-001"], [W20("spring-out-lit"), "w20-spring-out-lit-r2-20261116-001"]),
    "tetra": ([POINT[("tetra", "k")], "w20-tetra-spring-r2-20261117-001"], [W20("tetra-spring-lit"), "w20-tetra-spring-lit-r2-20261117-001"]),
    "cube": ([POINT[("cube", "k")], "w20-cube-spring-r2-20261117-001"], [W20("cube-spring-lit"), "w20-cube-spring-lit-r2-20261117-001"]),
    "msd": ([MSD[("spring", "visible")], "w20-msd-spring-visible-r2-20261116-001"], [W20("msd-spring-visible-lit"), "w20-msd-spring-visible-lit-r2-20261118-001"]),
    "cloth": (["w22-cloth-base-20261116-001", "w22-cloth-base-r2-20261116-001"], ["w22-cloth-base-lit-20261116-001", "w22-cloth-base-lit-r2-20261116-001"])}
LIGHT_NAME = {"spring-out": "Tunnel from outside: spring", "tetra": "Cluster on a spring, stiffness stated", "cube": "Cube on a spring, stiffness stated",
              "msd": "1D mass, visible spring", "cloth": "Cloth: uniform"}
HAND = {"prism": "w21-hand-prism-20261116-001", "prism-offset-com": "w21-hand-prism-offset-com-20261116-001", "smooth-mesh": "w21-hand-smooth-mesh-20261116-001",
        "spiky": "w21-hand-spiky-20261116-001", "spiky r2": "w21-hand-spiky-r2-20261116-001", "prism r2": "w21-hand-prism-r2-20261117-001",
        "prism-offset-com r2": "w21-hand-prism-offset-com-r2-20261117-001", "smooth-mesh r2": "w21-hand-smooth-mesh-r2-20261118-001"}
HAND_NAME = {"prism": "prism", "prism-offset-com": "prism, off-centre COM", "smooth-mesh": "smooth convex shape", "spiky": "two cones", "spiky r2": "two cones",
             "prism r2": "prism", "prism-offset-com r2": "prism, off-centre COM", "smooth-mesh r2": "smooth convex shape"}
CLOTH_LEVELS = ["base", "quadrants", "quadrants-colored", "quadrants-diagonal-colors", "holes", "center-hole-stiff-rim", "center-hole-soft-rim"]
CLOTH_NAME = {"base": "uniform", "quadrants": "four stiffness quadrants", "quadrants-colored": "quadrants, each coloured", "quadrants-diagonal-colors":
              "quadrants, colours on the diagonals (misleading)", "holes": "holes and cuts", "center-hole-stiff-rim": "centre hole, stiff rim",
              "center-hole-soft-rim": "centre hole, soft rim"}
CLOTH = {lv: [f"w22-cloth-{lv}-20261116-001"] + ([f"w22-cloth-{lv}-r2-20261116-001"] if lv in ("base", "quadrants", "quadrants-colored", "quadrants-diagonal-colors") else [])
         for lv in CLOTH_LEVELS}
LOST = ["w18-cylinder-r2-20261116-001", "w19-marble-equal-hint-r2-20261116-001", "w19-marble-heavy-hint-r2-20261116-001", "w20-cube-spring-r2-20261116-001",
        "w20-cube-spring-lit-r2-20261116-001", "w20-msd-spring-visible-lit-r2-20261116-001", "w20-spring-out-r2-20261116-001", "w20-tetra-spring-r2-20261116-001",
        "w20-tetra-spring-lit-r2-20261116-001", "w21-hand-prism-r2-20261116-001", "w21-hand-prism-offset-com-r2-20261116-001", "w21-hand-smooth-mesh-r2-20261116-001"]
GEM_LEVELS = {  # level: (scene name in this report, astra runs, primary error key)
    "w22-cloth-holes": ("Cloth: holes and cuts", ["w22-cloth-holes-20261116-001"], "stiff_rel"),
    "w11-tetra-spring-k": ("Spring level, stiffness stated", ["w11-tetra-spring-k-20261106-001", "w11-tetra-spring-k-r2-20261107-001", "w11-tetra-spring-k-r3-20261107-001"], "com_mm"),
    "w21-hand-smooth-mesh": ("Shadow hand: smooth convex shape", ["w21-hand-smooth-mesh-20261116-001", "w21-hand-smooth-mesh-r2-20261118-001"], "chamfer_mm"),
    "w19-marble-heavy-hint": ("Sealed box, heavy ball", ["w19-marble-heavy-hint-20261114-002", "w19-marble-heavy-hint-r2-20261118-001"], "traj_mm"),
    "w20-latch-out": ("Tunnel from outside: push-push latch", ["w20-latch-out-20261116-001"], "fp_rel")}
GEM_MODELS = {"neon": "Gemini 4 Neon", "hydrogen": "Gemini 4 Hydrogen"}
GEM_LABEL = {"stiff_rel": "lift-force error over a 4×4 grid", "com_mm": "COM error (mm)", "chamfer_mm": "Chamfer (mm)",
             "traj_mm": "composite COM path error, 4 throws (mm)", "fp_rel": "force-profile error"}
GEM = {(lv, m): [f"{lv}-{m}-{d}-001" for d in ("20261119", "20261120")] for lv in GEM_LEVELS for m in GEM_MODELS}  # set 1, set 2 (provider-outage hold)
GEM_DONE = {k: [r for r in runs if json.loads((NETRUNS / r / "status.json").read_text())["status"] == "completed"] for k, runs in GEM.items()}
GEM_FAILED = [r for k, runs in GEM.items() for r in runs if r not in GEM_DONE[k]]  # died of Google-side timeouts: infrastructure failures
MC = json.loads((MAIN / "validation" / "model-compare-20261120.json").read_text())  # model-compare-20261119.py: matched points, finish reasons
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
for (lv, m), runs in GEM_DONE.items():
    for r in runs:
        NAMES[r] = (GEM_LEVELS[lv][0], f"{GEM_MODELS[m]}, run {1 if '20261119' in r else 2}")
for lv, r in OUT.items():
    NAMES[r] = (f"Tunnel from outside: {LEVEL_NAME[lv]}", "point grasp")
for (k, v), r in MSD.items():
    NAMES[r] = (f"1D mass, {v} {'spring' if k == 'spring' else 'spring + damper'}", "point grasp")
for (o, k), r in POINT.items():
    NAMES[r] = (f"{'Cluster' if o == 'tetra' else 'Cube'} on a spring, {'stiffness stated' if k == 'k' else 'block mass stated'}", "point grasp, no force reading")
for fam, (unlit, lit) in LIGHT.items():
    for r in unlit + lit:
        NAMES[r] = (LIGHT_NAME[fam], ("true lighting given" if r in lit else "lighting not given") + f", run {rep_n(r)}")
for lv, r in HAND.items():
    NAMES[r] = (f"Shadow hand: {HAND_NAME[lv]}", f"touch only, run {rep_n(r)}")
for lv, runs in CLOTH.items():
    for r in runs:
        NAMES.setdefault(r, (f"Cloth: {CLOTH_NAME[lv]}", f"run {rep_n(r)}"))
ORDER = list(NAMES)
VIEWER = {"spring-k": SPRING_RUNS[("k", "movable")] + SPRING_RUNS[("k", "fixed")],
          "spring-block": SPRING_RUNS[("block", "movable")] + SPRING_RUNS[("block", "fixed")],
          "box": list(BOX.values()), **({"hand": list(HAND.values())} if WITH_MEDIA else {})}
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

EVS = (WORK2 / "eval-videos", WORK3 / "eval-videos")  # render-eval-videos.py: side-by-side evaluation videos and cloth stills, one clips.json per run
RENDERED = (list(BOX.values()) + list(TUNNEL.values()) + list(HINTED.values()) + [CUT_SHORT] + LIGHT["spring-out"][0] + LIGHT["spring-out"][1]
            + [OUT["latch"]] + [r for runs in CLOTH.values() for r in runs]
            + [r for (lv, m), runs in GEM_DONE.items() if lv in ("w22-cloth-holes", "w19-marble-heavy-hint", "w20-latch-out") for r in runs
               if r != "w20-latch-out-neon-20261120-001"])  # that run never had a scored submission (no scene.xml), so there is nothing to replay
EV = {r: next(d for d in EVS if (d / r / "clips.json").is_file()) for r in RENDERED} if WITH_MEDIA else {}
CLIPS = {r: json.loads((EV[r] / r / "clips.json").read_text()) for r in RENDERED} if WITH_MEDIA else {}
FIGURES.update({f"clips/{c['file']}": str(EV[c["run"]] / c["file"]) for cs in CLIPS.values() for c in cs})
SCALE = {x["run"]: x for d in (WORK2, WORK3) for f in sorted((d / "spring-scale").glob("*.json")) for x in json.loads(f.read_text())} if WITH_SCALE else {}  # spring-scale-usage.py

# ---------------- metrics ----------------
SPRING_KEYS = ["r:mass:cluster", "r:mass:spring_base", "r:mass:spring_top", "r:stiffness"]
METRICS = {"r:mass:cluster": ("sphere cluster mass", "ratio"), "r:mass:spring_base": ("bottom plate mass", "ratio"), "r:mass:spring_top": ("top plate mass", "ratio"),
           "r:mass:block": ("block mass", "ratio"), "r:mass:plug": ("plug mass", "ratio"), "r:stiffness": ("spring stiffness", "ratio"),
           "mech_rel": ("mechanism force error (÷ true force)", "err"), "fp_push_n": ("slow-push force error (N)", "err"),
           "fp_retract_n": ("slow-retract force error (N)", "err"), "fp_fast_n": ("fast-cycle force error (N)", "err"),
           "traj_mm": ("composite COM path error, mean of 4 throws (mm)", "err"), "r:mass:cube": ("cube mass", "ratio"),
           "fp_rel": ("force-profile error (÷ true force)", "err"), "abs:damping": ("damping, true value 0 (N·s/m)", "err"), "hand_traj_mm": ("object COM path error, squeeze and tilt (mm)", "err"),
           "tactile_n": ("touch-sensor error (N)", "err"), "rest_chamfer_mm": ("cloth shape at rest, Chamfer (mm)", "err"),
           "drop_chamfer_mm": ("cloth shape during 5 drops, Chamfer (mm)", "err"), "stiff_rel": ("lift-force error over a 4×4 grid (÷ true)", "err"),
           **{f"traj:{p}_mm": (f"COM path error, {PROBE_NAME[p]} (mm)", "err") for p in PROBES}}


def estimates(run, source, m, default):
    """Spring levels: mass on each source body (every positive-mass candidate body goes to the nearest source body by COM, so a split model
    still sums correctly) and the stiffest slide joint. Tunnel and box levels: none (their evaluator metrics are in extra_metrics)."""
    names = [b["name"] for b in source["bodies"]]
    if "-msd-" in run:  # the block's slide joint: mass it carries, stiffness and damping (the three constants of the mechanism)
        slides = [j for j in m["joints"] if j["type"] == 2]
        assert slides, (run, "no slide joint")
        j = max(slides, key=lambda j: j["stiffness"])
        return {"mass": next(b["subtree_mass"] for b in m["bodies"] if b["name"] == j["body"]), "stiffness": j["stiffness"], "damping": j["damping"]}
    if run.startswith(("w16", "w17", "w18", "w19", "w21", "w22")) or "-out" in run:
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
    if hist["status"] == "failed":  # rejected by the evaluator: a gap, as the builder records it
        return {}
    if run.startswith(("w16", "w17")):
        if run not in _FP:
            _FP[run] = json.loads((NETRUNS / run / "force-profile-rescore.json").read_text())["versions"]
        assert len(_FP[run]) >= i + 1, run
        return _fp_errors(_FP[run][f"version-{i + 1:03d}"])
    if run.startswith(("w20", "w21", "w22")):
        ev = json.loads((NETRUNS / run / f"version-{i + 1:03d}" / "evaluation.json").read_text())
        assert ev["version"] == hist["version"], (run, i)
        sc = ev["scores"]
        if "force_profile" in sc:  # tunnels from outside, 1D spring-mass: whole-probe RMS error ÷ RMS true resistance
            fp = sc["force_profile"]
            return {"fp_rel": None if fp["status"] != "ok" else fp["rms_resistance_error_n"] / float(np.sqrt(np.mean(np.asarray(fp["source"]["resistance_n"]) ** 2)))}
        if "arm_probes" in sc:  # Shadow hand: squeeze and tilt probes
            probes = [p for p in sc["arm_probes"] if p["status"] == "ok"]
            assert len(probes) == len(sc["arm_probes"]) == 2, (run, i)
            return {"hand_traj_mm": 1000 * float(np.mean([p["object_com"]["trajectory_mean_error_m"] for p in probes])),
                    "r:mass": probes[0]["object_com"]["candidate_mass_kg"] / probes[0]["object_com"]["source_mass_kg"],
                    "tactile_n": float(np.mean([p["tactile_mae_n"] for p in probes]))}
        if "cloth" in sc:
            c = sc["cloth"]
            return {} if c["status"] != "ok" else {"rest_chamfer_mm": 1000 * c["rest_chamfer_m"], "drop_chamfer_mm": 1000 * c["drop_mean_chamfer_m"],
                                                   "stiff_rel": c["stiffness_map"]["lift_force_relative_error"]}
        return {}
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


META = dict(title="Hidden mechanisms, hidden contents, touch and cloth: sim2sim results, 6–20 November 2026",
            subtitle=L("Runs finished 2026-11-06 to 2026-11-20 · {n} runs (two are baselines repeated from the previous report) · each run is one "
                       "<code>gpt-6-astra</code> agent (medium reasoning; section 11 compares Gemini 4 Neon and Hydrogen) working alone in an isolated container, up to 4 h (spring and geometry levels) or 6 h "
                       "(tunnel, box and later levels). <b>Mostly one run per condition</b>; the spring-level camera comparison has three runs per condition, "
                       "the lighting comparison two."),
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
                "mass</td></tr>"
                "<tr><td>Tunnel from outside</td><td>The tunnel spring and latch, with the stick starting 240 mm outside in an open trough</td>"
                "<td>force profile</td></tr>"
                "<tr><td>1D mass on a spring</td><td>A 0.5 kg block on a frictionless slide, held by a 32 N/m spring, with or without a damper "
                "(0.8 N·s/m). Visible: a drawn coil and dashpot. Hidden: a closed stage, identical whatever is inside</td><td>force profile</td></tr>"
                "<tr><td>Object on a spring, point grasp</td><td>The spring level of the first row without the Franka, with the sphere cluster or a "
                "48 g cube, an ideal spring, and no force reading; stiffness or block mass stated</td><td>shape, COM, each part's mass, stiffness</td></tr>"
                "<tr><td>Shadow hand</td><td>A dexterous hand, palm up, holding an unseen object (prism, prism with off-centre COM, smooth convex "
                "shape, two cones). No camera: joint readings and touch grids only</td><td>shape, COM, mass, squeeze and tilt probes</td></tr>"
                "<tr><td>Cloth</td><td>A square cloth whose bending stiffness is uniform, or differs by quadrant (plain, coloured, misleadingly "
                "coloured), or around holes, cuts or a centre hole with a stiff or soft rim</td><td>lift force at 16 points, five drops, shape at "
                "rest</td></tr></table></div>"
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
                "<tr><td>true lighting given</td><td>The task includes the hidden scene's exact light elements, so the agent need not fit lighting "
                "to match images</td></tr>"
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
    media = lambda make: make() if WITH_MEDIA else []
    scale_on = lambda make: make() if WITH_SCALE else []
    clip_items = lambda runs, fam=None: [dict(src=f"clips/{c['file']}", label=f"{' · '.join(NAMES[c['run']])} · {c['clip']}", run=c["run"])
                                         for r in runs for c in CLIPS[r] if fam is None or c["family"] == fam]
    mn = lambda t: "never" if t is None else f"min {t / 60:.0f}"

    def scale_rows(runs):
        """Did the agent seat the spring in the hole and weigh things on it (in the hidden scene), and in what order?"""
        rows = []
        for r in runs:
            sc, fw = SCALE[r], SCALE[r]["first_wall_s"]
            obj = "cluster" if "cluster" in sc["targets"] else "cube"
            block, ob = (fw.get("block_weighed_upright") if "block" in sc["targets"] else None), fw.get(f"{obj}_weighed_upright")
            order = "—" if "block" not in sc["targets"] else ("block first: calibrated k, then weighed the object" if block is not None and ob is not None and block < ob
                                                           else "object first, then the block" if block is not None and ob is not None
                                                           else "weighed the object only" if ob is not None else "weighed the block only" if block is not None else "neither")
            note = next((n for n in sc["notes"] if "hole" in n["text"].lower()), None) or next(iter(sc["notes"]), None)
            rows.append([lnk(r, " · ".join(NAMES[r])), hl(mn(fw.get("spring_in_hole")), fw.get("spring_in_hole") is not None),
                         hl(mn(fw.get(f"{obj}_weighed")), fw.get(f"{obj}_weighed") is not None), hl(mn(ob), ob is not None),
                         mn(block) if "block" in sc["targets"] else "—", order,
                         x(f(r, f"r:mass:{obj}")), f"<span class='small'>min {note['min']}: “{note['text'][:220]}{'…' if len(note['text']) > 220 else ''}”</span>" if note else "—"])
        return rows
    scale_head = ["run", "spring put in the hole", "object weighed on it in the hole", "object on the upright spring (anywhere)",
                  "block on the upright spring", "order (block levels)", "final object mass ÷ true", "first note about the hole or weighing"]
    upright = lambda runs: sum(any(k.endswith("_weighed_upright") and not k.startswith("block") for k in SCALE[r]["first_wall_s"]) for r in runs)
    seated = lambda runs: sum(SCALE[r]["first_wall_s"].get("spring_in_hole") is not None for r in runs)
    weighed = lambda runs: sum(any(k.endswith("_weighed") and not k.startswith("block") for k in SCALE[r]["first_wall_s"]) for r in runs)
    fmt_k = lambda k, v: "—" if v is None else x(v) if k.startswith("r:") else mm(v) if k.endswith("_mm") else pct(v) if k.endswith("_rel") else f"{v:.3g}"
    first_le = lambda r, k, v: next((u["t"] for u in R[r]["subs"] if u["m"].get(k) is not None and u["m"][k] <= v), None)
    minute = lambda t: "never" if t is None else f"min {t:.0f}"
    med = lambda vals: float(np.median([v for v in vals if v is not None]))
    sec = []
    FRANKA = [r for runs in SPRING_RUNS.values() for r in runs] + list(SPRING_VARIANTS.values())

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
                blk(type="table", head=["level", "Chamfer", "COM error", "cluster mass", "plate masses", "plug mass"], rows=var_rows),
                *scale_on(lambda: [cap("Table", L("<b>Did the agent use the spring as a scale?</b> The level is built for it: seat the spring in the hole (its top plate "
                               "then sits flush with the table), set the cluster on it, and read the compression, m = k·x/g; with the block mass stated "
                               "instead of k, weigh the block first to get k. From every 10th recorded state of the hidden scene: when the spring first "
                               f"sat in the hole, when the cluster first rested on it there, and when it first rested on the spring standing upright "
                               f"anywhere. {seated(FRANKA)} of {len(FRANKA)} Franka runs seated the spring in the hole, {weighed(FRANKA)} weighed the "
                               f"cluster on it there and {upright(FRANKA)} set the cluster on the upright spring anywhere.")),
                blk(type="table", head=scale_head, rows=scale_rows(FRANKA))])]))
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
                *media(lambda: [cap("Figure", L("<b>The scored force probe on each final model.</b> Left: hidden scene; right: the agent's model, zoomed on the stick. "
                                "Yellow arrow: the force the probe's handle applies at the grasp point (4 mm per N). Below: resistance against the push "
                                "along the tunnel, filling in as the probe runs (black: truth; orange: model).")),
                blk(type="clips", title=None, items=clip_items(list(TUNNEL.values()) + list(HINTED.values()), "tunnel"))]),
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
    sec.append(dict(id="box", nav="Sealed box", tag="one run per condition",
        title="4 · Contents sealed in a box: inferred only from how the load shifts, and only when the contents slide or the task says they exist",
        q=None, answer=L("Without a hint only the cylinder agent found the contents (a sliding body, from load-centre shifts in tilted grasps). Both ball agents "
                         "fitted contact and friction around a rigid box instead, and their models move worse than a plain solid box. With the hint, both "
                         "ball agents modeled a free sphere of the right mass inside a hollow shell, from loads alone (the camera can no longer see "
                         "inside), and beat the solid box on most throws."),
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
                *media(lambda: [cap("Figure", L("<b>The four scripted throws, hidden scene (left) against the agent's final model (right)</b>, re-simulated exactly as "
                                "scored, with the box walls drawn see-through so the contents show. The camera follows the hidden scene's box, and "
                                "the model is filmed along the same camera path.")),
                blk(type="clips", title=None, items=clip_items(BOX.values(), "box"))]),
                cap("Figure", "<b>Box reconstructions in 3D</b>: the hidden body is visible here (solid = truth, wireframe = reconstruction)."),
                blk(type="viewer", group="box", title="Sealed boxes")]))

    # ---------------- 5-10 · levels of 2026-11-16 ----------------
    at = lambda r, k, t=120: next((u["m"].get(k) for u in reversed(R[r]["subs"]) if u["t"] <= t and u["m"].get(k) is not None), None)
    exact_at = lambda r, k, tol: minute(first_le(r, k, tol))
    app = lambda r: R[r]["taxonomy"]["appearance"] / max(1, R[r]["taxonomy"]["notes"])
    fpcell = lambda r: hl(pct(f(r, "fp_rel")), f(r, "fp_rel") is not None and f(r, "fp_rel") < 0.05)
    out_rows = [[lnk(r, NAMES[r][0]), fpcell(r), exact_at(r, "fp_rel", 0.01), str(R[r]["submissions"]), f"{R[r]['wall_min']:.0f}"] for r in OUT.values()]
    sec.append(dict(id="tunnel-out", nav="Tunnel from outside", tag="one run per level",
        title="5 · Tunnels with the stick starting outside: the mechanism is found and modeled exactly",
        q=None, answer=L(f"With the stick lying outside the tunnel, so that any agent must insert it, the spring was modeled to "
                         f"{pct(f(OUT['spring'], 'fp_rel'))} force error and the latch to {pct(f(OUT['latch'], 'fp_rel'))}, without any hint."),
        blocks=[lead(L("Same mechanisms as section 3 (spring, push-push latch), but the stick starts 240 mm outside, in an open trough. Inserting it "
                       "is the obvious first experiment, so every run reaches the plunger. Point grasp with force reading, fixed camera, sequence tool.")),
                cap("Table", L("<b>Final force-profile error</b> (RMS difference over the evaluator's whole insertion and push/retract probe, ÷ RMS true "
                               "resistance; green = under 5%), and the first submission within 1%.")),
                blk(type="table", head=["level", "force error", "within 1% at", "submissions", "minutes"], rows=out_rows),
                *media(lambda: [cap("Figure", "<b>The scored probe</b> (insertion, then push and pull strokes) on each final model, as in section 3."),
                blk(type="clips", title=None, items=clip_items(LIGHT["spring-out"][0] + LIGHT["spring-out"][1] + [OUT["latch"]], "tunnel"))]),
                conv([[r, NAMES[r][0]] for r in OUT.values()], ["fp_rel", "chamfer_mm"], title="Time course", xmax=360)]))

    msd_rows = [[lnk(r, NAMES[r][0]), fpcell(r), exact_at(r, "fp_rel", 0.01), mm(f(r, "chamfer_mm")), f"{R[r]['wall_min']:.0f}"] for r in MSD.values()]
    sec.append(dict(id="msd", nav="1D spring-mass", tag="one run per level",
        title="6 · A mass on a spring, visible or hidden: identified exactly either way",
        q=None, answer=L("A 0.5 kg block on an ideal spring (32 N/m), with or without a damper, was identified to "
                         + ", ".join(f"{pct(f(r, 'fp_rel'))}" for r in MSD.values()) + " force error (visible spring, hidden spring, visible "
                         "damper, hidden damper). Hiding the mechanism inside a closed stage made no difference."),
        blocks=[lead(L("The visible levels show a coil (and a dashpot); the hidden levels are a closed linear stage, pixel-identical with and without the "
                       "damper, so only pulling and releasing the block tells them apart. Point grasp with force reading, movable camera.")),
                cap("Table", L("<b>Final force-profile error</b> (slow and fast pull cycles; green = under 5%) and the first submission within 1%.")),
                blk(type="table", head=["level", "force error", "within 1% at", "Chamfer", "minutes"], rows=msd_rows),
                cap("Table", L("<b>The three constants at every submission</b> (mass the slide joint carries, its stiffness and its damping, read from "
                               "the compiled submission; green = exact). True values: 0.5 kg, 32 N/m, and 0.8 N·s/m with the damper (0 without). Every "
                               "final model is one body on one slide joint with exactly these constants.")),
                blk(type="table", head=["level", "submission", "minute", "mass (kg)", "stiffness (N/m)", "damping (N·s/m)"],
                    rows=[[lnk(r, NAMES[r][0]) if i == 0 else "", str(i + 1), f"{u['t']:.0f}"] +
                          [hl(f"{u['est'][k]:.4g}", abs(u['est'][k] - R[r]['truth'][k]) < 1e-6) if u["est"] else "rejected" for k in ("mass", "stiffness", "damping")]
                          for r in MSD.values() for i, u in enumerate(R[r]["subs"])]),
                conv([[r, NAMES[r][0]] for r in MSD.values()], ["fp_rel", "r:mass", "r:stiffness", "r:damping", "abs:damping"], title="Time course", xmax=360)]))

    PG = list(POINT.values()) + [r for fam in ("tetra", "cube") for runs in LIGHT[fam] for r in runs if r not in POINT.values()]
    p_keys = lambda r: ["r:mass:cluster" if "tetra" in r else "r:mass:cube", "r:mass:spring_base", "r:mass:spring_top"]
    p_rows = [[lnk(r, NAMES[r][0]), mm(f(r, "chamfer_mm")), mm(f(r, "com_mm"))] + [hl(x(f(r, k)), within(r, k)) for k in p_keys(r)] +
              [hl(x(f(r, "r:stiffness")), within(r, "r:stiffness")) if "block" in r else "stated"] for r in POINT.values()]
    sec.append(dict(id="point-spring", nav="Spring, point grasp", tag="one run per level (two for the stiffness-stated levels, section 8)",
        title=f"7 · An object on a spring without force sensing: masses within 5%, but the spring was used as a scale in the hole in only {weighed(PG)} of {len(PG)} runs",
        q=None, answer=L("Without a force reading, the stated stiffness (or block mass) is the only scale. Final object mass ÷ true: " +
                         ", ".join(f"{NAMES[r][0].lower()} {x(f(r, p_keys(r)[0]))}" for r in POINT.values()) + "."),
        blocks=[lead(L("The spring levels of section 1 without the Franka: the agent grasps points directly but reads no force. The spring is ideal "
                       "(no damping). Weighing an object means setting it on the spring and measuring the compression in the camera or depth image.")),
                cap("Table", L("<b>Final submissions</b> (green = within 5%). Masses are estimate ÷ true.")),
                blk(type="table", head=["level", "Chamfer", "COM error", "object mass", "bottom plate", "top plate", "stiffness"], rows=p_rows),
                conv([[r, NAMES[r][0]] for r in POINT.values()], ["chamfer_mm", "r:mass:cluster", "r:mass:cube"], title="Time course", xmax=360),
                *scale_on(lambda: [cap("Table", L("<b>Did the agent use the spring as a scale?</b> As the table in section 1, for every point-grasp run (with and without the true "
                               f"lighting, both runs). {seated(PG)} of {len(PG)} runs seated the spring in the hole, {weighed(PG)} weighed the object on it "
                               f"there and {upright(PG)} set the object on the upright spring anywhere. On the block levels the order column says whether "
                               "the agent first weighed the known block to establish k and then the unknown object.")),
                blk(type="table", head=scale_head, rows=scale_rows(PG))])]))

    key = {"spring-out": "fp_rel", "tetra": "r:mass:cluster", "cube": "r:mass:cube", "msd": "fp_rel", "cloth": "stiff_rel"}
    lo_hi = lambda runs: f"{100 * min(app(r) for r in runs):.0f}–{100 * max(app(r) for r in runs):.0f}%"
    assert all(max(app(r) for r in lit) <= min(app(r) for r in unlit) + 0.01 for unlit, lit in LIGHT.values())  # the answer's claim
    l_rows = []
    for fam, (unlit, lit) in LIGHT.items():
        k = key[fam]
        for lab, runs in (("not given", unlit), ("given", lit)):
            l_rows.append([LIGHT_NAME[fam] if lab == "not given" else "", f"lighting {lab}", METRICS[k][0],
                           " / ".join(lnk(r, fmt_k(k, f(r, k))) for r in runs), " / ".join(fmt_k(k, at(r, k)) for r in runs),
                           " / ".join(f"{100 * app(r):.0f}%" for r in runs), " / ".join(f"{R[r]['wall_min']:.0f}" for r in runs)])
    sec.append(dict(id="lighting", nav="Lighting", tag="two runs per condition",
        title="8 · Giving the agent the true lighting: less appearance work, same physics",
        q=None, answer=L("In every one of the five levels, both runs given the true lighting wrote a smaller share of progress notes about appearance "
                         "and lighting than both runs without it (" + "; ".join(f"{LIGHT_NAME[fam]}: {lo_hi(lit)} vs {lo_hi(unlit)}"
                                                                               for fam, (unlit, lit) in LIGHT.items()) + "). The final physical scores "
                         "were the same within the run-to-run range. One tunnel replicate without lighting had not found the mechanism when it "
                         "stopped at its cost limit (2.5 h); with two runs per condition that is not evidence of an effect."),
        blocks=[lead(L("Agents spend much of a run matching shading and shadows, which is not physics. In the lit twins the task gives the exact light "
                       "elements and headlight of the hidden scene. First runs were extended to about 6 h; the replicates stopped at their cost "
                       "limit (2–4 h), so the 120-minute column is the fairer comparison.")),
                cap("Table", L("<b>Key physical score with and without the true lighting</b> (run 1 / run 2): final submission, at 120 minutes, share of "
                               "progress notes about appearance or lighting, and minutes used.")),
                blk(type="table", head=["level", "condition", "score", "final", "at 120 min", "appearance notes", "minutes"], rows=l_rows)] +
               [conv([[unlit, "lighting not given"], [lit, "lighting given"]], [key[fam], "drop_chamfer_mm" if fam == "cloth" else "chamfer_mm"],
                     title=LIGHT_NAME[fam], xmax=360) for fam, (unlit, lit) in LIGHT.items()]))

    h_rows = [[lnk(r, " · ".join(NAMES[r])), mm(f(r, "chamfer_mm")), mm(f(r, "com_mm")), x(f(r, "r:mass")), mm(f(r, "hand_traj_mm")), fmt_k("tactile_n", f(r, "tactile_n"))]
              for r in HAND.values()]
    sec.append(dict(id="hand", nav="Shadow hand", tag="one run per object (two for the cones)",
        title="9 · A dexterous hand with touch only: shape within millimetres and mass within 13%, from contact alone",
        q=None, answer=L(f"Without any image, agents recovered the held object's shape to {mm(min(f(r, 'chamfer_mm') for r in HAND.values()))}–"
                         f"{mm(max(f(r, 'chamfer_mm') for r in HAND.values()))} Chamfer and its mass to "
                         f"{x(min(f(r, 'r:mass') for r in HAND.values()))}–{x(max(f(r, 'r:mass') for r in HAND.values()))}. The two runs on the "
                         f"two-cone object differed widely (Chamfer {mm(f(HAND['spiky'], 'chamfer_mm'))} vs {mm(f(HAND['spiky r2'], 'chamfer_mm'))}), so "
                         "single runs here are trends. There is no reference model yet for the probe path error."),
        blocks=[lead(L("A Shadow Dexterous Hand, palm up, holds an object it cannot see. The agent sets 20 actuator targets and reads joint angles, "
                       "velocities, efforts and fingertip and palm touch grids; no camera. Scored on shape, COM, and two scripted probes (squeeze, tilt).")),
                cap("Table", "<b>Final submissions.</b> Path error and touch-sensor error are the evaluator's squeeze and tilt probes, averaged."),
                blk(type="table", head=["object", "Chamfer", "COM error", "mass ÷ true", "object path error", "touch-sensor error (N)"], rows=h_rows),
                conv([[r, " · ".join(NAMES[r])] for r in HAND.values()], ["chamfer_mm", "com_mm", "r:mass"], title="Time course", xmax=360),
                *media(lambda: [cap("Figure", L("<b>Each held object in 3D</b>: true shape (solid) and the agent's reconstruction (wireframe), the hand left out; pick a "
                                "run and drag the slider through its submissions.")),
                blk(type="viewer", group="hand", title="Shadow hand objects")])]))

    c_rows = [[lnk(r, NAMES[r][0]) if rep_n(r) == 1 else "", lnk(r, f"run {rep_n(r)}"), mm(f(r, "rest_chamfer_mm")), mm(f(r, "drop_chamfer_mm")),
               hl(pct(f(r, "stiff_rel")), f(r, "stiff_rel") is not None and f(r, "stiff_rel") < 0.05), f"{R[r]['wall_min']:.0f}"]
              for runs in CLOTH.values() for r in runs]
    sec.append(dict(id="cloth", nav="Cloth", tag="one or two runs per level",
        title="10 · Cloth with regions of different stiffness: stiffness maps recovered, misleading colours did not mislead",
        q=None, answer=L(f"The lift force at 16 points was within 5% in {sum(f(r, 'stiff_rel') < 0.05 for runs in CLOTH.values() for r in runs)} of "
                         f"{sum(len(runs) for runs in CLOTH.values())} runs, including both runs whose quadrant colours were placed on the diagonals to "
                         "mislead. The hardest level was holes and cuts: shape during drops "
                         f"{mm(f(CLOTH['holes'][0], 'drop_chamfer_mm'))}, against about 2 mm elsewhere."),
        blocks=[lead(L("A square cloth (a MuJoCo 2D flex) lies on the table. Levels differ in where the cloth is stiff: uniform; four quadrants of "
                       "very different bending stiffness; the same with each quadrant coloured, or coloured misleadingly along the diagonals; holes "
                       "and cuts; a centre hole with a stiff or soft rim. Point grasp with force reading, RGB + depth, movable camera. Scored by "
                       "lifting the cloth at 16 grid points, by five drops onto a bar or cone, and by shape at rest.")),
                cap("Table", L("<b>Final submissions</b> (green = lift force within 5%).")),
                blk(type="table", head=["level", "run", "shape at rest", "shape in drops", "lift-force error", "minutes"], rows=c_rows),
                *media(lambda: [cap("Figure", L("<b>Cloth levels, truth against the agent's final model.</b> Per level: the trusted-camera image at rest; the "
                                "bending-stiffness field (each cloth region coloured by the median magnitude of MuJoCo's compiled bending "
                                "coefficients, one log scale for both; relative units); the evaluator's lift force at 16 grid points (truth, model, "
                                "difference); and the evaluator's drops onto a bar, side by side."))])] +
               media(lambda: [b for lv, runs in CLOTH.items() for b in (
                   blk(type="html", html=f"<h3>{CLOTH_NAME[lv].capitalize()}</h3>"),
                   blk(type="images", items=[dict(src=f"clips/{c['file']}", label=f"{c['clip']}: truth (left) vs model (right), run 1", run=runs[0])
                                             for c in CLIPS[runs[0]] if c["family"] == "cloth-still"]),
                   blk(type="clips", title=None, items=clip_items(runs, "cloth")))]) + [
                conv([[runs[0], CLOTH_NAME[lv]] for lv, runs in CLOTH.items()], ["stiff_rel", "drop_chamfer_mm"], title="Time course (first runs)", xmax=360)]))

    # ---------------- 11 · Gemini vs astra ----------------
    effort = {r: json.loads((NETRUNS / r / "config-private.json").read_text())["inference"].get("reasoning_effort")
              for lv, (_, astra, _) in GEM_LEVELS.items() for r in astra + [x for m in GEM_MODELS for x in GEM_DONE[(lv, m)]]}
    assert set(effort.values()) == {"medium"}, effort  # same reasoning level for every compared run (Gemini: thinkingLevel = medium)
    point = lambda r, k, cut: next((u["m"].get(k) for u in reversed(R[r]["subs"]) if cut(u) and u["m"].get(k) is not None), None)
    avg = lambda vals: None if not [v for v in vals if v is not None] else float(np.mean([v for v in vals if v is not None]))
    by_model = lambda lv: [("gpt-6-astra", GEM_LEVELS[lv][1])] + [(GEM_MODELS[m], GEM_DONE[(lv, m)]) for m in GEM_MODELS]
    g_rows, ratios = [], {m: [] for m in GEM_MODELS}
    for lv, (scene, astra, key) in GEM_LEVELS.items():
        tmin, tmtok = MC[lv]["matched_wall_min"], MC[lv]["matched_mtok"]
        a = avg([f(r, key) for r in astra])
        for name, runs in by_model(lv):
            fin = [R[r]["subs"][-1]["m"].get(key) for r in runs]
            row = [f"{scene}<div class='muted small'>{GEM_LABEL[key]}</div>" if name == "gpt-6-astra" else "", name,
                   " / ".join(lnk(r, fmt_k(key, v)) for r, v in zip(runs, fin)) or "—",
                   " / ".join(fmt_k(key, point(r, key, lambda u: u["t"] <= tmin)) for r in runs) or "—",
                   " / ".join(fmt_k(key, point(r, key, lambda u: u["mtok"] <= tmtok)) for r in runs) or "—"]
            if name != "gpt-6-astra":
                m = next(k for k, v in GEM_MODELS.items() if v == name)
                g = avg(fin)
                ratios[m].append(None if g is None or a is None else g / a)
                row.append(hl(f"×{g / a:.2g}", g < a) if g is not None and a else "—")
            else:
                row.append("")
            g_rows.append(row)
    geo = {m: float(np.exp(np.mean(np.log([x for x in v if x is not None])))) for m, v in ratios.items()}
    wins = {m: sum(x is not None and x < 1 for x in v) for m, v in ratios.items()}
    q_rows = []
    for name, runs in (("gpt-6-astra", sorted({r for _, astra, _ in GEM_LEVELS.values() for r in astra})),) + tuple(
            (GEM_MODELS[m], [r for lv in GEM_LEVELS for r in GEM_DONE[(lv, m)]]) for m in GEM_MODELS):
        mc = [x for lv in GEM_LEVELS for x in MC[lv]["runs"] if x["run"] in runs]
        assert len(mc) == len(runs), (name, len(mc), len(runs))
        calls = sum(sum(x["finish_reasons"].values()) for x in mc)
        count = lambda *ks: sum(x["finish_reasons"].get(k, 0) for x in mc for k in ks)
        q_rows.append([name, str(len(runs)), str(calls), f"{100 * count('MALFORMED_FUNCTION_CALL', 'MALFORMED_RESPONSE') / calls:.1f}%",
                       f"{100 * count('MAX_TOKENS') / calls:.1f}%", str(sum(x["call_timeouts"] for x in mc)),
                       f"{sum(x['provider_held_min'] for x in mc):.0f}", f"{60 * calls / sum(x['wall_min'] for x in mc):.0f}",
                       f"{np.mean([x['final_mtok'] for x in mc]):.1f}"])
    gem_w11 = [r for m in GEM_MODELS for r in GEM_DONE[("w11-tetra-spring-k", m)]]
    sec.append(dict(id="gemini", nav="Gemini vs astra", tag="one or two runs per model per level",
        title=f"11 · Gemini 4 Neon and Hydrogen against gpt-6-astra: {'worse' if min(geo.values()) > 1 else 'better' if max(geo.values()) < 1 else 'mixed'} on average "
              f"(final error Neon ×{geo['neon']:.2g}, Hydrogen ×{geo['hydrogen']:.2g} astra's)",
        q=None, answer=L(f"On five hard levels, the average final error of Gemini 4 Neon was ×{geo['neon']:.2g} astra's (geometric mean over levels) and Gemini 4 "
                         f"Hydrogen's ×{geo['hydrogen']:.2g}; Neon was better than astra on {wins['neon']} of {len(GEM_LEVELS)} levels and Hydrogen on "
                         f"{wins['hydrogen']}. Both Gemini models did better on: {', '.join(GEM_LEVELS[lv][0].lower() for i, lv in enumerate(GEM_LEVELS) if all(ratios[m][i] is not None and ratios[m][i] < 1 for m in GEM_MODELS)) or 'none'}; "
                         f"astra did better on: {', '.join(GEM_LEVELS[lv][0].lower() for i, lv in enumerate(GEM_LEVELS) if all(ratios[m][i] is not None and ratios[m][i] > 1 for m in GEM_MODELS)) or 'none'}. "
                         "Gemini also made fewer than half astra's calls per hour, and 4–8% of its replies were malformed tool calls (Table below). Every compared run used the same medium reasoning level (astra <code>reasoning_effort</code>, Gemini "
                         "<code>thinkingLevel</code>), the same harness, task, tools, image and wall budget. With one or two runs per model, differences "
                         "smaller than astra's own run-to-run spread (up to ×4 on the marble box) are noise."),
        blocks=[lead(L("Each Gemini config equals its astra original except the model route and the cost lines; Google's API was reached directly. Set 1 "
                       "ran without protection against Google-side outages, and five of its ten runs died of call timeouts (infrastructure failures, "
                       "not results: " + ", ".join(f"<code>{r}</code>" for r in GEM_FAILED) + "). Set 2 held outages outside the run clock and all "
                       "ten completed. Final errors use each run's last scored submission; matched points use the comparison's equal run-clock "
                       "minutes and equal cumulative tokens (each provider's own token count).")),
                cap("Table", L("<b>Primary error per level</b> (lower is better; run 1 / run 2 for each model). Last column: the model's mean final error ÷ "
                               "astra's mean (green = better than astra). At equal minutes Gemini had made fewer calls, so the equal-token column is "
                               "the fairer one for effort.")),
                blk(type="table", head=["level", "model", "final", "at equal minutes", "at equal tokens", "mean ÷ astra"], rows=g_rows),
                cap("Table", L("<b>How the models used their calls</b> (completed runs). Malformed = replies with a malformed tool call or response; output "
                               "limit = replies that hit the 16,384-token output cap (Gemini counts thinking towards it); held = minutes of Google-side "
                               "outage held outside the run clock.")),
                blk(type="table", head=["model", "runs", "model calls", "malformed", "hit output limit", "call timeouts", "outage held (min)",
                                        "calls per hour", "tokens per run (M)"], rows=q_rows),
                *[conv([[GEM_LEVELS[lv][1] if len(GEM_LEVELS[lv][1]) > 1 else GEM_LEVELS[lv][1][0], "gpt-6-astra"]] +
                       [[r, NAMES[r][1], {"color": c, "dash": "6 3" if m == "hydrogen" else ""}] for m, c in (("neon", "--abl"), ("hydrogen", "--third"))
                        for r in GEM_DONE[(lv, m)]], [GEM_LEVELS[lv][2]], title=GEM_LEVELS[lv][0], xmax=360) for lv in GEM_LEVELS],
                *scale_on(lambda: [cap("Table", L("<b>Spring weighing by the Gemini runs</b>, as in the table of section 1.")),
                                   blk(type="table", head=scale_head, rows=scale_rows(gem_w11))]),
                *media(lambda: [cap("Figure", "<b>The scored tests on the Gemini runs' final models</b> (box throws, latch probe, cloth drops), as in sections 4, 5 and 10."),
                                blk(type="clips", title=None, items=clip_items([r for r in RENDERED if r in NAMES and ("-neon-" in r or "-hydrogen-" in r)]))])]))

    # ---------------- 12 · strategies ----------------
    sec.append(dict(id="strategy", nav="Strategies", tag=None, title="12 · How the agents work",
        q=None, answer=None, blocks=[lead("Every bash command and progress note, classified with the keyword rules of the previous report."), blk(type="strategy")]))

    # ---------------- appendix ----------------
    sec.append(dict(id="invalid", nav="A1", appendix=True, tag=None, title="A1 · Invalid runs and the fixes they prompted", q=None, answer=None,
        blocks=[lead(L("Three hinted box runs were invalid and are not in this report. In the first two (2026-11-14) the agents parked the movable camera "
                       "against the box, and its near clip plane cut through the 4 mm wall: \"The cavity view reveals a spherical internal weight\". The "
                       "camera now refuses positions closer than its near-plane reach to any surface. The relaunched light-ball run then carried the box "
                       "until the camera sat inside the cavity; now any surface inside the camera's clearance blacks out the image. Its third attempt "
                       "is the one reported. No unhinted box run did either.")),
                lead(L("Twelve replicates launched on 2026-11-16 stopped producing events a few minutes in, on nodes running three strict jobs at "
                       "once, and never finished; they are not reported (" + ", ".join(f"<code>{r}</code>" for r in LOST) + "). Several were "
                       "relaunched on 2026-11-17/18; three of those were still running when this report was built (a box and two hand replicates). "
                       "<code>w20-msd-spring-visible-20261116-001</code> was invalid (the controller hung after a code merge) and was relaunched as -002.")),
                lead(L(f"The hinted nonlinear tunnel run {lnk(CUT_SHORT, 'with the point grasp')} paused at its cost limit and was never reviewed "
                       f"(orchestrator error), so it stopped at {R[CUT_SHORT]['wall_min']:.0f} min; its rerun is the one in the tunnel table (section 3)."))]))
    sec.append(dict(id="effort", nav="A2", appendix=True, tag=None, title="A2 · Time and cost", q=None, answer=None,
        blocks=[blk(type="bars", title="Wall clock used (minutes)", unit="min", rows=[dict(run=r, label=" · ".join(NAMES[r]), v=R[r]["wall_min"],
                                                                                         color="--abl" if R[r]["reason"] == "budget_suspended" else "--base") for r in ORDER]),
                blk(type="bars", title="API cost (USD, gateway estimate)", unit="$", rows=[dict(run=r, label=" · ".join(NAMES[r]), v=R[r]["cost_usd"], color="--faint") for r in ORDER])]))
    sec.append(dict(id="runs", nav="A3", appendix=True, tag=None, title="A3 · All runs", q=None, answer=None,
                    blocks=[blk(type="html", html="<p>Click a name for its page with the full conversation and the images the model received.</p>"), blk(type="runs")]))
    return sec
