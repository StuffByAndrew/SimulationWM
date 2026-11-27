"""Report spec: every strict run completed since the 2026-11-24 reports (batch 2026-11-23 to 27), organized by question, plus the ablation
figures made since (grasp vs arm, resets, touch harness) and the data-integrity studies (jointed-body wrench scan, Inkling tool-call loop).
Baselines from earlier reports are included as comparison runs. Every number in the text is computed here from recorded data.
Build: .venv/bin/python experiments/validation/build-results-report.py data experiments/validation/report-spec-20261127-batch.py \\
           experiments/reports/report-work-20261127-batch experiments/reports/results-batch-20261127
       then `text` and `zip`. Run checkpoints-baselines.py-equivalent step first (WORK/checkpoints-baselines.json, see BASE_CHECKPOINTS)."""
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

VAL = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("gemini_spec", VAL / "report-spec-20261123-gemini.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
B = G.B
WORK = B.MAIN / "reports" / "report-work-20261127-batch"
REPORTS = B.MAIN / "reports"
L = lambda s: s

# ---------------- runs ----------------
HARD = list(G.GEM_LEVELS)  # the five hard levels
CARBON = {lv: [f"{lv}-carbon-{d}-001"] for lv, d in [("w22-cloth-holes", "20261123"), ("w11-tetra-spring-k", "20261125"), ("w19-marble-heavy-hint", "20261125"),
                                                    ("w20-latch-out", "20261125"), ("w21-hand-smooth-mesh", "20261125")]}
INKLING = {lv: f"{lv}-inkling-20261126-001" for lv in HARD}
MODELS_ORDER = [*G.MODEL_ORDER, "Gemini 4 Carbon"]
HARD_RUNS = {(lv, m): G.RUNS[(lv, m)] for lv in HARD for m in G.MODEL_ORDER} | {(lv, "Gemini 4 Carbon"): CARBON[lv] for lv in HARD}
FIXTURE = {"rigid": "w23-fixture-rigid-neon-20261125-002", "rubber": "w23-fixture-rubber-neon-20261125-002", "bimodal": "w23-fixture-bimodal-neon-20261125-001"}
DIE = {(arm, fps): f"w24-die-{arm}-neon{'-fps100' if fps == 100 else ''}-20261127-001" for arm in ("grasp", "franka") for fps in (25, 100)}
HAND_LEVELS = ["prism", "prism-offset-com", "smooth-mesh", "spiky"]
HARNESS = {(lv, c): f"w25-hand-{lv}{'' if c == 'harness' else '-control'}-20261127-001" for lv in HAND_LEVELS for c in ("control", "harness")}
LOWRANK = {"low-rank answer": "w26-cube-spring-block-lowrank-20261127-001", "full MJCF": "w26-cube-spring-block-mjcf-20261127-001"}
TABLETOP_LEVELS = ["shapes", "objects-unstated", "objects-stated", "bin-spiky"]
TABLETOP = {(lv, t): f"w27-{lv}{'-touch' if t else ''}-20261127-001" for lv in TABLETOP_LEVELS for t in (False, True)}
GEAR = {task: f"w28-gear-{task}-20261126-002" for task in ("recon", "cheap", "deep")}
PROMPT_LEVELS = {"tetra-spring": ("Cluster on a spring, stiffness stated", "r:mass:cluster"), "tetra-spring-block": ("Cluster on a spring, block mass stated", "r:mass:cluster"),
                 "cube-spring": ("Cube on a spring, stiffness stated", "r:mass:cube"), "cube-spring-block": ("Cube on a spring, block mass stated", "r:mass:cube")}
PROMPT = {lv: {"low-rank prompt": [f"w29-{lv}-lowrank-prompt-{'20261127-002' if lv.startswith('tetra') else '20261127-001'}"],
               "Neon, original prompt": [f"w29-{lv}-neon-20261127-001" if lv != "cube-spring-block" else LOWRANK["full MJCF"]],
               "gpt-6-astra, original prompt": [B.POINT[("tetra" if lv.startswith("tetra") else "cube", "block" if lv.endswith("block") else "k")]]}
          for lv in PROMPT_LEVELS}
PROMPT_W11 = {"low-rank prompt": ["w29-w11-tetra-spring-k-lowrank-prompt-20261127-001"], "Neon, original prompt": ["w11-tetra-spring-k-neon-20261120-001"]}
NORESET = {"w20-latch-out": "w31-latch-out-noreset-20261125-003", "w21-hand-smooth-mesh": "w31-hand-smooth-mesh-noreset-20261125-002",
           "w22-cloth-holes": "w31-cloth-holes-noreset-20261125-003", "w11-tetra-spring-k": "w31-tetra-spring-k-noreset-20261125-002",
           "w19-marble-heavy-hint": "w31-marble-heavy-hint-noreset-20261125-003", "w26-cube-spring-block-mjcf": "w31-cube-spring-block-mjcf-noreset-20261125-002"}
BOTTLE = {(c, g): f"w32-bottle-{c}-{g}grasp-20261127-001" for c in ("empty", "contents") for g in (1, 2)}
NO_SCENE = set(FIXTURE.values()) | set(DIE.values()) | {LOWRANK["low-rank answer"]}  # predictor programs and the parameter answer

NAMES = {}
for (lv, m), runs in HARD_RUNS.items():
    for i, r in enumerate(runs):
        NAMES[r] = (G.GEM_LEVELS[lv][0], f"{m}, run {i + 1}")
INKLING_STATUS = {r: B.json.loads((B.RUN(r) / "status.json").read_text()) for r in INKLING.values()}  # no submissions: not scored, not in NAMES
assert all(s["status"] == "completed" and s["result"]["submissions"] == 0 for s in INKLING_STATUS.values())
for k, r in FIXTURE.items():
    NAMES[r] = (f"Positioning fixture: {k} pad", "Gemini 4 Neon, predictor program")
for (arm, fps), r in DIE.items():
    NAMES[r] = ("Biased die", f"{'point grasp' if arm == 'grasp' else 'Franka arm'}, {fps} frames/s, predictor program")
for (lv, c), r in HARNESS.items():
    NAMES[r] = (f"Shadow hand: {B.HAND_NAME[lv]}", "touch harness" if c == "harness" else "control (no harness)")
for k, r in LOWRANK.items():
    NAMES[r] = ("Cube on a spring, block mass stated", k)
for (lv, t), r in TABLETOP.items():
    NAMES[r] = (f"Touch-only tabletop: {lv.replace('-', ' ')}", "with wave-25 touch tooling" if t else "plain touch")
for task, r in GEAR.items():
    NAMES[r] = ("Gear train (weld grasp)", {"recon": "reconstruct task", "cheap": "cheap-simulation task", "deep": "gear-removal task"}[task])
for lv, conds in PROMPT.items():
    for c, runs in conds.items():
        NAMES.setdefault(runs[0], (PROMPT_LEVELS[lv][0], c))
for c, runs in PROMPT_W11.items():
    NAMES.setdefault(runs[0], ("Spring level (Franka), stiffness stated", c))
for lv, r in NORESET.items():
    NAMES[r] = (G.GEM_LEVELS[lv][0] if lv in G.GEM_LEVELS else "Cube on a spring, full MJCF", "Gemini 4 Neon, no resets")
for (c, g), r in BOTTLE.items():
    NAMES[r] = (f"Child-proof bottle, {c}", f"{g} grasp{'s' if g > 1 else ''}")
ORDER = list(NAMES)
for r in NAMES:  # strict, completed (builder asserts strict; here every run must have finished, by agent finish, wall or hold stop)
    assert B.json.loads((B.RUN(r) / "status.json").read_text())["status"] == "completed", r

# ---------------- inputs ----------------
SUMMARIES = tuple(((f[0] if isinstance(f, tuple) else f), set(NAMES)) for f in B.SUMMARIES) + \
            tuple((f, set(NAMES)) for f in sorted((B.MAIN / "validation").glob("results-batchrun-*.json")))  # run-summary.py, one per batch run
INVALID, CONTINUATIONS, COMPARISONS, NOTE_FLAGS, EXAMPLES, VIEWER = set(), {}, {}, {}, {}, {}
MODELS = B.MODELS + (WORK / "models.json",)
CHECKPOINTS = (WORK / "checkpoints-baselines.json", WORK / "checkpoints-cost.json")  # baselines: one source per run (baseline-checkpoints step)
TRACES, CHAT = B.TRACES + (WORK / "traces",), B.CHAT + (WORK / "chat",)
VENDOR, TOOLS, EDITS = B.VENDOR, B.TOOLS, WORK / "edits.json"
RECORDINGS = B.recordings(NAMES)
FIG = {"embodiment": REPORTS / "embodiment-pairs-20261125" / "embodiment-pairs.png", "resets": REPORTS / "reset-ablation-20261126" / "reset-ablation.png",
       "harness": REPORTS / "harness-ablation-20261127" / "harness-ablation.png"}
FIGURES = {f"{k}.png": str(p) for k, p in FIG.items()}
PLOT_LOGS = {"embodiment": REPORTS / "embodiment-pairs-20261125" / "plot-63941899.log", "resets": REPORTS / "reset-ablation-20261126" / "plot-64068835.log",
             "harness": REPORTS / "harness-ablation-20261127" / "build-64096155.log"}  # ablation-pairs-plot.py ROW lines of the published figures


def _evaluation(run, i):
    return json.loads((B.RUN(run) / f"version-{i + 1:03d}" / "evaluation.json").read_text())


def estimates(run, source, m, default):
    if run.startswith(("w23", "w24", "w27", "w28", "w32")) or run == LOWRANK["low-rank answer"]:
        return {}  # their scores are the evaluator's (extra_metrics)
    names = [b["name"] for b in source["bodies"]]
    if "spring_base" in names and "-msd-" not in run:
        # Spring-as-a-scale scenes: a submitted body heavier than 10x the hidden scene's whole movable mass is a fixture the agent made a body (a
        # ~50 kg table, in several Neon runs). The nearest-body mass rule would load it onto the target; exclude it and report its mass apart.
        limit = 10 * sum(b["mass"] for b in source["bodies"])
        heavy = [b["mass"] for b in m["bodies"] if b["mass"] > limit]
        return B.estimates(run, source, dict(m, bodies=[b for b in m["bodies"] if b["mass"] <= limit]), default) | {"extra_mass": sum(heavy)}
    return B.estimates(run, source, m, default)


def extra_metrics(run, i, hist):
    """Batch evaluator families by run; everything else as the cumulative report scores it (B.extra_metrics, incl. its family mapping)."""
    if hist["status"] == "failed":
        return {}
    if not run.startswith(("w23", "w24", "w26-cube-spring-block-lowrank", "w27", "w28", "w32")):
        return B.extra_metrics(run, i, hist)
    ev = _evaluation(run, i)
    assert ev["version"] == hist["version"], (run, i)
    sc = ev["scores"]
    if run.startswith("w23"):
        f = sc["fixture"]
        assert f["status"] == "ok", (run, i, f["status"])
        p, t = f["prediction"], f["targets"]
        return {"fix_crps_mm": p["crps_mm"], "fix_skill": p["crps_skill_vs_pooled"], "fix_hit_target": t["target"]["measured_hit_fraction"],
                "fix_hit_narrow": t["narrow_target"]["measured_hit_fraction"]}
    if run.startswith("w24"):
        d = sc["die"]
        return {"die_brier": d["brier"], "die_acc": d["accuracy"]}
    if run.startswith("w26"):
        a, t = sc["lowrank"]["answer"], sc["lowrank"]["truth"]
        inertia = np.linalg.norm(np.asarray(a["inertia"]) - np.asarray(t["inertia"])) / np.linalg.norm(np.asarray(t["inertia"]))
        return {"r:mass:cube": a["mass"] / t["mass"], "r:stiffness": a["stiffness"] / t["stiffness"], "com_mm": 1000 * float(np.linalg.norm(np.subtract(a["com"], t["com"]))),
                "inertia_rel": float(inertia)}
    if run.startswith("w27"):
        o, out = sc["objects"], {}
        if o["status"] == "ok":
            per = o["per_object"]
            out = {"obj_found": float(np.mean([p["found"] for p in per])), "obj_chamfer_mm": 1000 * float(np.mean([p["chamfer_m"] for p in per if p["found"]])),
                   "obj_centroid_mm": 1000 * float(np.mean([p["centroid_error_m"] for p in per if p["found"]]))}
            labels = [p["label_correct"] for p in per if p["label_correct"] is not None]
            out["obj_labels"] = float(np.mean(labels)) if labels else None
            dims = [d["absolute_error_m"] for p in per if p["found"] for d in p.get("dimensions", []) if d.get("absolute_error_m") is not None]
            out["obj_dim_mm"] = 1000 * float(np.mean(dims)) if dims else None
        if "identification" in sc and sc["identification"].get("status") == "ok":
            out |= {"ident_correct": float(sc["identification"]["correct"]), "ident_err_mm": 1000 * sc["identification"]["distance_error_m"]}
        return out
    if run.startswith("w28"):
        g = sc["gear_train"]
        return {"gear_probe_rad": g["accuracy"]["probe_error_rad"], "gear_cost_rel": g["sim_cost"]["candidate"]["relative_to_shallow"],
                "gear_matched": g["structure"]["matched_fraction"]}
    b = sc["bottle"]
    return {"bottle_agree": b["outcome_agreement"], "bottle_engage_n": b["engage"].get("engage_force_error_n"), "bottle_angle_deg": b["angle"].get("opening_angle_error_deg")}


METRICS = {**B.METRICS, "fix_crps_mm": ("fixture CRPS (mm)", "err"), "fix_skill": ("CRPS skill vs pooled baseline", "score"),
           "fix_hit_target": ("measured hit rate, wide target", "score"), "fix_hit_narrow": ("measured hit rate, narrow target", "score"),
           "die_brier": ("die Brier score", "err"), "die_acc": ("die top-face accuracy", "score"), "inertia_rel": ("inertia error (relative)", "err"),
           "obj_found": ("objects found", "score"), "obj_chamfer_mm": ("object shape, Chamfer (mm)", "err"), "obj_centroid_mm": ("object position error (mm)", "err"),
           "obj_labels": ("object labels correct", "score"), "obj_dim_mm": ("key-dimension error (mm)", "err"), "ident_correct": ("spiky ball identified", "score"),
           "ident_err_mm": ("spiky-ball location error (mm)", "err"), "gear_probe_rad": ("gear probe error (rad)", "err"),
           "gear_cost_rel": ("simulation cost ÷ shallow model", "err"), "gear_matched": ("source gears matched", "score"),
           "bottle_agree": ("probe outcome agreement", "score"), "bottle_engage_n": ("engage-force error (N)", "err"), "bottle_angle_deg": ("opening-angle error (deg)", "err")}

# ---------------- researcher records read by the text ----------------
BOTTLE_OPEN = {Path(x["run"]).name: x for f in sorted((B.MAIN / "validation").glob("bottle-check-*.log"))
               for x in (json.loads(l) for l in f.read_text().splitlines() if l.startswith("{"))}
assert set(BOTTLE.values()) <= set(BOTTLE_OPEN), set(BOTTLE.values()) - set(BOTTLE_OPEN)  # python -m sim2sim.bottle, one per run
GRAB = {Path(x["run"]).name: x for x in (json.loads(l) for l in (B.MAIN / "validation" / "tabletop-post-64086418.log").read_text().splitlines() if l.startswith("{"))}
LOOP = {x["run"]: x for x in map(json.loads, (REPORTS / "report-work-20261127-inkling" / "tool-loop-study.jsonl").read_text().splitlines())}
assert set(INKLING.values()) <= set(LOOP)
WRENCH = [json.loads(l) for l in (B.MAIN / "validation" / "wrench-scan-20261126" / "wrench-scan.jsonl").read_text().splitlines()]


def plot_rows(name):
    """ROW lines of a published ablation figure: (row title, metric, condition, n, last, at matched minutes, at matched tokens)."""
    out = []
    for line in PLOT_LOGS[name].read_text().splitlines():
        if line.startswith("ROW\t"):
            _, title, key, cond, n, last, at_min, at_tok = line.split("\t")
            val = lambda s: None if s.split("=")[1] == "None" else float(s.split("=")[1])
            out.append(dict(title=title, key=key, cond=cond, n=int(n[2:]), last=val(last), at_min=val(at_min), at_tok=val(at_tok),
                            min_label=at_min.split("=")[0][3:], tok_label=at_tok.split("=")[0][3:]))
    assert out, name
    return out


META = dict(title="sim2sim batch results, 23–27 November 2026",
            subtitle=L("Every strict run completed since the 2026-11-24 reports, organized by question · one run per condition (n = 1) · each run is one "
                       "agent working alone in an isolated container on a hidden MuJoCo scene"),
            intro=L("<h2 style='border:0;margin-top:20px'>What this covers</h2>"
                    "<p>This batch asked new kinds of questions: models other than astra (Gemini 4 Carbon; the self-hosted Inkling 2 Mini), what the "
                    "agent's affordances do (resets, an arm instead of an ideal grasp, a local simulator harness, a low-rank answer format or prompt), and "
                    "new task types (stochastic predictors, touch-only tabletops, an explanation-depth gear train, a child-proof bottle). Every condition has "
                    "one complete run; attempts killed by infrastructure (quota, outages, cancellations) are discarded and rerun. Comparisons are early "
                    "observations: with n = 1, differences smaller than an earlier replicate's spread (up to ×4 on some levels) are not findings.</p>"))
GLOSSARY = [
    ("Last / at matched minutes / at matched tokens", "A run's last scored submission; its last one by the comparison's common run-clock minute; its last "
     "one before the common number of model tokens. The common budget is the smallest run total in the comparison."),
    ("Error ÷ astra (×k)", "A model's mean error on a level divided by gpt-6-astra's mean (below 1 = better than astra)."),
    ("CRPS", "Continuous ranked probability score of the predicted distribution of where the carriage settles, against held-out releases (mm; lower is better). "
             "Skill = 1 − CRPS ÷ the pooled baseline's CRPS."),
    ("Brier score", "Squared error of the predicted probabilities of the die's six faces over fresh throws (0 = perfect; uniform guessing scores 0.83)."),
    ("Chamfer, COM error", "Symmetric surface distance between true and reconstructed shapes (mm); distance between their centres of mass (mm)."),
    ("Depth class", "Gear train: 'deep' = every gear modelled and coupled; 'shallow' = a lumped two-body input-output model. Both predict the agent's probes equally."),
    ("Wrench bug", "Before the fix of 2026-11-26, the point grasp's force reading was garbage (up to 1e16 N) whenever it held a hinged or sliding body. No run "
                   "in this report is affected; earlier lever, spring, mass-spring and gear runs are flagged in section 10."),
]


def strategy(ctx):
    return dict(answer=L("Behaviour findings are in each section; the run pages have every conversation and a scrubbable recording."))


def methods(ctx):
    return [
        "<b>Strict runs only</b> (isolated container, model through its API or the self-hosted service; asserted by the builder), each completed.",
        "<b>Scores</b> are the private evaluators'. Reconstruction levels as in earlier reports; the fixture, die, low-rank answer, tabletop, gear train and "
        "bottle use their own evaluator families (docs/evaluation.md), read per submission from each version's evaluation.json.",
        "<b>Bottle opening and tabletop grab</b> come from the post-run state scans (<code>python -m sim2sim.bottle</code>, tabletop post-analysis), not from scores.",
        "<b>Two Gemini keys</b>: several sets ran on the second key after the first one's quota ran out; only the key and cost lines differ.",
        "<b>n = 1 per condition</b>; no replicates by the user's rule. Discarded infrastructure attempts are listed in docs/experiment-log.md.",
    ]


def sections(ctx):
    R = ctx.runs
    blk = lambda **kw: kw
    lnk = lambda r, t=None: f"<a href='run.html?run={r}'>{t or R[r]['condition']}</a>"
    num = lambda v, d=3: "—" if v is None else f"{v:.{d}g}"
    x = lambda v: "—" if v is None else ctx.x(v)
    mm = lambda v: "—" if v is None else ctx.mm(v)
    pct = lambda v: "—" if v is None else f"{100 * v:.0f}%"
    kg = lambda v: "—" if v is None else "none" if v == 0 else f"{v:.0f} kg"
    gate = {"ok": "passes", "fails_accuracy_gate": "fails the accuracy gate"}
    last = lambda r, k: next((u["m"].get(k) for u in reversed(R[r]["subs"]) if u["m"].get(k) is not None), None)
    err = lambda k, v: None if v is None else abs(v - 1) if k.startswith("r:") else v
    avg = lambda vals: None if not [v for v in vals if v is not None] else float(np.mean([v for v in vals if v is not None]))
    at = lambda r, k, axis, limit: next((u["m"].get(k) for u in reversed(R[r]["subs"]) if u[axis] <= limit and u["m"].get(k) is not None), None)
    first_within = lambda r, k, tol: next((u["t"] for u in R[r]["subs"] if u["m"].get(k) is not None and err(k, u["m"][k]) <= tol), None)
    ended = lambda r: {"agent_finish": "finished", "wall_deadline": "wall limit", "budget_suspended": "stopped at 4 h"}.get(R[r]["reason"], R[r]["reason"])
    counters = {}

    def cap(kind, text):
        counters[kind] = counters.get(kind, 0) + 1
        return blk(type="html", html=f"<p class='caption'><b>{kind} {counters[kind]}.</b> {text}</p>")

    lead = lambda text: blk(type="html", html=f"<p>{text}</p>")
    table = lambda head, rows: blk(type="table", head=head, rows=rows)
    sec, n = [], [0]

    def section(sid, nav, title, answer, blocks, tag=None):
        n[0] += 1
        sec.append(dict(id=sid, nav=nav, tag=tag, title=f"{n[0]} · {title}", q=None, answer=answer, blocks=blocks))

    # ---- 1 · models: Carbon and Inkling on the hard five
    ratio, rows = {}, []
    for lv in HARD:
        scene, astra, key = G.GEM_LEVELS[lv]
        members = [r for m in MODELS_ORDER for r in HARD_RUNS[(lv, m)]]
        bud_t, bud_k = min(R[r]["wall_min"] for r in members), min(R[r]["final_mtok"] for r in members)
        a = avg([err(key, last(r, key)) for r in astra])
        for m in MODELS_ORDER:
            runs = HARD_RUNS[(lv, m)]
            v = [avg([err(key, f(r)) for r in runs]) for f in (lambda r: last(r, key), lambda r: at(r, key, "t", bud_t), lambda r: at(r, key, "mtok", bud_k))]
            ratio[(lv, m)] = None if v[0] is None or not a else v[0] / a
            rows.append([scene if m == MODELS_ORDER[0] else "", m, str(len(runs)), ", ".join(lnk(r, f"{R[r]['wall_min']:.0f} min") for r in runs),
                         *(num(z) for z in v), "—" if ratio[(lv, m)] is None else f"×{ratio[(lv, m)]:.2g}"])
    geo = {m: float(np.exp(np.mean(np.log([ratio[(lv, m)] for lv in HARD if ratio[(lv, m)]])))) for m in MODELS_ORDER[1:]}
    carbon_better = [G.GEM_LEVELS[lv][0] for lv in HARD if ratio[(lv, "Gemini 4 Carbon")] and ratio[(lv, "Gemini 4 Carbon")] < 1]
    ink_ended = lambda r: {"agent_finish": "finished", "wall_deadline": "wall limit"}[INKLING_STATUS[r]["result"]["reason"]]
    loop_rows = [[G.GEM_LEVELS[lv][0], ink_ended(INKLING[lv]), "0", f"{LOOP[INKLING[lv]]['text_only']}/{LOOP[INKLING[lv]]['turns']}",
                  pct(LOOP[INKLING[lv]]["share_turns"]), pct(LOOP[INKLING[lv]]["share_time"]), str(LOOP[INKLING[lv]]["longest_streak"]),
                  f"“{LOOP[INKLING[lv]]['top_text_only'][0][0]}” ×{LOOP[INKLING[lv]]['top_text_only'][0][1]}"] for lv in HARD]
    section("models", "Models", f"Carbon's error was ×{geo['Gemini 4 Carbon']:.2g} astra's on average; Inkling 2 Mini never submitted",
            L(f"On the five hard levels, Gemini 4 Carbon's final error was ×{geo['Gemini 4 Carbon']:.2g} gpt-6-astra's (geometric mean over levels; Neon "
              f"×{geo['Gemini 4 Neon']:.2g}, Hydrogen ×{geo['Gemini 4 Hydrogen']:.2g}). Carbon beat astra on {', '.join(carbon_better).lower() or 'no level'}. "
              f"The self-hosted Inkling 2 Mini made no scored submission on any level: {min(LOOP[INKLING[lv]]['share_turns'] for lv in HARD):.0%}–"
              f"{max(LOOP[INKLING[lv]]['share_turns'] for lv in HARD):.0%} of its replies only announced a tool call without making one, taking "
              f"{min(LOOP[INKLING[lv]]['share_time'] for lv in HARD):.0%}–{max(LOOP[INKLING[lv]]['share_time'] for lv in HARD):.0%} of run time (a Gemini Neon "
              "control on the same harness: 3%)."),
            [cap("Table", L("<b>Primary error per level and model</b> (mean over runs): last submission, at the level's matched run-clock minutes and matched "
                            "tokens, and ratio to astra's last. Primary errors: cloth lift-force error, spring-level cluster COM (mm), hand shape Chamfer (mm), "
                            "sealed-box COM path (mm), latch force-profile error.")),
             table(["level", "model", "runs", "run pages", "last", "at matched minutes", "at matched tokens", "÷ astra (last)"], rows),
             cap("Table", L("<b>Inkling 2 Mini: the intention-only tool-call loop</b> (text-only replies, from the chat extracts; "
                            "experiments/validation/tool-loop-study.py).")),
             table(["level", "ended", "submissions", "text-only turns", "share of turns", "share of time", "longest streak", "most common reply"], loop_rows)],
            tag="Carbon n=1 per level; astra 1–3")

    # ---- 2–4 · affordance ablations from the published figures
    def ablation(sid, nav, name, title_fn, intro, cond_names):
        rows_ = plot_rows(name)
        conds = list(dict.fromkeys(r["cond"] for r in rows_))
        per = {}
        for r in rows_:
            per.setdefault(r["title"], {})[r["cond"]] = r
        # ROW values are already errors (ablation-pairs-plot.py prints |ratio - 1| for ratio metrics), so they are shown as they are
        trs = [[t, per[t][conds[0]]["key"]] + [f"{num(per[t][c]['last'])} / {num(per[t][c]['at_min'])} / {num(per[t][c]['at_tok'])}" for c in conds] +
               [cond_names.get(row_winner(per[t], conds), "tie")] for t in per]
        wins = {c: sum(1 for t in per if row_winner(per[t], conds) == c) for c in conds}
        section(sid, nav, title_fn(wins, len(per)), intro,
                [cap("Figure", L("Each row: both conditions' first camera frames and settings, then the tracked metrics over run-clock minutes (dots = scored "
                                 "submissions).")), blk(type="images", items=[dict(src=f"figs/{name}.png", label=nav)]),
                 cap("Table", L("<b>Primary error per row</b> (error; for ratios |estimate ÷ true − 1|): last / at matched minutes / at matched tokens; "
                                "lower is better.")),
                 table(["row", "primary metric"] + [cond_names[c] for c in conds] + ["lower at last submission"], trs)])

    def row_winner(p, conds):
        """The condition with the lower error at the last submission; None for a tie (within 1e-9) or no values."""
        vals = {c: p[c]["last"] for c in conds if p[c]["last"] is not None}
        if not vals or (len(vals) > 1 and max(vals.values()) - min(vals.values()) < 1e-9):
            return None
        return min(vals, key=vals.get)

    ablation("embodiment", "Grasp vs arm", "embodiment",
             lambda w, t: f"Ideal point grasp + force reading vs Franka arm: the arm ended lower in {w['arm']} of {t} scene-matched pairs",
             L("Same hidden scene, ideal point grasp with force reading vs a Franka/Panda arm (joint efforts), gpt-6-astra (earlier runs, new figure). "
               "Caveat: the lever and spring scenes' point-grasp force reading was hit by the jointed-body wrench bug (section 10), so on the lever "
               "the grasp arm did not have a usable force sensor."), {"grasp": "point grasp + wrench", "arm": "Franka arm"})
    ablation("resets", "Resets", "resets",
             lambda w, t: f"Resets vs no resets: resets helped on {w['resets']} of {t} levels — those where probing changes the scene",
             L("Wave 31: each level's Neon config with only <code>resets = false</code>. Agents with resets reset 5–32 times per run, almost always to start a "
               "scripted experiment from the same state (often in both worlds, for a head-to-head comparison). Without resets they turned cautious: the box "
               "agent pre-checked every move for being drop-free and never risked a hard forward throw (the one evaluator throw its model fails), the cloth "
               "agent saved the irreversible drop for last, and the cube agent spent about 45 minutes putting the spring back by hand. Details and quotes: "
               "experiments/reports/reset-ablation-20261126/README.md."), {"resets": "resets allowed", "noreset": "no resets"})
    ablation("harness", "Touch harness", "harness",
             lambda w, t: f"Touch harness (local controller + pausable bash): no clear effect; it ended lower on {w['harness']} of {t} hand levels",
             L("Wave 25: Shadow hand, touch only. Harness agents ran the local controller (their own model with the exact public observations) in 66–82% "
               "of tool calls, to rehearse probes on an empty hand, test shape hypotheses and fit friction and COM; the controls did the same work with "
               "their own MuJoCo in 65–79% of calls. Pausable bash was essentially unused (one resume). Shape errors were within 0.2 mm per pair. Details: "
               "experiments/reports/harness-ablation-20261127/README.md."), {"control": "control", "harness": "touch harness"})

    # ---- 5 · low-rank answer and low-rank prompt
    lr, mj = LOWRANK["low-rank answer"], LOWRANK["full MJCF"]
    lr_rows = [[lnk(r, k), ended(r), f"{R[r]['wall_min']:.0f}", str(R[r]["submissions"]), x(last(r, "r:mass:cube")), x(last(r, "r:stiffness")), mm(last(r, "com_mm")),
                num(last(r, "inertia_rel")) if r == lr else "—", kg(last(r, "abs:extra_mass")),
                "—" if first_within(r, "r:stiffness", 0.1) is None else f"min {first_within(r, 'r:stiffness', 0.1):.0f}"] for k, r in LOWRANK.items()]
    p_rows, faster = [], 0
    for lv, (scene, mkey) in PROMPT_LEVELS.items():
        t10 = {}
        for c, runs in PROMPT[lv].items():
            r = runs[0]
            t10[c] = first_within(r, mkey, 0.1)
            p_rows.append([scene if c == "low-rank prompt" else "", lnk(r, c), f"{R[r]['wall_min']:.0f}", x(last(r, mkey)), x(last(r, "r:stiffness")),
                           kg(last(r, "abs:extra_mass")), "never" if t10[c] is None else f"min {t10[c]:.0f}"])
        faster += t10["low-rank prompt"] is not None and (t10["Neon, original prompt"] is None or t10["low-rank prompt"] < t10["Neon, original prompt"])
    for c, runs in PROMPT_W11.items():
        r = runs[0]
        t = first_within(r, "r:mass:cluster", 0.1)
        p_rows.append(["Spring level (Franka), stiffness stated" if c == "low-rank prompt" else "", lnk(r, c), f"{R[r]['wall_min']:.0f}", x(last(r, "r:mass:cluster")),
                       x(last(r, "r:stiffness")), kg(last(r, "abs:extra_mass")), "never" if t is None else f"min {t:.0f}"])
    section("lowrank", "Low rank", f"A parameter answer got mass and stiffness exact; the low-rank prompt was faster on {faster} of {len(PROMPT_LEVELS)} levels",
            L(f"Wave 26 asked for the cube on a spring as five numbers (mass, stiffness, COM, inertia) instead of a MuJoCo scene: the answer had mass "
              f"{x(last(lr, 'r:mass:cube'))}, stiffness {x(last(lr, 'r:stiffness'))} and COM {mm(last(lr, 'com_mm'))}, in {R[lr]['wall_min']:.0f} run-minutes. "
              f"The full-MJCF run got cube mass {x(last(mj, 'r:mass:cube'))} and stiffness {x(last(mj, 'r:stiffness'))}, but also made the table a "
              f"{last(mj, 'abs:extra_mass'):.0f} kg body. Masses here exclude any submitted body heavier than 10× the hidden scene's movable mass and "
              f"report it in its own column ({sum(1 for c in PROMPT.values() for runs in c.values() if (last(runs[0], 'abs:extra_mass') or 0) > 0)} of the "
              f"{sum(len(c) for c in PROMPT.values())} spring-scale runs below did this). Wave 29 appended one sentence asking the agent to identify the scene's low-rank structure first; the time to get the target mass "
              f"within 10% was shorter with the prompt than the Neon baseline on {faster} of {len(PROMPT_LEVELS)} point-grasp levels."),
            [cap("Table", L("<b>Wave 26: low-rank answer vs full MJCF</b> (Neon; last submission; inertia error relative, answer only).")),
             table(["condition", "ended", "minutes", "submissions", "cube mass ÷ true", "stiffness ÷ true", "COM error", "inertia error", "extra heavy body",
                    "stiffness within 10% from"], lr_rows),
             cap("Table", L("<b>Wave 29: low-rank prompt vs the original prompt</b>: target mass and spring stiffness at the last submission, and the run-clock "
                            "minute the target mass first came within 10%. Cube-block Neon baseline = wave 26's MJCF run; astra baselines from wave 20.")),
             table(["level", "run", "minutes", "target mass ÷ true", "stiffness ÷ true", "extra heavy body", "mass within 10% from"], p_rows)])

    # ---- 6 · touch-only tabletop
    t_rows = []
    for lv in TABLETOP_LEVELS:
        for touch in (False, True):
            r = TABLETOP[(lv, touch)]
            g = GRAB.get(r, {}).get("grab_at_last_reference_state")
            t_rows.append([lv.replace("-", " ") if not touch else "", lnk(r, "+ touch tooling" if touch else "plain"), ended(r), str(R[r]["submissions"]),
                           pct(last(r, "obj_found")), mm(last(r, "obj_chamfer_mm")), mm(last(r, "obj_centroid_mm")), pct(last(r, "obj_labels")),
                           mm(last(r, "obj_dim_mm")), "—" if last(r, "ident_correct") is None else ("yes" if last(r, "ident_correct") else "no"),
                           "—" if g is None else ("yes" if g["success"] else "no")])
    grabs = [GRAB[TABLETOP[("bin-spiky", t)]]["grab_at_last_reference_state"]["success"] for t in (False, True)]
    idents = [last(TABLETOP[("bin-spiky", t)], "ident_correct") for t in (False, True)]
    section("tabletop", "Tabletop", f"Touch-only tabletop: the spiky ball was identified in {sum(bool(v) for v in idents)} of 2 runs and grabbed in {sum(grabs)} of 2",
            L("Wave 27: a Shadow hand on a Franka arm, touch only (joints, efforts, taxels; no camera), finds and rebuilds several free objects on a table. Levels: "
              "abstract shapes, recognizable objects without and with their categories stated, and a bin with one spiky ball among smooth ones (identify and "
              "grab it). Each level ran plain and with the wave-25 touch tooling."),
            [cap("Table", L("<b>Last submission per run</b>: objects found, mean shape error and position error of the found objects, labels correct (named "
                            "levels), mean key-dimension error; spiky ball identified (evaluator) and grabbed (post-run state scan: spiky ball above the bin rim "
                            "in the hand at the last state, the smooth balls still in the bin).")),
             table(["level", "run", "ended", "submissions", "found", "shape", "position", "labels", "dimensions", "spiky identified", "spiky grabbed"], t_rows)])

    # ---- 7 · gear train
    gev = {k: _evaluation(r, R[r]["submissions"] - 1)["scores"]["gear_train"] for k, r in GEAR.items()}
    g_rows = [[lnk(r, NAMES[r][1]), ended(r), str(R[r]["submissions"]), gev[k]["structure"]["depth_class"], pct(gev[k]["structure"]["matched_fraction"]),
               num(gev[k]["accuracy"]["probe_error_rad"]), f"×{gev[k]['sim_cost']['candidate']['relative_to_shallow']:.3g}",
               gate[gev[k]['cheap']['status']] + (f" (×{gev[k]['cheap']['cost_relative_to_shallow']:.3g})" if gev[k]["cheap"]["cost_relative_to_shallow"] else ""),
               f"{gev[k]['removal']['answered']}/{gev[k]['removal']['answered'] + gev[k]['removal']['cannot_answer'] + gev[k]['removal']['failed']}"]
              for k, r in GEAR.items()]
    depths = {k: gev[k]["structure"]["depth_class"] for k in GEAR}
    section("gear", "Gear train", f"Explanation depth followed the stated use: {depths['recon']} for reconstruction, {depths['cheap']} for cheap simulation, "
                                  f"{depths['deep']} for gear removal",
            L("Wave 28: a hidden gear train whose deep (every gear) and shallow (lumped input-output) models predict the agent's probes equally well. The "
              "task named the downstream use: plain reconstruction, a model that must simulate cheaply, or one that must answer what happens when gears are "
              "removed. One run each, weld grasp, Neon."),
            [cap("Table", L("<b>Final model per task</b>: depth class, source gears matched, probe error, simulation cost ÷ the shallow reference, the cheap-task "
                            "gate (accuracy within the gate at low cost) and gear-removal questions answered.")),
             table(["task", "ended", "submissions", "depth", "gears matched", "probe error (rad)", "cost ÷ shallow", "cheap gate", "removal answered"], g_rows)])

    # ---- 8 · bottle
    b_rows = []
    for (c, g), r in BOTTLE.items():
        o = BOTTLE_OPEN[r]
        b_rows.append([lnk(r, f"{c}, {g} grasp{'s' if g > 1 else ''}"), ended(r), str(R[r]["submissions"]), "yes" if o["opened"] else "no",
                       "—" if not o["opened"] else f"min {o['first_opening']['wall_elapsed_at_or_before_s'] / 60:.0f} (episode {o['first_opening']['episode']})",
                       "n/a" if o["contents_seen"] is None else ("yes: " + ", ".join(o["contents_tipped_out"]) if o["contents_seen"] else "no"),
                       pct(last(r, "bottle_agree")), num(last(r, "bottle_engage_n")), num(last(r, "bottle_angle_deg"))])
    opened = sum(BOTTLE_OPEN[r]["opened"] for r in BOTTLE.values())
    section("bottle", "Bottle", f"Child-proof bottle: opened in {opened} of {len(BOTTLE)} runs",
            L("Wave 32: a bottle with a hidden push-and-turn cap (it opens only from about 8 N of push plus a quarter turn), empty or holding objects, with one or "
              "two weld grasps. The agent must open it and model the mechanism. A single grasp can also open it (the table holds the bottle), so 1 vs 2 grasps "
              "measures whether a second grasp helps."),
            [cap("Table", L("<b>Per run</b>: opened (post-run state scan), first opening, contents seen, and the final model's probe-outcome agreement, "
                            "engage-force error (N) and opening-angle error (deg).")),
             table(["run", "ended", "submissions", "opened", "first opened", "contents", "outcome agreement", "engage error (N)", "angle error (deg)"], b_rows)])

    # ---- 9 · stochastic predictors
    f_rows = []
    for k, r in FIXTURE.items():
        p = _evaluation(r, R[r]["submissions"] - 1)["scores"]["fixture"]["prediction"]
        f_rows.append([lnk(r, k), ended(r), str(R[r]["submissions"]), num(p["crps_mm"]), num(p["pooled_crps_mm"]), num(p["mean_only_crps_mm"]), num(p["oracle_crps_mm"]),
                       f"{p['crps_skill_vs_pooled']:.2f}", pct(last(r, "fix_hit_target")), pct(last(r, "fix_hit_narrow"))])
    d_rows = []
    for (arm, fps), r in DIE.items():
        d = _evaluation(r, R[r]["submissions"] - 1)["scores"]["die"]
        bl = d["baselines"]
        d_rows.append([lnk(r, f"{'point grasp' if arm == 'grasp' else 'Franka'}, {fps} f/s"), ended(r), str(R[r]["submissions"]), num(d["brier"]), num(bl["uniform"]["brier"]),
                       num(bl["empirical"]["brier"]), num(bl["mechanics"]["brier"]), pct(d["accuracy"]), pct(bl["empirical"]["accuracy"]), f"{d['on_time']}/{d['trials']}"])
    beat = sum(1 for (arm, fps), r in DIE.items() if (d := _evaluation(r, R[r]["submissions"] - 1)["scores"]["die"])["brier"] < d["baselines"]["empirical"]["brier"])
    section("predictors", "Predictors", f"Stochastic models: the die predictor beat the empirical-frequency baseline in {beat} of {len(DIE)} runs",
            L("Waves 23 and 24 ask for a predictor program instead of a scene: where a perturbed carriage settles (positioning fixture, rigid, rubber or bimodal "
              "pad; distributions scored by CRPS on held-out releases) and which face a biased die lands on (fresh throws in real time; Brier score)."),
            [cap("Table", L("<b>Fixture</b> (last submission): CRPS of the predicted settling distribution and the baselines' (pooled over settings, mean-only, "
                            "oracle), skill vs pooled, and the measured hit rates at the agent's chosen settings for the wide and narrow targets.")),
             table(["pad", "ended", "submissions", "CRPS (mm)", "pooled", "mean-only", "oracle", "skill", "hit rate, wide", "hit rate, narrow"], f_rows),
             cap("Table", L("<b>Die</b> (last submission): Brier score vs uniform, empirical-frequency and mechanics baselines; top-face accuracy vs the "
                            "empirical baseline; answers on time.")),
             table(["setup", "ended", "submissions", "Brier", "uniform", "empirical", "mechanics", "accuracy", "empirical accuracy", "on time"], d_rows)])

    # ---- 10 · data integrity: the jointed-body wrench bug
    fam = lambda r: ("lever" if "lever" in r["run"] else "mass-spring(-damper)" if "msd" in r["run"] else "gear train" if "gear" in r["run"]
                     else "spring lever" if "spring" in r["run"] and not re.match(r"w1[67]-|w20-spring-out", r["run"]) else None)
    hit = [r for r in WRENCH if "worlds" in r and fam(r) and r["worlds"].get("reference", {}).get("over_1e3")]
    w_rows = [[f, str(sum(1 for r in hit if fam(r) == f)), pct(sum(r["worlds"]["reference"]["over_1e3"] for r in hit if fam(r) == f) /
                                                              sum(r["worlds"]["reference"]["readings"] for r in hit if fam(r) == f)),
               f"{max(r['worlds']['reference']['max_n'] for r in hit if fam(r) == f):.2g} N", ", ".join(sorted(r["run"] for r in hit if fam(r) == f))]
              for f in ("lever", "spring lever", "mass-spring(-damper)", "gear train")]
    assert not set(NAMES) & {r["run"] for r in hit}, "a run in this report is affected by the wrench bug"
    section("integrity", "Data integrity", f"Wrench bug: {len(hit)} earlier runs showed agents garbage grasp forces; none in this report",
            L("Before the fix of 2026-11-26, the point grasp's public force reading was wrong whenever it held a hinged or sliding body. An audit scan of every "
              "earlier strict run counted the readings agents saw above 1,000 N. Tunnel and cloth readings were genuine (jam forces of the stiff ideal grasp). "
              "<b>Flag:</b> results that credit force sensing on the runs below are confounded; they are marked here rather than removed."),
            [cap("Table", L("<b>Affected earlier runs</b> (reference world): runs, share of their readings above 1,000 N, largest reading "
                            "(experiments/validation/wrench-scan-20261126/).")),
             table(["level family", "runs", "readings > 1 kN", "largest", "runs (flagged)"], w_rows)])

    sec.append(dict(id="runs", nav="A1", appendix=True, tag=None, title="A1 · All runs", q=None, answer=None,
                    blocks=[lead("Click a name for its page: conversation, images the model received and a scrubbable recording."), blk(type="runs")]))
    return sec
