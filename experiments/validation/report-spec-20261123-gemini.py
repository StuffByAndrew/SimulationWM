"""Report spec: Gemini 4 Neon and Gemini 4 Hydrogen against gpt-6-astra on five hard levels (strict runs, 2026-11-06 to 2026-11-20).
Every Gemini config equals its astra original except the model route and the cost lines (same harness, image, task, tools,
observations, reasoning level, output cap and wall budget). Reuses the run sets, metrics and inputs of report-spec-20261114.py.
Build: .venv/bin/python experiments/validation/build-results-report.py data experiments/validation/report-spec-20261123-gemini.py \\
           experiments/reports/report-work-20261120 experiments/reports/results-gemini-vs-astra-20261123
       then `text` (seconds) and `zip`."""
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np

VAL = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("cumulative_spec", VAL / "report-spec-20261114.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
assert B.WITH_MEDIA and B.WITH_SCALE, "the comparison uses the rendered tests and the spring-scale scan"

GEM_LEVELS, GEM_MODELS, GEM_DONE, GEM_FAILED, MC = B.GEM_LEVELS, B.GEM_MODELS, B.GEM_DONE, B.GEM_FAILED, B.MC
MODEL_ORDER = ["gpt-6-astra", *GEM_MODELS.values()]
RUNS = {(lv, "gpt-6-astra"): GEM_LEVELS[lv][1] for lv in GEM_LEVELS} | {(lv, GEM_MODELS[m]): GEM_DONE[(lv, m)] for lv in GEM_LEVELS for m in GEM_MODELS}
NAMES = {r: (GEM_LEVELS[lv][0], f"{model}, run {i + 1}") for (lv, model), runs in RUNS.items() for i, r in enumerate(runs)}
MODEL_OF = {r: model for (lv, model), runs in RUNS.items() for r in runs}
LEVEL_OF = {r: lv for (lv, model), runs in RUNS.items() for r in runs}
ORDER = list(NAMES)

# inputs (the cumulative report's), restricted to these runs
SUMMARIES = tuple(((f[0] if isinstance(f, tuple) else f), set(NAMES)) for f in B.SUMMARIES)
INVALID, CONTINUATIONS, COMPARISONS, NOTE_FLAGS, EXAMPLES = set(), {}, {}, {}, {}
MODELS, TRACES, CHAT, VENDOR, TOOLS = B.MODELS, B.TRACES, B.CHAT, B.VENDOR, B.TOOLS
CHECKPOINTS = (B.MAIN / "reports" / "report-work-20261123" / "checkpoints-cost.json",)  # checkpoint-index.py with cumulative API cost per submission
EDITS = B.MAIN / "reports" / "report-work-20261123" / "edits.json"  # text edits from this report's edit mode
estimates, extra_metrics, METRICS = B.estimates, B.extra_metrics, B.METRICS
CLIPS = {r: B.CLIPS[r] for r in NAMES if r in B.CLIPS}
FIGURES = {f"clips/{c['file']}": str(B.EV[r] / c["file"]) for r, cs in CLIPS.items() for c in cs}
RECORDINGS = B.recordings(NAMES)
VIEWER = {"hand": RUNS[("w21-hand-smooth-mesh", "gpt-6-astra")] + [r for m in GEM_MODELS.values() for r in RUNS[("w21-hand-smooth-mesh", m)]]}
L = lambda s: s

LEVEL_TEXT = {
    "w22-cloth-holes": "A square cloth (a MuJoCo 2D flex) with holes and cuts lies on a table; its bending stiffness varies by region. Point grasp with "
                       "force reading, RGB + depth, movable camera, sequence tool. Scored by lifting the cloth at 16 grid points (lift-force error), "
                       "by five drops onto a bar or a cone (shape during the drops) and by shape at rest.",
    "w11-tetra-spring-k": "A Franka arm, a cluster of four fused spheres (masses 1:2:4:8) and a two-plate spring of stated stiffness lying next to a hole "
                          "it fits into. No force or joint-torque reading, movable camera. The scene is built for weighing the cluster on the seated "
                          "spring. Scored on the cluster's centre of mass (COM), shape, and each part's mass.",
    "w21-hand-smooth-mesh": "A Shadow dexterous hand, palm up, holds a smooth convex object it cannot see: no camera, only joint readings and fingertip "
                            "and palm touch grids. Scored on shape (Chamfer), COM, mass and two scripted probes (squeeze, tilt).",
    "w19-marble-heavy-hint": "An 80 g sealed box with a 68 g ball rolling inside, picked up by a Franka; the task says the box has moving contents. "
                             "Scored on the path of the composite COM (box + contents) during four scripted throws.",
    "w20-latch-out": "A stick lies outside a tunnel whose closed far end hides a push-push latch behind a spring-loaded plunger. Point grasp with force "
                     "reading, fixed camera. Scored on the resistance along the evaluator's insertion, push and pull probe (force-profile error).",
}
LEVEL_KEYS = {"w22-cloth-holes": ["stiff_rel", "drop_chamfer_mm", "rest_chamfer_mm"],
              "w11-tetra-spring-k": ["com_mm", "chamfer_mm", "r:mass:cluster", "r:mass:spring_base", "r:mass:spring_top"],
              "w21-hand-smooth-mesh": ["chamfer_mm", "com_mm", "r:mass", "hand_traj_mm"],
              "w19-marble-heavy-hint": ["traj_mm", "com_mm", "r:mass"],
              "w20-latch-out": ["fp_rel", "chamfer_mm"]}
COLORS = {"gpt-6-astra": "--base", "Gemini 4 Neon": "--abl", "Gemini 4 Hydrogen": "--third"}

META = dict(title="Gemini 4 Neon and Gemini 4 Hydrogen against gpt-6-astra on five hard sim2sim levels",
            subtitle=L("{n} strict runs, 2026-11-06 to 2026-11-20 · each run is one agent working alone in an isolated container on a hidden MuJoCo "
                       "scene · all three models at the same <b>medium</b> reasoning level · one to three runs per model per level"),
            intro=L("<h2 style='border:0;margin-top:20px'>The comparison</h2>"
                    "<p>Each run hides a MuJoCo scene and asks an LLM agent to rebuild it as a scene file whose geometry and physics match, by "
                    "looking at it (where a camera exists) and experimenting on it. A private evaluator scores every submission. Five of the hardest "
                    "levels were rerun with Google's <b>Gemini 4 Neon</b> and <b>Gemini 4 Hydrogen</b> through Google's API, in configurations "
                    "identical to the existing <b>gpt-6-astra</b> runs except the model route and cost accounting: same sandbox, harness, task text, "
                    "tools, observations, image, 16,384-token output cap and wall budget (6 h; 4 h for the spring level), and the same medium "
                    "reasoning level (astra <code>reasoning_effort</code>, Gemini <code>thinkingLevel</code>; asserted for every run). Gemini "
                    "spend was not capped; astra's spring-level runs stopped at their cost limit, so equal-minute and equal-token comparisons are "
                    "shown alongside the final submission.</p>"
                    "<p>Gemini ran in two batches. The first had no protection against Google-side outages, and five of its ten runs died of call "
                    "timeouts; they are infrastructure failures, not results, and are left out. The second held outages outside the run clock "
                    "(the agent never saw them) and all ten completed. Tokens are each provider's own count.</p>"))
GLOSSARY = [
    ("Final / at equal minutes / at equal tokens", "The last scored submission; the last one by the comparison's common run-clock minute; the last one "
     "before the run had used the comparison's common number of model tokens. Gemini made far fewer calls per minute, so equal tokens is the "
     "fairer view of effort."),
    ("Error ÷ astra (×k)", "A model's mean final error on a level divided by astra's mean (×1 = same, below 1 = better than astra). Across levels, "
     "the geometric mean of these ratios."),
    ("Chamfer, COM error", "Symmetric surface distance between true and reconstructed shapes (mm); distance between their centres of mass (mm)."),
    ("Force-profile error", "RMS difference in resistance along the evaluator's probe, ÷ the RMS true resistance (0 = exact; 100% = as wrong as no mechanism)."),
    ("Lift-force error", "Cloth: the evaluator lifts the cloth at 16 grid points; mean absolute force error ÷ mean true force."),
    ("COM path error", "Sealed box: mean distance between the true and reconstructed composite COM during four scripted throws (mm)."),
    ("Malformed reply / output limit", "A model reply whose tool call could not be parsed, or that hit the 16,384-token output cap (Gemini counts "
     "its thinking towards the cap); neither does anything, and the harness asks again."),
]


def strategy(ctx):
    return dict(answer=L("Per-model shares of bash calls and notes are in the section on how the models used their turns."))


def methods(ctx):
    return [
        "<b>Strict runs only</b>: isolated container, no access to scene files, model called through its API; asserted by the builder.",
        "<b>Same configuration</b>: each Gemini config equals its astra original except the model route and cost lines (diffed when the batch was "
        "built); the hidden scenes are byte-identical. Every compared run's private config has <code>reasoning_effort = \"medium\"</code> (asserted).",
        "<b>Scores</b> are the private evaluator's; tunnel and box scores come from each submission's evaluation, spring masses from compiling "
        "each submission with MuJoCo. Minutes, tokens and dollars at each submission come from the audit's per-call accounting "
        "(<code>checkpoint-index.py</code>); equal budgets are the largest every run on a level reached.",
        "<b>Infrastructure failures excluded</b>: " + ", ".join(f"<code>{r}</code>" for r in GEM_FAILED) + " died of Google-side call timeouts "
        "before the outage hold existed.",
        "<b>Few runs</b>: one to three per model per level. astra's own replicates differ by up to ×4 (marble box), so per-level differences smaller "
        "than that are trends, not findings.",
    ]


def sections(ctx):
    R, f = ctx.runs, ctx.final
    blk = lambda **kw: kw
    lnk = lambda r, t=None: f"<a href='run.html?run={r}'>{t or R[r]['condition']}</a>"
    x = lambda v, d=3: "—" if v is None else ctx.x(v, d)
    mm = lambda v, d=3: "—" if v is None else ctx.mm(v, d)
    pct = lambda v: "—" if v is None else "exact" if v < 1e-3 else f"{100 * v:.0f}%"
    fmt_k = lambda k, v: "—" if v is None else x(v) if k.startswith("r:") else mm(v) if k.endswith("_mm") else pct(v) if k.endswith("_rel") else f"{v:.3g}"
    hl = lambda txt, good: f"<span class='hl {'good' if good else 'bad'}'>{txt}</span>"
    conv = lambda series, keys, **kw: blk(type="convergence", keys=keys, **kw,
                                          series=[dict(**({"runs": s[0]} if isinstance(s[0], list) else {"run": s[0]}), label=s[1], **(s[2] if len(s) > 2 else {})) for s in series])
    counters = {}

    def cap(kind, text):
        counters[kind] = counters.get(kind, 0) + 1
        return blk(type="html", html=f"<p class='caption'><b>{kind} {counters[kind]}.</b> {text}</p>")

    lead = lambda text: blk(type="html", html=f"<p>{text}</p>")
    err = lambda k, v: None if v is None else abs(v - 1) if k.startswith("r:") else v
    avg = lambda vals: None if not [v for v in vals if v is not None] else float(np.mean([v for v in vals if v is not None]))
    point = lambda r, k, cut: next((u["m"].get(k) for u in reversed(R[r]["subs"]) if cut(u) and u["m"].get(k) is not None), None)
    final = lambda r, k: R[r]["subs"][-1]["m"].get(k)
    assert all("usd" in u for r in NAMES for u in R[r]["subs"]), "every compared submission needs its cumulative API cost"
    # Equal budgets: per level and axis, the largest budget every run reached (the smallest run total); each run is read at its last
    # scored submission within that budget. Axes: run-clock minutes, cumulative model tokens, cumulative API cost.
    AXES = {"min": ("t", lambda r: R[r]["wall_min"], lambda v: f"{v:.0f} min", "equal minutes"),
            "tok": ("mtok", lambda r: R[r]["final_mtok"], lambda v: f"{v:.1f}M tokens", "equal tokens"),
            "usd": ("usd", lambda r: R[r]["cost_usd"], lambda v: f"${v:.2f}", "equal dollars")}
    budget = {(lv, a): min(total(r) for model in MODEL_ORDER for r in RUNS[(lv, model)]) for lv in GEM_LEVELS for a, (_, total, _, _) in AXES.items()}
    at = lambda r, k, lv, a: point(r, k, lambda u: u[AXES[a][0]] <= budget[(lv, a)])
    WHEN = ("final", *AXES)
    WHEN_NAME = {"final": "final", **{a: v[3] for a, v in AXES.items()}}
    effort = {r: json.loads((B.RUN(r) / "config-private.json").read_text())["inference"].get("reasoning_effort") for r in NAMES}
    assert set(effort.values()) == {"medium"}, effort

    # ---- ratios to astra, per level and overall (final and at equal tokens)
    ratio = {}
    for lv, (_, astra, key) in GEM_LEVELS.items():
        for when, get in [("final", lambda r: final(r, key))] + [(a, lambda r, a=a: at(r, key, lv, a)) for a in AXES]:
            a = avg([err(key, get(r)) for r in astra])
            for model in GEM_MODELS.values():
                g = avg([err(key, get(r)) for r in RUNS[(lv, model)]])
                ratio[(lv, model, when)] = None if g is None or not a else g / a
    have = {(m, w): [ratio[(lv, m, w)] for lv in GEM_LEVELS if ratio[(lv, m, w)]] for m in GEM_MODELS.values() for w in WHEN}
    geo = {k: float(np.exp(np.mean(np.log(v)))) for k, v in have.items()}  # over the levels where both models had a score at that point
    better = {(m, w): [GEM_LEVELS[lv][0] for lv in GEM_LEVELS if ratio[(lv, m, w)] is not None and ratio[(lv, m, w)] < 1] for m in GEM_MODELS.values() for w in WHEN}
    cell = lambda v: "—" if v is None else hl(f"×{v:.2g}", v < 1)
    overview = [[f"<a href='#{lv}'>{GEM_LEVELS[lv][0]}</a><div class='muted small'>{B.GEM_LABEL[GEM_LEVELS[lv][2]]}</div>"] +
                [cell(ratio[(lv, m, w)]) for m in GEM_MODELS.values() for w in WHEN] for lv in GEM_LEVELS]
    overview.append(["<b>geometric mean</b> (levels with a score)"] + [f"<b>{cell(geo[(m, w)])}</b> <span class='muted'>({len(have[(m, w)])})</span>"
                                                                         for m in GEM_MODELS.values() for w in WHEN])
    LATCH = GEM_LEVELS["w20-latch-out"][1][0]
    spend = {model: (float(np.mean([R[r]["final_mtok"] for lv in GEM_LEVELS for r in RUNS[(lv, model)]])),
                     float(np.mean([R[r]["cost_usd"] for lv in GEM_LEVELS for r in RUNS[(lv, model)]]))) for model in MODEL_ORDER}
    verdict = lambda m: "worse" if geo[(m, "final")] > 1 else "better"
    neon, hyd = GEM_MODELS["neon"], GEM_MODELS["hydrogen"]
    sec = [dict(id="overview", nav="Overview", tag="one to three runs per model per level",
        title=f"1 · On average both Gemini models did worse than astra (final error ×{geo[(neon, 'final')]:.2g} and ×{geo[(hyd, 'final')]:.2g} astra's), "
              f"but not on every level",
        q=None, answer=L(f"Gemini 4 Neon's final error was ×{geo[(neon, 'final')]:.2g} astra's and Gemini 4 Hydrogen's ×{geo[(hyd, 'final')]:.2g} "
                         f"(geometric mean over five levels), so both did {verdict(neon)} on average. Neon beat astra on "
                         f"{', '.join(better[(neon, 'final')]).lower() or 'no level'}; Hydrogen on {', '.join(better[(hyd, 'final')]).lower() or 'no level'}. "
                         "Normalized by the resources spent, the comparison changes: at the largest budget every run on a level reached, the error ratios "
                         "(geometric mean over the levels where both models had a scored submission by then, count in brackets) were " +
                         "; ".join(f"{WHEN_NAME[a]}: Neon ×{geo[(neon, a)]:.2g} ({len(have[(neon, a)])}), Hydrogen ×{geo[(hyd, a)]:.2g} ({len(have[(hyd, a)])})"
                                   for a in AXES) + ". Per run, astra used " + f"{spend['gpt-6-astra'][0]:.1f}M tokens and ${spend['gpt-6-astra'][1]:.0f}; "
                         + ", ".join(f"{m} {spend[m][0]:.1f}M and ${spend[m][1]:.0f}" for m in GEM_MODELS.values()) + " (Gemini's tokens cost more at list "
                         "prices, and its thinking is billed as output), so fewer tokens did not mean fewer dollars. astra was also often still far "
                         f"from its final answer at Gemini's budget (latch: {pct(at(LATCH, 'fp_rel', 'w20-latch-out', 'tok'))} error at "
                         f"{AXES['tok'][2](budget[('w20-latch-out', 'tok')])}, {pct(final(LATCH, 'fp_rel'))} at the end)."),
        blocks=[cap("Table", L("<b>Mean error ÷ astra's mean error, per level</b> (green = better than astra): at the final submission, and normalized "
                               "by resources at equal run-clock minutes, equal model tokens and equal API dollars. The budget on each axis is the largest "
                               "every run on the level reached (its smallest run total; the level sections give the values); each run is read at its last "
                               "scored submission within it, and \"—\" means a model had no scored submission by then. Gemini dollars are priced at "
                               "Google's list prices (Google reports none), astra's are the gateway's. One to three runs per model; astra's own "
                               "replicates differ by up to ×4.")),
                blk(type="table", head=["level"] + [f"{m}: {WHEN_NAME[w]}" for m in GEM_MODELS.values() for w in WHEN], rows=overview)])]

    # ---- one section per level
    for i, (lv, (scene, astra, key)) in enumerate(GEM_LEVELS.items()):
        keys = LEVEL_KEYS[lv]
        rows = []
        for model in MODEL_ORDER:
            for r in RUNS[(lv, model)]:
                rows.append([model if r == RUNS[(lv, model)][0] else "", lnk(r, f"run {RUNS[(lv, model)].index(r) + 1}"),
                             "finished" if R[r]["reason"] == "agent_finish" else "cost limit" if R[r]["reason"] == "budget_suspended" else R[r]["reason"],
                             f"{R[r]['wall_min']:.0f}", f"{R[r]['final_mtok']:.1f}", str(R[r]["submissions"]),
                             f"${R[r]['cost_usd']:.2f}", f"<b>{fmt_k(key, final(r, key))}</b>"] + [fmt_k(key, at(r, key, lv, a)) for a in AXES] +
                            [fmt_k(k, final(r, k)) for k in keys[1:]])
        level_head = ["model", "run", "ended", "minutes", "tokens (M)", "submissions", "API cost", f"{B.GEM_LABEL[key]}: final"] + \
                     [f"at {AXES[a][2](budget[(lv, a)])}" for a in AXES] + [f"{ctx_label(k)} (final)" for k in keys[1:]]
        series = [[astra if len(astra) > 1 else astra[0], "gpt-6-astra", {"color": COLORS["gpt-6-astra"]}]] + \
                 [[r, f"{NAMES[r][1]}", {"color": COLORS[GEM_MODELS[m]], "dash": "6 3" if "run 2" in NAMES[r][1] else ""}] for m in GEM_MODELS for r in GEM_DONE[(lv, m)]]
        blocks = [lead(LEVEL_TEXT[lv]),
                  cap("Table", L(f"<b>Every run on this level.</b> Primary error ({B.GEM_LABEL[key]}) at the final submission and at equal minutes, "
                                 "tokens and dollars (the largest budget every run here reached); other scores at the final submission. Lower is better; "
                                 "masses are estimate ÷ true.")),
                  blk(type="table", head=level_head, rows=rows),
                  cap("Figure", L("<b>Error against resources spent</b>: by run-clock minutes, by cumulative model tokens and by cumulative API "
                                  "dollars (astra: mean and range when it has several runs).")),
                  conv(series, keys[:3], title="By run-clock minutes", xmax=360),
                  conv(series, keys[:3], title="By cumulative model tokens", x="mtok"),
                  conv(series, keys[:3], title="By cumulative API cost (USD)", x="usd")]
        if lv == "w11-tetra-spring-k":
            blocks += [cap("Table", L("<b>Did the agent use the spring as a scale?</b> From every 10th recorded state of the hidden scene: when the spring "
                                      "first sat in the hole, when the cluster first rested on it there, and when it rested on the upright spring "
                                      "anywhere.")),
                       blk(type="table", head=["run", "spring put in the hole", "cluster weighed on it in the hole", "cluster on the upright spring",
                                               "cluster mass ÷ true"],
                           rows=[[lnk(r, f"{MODEL_OF[r]}, {NAMES[r][1].split(', ')[-1]}")] +
                                 [("never" if B.SCALE[r]["first_wall_s"].get(k) is None else f"min {B.SCALE[r]['first_wall_s'][k] / 60:.0f}")
                                  for k in ("spring_in_hole", "cluster_weighed", "cluster_weighed_upright")] + [x(final(r, "r:mass:cluster"))]
                                 for model in MODEL_ORDER for r in RUNS[(lv, model)]])]
        if lv == "w21-hand-smooth-mesh":
            blocks += [cap("Figure", "<b>The held object in 3D</b>: true shape (solid) and each run's reconstruction (wireframe); pick a run."),
                       blk(type="viewer", group="hand", title="Shadow hand, smooth convex object")]
        clip_runs = [r for model in MODEL_ORDER for r in RUNS[(lv, model)] if r in CLIPS]
        if clip_runs:
            blocks += [cap("Figure", L("<b>The scored tests on each run's final model</b>, hidden scene (left) against the model (right), re-simulated as "
                                       "scored. Pick a run and test.")),
                       blk(type="clips", title=None, items=[dict(src=f"clips/{c['file']}", label=f"{MODEL_OF[r]} · {NAMES[r][1].split(', ')[-1]} · {c['clip']}", run=r)
                                                            for r in clip_runs for c in CLIPS[r] if c["family"] != "cloth-still"])]
            stills = [(r, c) for r in clip_runs for c in CLIPS[r] if c["family"] == "cloth-still" and c["clip"] != "RGB at rest"]
            if stills:
                blocks += [cap("Figure", "<b>Cloth stiffness, truth (left) against each final model (right)</b>: the compiled bending-stiffness field and the "
                                         "evaluator's lift force at 16 points."),
                           blk(type="images", items=[dict(src=f"clips/{c['file']}", label=f"{MODEL_OF[r]}, {NAMES[r][1].split(', ')[-1]}: {c['clip']}", run=r)
                                                     for r, c in stills])]
        a = avg([err(key, final(r, key)) for r in astra])
        sec.append(dict(id=lv, nav=scene, tag=f"astra {len(astra)} · Neon {len(RUNS[(lv, neon)])} · Hydrogen {len(RUNS[(lv, hyd)])} runs",
                        title=f"{i + 2} · {scene}: " + ", ".join(f"{m} ×{ratio[(lv, m, 'final')]:.2g}" for m in GEM_MODELS.values() if ratio[(lv, m, 'final')] is not None) + " astra's error",
                        q=None, answer=None, blocks=blocks))

    # ---- how the models work
    q_rows, s_rows, q, sh = [], [], {}, {}
    cats = [("fit_sim", "fit over local simulations"), ("fit_data", "fit on images or data"), ("simulate", "single local simulation"),
            ("analytic", "closed-form physics"), ("image", "image processing"), ("other", "housekeeping")]
    for model in MODEL_ORDER:
        runs = [r for lv in GEM_LEVELS for r in RUNS[(lv, model)]]
        mc = [m for lv in GEM_LEVELS for m in MC[lv]["runs"] if m["run"] in runs]
        assert len(mc) == len(runs), (model, len(mc), len(runs))
        calls = sum(sum(m["finish_reasons"].values()) for m in mc)
        count = lambda *ks: sum(m["finish_reasons"].get(k, 0) for m in mc for k in ks)
        q[model] = dict(hour=60 * calls / sum(m['wall_min'] for m in mc), tok=np.mean([R[r]['final_mtok'] for r in runs]),
                        bad=100 * count('MALFORMED_FUNCTION_CALL', 'MALFORMED_RESPONSE') / calls, cap=100 * count('MAX_TOKENS') / calls)
        q_rows.append([model, str(len(runs)), str(calls), f"{60 * calls / sum(m['wall_min'] for m in mc):.0f}",
                       f"{np.mean([R[r]['final_mtok'] for r in runs]):.1f}", f"{100 * count('MALFORMED_FUNCTION_CALL', 'MALFORMED_RESPONSE') / calls:.1f}%",
                       f"{100 * count('MAX_TOKENS') / calls:.1f}%", str(sum(m["call_timeouts"] for m in mc)), f"{sum(m['provider_held_min'] for m in mc):.0f}"])
        k = Counter()
        notes = sum(R[r]["taxonomy"]["notes"] for r in runs)
        for r in runs:
            k.update(R[r]["taxonomy"]["script_counts"])
        tot = sum(k.values())
        sh[model] = dict(fit=100 * (k["fit_sim"] + k["fit_data"]) / tot, sim=100 * k["simulate"] / tot, img=100 * k["image"] / tot,
                         app=100 * sum(R[r]['taxonomy']['appearance'] for r in runs) / notes)
        s_rows.append([model, str(tot)] + [f"{100 * k[c] / tot:.0f}%" for c, _ in cats] +
                      [f"{100 * sum(R[r]['taxonomy']['appearance'] for r in runs) / notes:.0f}%"])
    sec.append(dict(id="behaviour", nav="How they work", tag=None, title=f"{len(GEM_LEVELS) + 2} · How the models used their turns",
        q=None, answer=L(f"astra made {q['gpt-6-astra']['hour']:.0f} model calls per hour and Gemini {min(q[m]['hour'] for m in (neon, hyd)):.0f}–{max(q[m]['hour'] for m in (neon, hyd)):.0f}; "
                         f"{min(q[m]['bad'] for m in (neon, hyd)):.0f}–{max(q[m]['bad'] for m in (neon, hyd)):.0f}% of Gemini's replies were malformed and "
                         f"{min(q[m]['cap'] for m in (neon, hyd)):.0f}–{max(q[m]['cap'] for m in (neon, hyd)):.0f}% hit the output cap (thinking counts "
                         f"towards it), against {q['gpt-6-astra']['bad']:.0f}% and {q['gpt-6-astra']['cap']:.0f}% for astra. The models also worked "
                         f"differently: astra spent {sh['gpt-6-astra']['fit']:.0f}% of its bash calls fitting parameters by optimizer or sweep, Gemini "
                         f"{min(sh[m]['fit'] for m in (neon, hyd)):.0f}–{max(sh[m]['fit'] for m in (neon, hyd)):.0f}%; Gemini instead ran more single "
                         f"simulations ({min(sh[m]['sim'] for m in (neon, hyd)):.0f}–{max(sh[m]['sim'] for m in (neon, hyd)):.0f}% vs "
                         f"{sh['gpt-6-astra']['sim']:.0f}%) and more image processing, and wrote more notes about appearance "
                         f"({min(sh[m]['app'] for m in (neon, hyd)):.0f}–{max(sh[m]['app'] for m in (neon, hyd)):.0f}% vs {sh['gpt-6-astra']['app']:.0f}%)."),
        blocks=[cap("Table", L("<b>Calls and their quality</b> (completed runs). Outage-held minutes are Google-side outages held outside the run clock.")),
                blk(type="table", head=["model", "runs", "model calls", "calls per hour", "tokens per run (M)", "malformed replies", "hit output cap",
                                        "call timeouts", "outage held (min)"], rows=q_rows),
                cap("Table", L("<b>What their bash calls computed</b> (share of all bash calls, keyword rules of the cumulative report) and the share of "
                               "progress notes about appearance or lighting.")),
                blk(type="table", head=["model", "bash calls"] + [c[1] for c in cats] + ["notes about appearance"], rows=s_rows)]))
    sec.append(dict(id="effort", nav="A1", appendix=True, tag=None, title="A1 · Time and cost", q=None, answer=None,
        blocks=[lead("Gemini cost is host-priced at Google AI Studio list prices (Google reports none); astra's is the gateway's estimate."),
                blk(type="bars", title="Wall clock used (minutes)", unit="min", rows=[dict(run=r, label=" · ".join(NAMES[r]), v=R[r]["wall_min"], color=COLORS[MODEL_OF[r]]) for r in ORDER]),
                blk(type="bars", title="API cost (USD)", unit="$", rows=[dict(run=r, label=" · ".join(NAMES[r]), v=R[r]["cost_usd"], color=COLORS[MODEL_OF[r]]) for r in ORDER])]))
    sec.append(dict(id="runs", nav="A2", appendix=True, tag=None, title="A2 · All runs", q=None, answer=None,
                    blocks=[blk(type="html", html="<p>Click a name for its page with the full conversation and the images the model received.</p>"), blk(type="runs")]))
    return sec


def ctx_label(k):
    """Column label: the builder's names for its own keys, the spec's for the rest."""
    return {**{"chamfer_mm": "Chamfer", "com_mm": "COM error", "r:mass": "mass"}, **{key: v[0] for key, v in METRICS.items()}}.get(k, k)
