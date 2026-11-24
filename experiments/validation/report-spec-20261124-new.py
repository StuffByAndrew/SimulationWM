"""Report spec: strict runs started since 2026-11-17 that are not in the 2026-11-24 reports (results-report-20261124,
results-gemini-vs-astra-20261124). Status of every such run; scores where a run submitted: Gemini 4 Carbon on cloth-holes and on the
stated-stiffness spring level, and the wave-26 full-MJCF arm (Gemini 4 Neon) on the cube-on-a-spring level, each against the earlier
runs of the same level. Runs that ended early on infrastructure are labelled truncated and read up to their last submission.
Build: .venv/bin/python experiments/validation/build-results-report.py data experiments/validation/report-spec-20261124-new.py \\
           experiments/reports/report-work-20261124 experiments/reports/results-new-20261124
       then `text` (seconds) and `zip`."""
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VAL = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("gemini_spec", VAL / "report-spec-20261123-gemini.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
B = G.B
WORK = B.MAIN / "reports" / "report-work-20261124"
SINCE = "20261117"
REPORTED = set(B.NAMES) | set(G.NAMES)  # runs already in the two 2026-11-24 reports
MODEL_NAME = {"gpt-6-astra": "gpt-6-astra", "gemini-4-neon": "Gemini 4 Neon", "gemini-4-hydrogen": "Gemini 4 Hydrogen", "gemini-4-carbon": "Gemini 4 Carbon"}
L = lambda s: s

# ---------------- every new run and its state ----------------
BATCHES = [  # (run-id pattern, batch, question); each new run matches exactly one
    (r"-carbon-", "Gemini 4 Carbon on the five hard levels",
     "Is Gemini 4 Carbon better than Neon, Hydrogen and astra? Same configs as the Gemini batch; only the model route differs."),
    (r"-(neon|hydrogen)-20261119-", "Gemini batch 1 (no outage hold)",
     "The batch-1 runs that died of Google call timeouts; the Gemini report left them out as infrastructure failures."),
    (r"^w23-", "Wave 23: positioning fixture",
     "Can the agent predict where a perturbed carriage settles (rigid, rubber, bimodal pads)? Predictor program scored on held-out releases."),
    (r"^w24-", "Wave 24: biased die",
     "Can the agent predict a biased die from its own throws (point grasp or Franka; 25 or 100 frames/s)? Real-time predictor program."),
    (r"^w25-", "Wave 25: touch harness, Shadow hand",
     "Does a local controller with pausable bash help touch-only reconstruction? Each hand level paired with its control (the w21 config on Neon)."),
    (r"^w26-", "Wave 26: low-rank answer vs full MJCF",
     "Cube on a spring, block mass stated: is answering with a few parameters (mass, stiffness, COM, inertia) better than building a MuJoCo scene?"),
    (r"^w27-", "Wave 27: touch-only tabletop",
     "With a Shadow hand on a Franka and touch alone, can the agent find and rebuild several objects on a table, and tell a spiky ball from smooth "
     "ones? Each level paired with the wave-25 touch tooling."),
    (r"^w29-", "Wave 29: low-rank prompt",
     "Does asking the agent first to find the scene's low-rank structure make it converge faster on the target mass and spring stiffness?"),
]
CANCELLED = {"w20-latch-out-carbon-20261123-001": "cancelled to free a test slot for the live dashboard",
             "w20-latch-out-carbon-20261123-002": "cancelled to free a test slot for the live dashboard"}  # status.json still says running
CAUSES = [  # (regex over the run's status error and its job logs, cause); every failed run must match one
    (r"Disk quota exceeded", "netscratch disk quota exceeded"),
    (r"pre-action audit reservation", "runtime error: audit event over its reservation"),
    (r"podman', '(inspect|volume)", "podman command timed out at start (node)"),
]
ROOTS = B.RUN_ROOTS


def _new_runs():
    out = {}
    for root in ROOTS:
        for p in sorted(root.iterdir()):
            m = re.search(r"-(\d{8})-\d{3}$", p.name)
            if m and m.group(1) >= SINCE and p.name not in REPORTED:
                assert p.name not in out, p.name
                out[p.name] = p
    return out


def _state(run, path):
    """(state, cause, run-clock minutes, submissions) from the run's own records and its Slurm job logs."""
    s = json.loads((path / "status.json").read_text())
    prog = path / "progress.jsonl"
    lines = prog.read_text().splitlines() if prog.is_file() else []
    minutes = json.loads(lines[-1])["wall_elapsed"] / 60 if lines else 0.0
    subs = len(list(path.glob("submission-*.zip")))
    logs = " ".join(f.read_text(errors="replace") for f in (B.MAIN / "validation").glob(f"experiment-strict-*-{run}.log"))
    if s["status"] == "prepared":
        sib = re.sub(r"-001$", "-002", run)
        return "never started", ("superseded by " + sib) if sib != run and sib in NEW else "queued", minutes, subs
    if run in CANCELLED:
        return "cancelled", CANCELLED[run], minutes, subs
    if "Disk quota exceeded" in logs or "Disk quota exceeded" in str(s.get("error")):
        return "failed", "netscratch disk quota exceeded", minutes, subs
    if s["status"] in ("completed", "running"):
        return s["status"], s.get("result", {}).get("reason", ""), minutes, subs
    assert s["status"] == "failed", (run, s["status"])
    if s.get("error") == "Agent exited before completing protocol":
        held = json.loads((path / "termination.json").read_text()).get("held_seconds") or 0
        return "failed", ("Google outages outlasted the 4 h hold allowance" if held >= 14000 else "Google call timeouts (no outage hold)"), minutes, subs
    hits = [c for rule, c in CAUSES if re.search(rule, str(s.get("error")))]
    assert len(hits) == 1, (run, s.get("error"))
    return "failed", hits[0], minutes, subs


NEW = _new_runs()
STATE = {r: _state(r, p) for r, p in NEW.items()}
BATCH_OF = {}
for r in NEW:
    hit = [b for b in BATCHES if re.search(b[0], r)]
    assert len(hit) == 1, (r, [b[1] for b in hit])
    BATCH_OF[r] = hit[0]
MODEL_OF = {r: MODEL_NAME[json.loads((p / "config-private.json").read_text())["inference"]["model"]] for r, p in NEW.items()}
AS_OF = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ---------------- scored runs and their comparison runs ----------------
CARBON = {"w22-cloth-holes": ["w22-cloth-holes-carbon-20261123-001"], "w11-tetra-spring-k": ["w11-tetra-spring-k-carbon-20261123-001"]}
W26_MJCF = ["w26-cube-spring-block-mjcf-20261123-001", "w26-cube-spring-block-mjcf-20261123-002"]
CUBE_ASTRA = [B.POINT[("cube", "block")]]
TRUNCATED = {r: STATE[r][1] for r in CARBON["w11-tetra-spring-k"] + W26_MJCF}
for r in [x for v in CARBON.values() for x in v] + W26_MJCF:
    assert STATE[r][3] >= 1 and (STATE[r][0] == "completed" or r in TRUNCATED), (r, STATE[r])
RUNS = {(lv, m): G.RUNS[(lv, m)] for lv in CARBON for m in G.MODEL_ORDER} | {(lv, "Gemini 4 Carbon"): CARBON[lv] for lv in CARBON} | \
       {("w26", "gpt-6-astra"): CUBE_ASTRA, ("w26", "Gemini 4 Neon"): W26_MJCF}
MODEL_ORDER = [*G.MODEL_ORDER, "Gemini 4 Carbon"]
for r in CUBE_ASTRA:
    MODEL_OF[r] = "gpt-6-astra"
for (lv, model), runs in RUNS.items():
    for r in runs:
        MODEL_OF.setdefault(r, model)
        assert MODEL_OF[r] == model, (r, MODEL_OF[r], model)
SCENE = {"w22-cloth-holes": G.GEM_LEVELS["w22-cloth-holes"][0], "w11-tetra-spring-k": G.GEM_LEVELS["w11-tetra-spring-k"][0],
         "w26": "Cube on a spring, block mass stated"}
NAMES = {}
for (lv, model), runs in RUNS.items():
    for i, r in enumerate(runs):
        NAMES[r] = (SCENE[lv], f"{model}, run {i + 1}" + (", truncated" if r in TRUNCATED else "") + (", wave 26 MJCF arm" if r in W26_MJCF else ""))
ORDER = list(NAMES)

# ---------------- inputs ----------------
SUMMARIES = tuple(((f[0] if isinstance(f, tuple) else f), set(NAMES)) for f in B.SUMMARIES) + \
            ((B.MAIN / "validation" / "results-gemini-carbon-20261123.json", set(NAMES)), (B.MAIN / "validation" / "results-w26-mjcf-20261123.json", set(NAMES)))
INVALID, CONTINUATIONS, COMPARISONS, NOTE_FLAGS, EXAMPLES = set(), {}, {}, {}, {}
MODELS = B.MODELS + (WORK / "models.json",)
CHECKPOINTS = G.CHECKPOINTS + (WORK / "checkpoints-cube-astra.json", WORK / "checkpoints-cost.json")  # cube astra: its entry from the cumulative report (no cost)
TRACES, CHAT = B.TRACES + (WORK / "traces",), B.CHAT + (WORK / "chat",)
VENDOR, TOOLS = B.VENDOR, B.TOOLS
EDITS = WORK / "edits.json"
estimates, extra_metrics, METRICS = B.estimates, B.extra_metrics, B.METRICS
NEW_EV = WORK / "eval-videos"
EV = {r: B.EV[r] for r in NAMES if r in B.EV} | {r: NEW_EV for r in NAMES if (NEW_EV / r / "clips.json").is_file()}
CLIPS = {r: json.loads((EV[r] / r / "clips.json").read_text()) for r in EV}
assert all(r in CLIPS for r in CARBON["w22-cloth-holes"]), "render-eval-videos.py on the Carbon cloth run first"
FIGURES = {f"clips/{c['file']}": str(EV[r] / c["file"]) for r, cs in CLIPS.items() for c in cs}
SCALE = B.SCALE | {x["run"]: x for f in sorted((WORK / "spring-scale").glob("*.json")) for x in json.loads(f.read_text())}
RECORDINGS = B.recordings(NAMES)
VIEWER = {}

KEYS = {"w22-cloth-holes": ["stiff_rel", "drop_chamfer_mm", "rest_chamfer_mm"],
        "w11-tetra-spring-k": ["com_mm", "chamfer_mm", "r:mass:cluster", "r:mass:spring_base", "r:mass:spring_top"],
        "w26": ["r:mass:cube", "com_mm", "chamfer_mm", "r:stiffness", "r:mass:block"]}
LABEL = {**B.GEM_LABEL, "r:mass:cube": "cube mass ÷ true"}
COLORS = {**G.COLORS, "Gemini 4 Carbon": "--fourth"}

META = dict(title="New sim2sim experiments since 17 November 2026: status and first results",
            subtitle=L(f"Every strict run started since 2026-11-17 that is not in the 2026-11-24 reports · status as of {AS_OF} · scores for the runs "
                       "that submitted"),
            intro=L("<h2 style='border:0;margin-top:20px'>What this covers</h2>"
                    "<p>Each run hides a MuJoCo scene and asks an LLM agent, alone in an isolated container, to rebuild it (or, in the predictor "
                    "waves, to predict its outcomes); a private evaluator scores every submission. This page lists every strict run started since "
                    "17 November that the two 2026-11-24 reports do not include, and scores the ones that submitted. Most of these batches are still "
                    "queued or running, so the scored part is small: Gemini 4 Carbon on two levels and the full-MJCF arm of wave 26. Runs that ended "
                    "early on infrastructure (Google outages, a cancelled job) are marked <b>truncated</b>: their scores are real but they stopped "
                    "before their wall budget, so they are compared at equal minutes and tokens as well as at their last submission.</p>"))
GLOSSARY = [
    ("Truncated", "A run that ended early for an infrastructure reason (Google outages longer than the hold allowance), read up to its last "
                  "submission. It is rerun with a longer hold allowance; the rerun replaces it when it finishes."),
    ("Final / at equal minutes / at equal tokens", "The last scored submission; the last one by the comparison's common run-clock minute; the last one "
                                                   "before the run had used the comparison's common number of model tokens. The common budget is the "
                                                   "smallest run total on the level."),
    ("Error ÷ astra (×k)", "A model's mean error on a level divided by astra's mean (×1 = same, below 1 = better)."),
    ("Lift-force error", "Cloth: the evaluator lifts the cloth at 16 grid points; mean absolute force error ÷ mean true force."),
    ("Chamfer, COM error", "Symmetric surface distance between true and reconstructed shapes (mm); distance between their centres of mass (mm)."),
    ("Mass ÷ true", "A body's reconstructed mass divided by its true mass (1 = exact)."),
    ("Provider hold", "Google-side outages are held outside the run clock (the agent never sees them) up to an allowance: 4 h for the "
                      "2026-11-23 launch, 36 h for later runs. Past the allowance, failed calls reach the harness and the run dies."),
]


def strategy(ctx):
    return dict(answer=L("Too few new runs have submitted for a strategy comparison; the run pages have each conversation and recording."))


def methods(ctx):
    return [
        "<b>Strict runs only</b>: isolated container, no access to scene files, model called through its API; asserted by the builder.",
        "<b>New runs</b>: every run directory dated 2026-11-17 or later in the three experiment roots that is not in the two 2026-11-24 reports. "
        "States come from each run's own status, progress and termination records and its Slurm job log (a job killed by the disk quota could "
        "not update its status file).",
        "<b>Same configurations</b>: the Carbon configs equal the Gemini batch's (sources byte-identical); the wave-26 MJCF arm is the w20 "
        "cube-on-a-spring config on Neon. Every scored run used <code>reasoning_effort = \"medium\"</code> (asserted).",
        "<b>Dollars</b> are not compared: Carbon is host-priced at Neon's list prices as a placeholder.",
        "<b>Few runs</b>: one or two per model per level, and astra's own replicates differ by up to ×4, so these are first observations, not findings.",
    ]


def sections(ctx):
    R = ctx.runs
    blk = lambda **kw: kw
    lnk = lambda r, t=None: f"<a href='run.html?run={r}'>{t or R[r]['condition']}</a>"
    x = lambda v, d=3: "—" if v is None else ctx.x(v, d)
    mm = lambda v, d=3: "—" if v is None else ctx.mm(v, d)
    pct = lambda v: "—" if v is None else "exact" if v < 1e-3 else f"{100 * v:.0f}%"
    fmt = lambda k, v: "—" if v is None else x(v) if k.startswith("r:") else mm(v) if k.endswith("_mm") else pct(v) if k.endswith("_rel") else f"{v:.3g}"
    hl = lambda txt, good: f"<span class='hl {'good' if good else 'bad'}'>{txt}</span>"
    counters = {}

    def cap(kind, text):
        counters[kind] = counters.get(kind, 0) + 1
        return blk(type="html", html=f"<p class='caption'><b>{kind} {counters[kind]}.</b> {text}</p>")

    lead = lambda text: blk(type="html", html=f"<p>{text}</p>")
    err = lambda k, v: None if v is None else abs(v - 1) if k.startswith("r:") else v
    avg = lambda vals: None if not [v for v in vals if v is not None] else float(np.mean([v for v in vals if v is not None]))
    point = lambda r, k, cut: next((u["m"].get(k) for u in reversed(R[r]["subs"]) if cut(u) and u["m"].get(k) is not None), None)
    final = lambda r, k: point(r, k, lambda u: True)
    effort = {r: json.loads((B.RUN(r) / "config-private.json").read_text())["inference"].get("reasoning_effort")
              for (lv, m), runs in RUNS.items() if m != "gpt-6-astra" for r in runs}
    assert set(effort.values()) == {"medium"}, effort
    ended = lambda r: f"truncated: {TRUNCATED[r]}" if r in TRUNCATED else "finished" if R[r]["reason"] == "agent_finish" else \
        "cost limit" if R[r]["reason"] == "budget_suspended" else R[r]["reason"]

    # ---- 1 · status of every new run
    states = ["completed", "running", "failed", "cancelled", "never started"]
    by_batch = {b[1]: [r for r in NEW if BATCH_OF[r][1] == b[1]] for b in BATCHES}
    count = lambda runs, st: sum(STATE[r][0] == st for r in runs)
    scored = {r for runs in RUNS.values() for r in runs if r in NEW}
    rows = [[f"<b>{name}</b><div class='muted small'>{q}</div>", ", ".join(sorted({MODEL_OF[r] for r in by_batch[name]})), str(len(by_batch[name]))] +
            [str(count(by_batch[name], st) or "") for st in states] + [str(sum(STATE[r][3] for r in by_batch[name]) or "")]
            for _, name, q in BATCHES]
    fails = {}
    for r in NEW:
        if STATE[r][0] == "failed":
            fails[STATE[r][1]] = fails.get(STATE[r][1], 0) + 1
    n_sub = len(scored)
    batch1 = sum(1 for r in NEW if STATE[r][3] and BATCH_OF[r] is BATCHES[1])
    old = sorted(r for r in REPORTED if re.search(r"-(\d{8})-\d{3}$", r).group(1) >= SINCE)
    sec = [dict(id="status", nav="Status", tag=f"{len(NEW)} new runs · as of {AS_OF}",
        title=f"1 · {len(NEW)} new runs since 17 November: {count(NEW, 'completed')} completed, {count(NEW, 'running')} running, "
              f"{count(NEW, 'failed')} failed on infrastructure, {n_sub} with a scored submission",
        q=None, answer=L(f"Since 17 November, {len(NEW) + len(old)} strict runs were started. {len(old)} of them (astra replicates of 17–18 November "
                         f"and the Gemini Neon/Hydrogen batch of 19–20 November) are already in the 2026-11-24 reports. Of the other {len(NEW)}, only "
                         f"{count(NEW, 'completed')} completed: Carbon on cloth-holes. {count(NEW, 'running')} are running, "
                         f"{count(NEW, 'never started')} were prepared but never started (most superseded by a re-prepared copy), "
                         f"{count(NEW, 'cancelled')} were cancelled, and {count(NEW, 'failed')} failed, none for a reason the agent caused: " +
                         "; ".join(f"{v} × {k}" for k, v in sorted(fails.items(), key=lambda kv: -kv[1])) +
                         f". {n_sub} runs reached a scored submission; sections 2–4 compare them with earlier runs of the same levels ({batch1} batch-1 runs "
                         "also submitted before dying and stay excluded, as in the Gemini report). "
                         "Waves 23 (fixture), 24 (die), 25 (touch harness) and 29 (low-rank prompt) have no scored submission yet."),
        blocks=[cap("Table", L("<b>New runs by batch</b> (state from each run's records and Slurm job log; appendix A1 lists every run).")),
                blk(type="table", head=["batch and question", "model", "runs"] + states + ["submissions"], rows=rows)])]

    # ---- 2, 3 · scored levels
    def level(lv, title, tag, text, new):
        text = text.format(*(f"{R[r]['wall_min']:.0f}" for r in new))  # run-clock minutes of the level's new runs
        keys = KEYS[lv]
        key = keys[0]
        members = [r for m in MODEL_ORDER for r in RUNS.get((lv, m), [])]
        budget = {"t": min(R[r]["wall_min"] for r in members), "mtok": min(R[r]["final_mtok"] for r in members)}
        at = lambda r, k, a: point(r, k, lambda u: u[a] <= budget[a])
        rows = [[m if r == RUNS[(lv, m)][0] else "", lnk(r, f"run {RUNS[(lv, m)].index(r) + 1}"), ended(r), f"{R[r]['wall_min']:.0f}",
                 f"{R[r]['final_mtok']:.1f}", str(R[r]["submissions"]), f"<b>{fmt(key, final(r, key))}</b>", fmt(key, at(r, key, "t")),
                 fmt(key, at(r, key, "mtok"))] + [fmt(k, final(r, k)) for k in keys[1:]]
                for m in MODEL_ORDER for r in RUNS.get((lv, m), [])]
        astra = avg([err(key, final(r, key)) for r in RUNS[(lv, "gpt-6-astra")]])
        ratio = {m: (avg([err(key, final(r, key)) for r in RUNS[(lv, m)]]) or 0) / astra if astra else None
                 for m in MODEL_ORDER[1:] if (lv, m) in RUNS and avg([err(key, final(r, key)) for r in RUNS[(lv, m)]]) is not None}
        series = [[RUNS[(lv, "gpt-6-astra")] if len(RUNS[(lv, "gpt-6-astra")]) > 1 else RUNS[(lv, "gpt-6-astra")][0], "gpt-6-astra", {"color": COLORS["gpt-6-astra"]}]] + \
                 [[r, NAMES[r][1], {"color": COLORS[m], "dash": "6 3" if r in TRUNCATED else ""}] for m in MODEL_ORDER[1:] for r in RUNS.get((lv, m), [])]
        conv = lambda **kw: blk(type="convergence", keys=keys[:3], series=[dict(**({"runs": s[0]} if isinstance(s[0], list) else {"run": s[0]}), label=s[1], **s[2])
                                                                          for s in series], **kw)
        blocks = [lead(text),
                  cap("Table", L(f"<b>Every run on this level.</b> Primary error ({LABEL.get(key, key)}) at the last submission and at equal run-clock "
                                 f"minutes ({budget['t']:.0f}) and equal model tokens ({budget['mtok']:.1f}M), the smallest run totals here; other scores at "
                                 "the last submission. Lower is better; masses and stiffness are estimate ÷ true.")),
                  blk(type="table", head=["model", "run", "ended", "minutes", "tokens (M)", "submissions", f"{LABEL.get(key, key)}: last",
                                          f"at {budget['t']:.0f} min", f"at {budget['mtok']:.1f}M tokens"] + [G.ctx_label(k) for k in keys[1:]], rows=rows),
                  cap("Figure", "<b>Error against run-clock minutes and cumulative model tokens</b> (astra: mean and range over its runs; dashed = truncated)."),
                  conv(title="By run-clock minutes", xmax=360), conv(title="By cumulative model tokens", x="mtok")]
        body = {"w11-tetra-spring-k": "cluster", "w26": "cube"}.get(lv)
        if body:
            blocks += [cap("Table", L("<b>Did the agent use the spring as a scale?</b> From every 10th recorded state of the hidden scene: when the spring "
                                      f"first sat in the hole, and when the {body} first rested on it there.")),
                       blk(type="table", head=["run", "spring put in the hole", f"{body} weighed on it in the hole", f"{body} mass ÷ true (last)"],
                           rows=[[lnk(r, NAMES[r][1])] + [("never" if SCALE[r]["first_wall_s"].get(k) is None else f"min {SCALE[r]['first_wall_s'][k] / 60:.0f}")
                                                          for k in ("spring_in_hole", f"{body}_weighed")] + [x(final(r, f"r:mass:{body}"))] for r in members])]
        clip_runs = [r for r in members if r in CLIPS]
        if clip_runs:
            blocks += [cap("Figure", "<b>The scored tests on each run's final model</b>, hidden scene (left) against the model (right). Pick a run and test."),
                       blk(type="clips", title=None, items=[dict(src=f"clips/{c['file']}", label=f"{NAMES[r][1]} · {c['clip']}", run=r)
                                                            for r in clip_runs for c in CLIPS[r] if c["family"] != "cloth-still"])]
            stills = [(r, c) for r in clip_runs for c in CLIPS[r] if c["family"] == "cloth-still"]
            if stills:
                blocks += [cap("Figure", "<b>Cloth at rest, stiffness field and lift forces</b>: truth (left) against each final model (right)."),
                           blk(type="images", items=[dict(src=f"clips/{c['file']}", label=f"{NAMES[r][1]}: {c['clip']}", run=r) for r, c in stills])]
        head = (": " + ", ".join(f"{m} ×{v:.2g}" for m, v in ratio.items() if v is not None) + " astra's error (last submission)") if astra is not None and astra >= 1e-3 else \
            (f": astra {LABEL.get(key, key)} {fmt(key, final(RUNS[(lv, 'gpt-6-astra')][0], key))}; " +  # astra exact: a ratio to it is meaningless
             ", ".join(f"{m} run {RUNS[(lv, m)].index(r) + 1} {fmt(key, final(r, key))}" for m in MODEL_ORDER[1:] for r in RUNS.get((lv, m), [])))
        return dict(id=lv, nav=SCENE[lv], tag=tag, q=None, answer=None, blocks=blocks, title=title + head)

    sec.append(level("w22-cloth-holes", "2 · Gemini 4 Carbon, cloth with holes and cuts", "the only completed new run",
                     G.LEVEL_TEXT["w22-cloth-holes"] + " Carbon finished on its own at run-clock minute {} of 360 with a single submission.",
                     CARBON["w22-cloth-holes"]))
    sec.append(level("w11-tetra-spring-k", "3 · Gemini 4 Carbon, spring level with stated stiffness", "Carbon run truncated before its 240-minute budget",
                     G.LEVEL_TEXT["w11-tetra-spring-k"] + " The Carbon run died at run-clock minute {} of 240, when Google's outages outlasted the 4 h hold "
                     "allowance; it had submitted once. Its rerun (36 h allowance) is running.", CARBON["w11-tetra-spring-k"]))
    sec.append(level("w26", "4 · Wave 26, full-MJCF arm (Gemini 4 Neon), cube on a spring", "both runs truncated; the low-rank arm has no submission",
                     "A cube and a known-mass block lie next to a spring that fits a hole; point grasp, no force reading, camera. The MJCF arm builds a "
                     "full MuJoCo scene, as in wave 20. Both 2026-11-23 runs died when Google's outages outlasted the 4 h hold allowance, at run-clock "
                     "minutes {} and {}, each after one submission; the low-rank arm's two runs died before submitting, so the wave's question "
                     "(low-rank answer vs MJCF) cannot be answered yet. All four are rerunning with a 36 h allowance. Shown against astra's wave-20 run.", W26_MJCF))

    # ---- appendices
    order = {st: i for i, st in enumerate(states)}
    status_rows = [[f"<a href='run.html?run={r}'>{r}</a>" if r in NAMES else r, BATCH_OF[r][1].split(":")[0], MODEL_OF[r], STATE[r][0], STATE[r][1],
                    f"{STATE[r][2]:.0f}", str(STATE[r][3])]
                   for r in sorted(NEW, key=lambda r: ([b[1] for b in BATCHES].index(BATCH_OF[r][1]), order[STATE[r][0]], r))]
    sec.append(dict(id="all-new", nav="A1", appendix=True, tag=None, title=f"A1 · Every new run ({len(NEW)})", q=None, answer=None,
                    blocks=[lead(f"State as of {AS_OF}. Minutes are run-clock minutes (provider holds excluded) at the run's last record."),
                            blk(type="table", head=["run", "batch", "model", "state", "detail", "minutes", "submissions"], rows=status_rows)]))
    sec.append(dict(id="runs", nav="A2", appendix=True, tag=None, title="A2 · Scored and comparison runs", q=None, answer=None,
                    blocks=[lead("Click a name for its page: conversation, images the model received and a scrubbable recording of the run."), blk(type="runs")]))
    return sec
