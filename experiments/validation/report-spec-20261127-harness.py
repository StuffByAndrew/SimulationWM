"""Report data spec: wave 25 touch harness (tools.local_controller + pausable bash) vs control (the same Shadow-hand config without them),
Gemini 4 Neon, four hand levels, the -20261127-001 runs (the -20261125 attempts were cancelled for the first key's quota and are discarded).
Scored exactly as wave 21's hand levels (report-spec-20261114.family). Built for ablation-pairs-plot.py (ablation-configs/harness-ablation.json).
Build: .venv/bin/python experiments/validation/build-results-report.py data experiments/validation/report-spec-20261127-harness.py \\
    experiments/reports/report-work-20261127-harness experiments/reports/results-harness-20261127"""
import importlib.util
from pathlib import Path

VAL = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("cumulative_spec", VAL / "report-spec-20261114.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
WORK = B.MAIN / "reports" / "report-work-20261127-harness"
LEVELS = ["prism", "prism-offset-com", "smooth-mesh", "spiky"]
RUNS = {(lv, cond): f"w25-hand-{lv}{'' if cond == 'harness' else '-control'}-20261127-001" for lv in LEVELS for cond in ("harness", "control")}
for (lv, cond), r in RUNS.items():  # the factor, asserted from each private config
    c = B.json.loads((B.RUN(r) / "config-private.json").read_text())
    assert bool((c.get("tools") or {}).get("local_controller")) == (cond == "harness"), r
    assert ("pausable" in c["sandbox"].get("command", [])) == (cond == "harness"), r
NAMES = {r: (f"Shadow hand: {B.HAND_NAME.get(lv, lv)}", "touch harness" if cond == "harness" else "control") for (lv, cond), r in RUNS.items()}
ORDER = list(NAMES)
SUMMARIES = ((B.MAIN / "validation" / "results-w25-harness-20261127.json", set(NAMES)),)
INVALID, CONTINUATIONS, COMPARISONS, NOTE_FLAGS, EXAMPLES, VIEWER, FIGURES = set(), {}, {}, {}, {}, {}, {}
MODELS, CHECKPOINTS = (WORK / "models.json",), (WORK / "checkpoints-cost.json",)
TRACES, CHAT, EDITS = (WORK / "traces",), (WORK / "chat",), WORK / "edits.json"
VENDOR, TOOLS = B.VENDOR, B.TOOLS
estimates, extra_metrics, METRICS = B.estimates, B.extra_metrics, B.METRICS
RECORDINGS = B.recordings(NAMES)
META, GLOSSARY = dict(title="Wave 25: touch harness vs control", subtitle="", intro=""), []


def sections(ctx):
    return [dict(id="runs", nav="Runs", tag=None, title="Runs", q=None, answer=None, blocks=[dict(type="runs")])]


def strategy(ctx):
    return dict(answer="")


def methods(ctx):
    return []
