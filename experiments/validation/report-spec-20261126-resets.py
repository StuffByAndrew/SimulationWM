"""Report data spec: the six wave-31 no-reset runs (Gemini 4 Neon, resets disabled), scored exactly as their levels' original waves
(report-spec-20261114.family). Built for ablation-pairs-plot.py (ablation-configs/reset-ablation.json), whose baselines come from the Gemini and
new-runs reports' data. Build: .venv/bin/python experiments/validation/build-results-report.py data experiments/validation/report-spec-20261126-resets.py \\
    experiments/reports/report-work-20261126-resets experiments/reports/results-resets-20261126"""
import importlib.util
from pathlib import Path

VAL = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("cumulative_spec", VAL / "report-spec-20261114.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
WORK = B.MAIN / "reports" / "report-work-20261126-resets"
NORESET = {"latch-out": "w31-latch-out-noreset-20261125-003", "hand-smooth-mesh": "w31-hand-smooth-mesh-noreset-20261125-002",
           "cloth-holes": "w31-cloth-holes-noreset-20261125-003", "tetra-spring-k": "w31-tetra-spring-k-noreset-20261125-002",
           "marble-heavy-hint": "w31-marble-heavy-hint-noreset-20261125-003", "cube-spring-block-mjcf": "w31-cube-spring-block-mjcf-noreset-20261125-002"}
assert set(NORESET) == set(B.ABLATION_BASE)
for r in NORESET.values():
    assert B.json.loads((B.RUN(r) / "config-private.json").read_text())["simulation"]["resets"] is False, r
NAMES = {r: (lv, "Gemini 4 Neon, no resets") for lv, r in NORESET.items()}
ORDER = list(NAMES)
SUMMARIES = ((B.MAIN / "validation" / "results-w31-noreset-20261125.json", set(NAMES)),)
INVALID, CONTINUATIONS, COMPARISONS, NOTE_FLAGS, EXAMPLES, VIEWER, FIGURES = set(), {}, {}, {}, {}, {}, {}
MODELS, CHECKPOINTS = (WORK / "models.json",), (WORK / "checkpoints-cost.json",)
TRACES, CHAT, EDITS = (WORK / "traces",), (WORK / "chat",), WORK / "edits.json"
VENDOR, TOOLS = B.VENDOR, B.TOOLS
estimates, extra_metrics, METRICS = B.estimates, B.extra_metrics, B.METRICS
RECORDINGS = B.recordings(NAMES)
META, GLOSSARY = dict(title="Wave 31: no resets", subtitle="", intro=""), []


def sections(ctx):
    return [dict(id="runs", nav="Runs", tag=None, title="Runs", q=None, answer=None, blocks=[dict(type="runs")])]


def strategy(ctx):
    return dict(answer="")


def methods(ctx):
    return []
