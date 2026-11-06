"""Report spec: cumulative sim2sim results through 2026-11-04, waves 1-9 (every finished, valid strict-sandbox run), written for a reader new
to the project: the runs of report-spec-20261104b.py plus wave 9 (traces: report-work-plain/traces), with plain-language names, an intro (setup, scenes, conditions) and no internal jargon.
Build (login node is fine; JSON only):
  .venv/bin/python experiments/validation/build-results-report.py experiments/validation/report-spec-20261104-plain.py \
      experiments/reports/report-work-20261104 experiments/reports/results-report-20261104-plain
(Traces of the wave 7/8 runs come from scan.sh with RUNS=runs-new-b.txt.)
Work dir inputs (compute node): traces/ from report-scan-audit.py on runs-new.txt (old traces are read from the 11-03 work
dir) and checkpoints-old.json from checkpoint-index.py on runs-old-cp.txt. Every number in the text comes from ctx."""
import json, math, re
from pathlib import Path

VAL = Path(__file__).resolve().parent
OLDWORK = Path("/n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-experiments/report-work-20261103")
WORK = VAL.parent / "reports" / "report-work-20261104"
SUMMARIES = ("results-wave1-003.json", "results-wave2.json", "results-wave3a.json", "results-wave3c.json", "results-continuations.json",
             "results-rerunA-w5.json", "results-w5-spring-001.json", "results-rerunB-w3.json", "results-rerunB-w4.json", "results-wave6.json",
             "results-wave7.json", "results-wave8.json", "results-wave9.json", "results-wave10.json", "results-rep-r2.json", "results-rep-r3.json")
INVALID = {"w2-lever-hard-base-20261103-001"}  # blocked attempt, relaunched as -002
MODELS = (OLDWORK / "models.json", VAL / "rerun-analysis" / "models.json", VAL / "rerun-analysis" / "models-B.json",
          VAL / "rerun-analysis" / "models-C.json", VAL / "rerun-analysis" / "models-D.json", VAL / "rerun-analysis" / "models-E.json",
          VAL / "rerun-analysis-plain" / "models-F.json", VAL / "rerun-analysis-plain" / "models-G.json")
CHECKPOINTS = (VAL / "rerun-analysis" / "checkpoints.json", VAL / "rerun-analysis" / "checkpoints-B.json",
               VAL / "rerun-analysis" / "checkpoints-C.json", VAL / "rerun-analysis" / "checkpoints-D.json", VAL / "rerun-analysis" / "checkpoints-E.json",
               VAL / "rerun-analysis-plain" / "checkpoints-F.json", VAL / "rerun-analysis-plain" / "checkpoints-G.json", WORK / "checkpoints-old.json")
TRACES = (OLDWORK / "traces", WORK / "traces", VAL.parent / "reports" / "report-work-plain" / "traces")  # last: waves 9-10, repeats (report-scan-audit.py)
REPEATS = {  # repeat run of a key condition -> the original run (same prepared scene, same config except the cost pool)
    "w6-blocks-hard-nowrench-r2-20261105-001": "w6-blocks-hard-nowrench-20261104-001", "w6-spring-hard-base-r2-20261104-001": "w6-spring-hard-base-20261104-001",
    "w6-spring-hard-nowrench-mass-r2-20261105-001": "w6-spring-hard-nowrench-mass-20261104-001", "w6-spring-hard-nowrench-r2-20261105-001": "w6-spring-hard-nowrench-20261104-001",
    "w7-blocks-hard-base-r2-20261105-001": "w7-blocks-hard-base-20261104-001", "w7-lever-hard-nowrench-r2-20261104-001": "w7-lever-hard-nowrench-20261104-001",
    "w8-lever-hard-base-r2-20261104-001": "w8-lever-hard-base-20261104-001", "w8-lever-hard-nowrench-hint-r2-20261105-001": "w8-lever-hard-nowrench-hint-20261104-001",
    "w9-lever-hard-noaction-mass-hint-r2-20261105-001": "w9-lever-hard-noaction-mass-hint-20261104-001",
    "w9-lever-hard-nowrench-mass-hint-r2-20261105-001": "w9-lever-hard-nowrench-mass-hint-20261104-001"}
REPEATS.update({re.sub(r"-r2-\d{8}-001$", "-r3-20261106-001", r): o for r, o in list(REPEATS.items())})  # third runs, 2026-11-06
REPEAT_N = lambda r: int(re.search(r"-r(\d)-\d{8}-\d{3}$", r).group(1))
REPS = {o: sorted((r for r, oo in REPEATS.items() if oo == o), key=REPEAT_N) for o in set(REPEATS.values())}  # original -> [repeat 2, repeat 3]
EDITS = VAL.parent / "reports" / "report-work-plain" / "edits.json"  # text edits from the report's edit mode (serve-report.py)
CHAT = VAL.parent / "reports" / "report-work-plain" / "chat"  # report-chat-extract.py output, one directory per run
RUNROOT = Path("/n/home00/aejxie/sim2sim/experiments/runs")
EXPERT_RUNS = ["w10-assembly-base-20261105-001", "w10-tetra-base-20261105-001", "w10-boat-base-20261105-001"]


def final_evaluation(run):
    """Private evaluation of the run's final submission: scores and the expert side-by-side video (reference left, candidate right)."""
    d = RUNROOT / run
    last = json.loads((d / "metric-history.json").read_text())["checkpoints"][-1]
    v = d / f"version-{last['submission']:03d}"
    e = json.loads((v / "evaluation.json").read_text())
    assert e["version"] == last["version"], run
    (video,) = v.glob(f"evaluation/artifacts/{e['version']}/expert_000/side_by_side.mp4")
    return e["scores"], video


EXPERT = {r: final_evaluation(r) for r in EXPERT_RUNS}
EXTRA_VIDEOS = {f"expert-{r}.mp4": str(v) for r, (_, v) in EXPERT.items()}
VIDEOS = "/n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-experiments/results-viewer-20261103"  # new runs have record_video=false
VENDOR = OLDWORK / "vendor"
CONTINUATIONS = {"c1-assembly-base-20261103-002": "w1-assembly-base-20261102-003", "c1-assembly-depth-20261103-002": "w1-assembly-depth-20261102-003"}
TOOLS = {"segmentation": r"/opt/public/tools/segmentation|segment\.py|mobile_?sam", "planning": r"/opt/public/tools/planning",
         "planning_ran": r"planning/plan\.py\s+\S|from plan import"}
NOTE_FLAGS = {"scale": r"(mass|scale)\b[^.]{0,80}\b(not|cannot|can't|un)\s*(be\s+)?(identif|determin)|unidentifiable|not identifiable|cannot determine it"}
EXAMPLES = json.loads((VAL / "report-taxonomy-examples.json").read_text())

L = lambda s: s  # readability marker for long display strings
RGB3 = ["w1-assembly-base-20261102-003", "w3-assembly-toolsimg-base-20261103-001", "w3-assembly-seg-20261103-001"]
NAMES = {  # scene · condition. Run IDs appear only in the traceability table.
    "w1-assembly-base-20261102-003": ("Random assembly", "baseline, run A"),
    "w3-assembly-toolsimg-base-20261103-001": ("Random assembly", "baseline, run B"),
    "w3-assembly-seg-20261103-001": ("Random assembly", "baseline, run C (segmentation tool installed, not mentioned)"),
    "w1-assembly-depth-20261102-003": ("Random assembly", "camera + depth"),
    "w2-assembly-allangles-20261103-001": ("Random assembly", "task asks to match every viewing angle"),
    "c1-assembly-base-20261103-002": ("Random assembly", "baseline run A, extended +2 h"),
    "c1-assembly-depth-20261103-002": ("Random assembly", "camera + depth, extended +2 h"),
    "w5-assembly-noaction-20261103-002": ("Random assembly", "no-interaction"),
    "w3-assembly-nowrench-20261104-002": ("Random assembly", "no force reading"),
    "w4-assembly-seg-directed-20261104-002": ("Random assembly", "segmentation tool, task says when to use it"),
    "w1-tetra-base-20261102-003": ("Tetrahedral cluster", "colored spheres (baseline)"),
    "w2-tetra-gray-20261103-002": ("Tetrahedral cluster", "all spheres gray"),
    "w5-tetra-noaction-20261103-002": ("Tetrahedral cluster", "no-interaction"),
    "w3-tetra-nowrench-20261104-002": ("Tetrahedral cluster", "no force reading"),
    "w3-boat-base-20261104-002": ("Boat", "baseline"),
    "w3-boat-depth-20261104-002": ("Boat", "RGB + depth"),
    "w5-boat-noaction-20261103-002": ("Boat", "no-interaction"),
    "w3-ps-assembly-fixed-20261104-002": ("Small Panda + small assembly", "fixed camera"),
    "w3-ps-assembly-wrist-20261104-002": ("Small Panda + small assembly", "fixed + wrist camera"),
    "w1-blocks-base-20261102-003": ("Blocks", "baseline"),
    "w2-blocks-hard-base-20261103-001": ("Randomized blocks", "baseline"),
    "w2-blocks-hard-nowrench-20261103-001": ("Randomized blocks", "no force reading"),
    "w5-blocks-hard-noaction-20261103-002": ("Randomized blocks", "no-interaction"),
    "w7-blocks-hard-base-20261104-001": ("Randomized blocks, stop-when-done prompt", "baseline"),
    "w6-blocks-hard-nowrench-20261104-001": ("Randomized blocks, stop-when-done prompt", "no force reading"),
    "w6-blocks-hard-nowrench-mass-20261104-001": ("Randomized blocks, stop-when-done prompt", "no force reading + stated red-block mass"),
    "w1-lever-base-20261102-003": ("Lever", "baseline"),
    "w1-lever-hint-20261102-003": ("Lever", "task names hinge parameters"),
    "w2-lever-hard-base-20261103-002": ("Randomized lever", "baseline"),
    "w2-lever-hard-hint-20261103-001": ("Randomized lever", "task names hinge parameters"),
    "w2-lever-hard-nowrench-20261103-001": ("Randomized lever", "no force reading"),
    "w3-lever-hard-nowrench-hint-20261104-002": ("Randomized lever", "no force reading + task names hinge parameters"),
    "w5-lever-hard-noaction-20261103-002": ("Randomized lever", "no-interaction"),
    "w8-lever-hard-base-20261104-001": ("Randomized lever, stop-when-done prompt", "baseline"),
    "w7-lever-hard-nowrench-20261104-001": ("Randomized lever, stop-when-done prompt", "no force reading"),
    "w7-lever-hard-nowrench-mass-20261104-001": ("Randomized lever, stop-when-done prompt", "no force reading + stated lever mass"),
    "w8-lever-hard-nowrench-hint-20261104-001": ("Randomized lever, stop-when-done prompt", "no force reading + task names hinge parameters"),
    "w9-lever-hard-nowrench-mass-hint-20261104-001": ("Randomized lever, stop-when-done prompt", "no force reading + stated lever mass + task names hinge parameters"),
    "w9-lever-hard-noaction-mass-hint-20261104-001": ("Randomized lever, stop-when-done prompt", "no-interaction + stated lever mass + task names hinge parameters"),
    "w3-panda-lever-hard-20261104-002": ("Panda + randomized lever", "baseline (arm joint torques, no grasp force)"),
    "w4-panda-lever-hard-planning-20261104-002": ("Panda + randomized lever", "planning tool, task says when to use it"),
    "w5-panda-lever-hard-noaction-20261103-002": ("Panda + randomized lever", "no-interaction"),
    "w1-spring-base-20261102-003": ("Spring", "baseline"),
    "w3-spring-hard-base-20261103-001": ("Randomized spring", "baseline"),
    "w3-spring-hard-nowrench-20261103-001": ("Randomized spring", "no force reading"),
    "w5-spring-hard-noaction-20261103-001": ("Randomized spring", "no-interaction"),
    "w6-spring-hard-base-20261104-001": ("Randomized spring, stop-when-done prompt", "baseline"),
    "w6-spring-hard-nowrench-20261104-001": ("Randomized spring, stop-when-done prompt", "no force reading"),
    "w6-spring-hard-nowrench-mass-20261104-001": ("Randomized spring, stop-when-done prompt", "no force reading + stated lever mass"),
    "w7-spring-hard-noaction-20261104-001": ("Randomized spring, stop-when-done prompt", "no-interaction"),
    "w7-spring-hard-noaction-mass-20261104-001": ("Randomized spring, stop-when-done prompt", "no-interaction + stated lever mass"),
    "w1-panda-blind-20261102-003": ("Panda + blocks", "no camera (arm joint feedback only)"),
    "w2-panda-vision-20261103-001": ("Panda + blocks", "camera"),
    "w10-panda-blocks-vision-20261105-001": ("Panda + blocks, stop-when-done prompt", "camera"),
    "w10-panda-blocks-proprio-20261105-001": ("Panda + blocks, stop-when-done prompt", "no camera (arm joint feedback only)"),
    "w10-panda-blocks-tactile-20261105-001": ("Panda + blocks, stop-when-done prompt", "no camera, fingertip touch sensors"),
    "w10-panda-lever-hard-toolsimg-base-20261105-001": ("Panda + randomized lever, stop-when-done prompt", "baseline (planning tool not offered)"),
    "w10-panda-lever-hard-planning-20261105-001": ("Panda + randomized lever, stop-when-done prompt", "planning tool v2, task says when to use it"),
    "w10-assembly-base-20261105-001": ("Random assembly, stop-when-done prompt", "baseline"),
    "w10-tetra-base-20261105-001": ("Tetrahedral cluster, stop-when-done prompt", "baseline"),
    "w10-boat-base-20261105-001": ("Boat, stop-when-done prompt", "baseline"),
}
NAMES.update({r: (NAMES[o][0], NAMES[o][1] + f" · repeat {REPEAT_N(r)}") for r, o in sorted(REPEATS.items(), key=lambda kv: REPEAT_N(kv[0]))})
ORDER = list(NAMES)
VIEWER = {"assembly": RGB3 + ["w1-assembly-depth-20261102-003", "w2-assembly-allangles-20261103-001", "w5-assembly-noaction-20261103-002",
                              "w3-assembly-nowrench-20261104-002", "w4-assembly-seg-directed-20261104-002", "w10-assembly-base-20261105-001"],
          "tetra": ["w1-tetra-base-20261102-003", "w2-tetra-gray-20261103-002", "w5-tetra-noaction-20261103-002", "w3-tetra-nowrench-20261104-002",
                    "w10-tetra-base-20261105-001"],
          "boat": ["w3-boat-base-20261104-002", "w3-boat-depth-20261104-002", "w5-boat-noaction-20261103-002", "w10-boat-base-20261105-001"],
          "ps": ["w3-ps-assembly-fixed-20261104-002", "w3-ps-assembly-wrist-20261104-002"],
          "blocks": ["w2-blocks-hard-base-20261103-001", "w2-blocks-hard-nowrench-20261103-001", "w5-blocks-hard-noaction-20261103-002",
                     "w7-blocks-hard-base-20261104-001", "w6-blocks-hard-nowrench-20261104-001", "w6-blocks-hard-nowrench-mass-20261104-001",
                     "w1-blocks-base-20261102-003", "w7-blocks-hard-base-r2-20261105-001", "w6-blocks-hard-nowrench-r2-20261105-001"],
          "panda": ["w10-panda-blocks-proprio-20261105-001", "w10-panda-blocks-tactile-20261105-001", "w10-panda-blocks-vision-20261105-001",
                    "w1-panda-blind-20261102-003", "w2-panda-vision-20261103-001"]}
PANDA_BLOCKS = ["chamfer_mm", "com_mm", "r:mass:block_red", "r:mass:block_blue", "rgb_mse"]

LEVER = ["r:mass", "r:inertia", "r:damping", "r:frictionloss"]
SPRING = ["r:mass", "r:inertia", "r:damping", "r:stiffness"]
BLOCKS = ["r:mass:block_red", "r:mass:block_blue", "r:massratio"]
GEOM = ["chamfer_mm", "com_mm", "r:mass"]
SHORT = {"r:mass": "mass", "r:inertia": "inertia", "r:damping": "damping", "r:frictionloss": "friction", "r:stiffness": "stiffness",
         "r:mass:block_red": "red", "r:mass:block_blue": "blue"}
c = lambda label, base, abl, keys, bl="baseline", al=None: dict(label=label, base=base if isinstance(base, list) else [base], abl=abl, keys=keys,
                                                                 base_label=bl if not isinstance(base, list) else "3 baseline repeats (mean, range)", abl_label=al or NAMES[abl][1])
COMPARISONS = {
    "oo-lever": c("Randomized lever", "w2-lever-hard-base-20261103-002", "w5-lever-hard-noaction-20261103-002", LEVER),
    "oo-spring": c("Randomized spring", "w3-spring-hard-base-20261103-001", "w5-spring-hard-noaction-20261103-001", SPRING),
    "oo-panda": c("Panda + randomized lever", "w3-panda-lever-hard-20261104-002", "w5-panda-lever-hard-noaction-20261103-002", LEVER),
    "oo-blocks": c("Randomized blocks", "w2-blocks-hard-base-20261103-001", "w5-blocks-hard-noaction-20261103-002", BLOCKS + ["com_mm", "chamfer_mm"]),
    "oo-assembly": c("Random assembly", RGB3, "w5-assembly-noaction-20261103-002", GEOM + ["rgb_mse"]),
    "oo-tetra": c("Tetrahedral cluster", "w1-tetra-base-20261102-003", "w5-tetra-noaction-20261103-002", GEOM + ["sphere_err"]),
    "oo-boat": c("Boat", "w3-boat-base-20261104-002", "w5-boat-noaction-20261103-002", GEOM),
    "nf-lever": c("Randomized lever", "w2-lever-hard-base-20261103-002", "w2-lever-hard-nowrench-20261103-001", LEVER),
    "nf-spring": c("Randomized spring", "w3-spring-hard-base-20261103-001", "w3-spring-hard-nowrench-20261103-001", SPRING),
    "nf-blocks": c("Randomized blocks", "w2-blocks-hard-base-20261103-001", "w2-blocks-hard-nowrench-20261103-001", BLOCKS + ["com_mm", "chamfer_mm"]),
    "nf-assembly": c("Random assembly", RGB3, "w3-assembly-nowrench-20261104-002", GEOM),
    "nf-tetra": c("Tetrahedral cluster", "w1-tetra-base-20261102-003", "w3-tetra-nowrench-20261104-002", GEOM + ["sphere_err"]),
    "np-spring-nf": c("Randomized spring, stop-when-done prompt: removing the force reading", "w6-spring-hard-base-20261104-001", "w6-spring-hard-nowrench-20261104-001", SPRING),
    "np-spring-mass": c("Randomized spring, stop-when-done prompt: stating the lever mass", "w6-spring-hard-nowrench-20261104-001", "w6-spring-hard-nowrench-mass-20261104-001", SPRING,
                        bl="no force reading"),
    "np-blocks-mass": c("Randomized blocks, stop-when-done prompt: stating the red-block mass", "w6-blocks-hard-nowrench-20261104-001", "w6-blocks-hard-nowrench-mass-20261104-001",
                        BLOCKS + ["com_mm", "chamfer_mm"], bl="no force reading"),
    "np-blocks-nf": c("Randomized blocks, stop-when-done prompt: removing the force reading", "w7-blocks-hard-base-20261104-001", "w6-blocks-hard-nowrench-20261104-001",
                      BLOCKS + ["com_mm", "chamfer_mm"]),
    "np-lever-nf": c("Randomized lever, stop-when-done prompt: removing the force reading", "w8-lever-hard-base-20261104-001", "w7-lever-hard-nowrench-20261104-001", LEVER),
    "np-lever-mass": c("Randomized lever, stop-when-done prompt: stating the lever mass", "w7-lever-hard-nowrench-20261104-001", "w7-lever-hard-nowrench-mass-20261104-001", LEVER,
                       bl="no force reading"),
    "np-lever-mass-hint": c("Randomized lever, stop-when-done prompt, no force reading, stated mass: also naming the hinge parameters",
                            "w7-lever-hard-nowrench-mass-20261104-001", "w9-lever-hard-nowrench-mass-hint-20261104-001", LEVER,
                            bl="no force + stated mass", al="+ task names hinge parameters"),
    "np-lever-oo-mass-hint": c("Randomized lever, stop-when-done prompt, stated mass + hinge parameters named: grasping vs without interaction",
                               "w9-lever-hard-nowrench-mass-hint-20261104-001", "w9-lever-hard-noaction-mass-hint-20261104-001", LEVER,
                               bl="grasp, no force reading", al="no-interaction"),
    "np-spring-oo": c("Randomized spring, stop-when-done prompt: no-interaction", "w6-spring-hard-base-20261104-001", "w7-spring-hard-noaction-20261104-001", SPRING),
    "np-spring-oo-mass": c("Randomized spring, stop-when-done prompt, no-interaction: stating the lever mass", "w7-spring-hard-noaction-20261104-001",
                           "w7-spring-hard-noaction-mass-20261104-001", SPRING, bl="no-interaction"),
    "np-hint-nf": c("Randomized lever without force reading, stop-when-done prompt: naming the hinge parameters", "w7-lever-hard-nowrench-20261104-001",
                    "w8-lever-hard-nowrench-hint-20261104-001", LEVER, bl="no force reading", al="+ task names hinge parameters"),
    "prompt-lever": c("Randomized lever with force reading: full-time vs stop-when-done prompt", "w2-lever-hard-base-20261103-002", "w8-lever-hard-base-20261104-001", LEVER,
                      bl="full-time prompt", al="stop-when-done prompt"),
    "hint-lever": c("Lever", "w1-lever-base-20261102-003", "w1-lever-hint-20261102-003", ["damping_rel", "frictionloss_rel"]),
    "hint-hardlever": c("Randomized lever", "w2-lever-hard-base-20261103-002", "w2-lever-hard-hint-20261103-001", ["damping_rel", "frictionloss_rel"]),
    "hint-nf": c("Randomized lever without force reading", "w2-lever-hard-nowrench-20261103-001", "w3-lever-hard-nowrench-hint-20261104-002", LEVER,
                 bl="no force reading", al="+ task names hinge parameters"),
    "angles": c("Random assembly", RGB3, "w2-assembly-allangles-20261103-001", ["chamfer_mm", "com_mm", "rgb_mse"]),
    "depth-assembly": c("Random assembly", RGB3, "w1-assembly-depth-20261102-003", GEOM),
    "depth-boat": c("Boat", "w3-boat-base-20261104-002", "w3-boat-depth-20261104-002", GEOM),
    "wrist": c("Small Panda + small assembly", "w3-ps-assembly-fixed-20261104-002", "w3-ps-assembly-wrist-20261104-002", GEOM + ["rgb_mse"], bl="fixed camera"),
    "gray": c("Tetrahedral cluster", "w1-tetra-base-20261102-003", "w2-tetra-gray-20261103-002", ["chamfer_mm", "com_mm", "sphere_err"], bl="colored spheres"),
    "arm-lever": c("Randomized lever: Panda arm vs point grasp", "w2-lever-hard-base-20261103-002", "w3-panda-lever-hard-20261104-002", LEVER,
                   bl="point grasp + force reading", al="Panda arm (joint torques)"),
    "touch": c("Panda + blocks", "w2-panda-vision-20261103-001", "w1-panda-blind-20261102-003", ["chamfer_mm", "com_mm", "r:mass:block_red", "r:mass:block_blue", "rgb_mse"],
               bl="camera"),
    "seg": c("Random assembly", RGB3, "w4-assembly-seg-directed-20261104-002", GEOM + ["rgb_mse"]),
    "planning": c("Panda + randomized lever", "w3-panda-lever-hard-20261104-002", "w4-panda-lever-hard-planning-20261104-002", LEVER),
    "planning2": c("Panda + randomized lever, stop-when-done prompt: planning tool v2", "w10-panda-lever-hard-toolsimg-base-20261105-001",
                   "w10-panda-lever-hard-planning-20261105-001", LEVER, bl="tool not offered"),
    "blind-proprio": c("Panda + blocks, stop-when-done prompt: removing the camera", "w10-panda-blocks-vision-20261105-001",
                       "w10-panda-blocks-proprio-20261105-001", PANDA_BLOCKS, bl="camera"),
    "blind-tactile": c("Panda + blocks, stop-when-done prompt: no camera, adding fingertip touch sensors", "w10-panda-blocks-proprio-20261105-001",
                       "w10-panda-blocks-tactile-20261105-001", PANDA_BLOCKS, bl="arm joint feedback only", al="+ touch sensors"),
    "prompt-assembly": c("Random assembly: full-time vs stop-when-done prompt", RGB3, "w10-assembly-base-20261105-001", GEOM + ["rgb_mse"],
                         al="stop-when-done prompt"),
    "prompt-tetra": c("Tetrahedral cluster: full-time vs stop-when-done prompt", "w1-tetra-base-20261102-003", "w10-tetra-base-20261105-001",
                      GEOM + ["sphere_err"], bl="full-time prompt", al="stop-when-done prompt"),
    "prompt-boat": c("Boat: full-time vs stop-when-done prompt", "w3-boat-base-20261104-002", "w10-boat-base-20261105-001", GEOM,
                     bl="full-time prompt", al="stop-when-done prompt"),
}

META = dict(title="Can an LLM agent recover a scene's geometry and physics by experimenting on it?",
            subtitle=L("Results through 2026-11-06 · {n} finished runs · each run is one <code>gpt-6-astra</code> agent (medium reasoning) working alone for up to 4 h. "
                       "<b>Mostly one run per condition</b>; ten key conditions have three runs (section 7), and the random-assembly baseline has three."),
            intro=L(
                "<h2 style='border:0;margin-top:20px'>The experiment</h2>"
                "<p>Each run hides a 3D scene inside MuJoCo, a rigid-body physics simulator, and asks an LLM agent to rebuild it: write a MuJoCo scene file whose "
                "<b>geometry</b> (shape and placement of every object) and <b>physics</b> (masses, center of mass, and for hinged objects the joint's inertia, damping, "
                "dry friction and spring stiffness) match the hidden one. The agent never sees the hidden file. It learns only by looking at the scene and "
                "experimenting on it, so this is a test of active system identification: <b>which observations and interactions does an agent need to recover "
                "which physical quantities, and how does it go about it?</b></p>"
                "<div class='card'><div class='flow'>"
                "<div><b>1 · Hidden scene</b>Seen through one fixed 320×240 RGB camera with known intrinsics and pose. By default the agent can also "
                "pin any visible point of an object to a virtual handle (an ideal point grasp that leaves orientation free), move it, release it, and read the "
                "force the object exerts on the handle.</div>"
                "<div><b>2 · Agent</b>Works in an isolated container with no access to the hidden files: it writes candidate scene files, runs its own Python and "
                "MuJoCo scripts (fitting, rendering, simulating) and looks at images.</div>"
                "<div><b>3 · Its own copy</b>The agent can load a candidate into a second simulator with the same interface and replay the same experiment there, "
                "comparing images and forces side by side with the hidden scene.</div>"
                "<div><b>4 · Scoring</b>The agent submits candidates whenever it likes. A private evaluator it never sees compares each submission with the hidden "
                "scene: shape (Chamfer distance), center of mass (COM), physical parameters, and images from viewpoints the agent never saw. "
                "The newest runs are also scored on a scripted manipulation and on a planning task (section 8).</div>"
                "</div></div>"
                "<h3>Scenes</h3><div class='card scroll'><table class='intro'>"
                "<tr><th>scene</th><th>what it is</th><th>what is scored</th></tr>"
                "<tr><td>Random assembly</td><td>Ten boxes, spheres, capsules and cylinders fused into one rigid object that partly hides itself from the camera; "
                "each part has its own density</td><td>shape, COM, total mass</td></tr>"
                "<tr><td>Tetrahedral cluster</td><td>Four spheres of very different masses fused into one rigid body</td><td>shape, COM, each sphere's mass</td></tr>"
                "<tr><td>Boat</td><td>A free-standing boat built from primitives (hull, cabin, roof, stack)</td><td>shape, COM, total mass</td></tr>"
                "<tr><td>Blocks</td><td>Two separate boxes, red and blue, resting on a table</td><td>each block's mass, their ratio, COM, shape</td></tr>"
                "<tr><td>Lever</td><td>A rod with a ball at the end on a horizontal hinge, with viscous damping, dry friction and a stop at ±65°</td>"
                "<td>mass, inertia about the hinge, damping, dry friction</td></tr>"
                "<tr><td>Spring</td><td>The same kind of hinged rod, held up by a torsional spring, with damping</td><td>mass, inertia, damping, stiffness</td></tr>"
                "<tr><td>Panda + …</td><td>A simulated Franka Panda arm replaces the point grasp. The agent sets joint targets and reads joint angles and joint "
                "torques (an indirect force signal), but gets no grasp-force reading</td><td>as for the scene it is paired with</td></tr>"
                "<tr><td>Small Panda + small assembly</td><td>The random assembly scaled to 12% of its size, placed within the arm's reach</td><td>shape, COM, mass</td></tr>"
                "</table><p class='note'><b>Randomized</b> scenes (lever, spring, blocks, Panda + lever) have every size scaled by a seeded random factor in 0.8–1.25 "
                "and every mass, density, damping, friction and stiffness by a factor in 0.6–1.6, so true values are non-round (damping 0.1157 rather than 0.12). "
                "On the original round-valued scenes agents often land on the exact value (Appendix A3), which hides any difference between conditions.</p></div>"
                "<h3>Conditions</h3><div class='card scroll'><table class='intro'>"
                "<tr><th>condition</th><th>what changes from the baseline</th></tr>"
                "<tr><td>baseline</td><td>Camera + point grasp + force reading (Panda scenes: camera + arm)</td></tr>"
                "<tr><td>no force reading</td><td>Grasping still works, but the force on the handle is no longer reported; the agent sees only motion</td></tr>"
                "<tr><td>no-interaction</td><td>No grasping or moving in the hidden scene: the agent can only watch it and reset it. Gravity still acts, so objects "
                "fall, swing and settle. Everything is still allowed in the agent's own copy</td></tr>"
                "<tr><td>stated mass</td><td>The task text gives one true mass (the lever's, or the red block's) as a known fact</td></tr>"
                "<tr><td>task names hinge parameters</td><td>One added sentence asks the agent to estimate the hinge's damping and dry friction (names, not values)</td></tr>"
                "<tr><td>match every viewing angle</td><td>The task asks for a reconstruction that looks right from every angle</td></tr>"
                "<tr><td>camera + depth / wrist camera</td><td>Adds a depth image from the same camera, or a second camera on the Panda's hand</td></tr>"
                "<tr><td>all spheres gray</td><td>The tetrahedral cluster's spheres all get the same color</td></tr>"
                "<tr><td>no camera (Panda)</td><td>The agent gets only the arm's joint angles and torques and must find objects by touch</td></tr>"
                "<tr><td>fingertip touch sensors (Panda)</td><td>No camera, but each finger pad reports a 3×3 grid of contact pressures plus the finger's total "
                "normal force</td></tr>"
                "<tr><td>segmentation / planning tool</td><td>A MobileSAM image segmenter, or Panda motion-planning code, is installed in the container; in some runs one "
                "added sentence says when it helps</td></tr>"
                "<tr><td>extended +2 h</td><td>A new 2 h run that starts from a finished run's last submission and notes</td></tr>"
                "</table><p class='note'><b>Two task prompts.</b> Most runs were told \"this session lasts the full time budget and "
                "finishing early earns nothing\" (<b>full-time prompt</b>). The latest group, labeled in the run names, was told to finish when nothing about "
                "the reconstruction differs from the hidden scene (<b>stop-when-done prompt</b>). Because this changes how long agents work, runs are compared only with runs under the same "
                "prompt, except one labeled pair. Each comparison changes exactly one thing.</p></div>"))
GLOSSARY = [
    ("Estimate ÷ true (×k)", "The agent's value divided by the true value, read from its submitted scene file. ×1 = exact, ×0.25 = four times too small, \"0 (dropped)\" = the agent left the parameter out."),
    ("Uniform scale factor", "When every mass, inertia and joint parameter has the same estimate ÷ true, the reconstruction is the true scene with all of them multiplied by one number. Its motion is identical (section 1 explains why), so only a force measurement or a known mass can tell the two apart."),
    ("Round default", "With the overall scale undetermined, agents tended to fix one part at a round value (e.g. 0.1 kg, or MuJoCo's default density of 1000 kg/m³) and fit everything else relative to it."),
    ("Hinge parameters", "Inertia: moment of inertia about the hinge axis (kg·m²). Damping: torque proportional to angular velocity (N·m·s/rad). Dry friction: constant resisting torque (N·m). Stiffness: spring constant (N·m/rad)."),
    ("COM error", "Distance between the true and reconstructed center of mass (mm). It depends on the mass ratios between parts, not on total mass."),
    ("Final / at 120 min / at equal tokens", "Comparison tables show three points per run: the last submission; the last submission by minute 120; and the last submission before the run had used as many model tokens as the lower-spending run of the pair. Tokens track compute better than wall time."),
    ("Run-to-run range", "Spread of the three random-assembly baseline repeats. A single-run difference smaller than this spread is not evidence of an effect."),
    ("API cost limit", "A per-run spending limit enforced outside the agent's container and never shown to it. A run that hits it is stopped, and its last submission counts as final."),
    ("Image MSE", "Mean squared pixel error between true and reconstructed renders from a viewpoint the agent never saw."),
]


def first(ctx, run, key):
    return ctx.first(run, key)


def round_anchor(ctx, run):
    """Round masses (multiples of 0.05 kg) or MuJoCo's default density (1000 kg/m³) in the final submission's object bodies: the likely scale anchors."""
    p, out = ctx.runs[run]["parts"]["final"], []
    is_round = lambda x: x > 0 and abs(20 * x - round(20 * x)) < 1e-6
    for body, mass in p["bodies"].items():
        geoms = [g for g in p["geoms"] if g["body"] == body]
        if is_round(mass) and not any(g["mass"] is not None and is_round(g["mass"]) for g in geoms):
            out.append(f"body {mass:g} kg")
        for g in geoms:
            if g["mass"] is not None and is_round(g["mass"]):
                out.append(f"{g['name']} {g['mass']:g} kg")
            elif g["density"] is not None and abs(g["density"] - 1000) < 0.5:
                out.append(f"{g['name']} at density 1000 kg/m³ (MuJoCo default)")
    return out


def table_rows_points(ctx, cid, key):
    """(final, 120 min, matched tokens) values of one metric for the ablation run of a comparison."""
    cmp = ctx.comparisons[cid]
    return [None if cmp["points"][p][cmp["abl"]] is None else ctx.runs[cmp["abl"]]["subs"][cmp["points"][p][cmp["abl"]]]["m"][key] for p in ("final", "at_120min", "at_matched_tokens")]


def sections(ctx):
    """Paper-style results: each numbered section states one finding; every figure and table has a numbered caption saying what it
    demonstrates. Supporting detail (all pairwise comparisons, per-submission curves, other videos, ceiling check, cost, all runs) is
    in the appendix sections, whose titles start with "A"."""
    R, x0, f = ctx.runs, ctx.x, ctx.final
    blk = lambda **kw: kw
    conv = lambda series, keys, **kw: blk(type="convergence", series=[dict(run=s[0], label=s[1], **(s[2] if len(s) > 2 else {})) for s in series], keys=keys, **kw)
    cmps = lambda *ids: blk(type="compare", ids=list(ids))
    common = lambda run, keys: [f(run, k) for k in keys]
    spread = lambda vals: f"×{min(vals):.2f}–{max(vals):.2f}" if max(vals) / min(vals) > 1.01 else f"×{vals[0]:.3g}"
    x = lambda v, d=3: "—" if v is None else x0(v, d)
    mm = lambda v, d=3: "—" if v is None else ctx.mm(v, d)
    lnk = lambda r, t=None: f"<a href='run.html?run={r}'>{t or R[r]['condition']}</a>"
    worst = lambda r: max(abs(v - 1) for k, v in R[r]["subs"][-1]["m"].items() if k.startswith("r:") and v is not None)
    counters = {}

    def cap(kind, text, appendix=False):
        key = (kind, appendix)
        counters[key] = counters.get(key, 0) + 1
        return blk(type="html", html=f"<p class='caption'><b>{kind} {'A' if appendix else ''}{counters[key]}.</b> {text}</p>")

    def lead(text):
        return blk(type="html", html=f"<p>{text}</p>")

    S = dict(sp_oo="w5-spring-hard-noaction-20261103-001", sp_nf="w3-spring-hard-nowrench-20261103-001", sp_b="w3-spring-hard-base-20261103-001",
             sp6="w6-spring-hard-base-20261104-001", sp6_nf="w6-spring-hard-nowrench-20261104-001", sp6_m="w6-spring-hard-nowrench-mass-20261104-001",
             sp7_oo="w7-spring-hard-noaction-20261104-001", sp7_m="w7-spring-hard-noaction-mass-20261104-001",
             lv_b="w2-lever-hard-base-20261103-002", lv_hb="w2-lever-hard-hint-20261103-001", lv_oo="w5-lever-hard-noaction-20261103-002",
             lv_nf="w2-lever-hard-nowrench-20261103-001", lv_h="w3-lever-hard-nowrench-hint-20261104-002", lv8="w8-lever-hard-base-20261104-001",
             lv7_nf="w7-lever-hard-nowrench-20261104-001", lv7_m="w7-lever-hard-nowrench-mass-20261104-001", lv8_h="w8-lever-hard-nowrench-hint-20261104-001",
             lv9_mh="w9-lever-hard-nowrench-mass-hint-20261104-001", lv9_oo="w9-lever-hard-noaction-mass-hint-20261104-001",
             pl="w3-panda-lever-hard-20261104-002", pl_oo="w5-panda-lever-hard-noaction-20261103-002", pl1="w4-panda-lever-hard-planning-20261104-002",
             pl_b2="w10-panda-lever-hard-toolsimg-base-20261105-001", pl2="w10-panda-lever-hard-planning-20261105-001",
             bl_b="w2-blocks-hard-base-20261103-001", bl_nf="w2-blocks-hard-nowrench-20261103-001", bl_oo="w5-blocks-hard-noaction-20261103-002",
             bl7="w7-blocks-hard-base-20261104-001", bl6="w6-blocks-hard-nowrench-20261104-001", bl6_m="w6-blocks-hard-nowrench-mass-20261104-001",
             tt="w1-tetra-base-20261102-003", tt_nf="w3-tetra-nowrench-20261104-002", tt_oo="w5-tetra-noaction-20261103-002", tt_g="w2-tetra-gray-20261103-002",
             as_nf="w3-assembly-nowrench-20261104-002", as_oo="w5-assembly-noaction-20261103-002", as_d="w1-assembly-depth-20261102-003",
             as_ang="w2-assembly-allangles-20261103-001", as_seg="w3-assembly-seg-20261103-001", as_segd="w4-assembly-seg-directed-20261104-002",
             bt="w3-boat-base-20261104-002", bt_oo="w5-boat-noaction-20261103-002", bt_d="w3-boat-depth-20261104-002",
             ps="w3-ps-assembly-fixed-20261104-002", ps_w="w3-ps-assembly-wrist-20261104-002",
             pv="w2-panda-vision-20261103-001", pbl="w1-panda-blind-20261102-003", pv10="w10-panda-blocks-vision-20261105-001",
             pp10="w10-panda-blocks-proprio-20261105-001", pt10="w10-panda-blocks-tactile-20261105-001",
             a10="w10-assembly-base-20261105-001", t10="w10-tetra-base-20261105-001", b10="w10-boat-base-20261105-001")
    sec = []

    # ---------------- 1 · scale ----------------
    scale_lost = [S["lv_oo"], S["lv_nf"], S["lv_h"], S["lv7_nf"], S["lv8_h"], S["pl_oo"], S["sp_oo"], S["sp_nf"], S["sp6_nf"], S["sp7_oo"],
                  S["bl_oo"], S["bl_nf"], S["bl6"], S["bl6_m"], S["as_oo"], S["as_nf"], S["tt_oo"], S["tt_nf"], S["bt_oo"]]
    anchor_rows = []
    for r in scale_lost:
        keys = [k for k in R[r]["subs"][-1]["m"] if k.startswith("r:") and not k.startswith(("r:sphere", "r:massratio")) and R[r]["subs"][-1]["m"][k] is not None]
        flags = R[r]["flags"]["scale"]
        anchor_rows.append([f"{R[r]['scene']} · {lnk(r)}", ", ".join(f"{SHORT[k]} {x(f(r, k))}" for k in keys), ", ".join(round_anchor(ctx, r)) or "<span class='muted'>none</span>",
                            f"{len(flags)} of {R[r]['taxonomy']['notes']}" + (f" (first at min {flags[0]['min']})" if flags else "")])
    quotes = [(r, R[r]["flags"]["scale"][0]) for r in scale_lost if R[r]["flags"]["scale"]][:4]
    bl_nf_runs = [S["bl_nf"], S["bl6"]] + REPS[S["bl6"]]  # every no-force blocks run without a stated mass
    rng = lambda rs, key, fm: fm(f(rs[0], key)) if len(rs) == 1 else f"{fm(min(f(r, key) for r in rs))} – {fm(max(f(r, key) for r in rs))}"
    # Table 1 highlighting: mass within 5% of true = good, else bad; shape/COM exact = good, worse than every run of the scene with interaction = warn.
    hl = lambda cls, t: f"<span class='hl {cls}'>{t}</span>" if cls else t
    chip = {"baseline": "c-base", "no force reading": "c-nf", "no-interaction": "c-oo"}
    mass_cell = lambda v: hl("good" if abs(v - 1) <= 0.05 else "bad", x(v))
    obj_rows = []
    for scene, cond in (("Random assembly", [(RGB3, "baseline"), ([S["as_nf"]], "no force reading"), ([S["as_oo"]], "no-interaction")]),
                        ("Tetrahedral cluster", [([S["tt"]], "baseline"), ([S["tt_nf"]], "no force reading"), ([S["tt_oo"]], "no-interaction")]),
                        ("Boat", [([S["bt"]], "baseline"), ([S["bt_oo"]], "no-interaction")]),
                        ("Randomized blocks (two bodies)", [([S["bl_b"]], "baseline"), ([S["bl_nf"]], "no force reading"), ([S["bl_oo"]], "no-interaction")])):
        inter = [r for rs, label in cond if label != "no-interaction" for r in rs]  # runs of this scene with interaction
        shape_cell = lambda rs, key: hl("good" if max(f(r, key) for r in rs) < 0.01 else
                                        "warn" if min(f(r, key) for r in rs) > max(f(r, key) for r in inter) else "", rng(rs, key, mm))
        for rs, label in cond:
            runs_txt = lnk(rs[0], "run") if len(rs) == 1 else " · ".join(lnk(r, "run " + "ABC"[i]) for i, r in enumerate(rs))
            mass = (" ".join(f"{c} {mass_cell(f(rs[0], 'r:mass:block_' + c))}" for c in ("red", "blue")) if "blocks" in scene
                    else hl("good" if all(abs(f(r, "r:mass") - 1) <= 0.05 for r in rs) else "bad", rng(rs, "r:mass", x)))
            obj_rows.append([scene, f"<span class='chip {chip[label]}'>{label}</span> {runs_txt}", shape_cell(rs, "chamfer_mm"), shape_cell(rs, "com_mm"), mass])
    sec.append(dict(id="scale", nav="Scale", tag="consistent across 7 scenes, both prompts and repeats",
        title="1 · Without force sensing at the grasp, physics is reconstructed only up to a scale factor; removing interaction costs little shape accuracy but all of the mass scale",
        q=None, answer=L("Every run that could not measure a force reproduced the observed motion with every mass, inertia and joint parameter "
                         "multiplied by one shared factor. Its shape was only modestly worse than with full sensing, because all conditions see the "
                         "scene from the same single camera."),
        blocks=[lead(L(
            "We compare three sensing conditions on the same hidden scenes: the <b>baseline</b> (camera, point grasp and force reading), <b>no force reading</b> "
            "(the agent can still grasp and move objects but does not see the force), and <b>no-interaction</b> (the agent cannot act on the hidden scene; objects "
            "still fall, swing and settle under gravity). The reason a factor survives is mechanical. Gravity's torque on a lever is <i>m</i>·<i>g</i>·<i>r</i>, "
            "and inertia, damping, dry friction and spring torques enter the equation of motion linearly. Multiplying every mass, inertia and joint parameter by "
            "one <i>k</i> multiplies every torque by <i>k</i>, so the motion is unchanged. Gravity cannot fix <i>k</i> because its torque scales with mass "
            "too, and a position-controlled grasp imposes motion without measuring the force needed. Only a force measurement, the Panda's joint torques "
            "(section 5) or a known mass (section 2) sets <i>k</i>. Geometry is different: the visible surfaces are measured from one fixed camera "
            "in every condition, and the hidden ones are filled from priors on simple primitives (boxes, spheres, capsules, cylinders). Interaction "
            "can add views of hidden faces by lifting or tipping an object, so removing it should cost some shape accuracy, but much less than mass.")),
            cap("Figure", L("<b>Estimated ÷ true value of every physical parameter</b>, final submission, log scale (×1 = exact). Each row is one run (click a label "
                            "to open it); dots in a row are that run's parameters. With a force reading (rows labeled \"baseline\") every dot sits at ×1. Without "
                            "it, the dots of a row stack on one vertical line away from ×1: the reconstruction is the true scene with all inertial and joint "
                            "parameters scaled by one factor. Hollow dots at the left edge are parameters the agent left out (usually dry friction; section 3). "
                            "On the two blocks, which are separate bodies, the factor can differ per block.")),
            blk(type="ratios", title=None, panels=[
                dict(title="Randomized lever", keys=LEVER, rows=[[S["lv_b"], "baseline (force)"], [S["lv_nf"], "no force reading"], [S["lv_oo"], "no-interaction"],
                                                               {"header": "stop-when-done prompt"}, [S["lv8"], "baseline (force)"], [S["lv7_nf"], "no force reading"]]),
                dict(title="Randomized spring", keys=SPRING, rows=[[S["sp_b"], "baseline (force)"], [S["sp_nf"], "no force reading"], [S["sp_oo"], "no-interaction"],
                                                                 {"header": "stop-when-done prompt"}, [S["sp6"], "baseline (force)"], [S["sp6_nf"], "no force reading"],
                                                                 [S["sp7_oo"], "no-interaction"]]),
                dict(title="Panda + randomized lever (joint torques instead of a force reading)", keys=LEVER, rows=[[S["pl"], "baseline (joint torques)"], [S["pl_oo"], "no-interaction"]]),
                dict(title="Randomized blocks (red, blue, red ÷ blue)", keys=BLOCKS, rows=[[S["bl_b"], "baseline (force)"], [S["bl_nf"], "no force reading"],
                                                                                    [S["bl_oo"], "no-interaction"], {"header": "stop-when-done prompt"}, [S["bl7"], "baseline (force)"],
                                                                                    [S["bl6"], "no force reading"]]),
                dict(title="Free single objects: total mass", keys=["r:mass"], rows=[[r, f"{R[r]['scene'].replace('Tetrahedral cluster', 'Tetrahedral').replace('Random assembly', 'Assembly')} · {R[r]['condition']}"]
                     for r in [RGB3[0], S["as_nf"], S["as_oo"], S["tt"], S["tt_nf"], S["tt_oo"], S["bt"], S["bt_oo"]]])]),
            cap("Table", L("<b>Free objects: removing interaction costs a little shape accuracy and all of the mass scale.</b> Final submission of each run. "
                           "<span class='hl good'>Green</span>: exact (mass within 5%). <span class='hl warn'>Amber</span>: shape or COM error worse than every run "
                           "of the same scene with interaction. <span class='hl bad'>Red</span>: mass off by more than 5%. "
                           f"Every no-interaction run had a larger shape error than every run of the same scene with interaction, by 1–4 mm "
                           f"(assembly {mm(f(S['as_oo'], 'chamfer_mm'))} against "
                           f"{rng(RGB3, 'chamfer_mm', mm)} for the three baselines; boat {mm(f(S['bt_oo'], 'chamfer_mm'))} against {mm(f(S['bt'], 'chamfer_mm'))}; "
                           f"blocks {mm(f(S['bl_oo'], 'chamfer_mm'))} against exact), consistent with interaction revealing some hidden faces. The tetrahedral "
                           f"cluster's four spheres were exact even without interaction, because the hidden part of a symmetric sphere cluster is easy to infer. "
                           f"Every condition keeps some shape error on the self-occluding assembly and boat because the single fixed camera never sees their "
                           f"hidden faces; adding depth gave the best assembly shape ({mm(f(S['as_d'], 'chamfer_mm'))}, section 4). Total mass, in contrast, "
                           f"moved off by one factor without a force reading, as far as {x(min(f(r, 'r:mass') for r in (S['as_oo'], S['tt_oo'], S['bt_oo'])), 2)} "
                           f"(about {1 / min(f(r, 'r:mass') for r in (S['as_oo'], S['tt_oo'], S['bt_oo'])):.0f} times too light) without interaction.The assembly's no-interaction COM error ({mm(f(S['as_oo'], 'com_mm'), 2)}) is inside the baseline range, "
                           f"but the tetrahedral cluster's four sphere masses, which differ 8-fold and are hidden inside one rigid body, were lost without interaction "
                           f"(COM {mm(f(S['tt_oo'], 'com_mm'), 2)} against {mm(f(S['tt'], 'com_mm'), 2)}). For the two separate blocks, grasping without a force "
                           f"reading recovered the red-to-blue mass ratio in {len([r for r in bl_nf_runs if abs(f(r, 'r:massratio') - 1) < 0.01])} of "
                           f"{len(bl_nf_runs)} runs (ratio " + ", ".join(x(f(r, 'r:massratio')) for r in bl_nf_runs) + f"), while the no-interaction run did not "
                           f"({x(f(S['bl_oo'], 'r:massratio'))}): interaction can reveal how mass is split between separate bodies, but not reliably.")),
            blk(type="table", title=None, head=["scene", "condition", "Chamfer", "COM error", "mass ÷ true"], rows=obj_rows),
            cap("Table", L("<b>Where the shared factor comes from: a round default.</b> For every run that lost the scale: its final estimate ratios, the round "
                           "values in its submission (a body or part mass that is a multiple of 0.05 kg, or MuJoCo's default density of 1000 kg/m³, which a part "
                           "gets when no mass is given), and how many of its progress notes say the mass or scale cannot be identified. The agents anchored one "
                           "part at a round default (often 0.1 kg) and fitted everything else relative to it; the factor is therefore arbitrary and differs "
                           "between repeats of the same condition (section 7).")),
            blk(type="table", title=None, head=["scene · condition", "final estimate ÷ true", "round values in the final submission", "notes saying the scale is unidentifiable"],
                rows=anchor_rows),
            cap("Figure", "<b>Some agents noticed the problem and still committed to a default.</b> Verbatim progress notes, with the run and minute."),
            blk(type="quotes", title="", items=[dict(run=r, name=f"{R[r]['scene']} · {R[r]['condition']}", minute=q["min"], text=q["text"]) for r, q in quotes]),
            cap("Figure", "<b>Scaled reconstructions move like the hidden scene.</b> Recorded camera frames (runs from 2026-11-03; later runs did not record video). "
                          "Left: hidden scene; right: the agent's reconstruction under the same actions. The runs without a force reading reproduce the motion "
                          "although their parameters are 2–4× off, which is why the agents had no visual signal that anything was wrong."),
            blk(type="videos", runs=[S["lv_b"], S["lv_nf"], S["sp_b"], S["sp_nf"]]),
            cap("Figure", "<b>Shape without interaction.</b> True geometry (solid) and each run's reconstruction (wireframe) for the random assembly; pick a run, "
                          "including the no-interaction and no-force runs, and drag the slider through its submissions."),
            blk(type="viewer", group="assembly", title="Random assembly, every condition"),
            blk(type="html", html="<p class='note'>Ratios are read from each compiled submission: the hinge body's mass and inertia about the axis, joint "
                                  "parameters, and for free bodies the mass of the candidate body nearest each true body's COM. Parameters that are zero in the true "
                                  "scene (spring friction, lever stiffness) are omitted. The same comparisons at 120 minutes and at equal token use are in Appendix A1; "
                                  "per-submission curves are in Appendix A2.</p>")]))

    # ---------------- 2 · stated mass ----------------
    sec.append(dict(id="stated", nav="Known mass", tag="one run per condition (two for spring and lever) · stop-when-done prompt",
        title="2 · One known mass fixes the scale of a single body, even from video alone",
        q=None, answer=L("Stating one true mass in the task removed the shared factor for the spring and the lever, with or without interaction, but only "
                         "fixed the named block of the two."),
        blocks=[lead(L("If the only missing quantity is the scale <i>k</i> of section 1, one known mass should determine everything else. We gave the true mass "
                       "of one part in the task text, without force reading or without interaction.")),
                cap("Figure", L("<b>A stated mass collapses the shared factor to ×1.</b> Same plot as Figure 1, stop-when-done prompt. Spring: exact with the stated "
                                f"mass, both with grasping and without interaction (damping {x(f(S['sp7_m'], 'r:damping'))} against {x(f(S['sp7_oo'], 'r:damping'))}); "
                                "its two repeats were within " + f"{100 * max(worst(r) for r in REPS[S['sp6_m']]):.0f}%" + " (section 7). "
                                f"Lever: mass, inertia and damping become exact, but dry friction stays out of the model ({x(f(S['lv7_m'], 'r:frictionloss'))}) "
                                f"until the task also names the hinge parameters ({x(f(S['lv9_mh'], 'r:frictionloss'), 2)}; {x(f(S['lv9_oo'], 'r:frictionloss'), 3)} "
                                f"without interaction). Blocks: the named red block is exact, but the unnamed blue block is not "
                                f"({x(f(S['bl6_m'], 'r:mass:block_blue'))}), because the two bodies have independent scales.")),
                blk(type="ratios", title=None, panels=[
                    dict(title="Randomized spring", keys=SPRING, rows=[[S["sp6_nf"], "no force reading"], [S["sp6_m"], "no force + stated lever mass"],
                                                                     [S["sp7_oo"], "no-interaction"], [S["sp7_m"], "no-interaction + stated lever mass"]]),
                    dict(title="Randomized lever", keys=LEVER, rows=[[S["lv7_nf"], "no force reading"], [S["lv7_m"], "no force + stated lever mass"],
                                                                   [S["lv9_mh"], "no force + stated mass + names hinge parameters"],
                                                                   [S["lv9_oo"], "no-interaction + stated mass + names hinge parameters"]]),
                    dict(title="Randomized blocks (red, blue, red ÷ blue)", keys=BLOCKS, rows=[[S["bl6"], "no force reading"], [S["bl6_m"], "no force + stated red-block mass"]])])]))

    # ---------------- 3 · task wording ----------------
    lever_conditions = [("full-time", "force reading", "—", S["lv_b"]), ("full-time", "force reading", "hinge parameters", S["lv_hb"]),
                        ("full-time", "no force reading", "—", S["lv_nf"]), ("full-time", "no force reading", "hinge parameters", S["lv_h"]),
                        ("full-time", "no-interaction", "—", S["lv_oo"]), ("stop-when-done", "force reading", "—", S["lv8"]),
                        ("stop-when-done", "no force reading", "—", S["lv7_nf"]), ("stop-when-done", "no force reading", "lever mass", S["lv7_m"]),
                        ("stop-when-done", "no force reading", "hinge parameters", S["lv8_h"]),
                        ("stop-when-done", "no force reading", "lever mass + hinge parameters", S["lv9_mh"]),
                        ("stop-when-done", "no-interaction", "lever mass + hinge parameters", S["lv9_oo"])]
    lever_rows = [(p, s, t, r) for p, s, t, o in lever_conditions for r in [o] + REPS.get(o, [])]  # each condition, then its repeats
    friction_rows = [[p, s, t, lnk(r, f"repeat {REPEAT_N(r)}" if r in REPEATS else "run 1"), x(f(r, "r:mass")), x(f(r, "r:damping")), f"<b>{x(f(r, 'r:frictionloss'))}</b>"]
                     for p, s, t, r in lever_rows]
    unnamed = [r for p, s, t, r in lever_rows if s != "force reading" and "hinge" not in t]
    unnamed_zero = [r for r in unnamed if f(r, "r:frictionloss") == 0]
    unnamed_late = [r for r in unnamed if f(r, "r:frictionloss")]
    pair_rows = []
    for scene, a, b in (("Randomized lever", [S["lv_b"]], S["lv8"]), ("Randomized spring", [S["sp_b"]], S["sp6"]), ("Randomized blocks", [S["bl_b"]], S["bl7"]),
                        ("Random assembly", RGB3, S["a10"]), ("Tetrahedral cluster", [S["tt"]], S["t10"]), ("Boat", [S["bt"]], S["b10"])):
        summ = (lambda r: "exact" if worst(r) < 1e-3 else f"worst parameter off by {100 * worst(r):.0f}%") if "Randomized" in scene else \
               (lambda r: f"Chamfer {mm(f(r, 'chamfer_mm'))}, COM {mm(f(r, 'com_mm'))}, mass {x(f(r, 'r:mass'))}")
        pair_rows.append([scene, "<br>".join(f"{lnk(r, 'run ' + 'ABC'[i] if len(a) > 1 else 'run')}: {summ(r)} ({R[r]['wall_min']:.0f} min)" for i, r in enumerate(a)),
                          f"{lnk(b, 'run')}: {summ(b)} ({R[b]['wall_min']:.0f} min)"])
    sec.append(dict(id="prompt", nav="Task wording", tag="16 lever runs · prompt pairs one run each",
        title="3 · Parameters the task does not name can be silently dropped; the prompt style changes little",
        q=None, answer=L("Without a force reading, agents usually left the lever's dry friction out of the model unless the task named it. Switching from "
                         "the full-time to the stop-when-done prompt did not change what was recovered."),
        blocks=[lead(L("Dry friction is a small, constant torque that the lever's motion reveals only weakly. We varied whether the task text names the hinge "
                       "parameters (damping and dry friction, without values) and tabulate every randomized-lever run with a point grasp.")),
                cap("Table", L("<b>Dry friction is fitted when it is measurable or named, and otherwise usually omitted.</b> Final estimate ÷ true for every "
                               "randomized-lever run with the point grasp. With a force reading, friction was exact whether or not it was named. Without one, "
                               f"{len(unnamed_zero)} of the {len(unnamed)} runs whose task did not name the hinge parameters left friction out entirely (0), and the "
                               f"other {len(unnamed_late)} fitted a value far below the truth (" +
                               ", ".join(x(f(r, 'r:frictionloss'), 2) for r in unnamed_late) + "). Every run whose task named the parameters fitted friction, on "
                               "roughly the same scale as its other parameters. "
                               "Without a force reading, friction is identifiable (as a ratio) but easy to overlook.")),
                blk(type="table", title=None, head=["prompt", "sensing", "task states", "run", "mass ÷ true", "damping ÷ true", "dry friction ÷ true"], rows=friction_rows),
                cap("Table", L("<b>The prompt style did not change what was recovered.</b> Each scene's baseline under the full-time prompt (\"the session lasts the "
                               "full time budget\") and the stop-when-done prompt (\"finish when nothing differs\"); minutes in parentheses. Exact baselines stayed "
                               f"exact. Free-object shape errors stayed in the range of earlier runs (boat {mm(f(S['b10'], 'chamfer_mm'))} against "
                               f"{mm(f(S['bt'], 'chamfer_mm'))}; assembly inside the three full-time runs' range), and the tetrahedral cluster became exact. "
                               "One run each, so these differences are within the run-to-run variation seen elsewhere.")),
                blk(type="table", title=None, head=["scene", "full-time prompt", "stop-when-done prompt"], rows=pair_rows),
                blk(type="html", html=L(f"<p class='note'>Asking the agent to match the appearance from every viewing angle (one random-assembly run) left Chamfer "
                                        f"{mm(f(S['as_ang'], 'chamfer_mm'))} and COM {mm(f(S['as_ang'], 'com_mm'))}, inside or just past the range of the three "
                                        f"baseline repeats (Appendix A1): no clear effect from one run.</p>"))]))

    # ---------------- 4 · camera variants ----------------
    cam = lambda label, rs, rv, extra=None: [label, " · ".join(lnk(r, "ABC"[i] if len(rs) > 1 else "baseline") for i, r in enumerate(rs)) + f" → {lnk(rv, 'variant')}",
                                            f"{rng(rs, 'chamfer_mm', mm)} → {mm(f(rv, 'chamfer_mm'))}", f"{rng(rs, 'com_mm', mm)} → {mm(f(rv, 'com_mm'))}",
                                            f"{rng(rs, extra or 'r:mass', x if extra is None else (lambda v: '—' if v is None else f'{100 * v:.0f}%'))} → "
                                            f"{(x if extra is None else (lambda v: '—' if v is None else f'{100 * v:.0f}%'))(f(rv, extra or 'r:mass'))}"]
    sec.append(dict(id="sensing", nav="Camera", tag="one run per condition: trends only",
        title="4 · Richer camera input: depth helps where parts hide each other; a wrist camera trades shape for mass; color is not needed",
        q=None, answer=L("Depth improved the occluded random assembly but not the boat, a wrist camera improved shape but worsened mass, and gray spheres were "
                         "reconstructed as well as colored ones."),
        blocks=[cap("Table", L("<b>Camera variants against their baselines</b> (baseline → variant; final submissions). The random assembly hides parts of itself "
                               "from the fixed camera, so a depth image adds information there; the boat's shape is mostly visible, and depth did not help. The "
                               "wrist camera on the small Panda assembly gave closer views and better shape, but that run's mass estimate was worse. The tetrahedral cluster was recovered exactly with gray spheres, so color was not what "
                               "identified the spheres. One run per variant: read these as trends.")),
                blk(type="table", title=None, head=["comparison", "runs", "Chamfer", "COM error", "mass ÷ true (tetrahedral: worst sphere-mass error)"],
                    rows=[cam("Random assembly + depth", RGB3, S["as_d"]), cam("Boat + depth", [S["bt"]], S["bt_d"]),
                          cam("Small Panda assembly + wrist camera", [S["ps"]], S["ps_w"]), cam("Tetrahedral cluster, all spheres gray", [S["tt"]], S["tt_g"], "sphere_err")]),
                cap("Figure", "<b>The camera variants in 3D.</b> True geometry (solid) against each reconstruction (wireframe)."),
                blk(type="viewer", group="boat", title="Boat"), blk(type="viewer", group="ps", title="Small Panda assembly"),
                blk(type="viewer", group="tetra", title="Tetrahedral cluster")]))

    # ---------------- 5 · Panda ----------------
    panda_lever = [[lnk(S["lv_b"], "point grasp + force reading"), *(x(f(S["lv_b"], k)) for k in LEVER)]] + \
                  [[lnk(r, t), *(x(f(r, k)) for k in LEVER)] for r, t in ((S["pl"], "Panda, joint torques (run 1)"), (S["pl_b2"], "Panda, joint torques (run 2)"),
                                                                        (S["pl_oo"], "Panda, no-interaction"))]
    blocks_rows = [[p, lnk(r, t), mm(f(r, "chamfer_mm")), mm(f(r, "com_mm")), x(f(r, "r:mass:block_red")), x(f(r, "r:mass:block_blue"))]
                   for p, r, t in (("full-time", S["pv"], "camera"), ("full-time", S["pbl"], "no camera, arm feedback"),
                                   ("stop-when-done", S["pv10"], "camera"), ("stop-when-done", S["pp10"], "no camera, arm feedback"),
                                   ("stop-when-done", S["pt10"], "no camera, + fingertip touch sensors"))]
    sec.append(dict(id="panda", nav="Panda arm", tag="one run per condition: trends only",
        title="5 · A Panda arm: joint torques act as a coarse force sensor; without a camera, finding objects by touch is unreliable",
        q=None, answer=L("With a simulated Franka Panda instead of the ideal grasp, the arm's joint torques gave the lever's scale to within about 10–50%, "
                         "but dry friction was dropped. Without a camera, the blocks were found exactly once and missed in two later runs."),
        blocks=[cap("Table", L("<b>Joint torques give the scale roughly.</b> Randomized lever, final estimate ÷ true. The arm's torques carry force information, "
                               "so the scale is closer to ×1 than without any force signal (section 1), but less precise than the point grasp's force reading, "
                               "and dry friction came out at " + ", ".join(x(f(r, 'r:frictionloss')) for r in (S['pl'], S['pl_b2'], S['pl_oo'])) +
                               " in the three Panda runs. The two Panda runs of the same condition differ (damping "
                               f"{x(f(S['pl'], 'r:damping'))} and {x(f(S['pl_b2'], 'r:damping'))}), so single-run Panda differences are noisy.")),
                blk(type="table", title=None, head=["condition", *(f"{SHORT[k]} ÷ true" for k in LEVER)], rows=panda_lever),
                cap("Table", L("<b>Finding objects without a camera.</b> Panda + two blocks (original round-valued scene), final submission. With a camera all runs "
                               "were exact. The first camera-less run found both blocks exactly by touching them, but two later camera-less runs, one with "
                               "fingertip touch sensors, missed the red block's position and mass. Touch sensing narrowed the error (COM "
                               f"{mm(f(S['pt10'], 'com_mm'))} against {mm(f(S['pp10'], 'com_mm'))}). The runs' pages show every arm and touch reading the model received.")),
                blk(type="table", title=None, head=["prompt", "condition", "Chamfer", "COM error", "red mass ÷ true", "blue mass ÷ true"], rows=blocks_rows),
                cap("Figure", "<b>The camera-less reconstructions in 3D.</b> Pick a run and drag the slider to watch it search for the blocks; no video exists for these runs."),
                blk(type="viewer", group="panda", title="Panda + blocks, every condition")]))

    # ---------------- 6 · tools ----------------
    up = lambda r, t: R[r]["tools"][t]
    first = lambda r, t: "never" if up(r, t)["first_min"] is None else f"min {up(r, t)['first_min']:.0f}"
    tool_rows = [["segmentation (MobileSAM)", "no", lnk(S["as_seg"]), str(up(S["as_seg"], "segmentation")["calls"]), first(S["as_seg"], "segmentation"),
                  f"Chamfer {mm(f(S['as_seg'], 'chamfer_mm'))}; repeats without the tool {rng(RGB3[:2], 'chamfer_mm', mm)}"],
                 ["segmentation (MobileSAM)", "yes", lnk(S["as_segd"]), str(up(S["as_segd"], "segmentation")["calls"]), first(S["as_segd"], "segmentation"),
                  f"Chamfer {mm(f(S['as_segd'], 'chamfer_mm'))} (stopped by its cost limit at {R[S['as_segd']]['wall_min']:.0f} min)"],
                 ["Panda planning v1", "yes", lnk(S["pl1"]), str(up(S["pl1"], "planning")["calls"]), first(S["pl1"], "planning"),
                  f"damping {x(f(S['pl1'], 'r:damping'))}, friction {x(f(S['pl1'], 'r:frictionloss'))}; without the tool {x(f(S['pl'], 'r:damping'))}, {x(f(S['pl'], 'r:frictionloss'))}"],
                 ["Panda planning v2", "yes", lnk(S["pl2"]), str(up(S["pl2"], "planning")["calls"]), first(S["pl2"], "planning"),
                  f"damping {x(f(S['pl2'], 'r:damping'))}, friction {x(f(S['pl2'], 'r:frictionloss'))}; without the tool {x(f(S['pl_b2'], 'r:damping'))}, {x(f(S['pl_b2'], 'r:frictionloss'))}"]]
    sec.append(dict(id="tools", nav="Tools", tag="one run per condition: trends only",
        title="6 · Offered tools: segmentation was never used; the revised planning tool was used and the lever came out exact",
        q=None, answer=L("Whether a tool helps depends first on whether the agent uses it: the image segmenter was never called, even when recommended, while the "
                         "Panda planning tools were used from the first minutes, and only the revised one improved the result."),
        blocks=[cap("Table", L("<b>Tool uptake and outcome.</b> Bash calls that run, import or reference the tool (from the full traces), the first use, and the "
                               "final result against the same setup without the tool. Segmentation was never touched, so its runs measure nothing about the tool. "
                               "Planning v1 was used but the lever came out worse than without it. Planning v2 (revised tool and instructions) gave an exact lever, "
                               "every parameter including friction. One run each, and the v2-era run without the tool was itself worse than the earlier one, so part "
                               "of the gap may be run-to-run variation.")),
                blk(type="table", title=None, head=["tool", "task recommends it", "run", "bash calls using it", "first use", "result (without the tool)"], rows=tool_rows)]))

    # ---------------- 7 · repeats ----------------
    pair = lambda o, lab: [[o, lab]] + [[r, f"  repeat {REPEAT_N(r)}"] for r in REPS[o]]
    runs3 = lambda o: [o] + REPS[o]
    sec.append(dict(id="repeats", nav="Repeats", tag="three runs per condition",
        title="7 · Repeats: every key finding holds in three runs; the size of a wrong scale varies",
        q=None, answer=L("Three runs of each of ten key conditions reproduced every finding in direction; only the value of the arbitrary scale factor "
                         "changed between runs."),
        blocks=[cap("Figure", L("<b>Three runs per condition, side by side</b> (stop-when-done prompt; same plot as Figure 1). With a force reading the parameters "
                                "sit at ×1 in every run. With a stated mass (plus named hinge parameters for the lever) they sit at or near ×1. Without a force "
                                "reading the scale is wrong in every run, by a factor that changes from run to run (spring " +
                                ", ".join(spread(common(r, SPRING)) for r in runs3(S['sp6_nf'])) + "; blocks' red-to-blue ratio " +
                                ", ".join(x(f(r, 'r:massratio')) for r in runs3(S['bl6'])) + "), as expected if the factor comes from whichever round default "
                                "the agent anchors on (Table 2). Repeats used the same prepared scene file and configuration; the only change visible to the agent "
                                "was one extra field (<code>\"tactile\": false</code>) in the tool configuration.")),
                blk(type="ratios", title=None, panels=[
                    dict(title="Randomized lever", keys=LEVER, rows=pair(S["lv8"], "baseline (force)") + pair(S["lv7_nf"], "no force reading") +
                         pair(S["lv8_h"], "no force + names parameters") + pair(S["lv9_mh"], "no force + stated mass + names parameters") +
                         pair(S["lv9_oo"], "no-interaction + stated mass + names parameters")),
                    dict(title="Randomized spring", keys=SPRING, rows=pair(S["sp6"], "baseline (force)") + pair(S["sp6_nf"], "no force reading") +
                         pair(S["sp6_m"], "no force + stated lever mass")),
                    dict(title="Randomized blocks (red, blue, red ÷ blue)", keys=BLOCKS, rows=pair(S["bl7"], "baseline (force)") + pair(S["bl6"], "no force reading"))])]))

    # ---------------- 8 · manipulation and planning ----------------
    ex = {r: EXPERT[r][0] for r in EXPERT_RUNS}
    mmv = lambda v: f"{1000 * v:.0f} mm"
    ex_rows = [[lnk(r, R[r]["scene"].replace(", stop-when-done prompt", "")), ex[r]["expert"][0]["name"].replace("_", " "), "yes" if ex[r]["expert"][0]["candidate_grasp_success"] else "no",
                f"{ex[r]['expert'][0]['rgb_mse']:.4f}", f"{ex[r]['expert'][0]['silhouette_error']:.3f}", mm(f(r, "chamfer_mm")),
                mmv(ex[r]["policy_transfer"][0]["candidate_predicted_error_m"]), mmv(ex[r]["policy_transfer"][0]["transfer_error_m"]),
                mmv(ex[r]["policy_transfer"][0]["oracle_error_m"]), mmv(ex[r]["policy_transfer"][0]["transfer_regret_m"]), mmv(ex[r]["policy_transfer"][0]["naive_error_m"])]
               for r in EXPERT_RUNS]
    asm = ex[S["a10"]]["policy_transfer"][0]
    sec.append(dict(id="newscores", nav="Manipulation", tag="one run per scene · stop-when-done prompt",
        title="8 · Beyond static scores: matching shape and mass is not always enough to plan with",
        q=None, answer=L(f"Two of three reconstructions behaved like the hidden scene under a scripted manipulation and supported planning; the random assembly, "
                         f"despite exact mass and COM, misled a planner by {mmv(asm['transfer_regret_m'])}."),
        blocks=[lead(L("Static scores compare the reconstruction at rest. Two newer scores test it in use. <b>Scripted manipulation</b>: the evaluator replays a "
                       "fixed grasp-and-move script from reset in fresh copies of the hidden scene and the reconstruction and compares the two camera videos. "
                       "<b>Policy transfer</b>: the evaluator searches for an open-loop fling (grasp, move, release, settle) that brings the object's COM to a "
                       "goal using only the reconstruction, then runs it in the hidden scene. <i>Transfer error</i> is the distance from the goal there; "
                       "<i>oracle</i> is the same search done in the hidden scene; <i>regret</i> = transfer − oracle; <i>naive</i> is the unoptimized move "
                       "straight toward the goal.")),
                cap("Table", L("<b>Final submissions of the three newest free-object baselines.</b> The tetrahedral cluster and boat replay identically (RGB and "
                               "silhouette error 0) and their planned flings land within a few millimeters of the best possible. The random assembly replays "
                               f"closely but its plan missed the goal by {mmv(asm['transfer_error_m'])} in the hidden scene against {mmv(asm['oracle_error_m'])} "
                               f"for the oracle, while the reconstruction itself predicted {mmv(asm['candidate_predicted_error_m'])}: its remaining shape error "
                               "matters once the object collides with the table.")),
                blk(type="table", title=None, head=["scene", "script", "grasp ok", "RGB MSE", "silhouette error", "Chamfer", "reconstruction's own prediction",
                                                    "transfer error", "oracle", "regret", "naive"], rows=ex_rows),
                cap("Figure", "<b>The scripted manipulation, rendered by the evaluator.</b> Left: hidden scene. Right: final reconstruction under the same script."),
                blk(type="html", html="<div class='grid2'>" + "".join(
                    f"<div class='card vid'><video controls loop muted playsinline preload='metadata' src='videos/expert-{r}.mp4'></video>"
                    f"<p><b>{lnk(r, R[r]['scene'].replace(', stop-when-done prompt', ''))} · {ex[r]['expert'][0]['name'].replace('_', ' ')}</b></p></div>" for r in EXPERT_RUNS) + "</div>")]))

    # ---------------- 9 · strategies ----------------
    sec.append(dict(id="strategy", nav="Strategies", tag=None, title="9 · How the agents work: fit, then check",
        q=None, answer=None,
        blocks=[lead("How do the agents reach their estimates: by guessing and adjusting, by closed-form physics, or by systematic fitting? We classified every "
                     "bash command and every progress note of every run with keyword rules (validated against hand labels; Methods)."),
                blk(type="strategy")]))

    # ---------------- appendix ----------------
    sec.append(dict(id="appx-compare", nav="A1", appendix=True, tag=None, title="A1 · Every pairwise comparison at three checkpoints",
        q=None, answer=None,
        blocks=[lead(L("Each table compares one baseline with one changed condition at three points: the final submission; the last submission by minute 120; "
                       "and the last submission before the run had used as many model tokens as the lower-spending run of the pair (tokens track compute better "
                       "than wall time). Under each changed-condition value is how its error compares with the baseline. These tables back the summaries in "
                       "sections 1–6; a conclusion stated there holds at all three points unless noted.")),
                blk(type="html", html="<h3>Section 1: removing the force reading, or all interaction</h3>"),
                cmps("nf-lever", "nf-spring", "nf-blocks", "nf-assembly", "nf-tetra", "np-lever-nf", "np-blocks-nf", "np-spring-nf",
                     "oo-lever", "oo-spring", "oo-panda", "oo-blocks", "oo-assembly", "oo-tetra", "oo-boat", "np-spring-oo"),
                blk(type="html", html="<h3>Section 2: stating a mass</h3>"),
                cmps("np-spring-mass", "np-spring-oo-mass", "np-lever-mass", "np-lever-mass-hint", "np-lever-oo-mass-hint", "np-blocks-mass"),
                blk(type="html", html="<h3>Section 3: task wording and prompt style</h3>"),
                cmps("hint-lever", "hint-hardlever", "hint-nf", "np-hint-nf", "prompt-lever", "prompt-assembly", "prompt-tetra", "prompt-boat", "angles"),
                blk(type="html", html="<h3>Sections 4–6: camera variants, Panda arm, tools</h3>"),
                cmps("depth-assembly", "depth-boat", "wrist", "gray", "arm-lever", "touch", "blind-proprio", "blind-tactile", "seg", "planning", "planning2")]))
    sec.append(dict(id="appx-curves", nav="A2", appendix=True, tag=None, title="A2 · Estimates at every submission",
        q=None, answer=None,
        blocks=[lead("How each run's estimates evolved. One point per submission at its submission time, log scale; the dashed line is the true value on ratio "
                     "plots. These show whether a wrong value was there from the start or appeared late, and whether a run was still improving when it ended."),
                conv([[S["lv_b"], "baseline"], [S["lv_nf"], "no force reading"], [S["lv_oo"], "no-interaction", {"color": "--third"}]], LEVER, title="Randomized lever, full-time prompt"),
                conv([[S["sp_b"], "baseline"], [S["sp_nf"], "no force reading"], [S["sp_oo"], "no-interaction", {"color": "--third"}]], SPRING, title="Randomized spring, full-time prompt"),
                conv([[S["pl"], "baseline"], [S["pl_oo"], "no-interaction"]], LEVER, title="Panda + randomized lever"),
                conv([[S["bl_b"], "baseline"], [S["bl_nf"], "no force reading"], [S["bl_oo"], "no-interaction", {"color": "--fourth"}]], BLOCKS + ["com_mm"], title="Randomized blocks, full-time prompt"),
                conv([[r, NAMES[r][1], {"color": "--base", "dash": d}] for r, d in zip(RGB3, ("", "6 3", "2 3"))] +
                     [[S["as_nf"], "no force reading", {"color": "--abl"}], [S["as_oo"], "no-interaction", {"color": "--third"}], [S["as_d"], "camera + depth", {"color": "--fourth"}],
                      ["c1-assembly-base-20261103-002", "baseline A +2 h", {"color": "--base", "faded": True}], ["c1-assembly-depth-20261103-002", "depth +2 h", {"color": "--fourth", "faded": True}]],
                     ["chamfer_mm", "com_mm", "r:mass"], title="Random assembly (faded: +2 h extensions)", xmax=360),
                conv([[S["tt"], "tetrahedral baseline"], [S["tt_nf"], "tetrahedral no force"], [S["tt_oo"], "tetrahedral no-interaction", {"color": "--third"}],
                      [S["bt"], "boat baseline", {"color": "--fourth"}], [S["bt_oo"], "boat no-interaction", {"color": "--s5"}]], ["chamfer_mm", "com_mm", "r:mass"],
                     title="Tetrahedral cluster and boat"),
                conv([[S["sp6"], "baseline"], [S["sp6_nf"], "no force reading"], [S["sp6_m"], "no force + stated mass", {"color": "--third"}],
                      [S["sp7_oo"], "no-interaction", {"color": "--fourth"}], [S["sp7_m"], "no-interaction + stated mass", {"color": "--s5"}]], SPRING, title="Randomized spring, stop-when-done prompt"),
                conv([[S["lv8"], "baseline"], [S["lv7_nf"], "no force reading"], [S["lv7_m"], "no force + stated mass", {"color": "--third"}],
                      [S["lv9_mh"], "+ names parameters", {"color": "--fourth"}], [S["lv9_oo"], "no-interaction + mass + names", {"color": "--s5"}]], LEVER,
                     title="Randomized lever, stop-when-done prompt"),
                conv([[S["lv_nf"], "full-time, no force"], [S["lv_h"], "full-time, no force + names"], [S["lv7_nf"], "stop-when-done, no force", {"color": "--third"}],
                      [S["lv8_h"], "stop-when-done, no force + names", {"color": "--fourth"}]], LEVER, title="Randomized lever without force reading: naming the hinge parameters"),
                conv([[S["bl7"], "baseline"], [S["bl6"], "no force reading"], [S["bl6_m"], "no force + stated red mass", {"color": "--third"}]], BLOCKS + ["com_mm"],
                     title="Randomized blocks, stop-when-done prompt"),
                conv([[S["pv10"], "camera"], [S["pp10"], "no camera, arm feedback"], [S["pt10"], "no camera, + touch", {"color": "--third"}],
                      [S["pbl"], "no camera (first run)", {"color": "--fourth"}]], ["chamfer_mm", "com_mm", "r:mass:block_red", "r:mass:block_blue"], title="Panda + blocks"),
                conv([[S["pl"], "Panda (run 1)"], [S["pl1"], "+ planning v1"], [S["pl_b2"], "Panda (run 2)", {"color": "--third"}], [S["pl2"], "+ planning v2", {"color": "--fourth"}]],
                     LEVER, title="Panda + randomized lever: planning tools"),
                conv([[S["bt"], "boat baseline"], [S["bt_d"], "boat + depth"], [S["ps"], "small assembly, fixed camera", {"color": "--third"}],
                      [S["ps_w"], "small assembly, + wrist camera", {"color": "--fourth"}]], ["chamfer_mm", "com_mm", "r:mass"], title="Camera variants")]))
    easy = ["w1-blocks-base-20261102-003", "w1-spring-base-20261102-003", "w1-lever-base-20261102-003", "w1-lever-hint-20261102-003", S["bl_b"], S["pbl"], S["pv"], S["sp6"]]
    fmt_err = lambda v: "exact" if v < 1e-3 else f"{100 * v:.1f}%"
    sec.append(dict(id="saturate", nav="A3", appendix=True, tag=None, title="A3 · Round-valued scenes are recovered exactly, which is why the randomized variants exist",
        q=None, answer=None,
        blocks=[cap("Table", L("<b>Ceiling check.</b> The original scenes use round values (for example 0.25 kg or damping 0.12), and these runs recovered "
                               "them exactly, as did the force-reading baselines of the randomized scenes. A condition that changes nothing on these scenes may "
                               "simply be at the ceiling, so the main comparisons use the randomized scenes."), appendix=True),
                blk(type="table", title=None, head=["scene · condition", "Chamfer", "COM error", "worst parameter error", "submissions"],
                    rows=[[f"{R[r]['scene']} · {lnk(r)}", mm(f(r, "chamfer_mm")), mm(f(r, "com_mm")), fmt_err(worst(r)), str(R[r]["submissions"])] for r in easy])]))
    sec.append(dict(id="appx-videos", nav="A4", appendix=True, tag=None, title="A4 · Other recorded videos",
        q=None, answer=None,
        blocks=[cap("Figure", "<b>Recorded camera frames of the remaining runs with video</b> (runs from 2026-11-03). Left: hidden scene; right: the agent's "
                              "reconstruction under the same actions.", appendix=True),
                blk(type="videos", runs=[RGB3[0], S["as_d"], S["as_ang"], S["as_seg"], S["tt_g"], S["bl_nf"], S["pv"]])]))
    new = [r for r in ORDER if r.startswith(("w6", "w7", "w8", "w9", "w10"))]
    main_runs = [r for r in ORDER if r not in CONTINUATIONS]
    capped_ids = [r for r in main_runs if R[r]["budget_stop"] or R[r]["reason"] == "budget_suspended"]
    sec.append(dict(id="effort", nav="A5", appendix=True, tag=None, title="A5 · Time, cost and when agents stop",
        q=None, answer=None,
        blocks=[cap("Figure", L("<b>When do agents decide they are done?</b> Wall clock used under the stop-when-done prompt (of 240 min; orange = stopped by "
                                "the API cost limit). Exact runs did not reliably stop early: the force-reading lever baseline was exact from submission "
                                f"{next(i for i in range(len(R[S['lv8']]['subs'])) if worst_at(R, S['lv8'], i) < 1e-3) + 1} yet kept testing until minute "
                                f"{R[S['lv8']]['wall_min']:.0f}, while runs with a wrong scale also finished on their own (they had no signal that anything was "
                                "wrong; Figure 3)."), appendix=True),
                blk(type="bars", title=None, unit="min", max=240,
                    rows=[dict(run=r, label=f"{R[r]['scene'].replace(', stop-when-done prompt', '')} · {R[r]['condition']}", v=R[r]["wall_min"],
                               color="--abl" if R[r]["reason"] == "budget_suspended" else "--base") for r in new]),
                cap("Figure", L(f"<b>Wall clock, tokens and cost of every run.</b> Full-time-prompt runs used most of their 4 hours; {len(capped_ids)} runs across "
                                "both prompts were stopped by their API cost limit (orange), and their last submission counts as final."), appendix=True),
                blk(type="bars", title="Wall clock used (% of 4 h)", unit="%", max=100,
                    rows=[dict(run=r, label=f"{R[r]['scene']} · {R[r]['condition']}", v=100 * R[r]["budget_used"], color="--abl" if r in capped_ids else "--base") for r in main_runs]),
                blk(type="bars", title="Model tokens used (millions)", unit="M", rows=[dict(run=r, label=f"{R[r]['scene']} · {R[r]['condition']}", v=R[r]["final_mtok"], color="--faint") for r in main_runs]),
                blk(type="bars", title="API cost (USD, gateway estimate)", unit="$", rows=[dict(run=r, label=f"{R[r]['scene']} · {R[r]['condition']}", v=R[r]["cost_usd"], color="--faint") for r in ORDER])]))
    sec.append(dict(id="runs", nav="A6", appendix=True, tag=None, title="A6 · All finished runs", q=None, answer=None,
                    blocks=[blk(type="html", html="<p>Every run in this report; click a name for its page with the full conversation and the images the model received.</p>"),
                            blk(type="runs")]))
    return sec


def worst_at(R, r, i):
    return max(abs(v - 1) for k, v in R[r]["subs"][i]["m"].items() if k.startswith("r:") and v is not None)


def strategy(ctx):
    """The section-10 answer: aggregate shares over every run."""
    from collections import Counter
    kinds, seconds = Counter(), Counter()
    for v in ctx.runs.values():
        kinds.update(v["taxonomy"]["script_counts"])
        seconds.update(v["call_seconds"])
    share = lambda *ks: round(100 * sum(kinds[k] for k in ks) / sum(kinds.values()))
    world = sum(seconds[k] for k in ("observe", "grasp", "move", "release", "reset", "arm_target", "load_reconstruction"))
    tshare = lambda v: round(100 * v / sum(seconds.values()))
    answer = L(f"Mostly systematic fitting, then checking. {share('fit_sim', 'fit_data')}% of all {sum(kinds.values())} bash calls ran an optimizer or parameter "
               f"sweep ({share('fit_sim')}% over local MuJoCo simulations or renders), {share('simulate')}% ran a single local simulation or render to check a "
               f"candidate, and {share('analytic')}% did closed-form physics. Bash scripts took {tshare(seconds['bash'])}% of the wall time between model turns; "
               f"actions on the hidden scene and the agent's copy took {tshare(world)}%. The agents re-implemented the grasp interface in their own local "
               f"simulator and fitted masses, inertia, damping, friction and shapes "
               f"by least squares to recorded forces, poses and pixels, then validated candidates on new interactions. When a quantity was unidentifiable they "
               f"did not stop fitting; they filled the gap with a round default (section 1).")
    return dict(answer=answer, glance=f"{share('fit_sim', 'fit_data')}% of bash calls fit parameters by optimizer or sweep.")


def methods(ctx):
    oo = [r for r in ctx.runs if 'noaction' in r]
    attempts = {r: sum(n for k, n in ctx.runs[r]['actions'].items() if k.split('|')[0] in ('grasp', 'move', 'release', 'arm_target') and k.endswith('|reference')) for r in oo}
    return [
    L(f"<b>Mostly one run per condition</b> (ten key conditions have three runs, section 7). {len([r for r in ctx.runs if r.startswith(('w6', 'w7', 'w8', 'w9', 'w10')) and ctx.runs[r]['reason'] == 'budget_suspended'])} stop-when-done runs ({', '.join(NAMES[r][0].lower() + ' · ' + NAMES[r][1] for r in [r for r in ctx.runs if r.startswith(('w6', 'w7', 'w8', 'w9', 'w10')) and ctx.runs[r]['reason'] == 'budget_suspended'])}) were stopped by their API cost limit before finishing; their last submission is their final one, and the 120 min and equal-token columns compare like with like. "
      "The three random-assembly baseline repeats differ several-fold in final COM error, so single-run differences smaller than that are not evidence. The scale effect (exact vs one uniform scale factor, with a physical explanation, on five scenes and both prompts) is large enough to read from single runs."),
    "<b>Every run is a strict run</b>: the agent works in an isolated container with no access to scene files, and the model is called through the API. "
    "Earlier development-mode runs, in which a spawned agent shared the workspace filesystem without an API boundary, are excluded "
    "(<code>experiments/validation/excluded-runs.json</code>); the report builder asserts that no excluded or non-strict run is included.",
    "<b>Physical parameters</b> are read by compiling each hidden scene and each submission with MuJoCo (XML compile and one forward pass, no plugins). The compiled COM matches the evaluator's COM error to 0.01 mm wherever the evaluator reports it (checked by assertion); for tetrahedral runs whose submission sets one explicit body inertia, the evaluator reports none and the compiled value is shown.",
    "<b>Comparison points</b> come from an index of every scored submission by wall time and cumulative model tokens (<code>checkpoint-index.py</code>). The equal-token point uses the smaller token total of the runs compared; for the three baseline repeats, the smallest of all four.",
    "<b>Strategy classification</b>: keyword rules over every progress note the agent wrote and every bash command it ran (a script run by name is classified by its latest written content). Rules and all labels are in <code>taxonomy-labels.json</code>; agreement with hand labels (on runs from 2026-11-03) is shown under the table. Run logs were read as recorded, without re-verifying their integrity checksums.",
    "<b>Prompts.</b> All runs except the stop-when-done group used the full-time prompt. Runs from 2026-11-03 also showed a fixed API cost limit in the task text; later runs never saw any cost information. Runs under different prompts are not compared, except one labeled pair (randomized lever with force reading), where both are exact.",
    "<b>Videos</b> exist only for runs from 2026-11-03 (recorded camera frames, not re-simulations). Later runs did not record video; the 3D overlays and per-submission plots are their visual evidence.",
    L(f"<b>Validity checks (not results).</b> No-interaction runs performed no grasp, move, release or arm action in the hidden scene "
      f"(recorded action counts: {', '.join(f'{NAMES[r][0].lower()} {n}' for r, n in attempts.items())}). To check that the camera-less Panda blocks run "
      f"could not see the scene, its block colors were randomized; it guessed a generic blue/orange, but this run's random colors happened to be close to "
      f"blue/orange, so the check is weak. The two later camera-less runs also had randomized colors. The arm-feedback-only run guessed colors far from both the "
      f"random and the original ones. The touch-sensor run's colors landed near the original red and blue, but there is no channel for that "
      f"to leak through: it ran in the isolated container with no scene files, its touch observation reports only forces (no names or positions), and its own "
      f"notes label both colors as guesses (\"blue guessed\" for the box it found first, \"guessed red\" for the one it found last). Calling two blocks red and "
      f"blue is a coin-flip prior that happened to match. "
      f"Runs lost in the 2026-11-03 storage incident and one blocked randomized-lever attempt are excluded; each was "
      f"rerun under a new ID."),
    ]
