"""Researcher figure: same hidden scene, ideal point grasp + wrench vs Panda/Franka arm. One row per scene-matched pair, one panel per
tracked metric, each run's per-submission value against run-clock minutes (step until the next submission, to the run's end).
Reads the built report data (per-submission metrics computed by build-results-report.py); prints when each run first got within
tolerance. Usage (Slurm): .venv/bin/python experiments/validation/embodiment-pairs-plot.py OUT.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TUNNEL_DATA = Path("/n/home00/aejxie/sim2sim/experiments/reports/results-report-20261124/data.js")
LEVER_DATA = Path("/n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-experiments/reports/results-report-20261104-plain/data.js")
GRASP, ARM = "#257de0", "#f47230"  # validated default categorical slots 1 and 2
TEXT, MUTED, GRID = "#191919", "#666666", "#e6e6e6"
TUNNEL_KEYS = [("mech_rel", "mechanism force error ÷ true force", 0.2), ("fp_push_n", "slow-push force error (N)", None),
               ("fp_retract_n", "slow-retract force error (N)", None), ("fp_fast_n", "fast-cycle force error (N)", None)]
LEVER_KEYS = [("r:mass", "lever mass ÷ true", 0.1), ("r:inertia", "inertia ÷ true", 0.1), ("r:damping", "hinge damping ÷ true", 0.1),
              ("r:frictionloss", "hinge friction loss ÷ true", 0.1)]
LEVEL = {"spring": "spring", "damper": "spring + damper", "nonlinear": "nonlinear spring", "latch": "push-push latch", "curved": "curved tunnel, spring"}
# (row title, family, {embodiment: [runs]}); a list longer than one would be repetitions, drawn dashed
PAIRS = [(f"Tunnel: {LEVEL[lv]}" + (",\nsequence tool" if seq else ""), "tunnel",
          {"grasp": [f"w16-{lv}-grasp{seq}-20261109-001"], "arm": [f"w16-{lv}-franka{seq}-20261109-001"]})
         for lv in LEVEL for seq in ("", "-seq")] + \
        [("Randomized\ndamped lever", "lever", {"grasp": ["w2-lever-hard-base-20261103-002"], "arm": ["w3-panda-lever-hard-20261104-002"]}),
         ("Randomized\ndamped lever,\nno interaction", "lever",
          {"grasp": ["w5-lever-hard-noaction-20261103-002"], "arm": ["w5-panda-lever-hard-noaction-20261103-002"]})]
NAME = {"grasp": "point grasp + wrench", "arm": "Franka/Panda arm (joint efforts)"}
load = lambda p: json.loads(p.read_text()[len("window.R="):-1])["runs"]
RUNS = {"tunnel": load(TUNNEL_DATA), "lever": load(LEVER_DATA)}


def series(run, key):
    """(minutes, value) per submission; a rejected or unscored submission is a gap (None), never a fabricated value."""
    return [(u["t"], u["m"].get(key)) for u in run["subs"]]


def first_within(pts, key, tol):
    """Run-clock minute when the value first got within tolerance: |ratio - 1| <= tol for ratios, error <= tol otherwise."""
    ok = (lambda v: abs(v - 1) <= tol) if key.startswith("r:") else (lambda v: v <= tol)
    return next((t for t, v in pts if v is not None and ok(v)), None)


def main(out):
    keys = {"tunnel": TUNNEL_KEYS, "lever": LEVER_KEYS}
    fig, axes = plt.subplots(len(PAIRS), 4, figsize=(19, 3.0 * len(PAIRS)), squeeze=False)
    summary = []
    for row, (title, fam, sides) in zip(axes, PAIRS):
        R = RUNS[fam]
        end = max(R[r]["wall_min"] for runs in sides.values() for r in runs)
        for ax, (key, label, tol) in zip(row, keys[fam]):
            for side, color in (("grasp", GRASP), ("arm", ARM)):
                for i, r in enumerate(sides[side]):
                    pts = series(R[r], key)
                    xs, ys = [p[0] for p in pts] + [R[r]["wall_min"]], [p[1] for p in pts] + [pts[-1][1]]
                    ax.step(xs, [float("nan") if y is None else y for y in ys], where="post", color=color, lw=2, ls="-" if i == 0 else "--")
                    ax.plot([p[0] for p in pts], [float("nan") if p[1] is None else p[1] for p in pts], "o", color=color, ms=5,
                            mec="white", mew=1.2, ls="none")
                    last = next((v for _, v in reversed(pts) if v is not None), None)
                    if last is not None:
                        ax.annotate(f"{last:.3g}", (R[r]["wall_min"], last), xytext=(4, 0), textcoords="offset points", fontsize=8,
                                    color=TEXT, va="center")
                    summary.append((title.replace("\n", " "), key, side, r, last, first_within(pts, key, tol) if tol else None, len(pts)))
            if key.startswith("r:"):
                ax.axhline(1, color=MUTED, lw=1, ls=":")
                ax.set_ylim(0, max(1.6, max((v for runs in sides.values() for r in runs for _, v in series(R[r], key) if v is not None), default=1) * 1.1))
            else:
                ax.set_yscale("log")
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                if key == "mech_rel":
                    ax.axhline(1, color=MUTED, lw=1, ls=":")  # 1 = as wrong as no mechanism
            ax.set_xlim(0, end * 1.12)
            ax.set_title(label, fontsize=9, color=TEXT, loc="left")
            ax.grid(color=GRID, lw=0.8)
            ax.tick_params(labelsize=8, colors=MUTED)
            for s in ax.spines.values():
                s.set_color(GRID)
        row[0].set_ylabel(title, fontsize=10, color=TEXT, rotation=0, ha="right", va="center", labelpad=12)
    for ax in axes[-1]:
        ax.set_xlabel("run-clock minutes", fontsize=9, color=MUTED)
    handles = [plt.Line2D([], [], color=GRASP, lw=2, marker="o", mec="white"), plt.Line2D([], [], color=ARM, lw=2, marker="o", mec="white")]
    fig.legend(handles, [NAME["grasp"], NAME["arm"]], loc="upper center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.55, 0.998))
    fig.suptitle("Same hidden scene: ideal point grasp + wrench vs Franka/Panda arm (gpt-6-astra, strict runs). Each dot is a scored submission; the line "
                 "holds its value until the next one.\nDotted line: truth (ratios) or no-mechanism error (1). One run per embodiment: no repetitions "
                 "exist on these scenes, so no dashed repeat lines appear.", fontsize=10, color=MUTED, y=0.99)
    fig.tight_layout(rect=(0.03, 0, 1, 0.975))
    fig.savefig(out, dpi=110)
    for s in summary:
        print(*s, sep="\t")


if __name__ == "__main__":
    main(sys.argv[1])
