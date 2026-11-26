"""Researcher figure for a controlled ablation: the same hidden scene under two conditions (e.g. point grasp + wrench vs Franka arm, resets vs
no resets, touch harness vs control). One row per scene: each condition's first recorded reference-camera frame (verified audit) with its
settings from the private config, then one panel per tracked metric: each run's per-submission value against run-clock minutes (step until the
next submission, to the run's end); repetitions dashed in their condition's colour. Per-submission metrics come from built report data
(build-results-report.py data.js). Prints per run when it first got within tolerance, and per row each condition's primary error at its last
submission, at matched run-clock minutes and at matched cumulative tokens (the smallest run total in the row).
Usage (Slurm): .venv/bin/python experiments/validation/ablation-pairs-plot.py CONFIG.json OUT.png
CONFIG: {"title", "data": [data.js...], "conditions": [{"id", "name", "color"}, ...], "families": {name: [[key, label, tol, scale]]},
         "rows": [[title, family, {condition id: [runs]}]]}; scale is "ratio" (estimate ÷ true, line at 1), "log", "log1" (log, line at 1 =
         no-mechanism error) or "zero" (linear from 0)."""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from sim2sim.audit import read_audit
from sim2sim.replay import _observations

ROOTS = (Path("/n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-experiments/runs"), Path("/n/netscratch/qianqian_lab/Lab/aejxie/sim2sim-experiments/runs"),
         Path("/n/home00/aejxie/sim2sim/experiments/runs"))
TEXT, MUTED, GRID = "#191919", "#666666", "#e6e6e6"


def series(run, key):
    """(minutes, tokens, value) per submission; a rejected or unscored submission is a gap (None), never a fabricated value."""
    return [(u["t"], u["mtok"], u["m"].get(key)) for u in run["subs"]]


def run_dir(run):
    hits = [d / run for d in ROOTS if (d / run / "config-private.json").is_file()]
    assert len(hits) == 1, (run, hits)
    return hits[0]


def first_view(run, cache):
    """The first reference-world RGB frame the run recorded: the scene as its agent first saw it (verified audit, nothing rendered).
    Saved once as CACHE/<run>.png (the exact audit PNG bytes), so re-renders need not read audits again."""
    path = cache / f"{run}.png"
    if not path.is_file():
        audit = run_dir(run) / "audit"
        for _ in range(720):  # mid-compaction (pack written, loose blobs being removed): wait for sim2sim.compact to finish the run
            if not ((audit / "pack").is_dir() and (audit / "blobs").is_dir() and any((audit / "blobs").iterdir())):
                break
            time.sleep(10)
        else:
            raise AssertionError(f"{run}: audit still mid-compaction after 2 h")
        png = next((images["rgb"] for _, event in read_audit(run_dir(run)) if event.get("world") == "reference"
                    for _, _, images in _observations(event) if "rgb" in images), None)
        assert png is not None, f"{run}: no reference RGB frame in its audit"
        cache.mkdir(exist_ok=True)
        path.write_bytes(png)
    return mpimg.imread(path, format="png")


def settings(run):
    """The setup's distinguishing settings, from the run's private config."""
    c = json.loads((run_dir(run) / "config-private.json").read_text())
    sim, tools = c["simulation"], c.get("tools") or {}
    body = {"panda": "Franka arm: joint targets; reads joint angles + efforts", "shadow_hand": "Shadow hand: joint targets; reads joints + touch"}.get(
        sim.get("robot"), "ideal point grasp: " + ("reads grasp force (wrench)" if sim.get("wrench") else "no force reading"))
    extra = [s for s, on in (("sequence tool", tools.get("sequence")), ("local controller", tools.get("local_controller")),
                             ("pausable bash", "pausable" in c["sandbox"].get("command", [])), ("NO resets", sim.get("resets") is False),
                             ("no interaction with the hidden scene", sim.get("reference_interaction") is False)) if on]
    view = f"{sim['camera']} camera, " + " + ".join(k.upper() for k in ("rgb", "depth") if sim.get(k)) if sim.get("rgb") or sim.get("depth") else "no camera"
    return f"{body}\n{view}, {c['budget']['wall_seconds'] / 60:.0f} min wall, {c['inference']['model']}" + (f"\n{', '.join(extra)}" if extra else "")


def err(key, v):
    return None if v is None else abs(v - 1) if key.startswith("r:") else v


def first_within(pts, key, tol):
    """Run-clock minute when the value first got within tolerance: |ratio - 1| <= tol for ratios, error <= tol otherwise."""
    return next((t for t, _, v in pts if v is not None and err(key, v) <= tol), None)


def last_by(pts, axis, limit):
    """The last scored value at or before a budget on axis 0 (minutes) or 1 (tokens)."""
    return next((v for p in reversed(pts) if p[axis] <= limit and (v := p[2]) is not None), None)


def main(config, out):
    C = json.loads(Path(config).read_text())
    R = {}
    for p in C["data"]:
        for k, v in json.loads(Path(p).read_text()[len("window.R="):-1])["runs"].items():
            assert R.get(k, v)["subs"] == v["subs"], (p, k)  # a run in two data files must agree
            R[k] = v
    conds, rows = C["conditions"], C["rows"]
    fig, grid = plt.subplots(len(rows), 2 + 4, figsize=(26, 3.6 * len(rows)), squeeze=False, gridspec_kw=dict(width_ratios=[1.25, 1.25, 1, 1, 1, 1]))
    for cells, (title, fam, sides) in zip(grid, rows):
        for i, (ax, cond) in enumerate(zip(cells[:2], conds)):
            run = sides[cond["id"]][0]
            ax.imshow(first_view(run, Path(out).parent / "first-frames"))
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(cond["color"]), s.set_linewidth(3)
            ax.set_title(title.replace("\n", " ") if i == 0 else f"same scene, {cond['name']}", fontsize=10 if i == 0 else 9, color=TEXT, loc="left",
                         fontweight="bold" if i == 0 else "normal")
            ax.set_xlabel(settings(run), fontsize=8, color=MUTED, loc="left")
        runs = [r for c in conds for r in sides[c["id"]]]
        end = max(R[r]["wall_min"] for r in runs)
        budget = (min(R[r]["wall_min"] for r in runs), min(R[r]["final_mtok"] for r in runs))
        metrics = C["families"][fam]
        for ax, (key, label, tol, scale) in zip(cells[2:], metrics):
            for cond in conds:
                for i, r in enumerate(sides[cond["id"]]):
                    pts = series(R[r], key)
                    xs, ys = [p[0] for p in pts] + [R[r]["wall_min"]], [p[2] for p in pts] + [pts[-1][2]]
                    ax.step(xs, [np.nan if y is None else y for y in ys], where="post", color=cond["color"], lw=2, ls="-" if i == 0 else "--")
                    ax.plot([p[0] for p in pts], [np.nan if p[2] is None else p[2] for p in pts], "o", color=cond["color"], ms=5, mec="white", mew=1.2, ls="none")
                    last = next((v for *_, v in reversed(pts) if v is not None), None)
                    if last is not None:
                        ax.annotate(f"{last:.3g}", (R[r]["wall_min"], last), xytext=(4, 0), textcoords="offset points", fontsize=8, color=TEXT, va="center")
                    print("RUN", title.replace("\n", " "), key, cond["id"], r, last, first_within(pts, key, tol) if tol is not None else None, len(pts), sep="\t")
            if scale == "ratio":
                ax.axhline(1, color=MUTED, lw=1, ls=":")
                ax.set_ylim(0, max(1.6, max((p[2] for r in runs for p in series(R[r], key) if p[2] is not None), default=1) * 1.1))
            elif scale == "zero":
                ax.set_ylim(bottom=0)
            else:
                assert scale in ("log", "log1"), scale
                ax.set_yscale("log")
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                if scale == "log1":
                    ax.axhline(1, color=MUTED, lw=1, ls=":")  # 1 = as wrong as no mechanism
            ax.set_xlim(0, end * 1.12)
            ax.set_title(label, fontsize=9, color=TEXT, loc="left")
            ax.grid(color=GRID, lw=0.8)
            ax.tick_params(labelsize=8, colors=MUTED)
            for s in ax.spines.values():
                s.set_color(GRID)
        for ax in cells[2 + len(metrics):]:
            ax.axis("off")
        key = metrics[0][0]  # primary metric: each condition's mean error at its last submission and at the row's matched budgets
        for cond in conds:
            pts = [series(R[r], key) for r in sides[cond["id"]]]
            mean = lambda vals: None if not [v for v in vals if v is not None] else float(np.mean([v for v in vals if v is not None]))
            print("ROW", title.replace("\n", " "), key, cond["id"], f"n={len(pts)}",
                  f"last={mean([err(key, next((v for *_, v in reversed(p) if v is not None), None)) for p in pts])}",
                  f"at_{budget[0]:.0f}min={mean([err(key, last_by(p, 0, budget[0])) for p in pts])}",
                  f"at_{budget[1]:.1f}Mtok={mean([err(key, last_by(p, 1, budget[1])) for p in pts])}", sep="\t")
    for ax in grid[-1][2:]:
        ax.set_xlabel("run-clock minutes", fontsize=9, color=MUTED)
    handles = [plt.Line2D([], [], color=c["color"], lw=2, marker="o", mec="white") for c in conds]
    height = fig.get_size_inches()[1]  # title and legend a fixed distance (inches) from the top, whatever the row count
    fig.suptitle(C["title"], fontsize=10, color=MUTED, y=1 - 0.12 / height, va="top")
    fig.legend(handles, [c["name"] for c in conds], loc="upper center", ncol=len(conds), frameon=False, fontsize=11, bbox_to_anchor=(0.55, 1 - 0.55 / height))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.95 / height))
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
