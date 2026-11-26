"""Researcher audit scan for the jointed-body wrench bug: how often agents saw implausible grasp-force magnitudes.

Selects every strict run with the ideal point grasp's wrench observation whose hidden source scene compiles with a hinge or
slide joint (lever, spring, msd, gear, bottle, and as controls the tunnels and cloth), then walks its verified audit
(sim2sim.audit.read_audit; nothing simulated) and counts the distinct active wrench readings per world, those above 1e3 N
and 1e6 N, the largest, and the run-clock minute of the first above 1e3 N. One JSON line per run.
Usage (Slurm test node): .venv/bin/python experiments/validation/wrench-scan.py OUT.jsonl [WORKERS]"""
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco

from sim2sim.audit import read_audit

ROOTS = (Path("/n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-experiments/runs"), Path("/n/netscratch/qianqian_lab/Lab/aejxie/sim2sim-experiments/runs"),
         Path("/n/home00/aejxie/sim2sim/experiments/runs"))
EXCLUDED = set(json.loads((Path(__file__).resolve().parent / "excluded-runs.json").read_text())["runs"])
LIMITS = (1e3, 1e6)


def candidates():
    """Strict, non-excluded runs with a point-grasp wrench observation and at least one audit."""
    out = []
    for root in ROOTS:
        for d in sorted(root.iterdir()):
            cfg, exp = d / "config-private.json", d / "experiment.json"
            if d.name in EXCLUDED or not (cfg.is_file() and exp.is_file() and (d / "audit").is_dir()):
                continue
            c = json.loads(cfg.read_text())
            if json.loads(exp.read_text()).get("execution_mode") == "strict_oci" and c["simulation"].get("wrench") and not c["simulation"].get("robot"):
                out.append(d)
    return out


def jointed(run, c):
    """Hinge and slide joints in the compiled hidden source scene."""
    parts = Path(c["source"]["scene"]).parts
    m = mujoco.MjModel.from_xml_path(str(run / Path(*parts[parts.index(run.name) + 1:])))
    return sum(int(t) in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) for t in m.jnt_type)


def wrenches(value, world, found):
    """Every active wrench reading in an event, keyed by (world, observation time, force) so a reading stored twice counts once."""
    if isinstance(value, dict):
        w = value.get("wrench")
        for one in (w if isinstance(w, list) else [w] if isinstance(w, dict) else []):
            if one.get("active") and isinstance(one.get("force"), list):
                found.setdefault((world, value.get("time"), tuple(one["force"])), math.hypot(*one["force"]))
        for k, v in value.items():
            if k not in ("state", "before_state", "after_state", "beforestate", "afterstate", "request"):
                wrenches(v, world, found)
    elif isinstance(value, list):
        for v in value:
            wrenches(v, world, found)


def scan(run):
    c = json.loads((run / "config-private.json").read_text())
    n_joints = jointed(run, c)
    row = dict(run=run.name, root=run.parent.parent.name, scene=Path(c["source"]["scene"]).parent.name, joints=n_joints,
               model=c["inference"].get("model"), status=json.loads((run / "status.json").read_text())["status"],
               submissions=len(list(run.glob("submission-*.zip"))))
    if not n_joints:
        return dict(row, skipped="no hinge or slide joint")
    audit = run / "audit"
    for _ in range(720):  # wait out a compaction in progress (pack written, loose blobs being removed)
        if not ((audit / "pack").is_dir() and (audit / "blobs").is_dir() and any((audit / "blobs").iterdir())):
            break
        time.sleep(10)
    found, first = {}, {}
    for _, event in read_audit(run):
        world, wall = event.get("world"), event.get("wall_elapsed")
        if world is None or (event.get("kind") == "boundary" and event.get("direction") == "request"):
            continue
        before = len(found)
        wrenches(event, world, found)
        if len(found) > before and isinstance(wall, (int, float)):
            for (w, _, _), mag in list(found.items())[before:]:
                if mag > LIMITS[0] and w not in first:
                    first[w] = round(wall / 60, 1)
    worlds = {}
    for (w, _, _), mag in found.items():
        s = worlds.setdefault(w, dict(readings=0, over_1e3=0, over_1e6=0, max_n=0.0))
        s["readings"] += 1
        s["over_1e3"] += mag > LIMITS[0]
        s["over_1e6"] += mag > LIMITS[1]
        s["max_n"] = max(s["max_n"], mag)
    for w, s in worlds.items():
        s["first_over_1e3_min"] = first.get(w)
    return dict(row, worlds=worlds)


def main(out, workers):
    runs = candidates()
    print(len(runs), "candidate runs", flush=True)
    with ProcessPoolExecutor(workers) as pool, open(out, "w") as f:
        for row in pool.map(scan, runs):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(row["run"], row.get("skipped") or {w: (s["readings"], s["over_1e3"], f"{s['max_n']:.3g}") for w, s in row["worlds"].items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
