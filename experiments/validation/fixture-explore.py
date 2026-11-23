"""Tuning aid for the fixture levels (Slurm only): per level and setting, release statistics and pad state after settling.
.venv/bin/python experiments/validation/fixture-explore.py [LEVEL ...] [--override 'LEVELS["fixture_rubber"]["flap"]=(...)']"""
import argparse
import os
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from sim2sim.simulation import Simulation
from sim2sim.fixture import release_trial

spec = importlib.util.spec_from_file_location("scenes", "experiments/validation/fixture-scenes.py")
scenes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scenes)
parser = argparse.ArgumentParser()
parser.add_argument("levels", nargs="*", default=list(scenes.LEVELS))
parser.add_argument("--override", action="append", default=[])
parser.add_argument("--trials", type=int, default=12)
parser.add_argument("--settings", default="[[2,2],[5,5],[8,3],[8,8],[3,8],[9,1]]")
args = parser.parse_args()
for line in args.override:
    exec(line, vars(scenes))
for level in args.levels:
    path = Path(f"/tmp/fixture-explore-{os.environ['SLURM_JOB_ID']}/{level}/scene.xml")
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent.parent / "robot").exists() or (path.parent.parent / "robot").symlink_to(Path("environments/panda_small/robot").resolve())
    path.write_text(scenes.scene(level))
    sim = Simulation(path, {"robot": "panda", "width": 64, "height": 48})
    mech = sim.fixture
    start = time.time()
    sim._advance(500)
    step_us = (time.time() - start) / 500 * 1e6
    print(f"== {level}: {step_us:.0f} us/step; rest drift carriage {mech.position_mm():.3f} mm", flush=True)
    for u, v in json.loads(args.settings):
        stops, folds, heights, ticks = [], [], [], []
        for trial in range(args.trials):
            rng = np.random.default_rng([7, trial])
            sim.action({"op": "reset"}, render=False)
            mech.set_dials(u, v)
            mech.reset_carriage(rng)
            folds.append(float(np.sum(sim._data.qpos[mech.hinge_qpos])) if len(mech.hinge_qpos) else 0.)
            heights.append(1000 * float(sim._data.qpos[mech.qpos["pad_z"]]))
            result = release_trial(sim, u, v, np.random.default_rng([7, trial]), 2500, 5)
            stops.append(result["stop_mm"])
            ticks.append(result["ticks"])
        stops = np.array(stops)
        order = np.argsort(stops)
        print(f"  u={u} v={v}: stop mean {stops.mean():7.2f} sd {stops.std():6.2f} min {stops.min():7.2f} max {stops.max():7.2f} ticks<= {max(ticks)}")
        print("    sorted stops", " ".join(f"{s:.1f}" for s in stops[order]))
        print("    fold sum (rad)", " ".join(f"{folds[i]:+.2f}" for i in order))
        print("    pad_z mm     ", " ".join(f"{heights[i]:+.2f}" for i in order), flush=True)
