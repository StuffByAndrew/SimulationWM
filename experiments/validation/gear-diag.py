"""Diagnostic aid (Slurm): per probe, the deep and shallow (source timestep) trains' twist (hand angle + 7.5 crank angle),
crank angle at release and at the end, and per-body RMS difference, for cumulative JSON overrides of gear-scenes.py constants.
PYTHONPATH=src .venv/bin/python experiments/validation/gear-diag.py '[{}, {"GRASP_SOLREF": [0.004, 1]}]'"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, "src")
from sim2sim.gear_train import compile_plain, panel_normal, probe_actions, rollout  # noqa: E402
from sim2sim.simulation import Simulation  # noqa: E402

spec = importlib.util.spec_from_file_location("scenes", "experiments/validation/gear-scenes.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
for variant in json.loads(sys.argv[1]):
    for name, value in variant.items():
        setattr(S, name, value)
    for config in (S.SOURCE_CONFIG, S.CANDIDATE_CONFIG):
        config.update(grasp_solref=S.GRASP_SOLREF, grasp_solimp=S.GRASP_SOLIMP)
    with tempfile.TemporaryDirectory() as temp:
        deep, shallow = Path(temp) / "scene.xml", Path(temp) / "shallow.xml"
        deep.write_text(S.scene())
        shallow.write_text(S.scene(True))
        normal = panel_normal(*compile_plain(deep))
        sims = Simulation(deep, S.SOURCE_CONFIG), Simulation(shallow, S.CANDIDATE_CONFIG)
        out = {"variant": variant}
        for probe in S.probe_tables():
            actions = probe_actions(sims[0], probe)
            (_, a, ga), (_, b, gb) = (rollout(sim, actions, S.TIMESTEP, normal) for sim in sims)
            release = 4 * (probe["ticks"] // 4) + 1
            twist = lambda x: np.abs(x[:, 1] + 7.5 * x[:, 0]).max()  # noqa: E731
            out[probe["name"]] = {"grasped": [ga, gb], "twist_max": [round(float(twist(a)), 4), round(float(twist(b)), 4)],
                                  "crank_release": [round(float(a[release, 0]), 4), round(float(b[release, 0]), 4)],
                                  "crank_end": [round(float(a[-1, 0]), 4), round(float(b[-1, 0]), 4)],
                                  "hand_range": [round(float(np.ptp(a[:, 1])), 3), round(float(np.ptp(b[:, 1])), 3)],
                                  "rms": [round(float(v), 4) for v in np.sqrt(np.mean((a - b) ** 2, axis=0))]}
            if probe["name"] in variant.get("_series", ()):  # crank angle (deep, shallow) around the release, every 40 steps
                out[probe["name"]]["series"] = [[k - release, round(float(a[k, 0]), 4), round(float(b[k, 0]), 4)]
                                                for k in range(max(0, release - 800), min(len(a), release + 1200), 40)]
        print(json.dumps(out), flush=True)
