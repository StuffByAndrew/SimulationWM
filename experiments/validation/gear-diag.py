"""Diagnostic aid (Slurm): for cumulative JSON overrides of gear-scenes.py constants, every manifest probe and check over-pull
run in the deep source and the shallow reference (both at the source timestep), with and without the grasp slip: peak grasp
stretch (m), the time the grip slipped (s, or null), crank angle at the end and per-body RMS difference (rad).
PYTHONPATH=src .venv/bin/python experiments/validation/gear-diag.py '[{}, {"SLIP_M": 0.003}]'"""
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


def watched(sim):
    """Record the grasp stretch after every step and when the grip slipped."""
    record, slip = {"stretch": 0., "slipped_s": None}, sim.slip

    def check():
        if sim._grasp is not None:
            eq, mocap = sim._grasp
            body = sim._model.eq_obj1id[eq]
            point = sim._data.xpos[body] + sim._data.xmat[body].reshape(3, 3) @ sim._model.eq_data[eq, :3]
            record["stretch"] = max(record["stretch"], float(np.linalg.norm(point - sim._data.mocap_pos[mocap])))
            slip()
            if sim._grasp is None:
                record["slipped_s"] = float(sim._data.time)
    sim.slip = check
    return record


for variant in json.loads(sys.argv[1]):
    for name, value in variant.items():
        setattr(S, name, value)
    with tempfile.TemporaryDirectory() as temp:
        deep, shallow = Path(temp) / "scene.xml", Path(temp) / "shallow.xml"
        deep.write_text(S.scene())
        shallow.write_text(S.scene(True))
        normal = panel_normal(*compile_plain(deep))
        for slip in (None, S.SLIP_M):
            configs = [{k: v for k, v in c.items() if k != "grasp_slip_m"} | ({"grasp_slip_m": slip} if slip else {})
                       for c in (S.SOURCE_CONFIG, S.CANDIDATE_CONFIG)]
            sims = Simulation(deep, configs[0]), Simulation(shallow, configs[1])
            out = {"variant": variant, "slip_m": slip}
            for probe in S.probe_tables() + S.overpull_tables():
                actions = probe_actions(sims[0], probe)
                records = [watched(sim) for sim in sims]
                (_, a, _), (_, b, _) = (rollout(sim, actions, S.TIMESTEP, normal) for sim in sims)
                for sim in sims:
                    del sim.slip
                out[probe["name"]] = {"stretch_mm": [round(r["stretch"] * 1e3, 2) for r in records],
                                      "slipped_s": [r["slipped_s"] for r in records],
                                      "crank_end": [round(float(a[-1, 0]), 4), round(float(b[-1, 0]), 4)],
                                      "rms": [round(float(v), 4) for v in np.sqrt(np.mean((a - b) ** 2, axis=0))]}
            print(json.dumps(out), flush=True)
