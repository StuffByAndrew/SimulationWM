"""Diagnostic (Slurm): per probe, the deep and shallow trains' twist (hand angle + 7.5 crank angle) and the crank grasp lag."""
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
BASE_PROBES = list(S.PROBES)
for variant in json.loads(sys.argv[1]):
    for k, v in variant.items():
        setattr(S, k, v)
    if "TIMESTEP" in variant:  # same physical probe durations
        k = round(.002 / S.TIMESTEP)
        S.PROBES = [(n, b, p, d, t * k, s * k) for n, b, p, d, t, s in BASE_PROBES]
        S.CANDIDATE_CONFIG = {**S.CANDIDATE_CONFIG, "timestep": S.TIMESTEP}
    with tempfile.TemporaryDirectory() as temp:
        deep, shallow = Path(temp) / "scene.xml", Path(temp) / "shallow.xml"
        deep.write_text(S.scene())
        shallow.write_text(S.scene(True))
        normal = panel_normal(*compile_plain(deep))
        src = Simulation(deep, S.SOURCE_CONFIG)
        out = {"variant": variant}
        for probe in S.probe_tables():
            actions = probe_actions(src, probe)
            a, _ = rollout(src, actions, normal)
            b, _ = rollout(Simulation(shallow, S.CANDIDATE_CONFIG), actions, normal)
            twist = lambda x: x[:, 1] + 7.5 * x[:, 0]  # noqa: E731
            grasped = 4 * (probe["ticks"] // 4) + 1
            out[probe["name"]] = {"deep_twist_max": float(np.abs(twist(a)).max()), "shallow_twist_max": float(np.abs(twist(b)).max()),
                                  "crank_at_release": [float(a[grasped, 0]), float(b[grasped, 0])],
                                  "crank_final": [float(a[-1, 0]), float(b[-1, 0])],
                                  "crank_diff_max_during_grasp": float(np.abs(a[:grasped, 0] - b[:grasped, 0]).max())}
        print(json.dumps(out), flush=True)
