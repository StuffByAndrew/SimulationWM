"""Tuning aid for the gear-train level (Slurm test node): deep-vs-shallow probe error and step-cost ratio for variants of
gear count, coupling softness, timestep and grasp softness. Monkeypatches experiments/validation/gear-scenes.py constants;
the camera is pulled back so long trains stay in view. PYTHONPATH=src .venv/bin/python experiments/validation/gear-explore.py"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, "src")
from sim2sim.gear_train import compile_plain, panel_normal, probe_errors, reference_runs, sim_cost  # noqa: E402
from sim2sim.simulation import Simulation  # noqa: E402

spec = importlib.util.spec_from_file_location("scenes", "experiments/validation/gear-scenes.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
BASE = list(S.GEARS)
S.PANEL, S.EYE = (5., .01, .3), np.array([0., -2.4, 1.1])
S.SOURCE_CONFIG, S.CANDIDATE_CONFIG = ({**c, "width": 1280, "height": 960} for c in (S.SOURCE_CONFIG, S.CANDIDATE_CONFIG))


IDLERS = [g[1] for g in BASE if g[1] == g[3]]


def gears(n, late):
    """n gears: the design's (compound pairs 7 and 14) plus extra L0 idlers, or with late=True all idlers on L0 and the two
    compound pairs third- and last-but-one from the hand."""
    if not late:
        return BASE + [S.IDLER(0, IDLERS[k % 12]) for k in range(n - len(BASE))]
    return ([S.IDLER(0, IDLERS[k % 12]) for k in range(n - 3)] + [(0, .016, 1, .032), S.IDLER(1, .022), (1, .016, 0, .032)])


VARIANTS = [dict(n=20, late=False), dict(n=20, late=False, rho=1100, rho_crank=7800), dict(n=20, late=True),
            dict(n=20, late=True, rho=1100, rho_crank=7800), dict(n=40, late=True, rho=1100, rho_crank=7800),
            dict(n=20, late=True, rho=1100, rho_crank=7800, solimp=".95 .99 .001 .5 2")]
for v in VARIANTS:
    S.GEARS, S.SOLIMP, S.TIMESTEP = gears(v["n"], v["late"]), v.get("solimp", ".99 .999 .001 .5 2"), v.get("dt", .002)
    S.RHO, S.RHO_CRANK = v.get("rho", 2700.), v.get("rho_crank", 2700.)
    S.TOOTH_PITCH = .008 if v.get("teeth", True) else 1e3
    grasp = {"grasp_solimp": v["grasp"]} if "grasp" in v else {}
    with tempfile.TemporaryDirectory() as temp:
        deep, shallow = Path(temp) / "scene.xml", Path(temp) / "shallow.xml"
        deep.write_text(S.scene())
        shallow.write_text(S.scene(True))
        normal = panel_normal(*compile_plain(deep))
        m = compile_plain(deep)[0]
        crank_inertia = float(m.dof_M0[m.jnt_dofadr[m.joint("crank_hinge").id]])
        probes = S.probe_tables()
        if S.TIMESTEP != .002:
            probes = [{**p, "ticks": p["ticks"] * 2, "settle_ticks": p["settle_ticks"] * 2} for p in probes]
        runs = reference_runs(Simulation(deep, {**S.SOURCE_CONFIG, **grasp}), probes, normal)
        err = probe_errors(runs, Simulation(shallow, {**S.CANDIDATE_CONFIG, "timestep": S.TIMESTEP, **grasp}), normal)
        cost = sim_cost({"deep": compile_plain(deep)[0], "shallow": compile_plain(shallow)[0]}, normal,
                        {"cost_runs": 5, "cost_seconds": 1.0, "cost_torque_nm": .02}, S.TIMESTEP)
    print(json.dumps({**{k: v[k] for k in v}, "probe_error_rad": round(err["probe_error_rad"], 5),
                      "per_probe": [(p["name"], round(p["crank_rms_rad"], 5), round(p["hand_rms_rad"], 5)) for p in err["probes"]],
                      "us_per_step": {k: round(c["seconds_per_simulated_second"] * S.TIMESTEP * 1e6, 2) for k, c in cost.items()},
                      "ratio": round(cost["deep"]["seconds_per_simulated_second"] / cost["shallow"]["seconds_per_simulated_second"], 2),
                      "geoms": cost["deep"]["geoms"], "lumped_over_crank": round(S.lumped()[0] / crank_inertia, 3),
                      "gate_rad": round(S.gate(), 4)}), flush=True)
