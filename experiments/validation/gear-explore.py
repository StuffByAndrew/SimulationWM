"""Tuning aid for the gear-train level (Slurm test node): for variants of the generator's constants (JSON list of
{NAME: value} overrides, applied cumulatively), the G-cheap gate, the shallow reference's probe error at each candidate
timestep, the chosen shallow timestep and the deep / shallow cost per simulated second.
PYTHONPATH=src .venv/bin/python experiments/validation/gear-explore.py '[{"GRASP_SOLIMP": [0.9, 0.95, 0.001, 0.5, 2]}]'"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, "src")
from sim2sim.gear_train import compile_plain, panel_normal, sim_cost  # noqa: E402

spec = importlib.util.spec_from_file_location("scenes", "experiments/validation/gear-scenes.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
for variant in json.loads(sys.argv[1]):
    for name, value in variant.items():
        setattr(S, name, value)
    for config in (S.SOURCE_CONFIG, S.CANDIDATE_CONFIG):
        config.update(grasp_solref=S.GRASP_SOLREF, grasp_solimp=S.GRASP_SOLIMP)
    gate, shallow_dt, errors = S.calibrate()
    with tempfile.TemporaryDirectory() as temp:
        deep, shallow = Path(temp) / "scene.xml", Path(temp) / "shallow.xml"
        deep.write_text(S.scene())
        shallow.write_text(S.scene(True, timestep=shallow_dt))
        cost = sim_cost({"deep": compile_plain(deep)[0], "shallow": compile_plain(shallow)[0]}, panel_normal(*compile_plain(deep)),
                        {"cost_runs": 5, "cost_seconds": 1.0, "cost_torque_nm": .02})
    print(json.dumps({"variant": variant, "gate_rad": gate, "shallow_error_at_source_step_share_of_gate": round(errors[S.TIMESTEP] / gate, 4),
                      "errors": {f"{k:g}": round(v, 5) for k, v in errors.items()}, "shallow_dt": shallow_dt,
                      "ms_per_simulated_s": {k: round(c["seconds_per_simulated_second"] * 1e3, 3) for k, c in cost.items()},
                      "ratio": round(cost["deep"]["seconds_per_simulated_second"] / cost["shallow"]["seconds_per_simulated_second"], 2)}), flush=True)
