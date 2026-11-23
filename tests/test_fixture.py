"""Positioning-fixture levels: hidden mechanism, isolated predictor worker and scoring (physics: Slurm test node)."""
import base64
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from sim2sim.candidate_worker import public_observation
from sim2sim.fixture import collect_predictions, crps, evaluate, parse_settings
from sim2sim.simulation import Simulation

LEVELS = Path(__file__).resolve().parents[1] / "environments/panda_small"
CONFIG = {"robot": "panda", "width": 64, "height": 48, "rgb": False, "depth": False, "calibration": False}


def test_crps_and_settings_validation():
    assert crps([3.0] * 4, [3.0]) == 0
    assert crps([0.0, 10.0], [0.0, 10.0]) == pytest.approx(2.5) and crps([5.0], [0.0, 10.0]) == pytest.approx(5)
    assert parse_settings({"t": {"compression": 2, "pressure": 9.5}}, ["t"]) == {"t": (2.0, 9.5)}
    for bad in ({"t": {"compression": 11, "pressure": 1}}, {"t": {"compression": 1}}, {"u": {"compression": 1, "pressure": 1}}, [1, 2]):
        with pytest.raises(ValueError):
            parse_settings(bad, ["t"])


def test_mechanism_requires_cocking_and_hides_pad_state():
    sim = Simulation(LEVELS / "fixture_rubber/scene.xml", {**CONFIG, "fixture_seed": 7})
    observation = public_observation(sim.observe(full=True), CONFIG)
    assert observation["fixture"] == {"compression_dial": pytest.approx(5), "pressure_dial": pytest.approx(5), "carriage_cocked": False}
    assert set(observation) == {"time", "ticks", "total_ticks", "proprioception", "fixture"}
    with pytest.raises(ValueError, match="not cocked"):
        sim.action({"op": "release_carriage"}, render=False)
    sim.action({"op": "reset_carriage"}, render=False)
    assert sim.total_ticks == sim.fixture.settle_ticks and sim.fixture.readings()["carriage_cocked"]
    start = sim.fixture.position_mm()
    sim.action({"op": "release_carriage", "ticks": 400}, render=False)
    assert sim.fixture.position_mm() > start + 10 and not sim.fixture.readings()["carriage_cocked"]
    with pytest.raises(ValueError, match="positioning-fixture"):
        Simulation(LEVELS / "blocks/scene.xml", CONFIG).action({"op": "reset_carriage"}, render=False)


PREDICTOR = '''
import os
print("stdout noise from the predictor")
os.write(1, b"raw fd noise\\n")
def predict(compression, pressure, n):
    return [10.0 * compression + pressure] * n
def slow():
    while True:
        pass
def swallow():
    try:
        while True:
            pass
    except Exception:
        return 1
def huge():
    return [0.0] * 400000
def nan():
    return float("nan")
'''


def test_predictor_worker_bounds_time_output_and_protocol(tmp_path):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("predictor.py", PREDICTOR)
    worker = subprocess.Popen([sys.executable, "-c", "import sys; from pathlib import Path; from sim2sim.predictor import main; main(root=Path(sys.argv[1]))",
                               str(tmp_path / "predictor")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    messages = [{"op": "call", "function": "predict", "kwargs": {"compression": 1, "pressure": 2, "n": 2}},
                {"op": "load", "project_b64": base64.b64encode(data.getvalue()).decode(), "entrypoint": "predictor.py", "call_seconds": 1},
                *({"op": "call", "function": name, "kwargs": kwargs} for name, kwargs in
                  (("predict", {"compression": 1, "pressure": 2, "n": 2}), ("slow", {}), ("swallow", {}), ("huge", {}), ("nan", {}),
                   ("missing", {}), ("predict", {"bogus": 1})))]
    out, err = worker.communicate("".join(json.dumps(m) + "\n" for m in messages).encode(), timeout=60)
    replies = [json.loads(line) for line in out.decode().splitlines()]
    assert [r["ok"] for r in replies] == [False, True, True, False, False, False, False, False, False]
    assert replies[2]["value"] == [12.0, 12.0]
    assert "CallTimeout" in replies[3]["error"] and "CallTimeout" in replies[4]["error"]
    assert "exceeds" in replies[5]["error"] and "ValueError" in replies[6]["error"]
    assert b"stdout noise" in err and b"raw fd noise" in err


def test_evaluate_scores_a_predictor_and_writes_the_view(tmp_path):
    sim = Simulation(LEVELS / "fixture_rigid/scene.xml", CONFIG)
    table = {"seed": 1, "settings": [[8.0, 3.0], [6.0, 6.0]], "repeats": 3, "chosen_repeats": 3, "samples": 5, "max_release_ticks": 1500,
             "trajectory_every": 10, "targets": [{"name": "target", "lo_mm": 80.0, "hi_mm": 110.0}, {"name": "narrow_target", "lo_mm": 30.0, "hi_mm": 32.0}]}
    def call(function, **kwargs):
        if function == "settings":
            return {"ok": True, "value": {"target": {"compression": 8, "pressure": 3}, "narrow_target": {"compression": 6, "pressure": 6}}}
        return {"ok": True, "value": [95.0] * kwargs["n"]} if kwargs["compression"] == 8 else {"ok": False, "error": "ValueError: no"}
    scores = evaluate(sim, table, collect_predictions(table, call), tmp_path / "artifacts/v/fixture")
    assert scores["prediction"]["status"] == "failed" and scores["prediction"]["crps_mm"] is None and scores["prediction"]["failed_queries"] == 1
    target = scores["targets"]["target"]
    assert target["trials"] == 3 and target["predicted_hit_fraction"] == 1 and 0 <= target["measured_hit_fraction"] <= 1
    assert scores["targets"]["narrow_target"]["predicted_hit_fraction"] is None
    page = (tmp_path / "artifacts/v/fixture/view.html").read_text()
    assert scores["view"]["path"] == "artifacts/v/fixture/view.html" and '"x_mm"' in page and "<script" in page
    json.dumps(scores, allow_nan=False)
