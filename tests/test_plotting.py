import json
import math
from pathlib import Path

from PIL import Image

from sim2sim.plotting import checkpoint_series, update_metric_history


def report(version, chamfer=.1, center=.2, density_error=50):
    return {"status": "ok", "version": version, "scores": {"reconstruction": {
        "chamfer_distance_m": chamfer,
        "center_of_mass_error_m": center,
        "center_of_mass_error_vector_m": [.1, .1, .141421356],
        "source_center_of_mass_m": [0., 0., .2],
        "predicted_center_of_mass_m": [.1, .1, .341421356],
        "density": {"status": "ok", "coverage": 1., "unmatched_source": [], "unmatched_candidate": [],
                    "per_primitive": [{"source_name": "part_a", "candidate_name": "renamed_candidate",
                                       "source_density_kg_m3": 450., "predicted_density_kg_m3": 450. + density_error,
                                       "absolute_error_kg_m3": density_error, "relative_error": density_error / 450.}]}}}}


def test_aligned_series_preserves_missing_gaps_and_original_labels():
    reconstruction = report("one")["scores"]["reconstruction"]
    checkpoints = [{"submission": 1, "status": "ok", "reconstruction": reconstruction},
                   {"submission": 2, "status": "failed", "reconstruction": reconstruction},
                   {"submission": 3, "status": "ok", "reconstruction": {"chamfer_distance_m": 0., "density": {"unmatched_source": ["part_a"]}}}]
    curves = checkpoint_series(checkpoints)
    assert curves["submission"] == [1, 2, 3]
    assert curves["chamfer_distance_m"][0] == .1
    assert math.isnan(curves["chamfer_distance_m"][1])
    assert curves["chamfer_distance_m"][2] == 0.
    assert list(curves["density_absolute_error_kg_m3"]) == ["part_a"]
    assert math.isnan(curves["density_absolute_error_kg_m3"]["part_a"][2])
    assert math.isnan(curves["center_of_mass_error_m"][2])


def test_checkpoint_history_png_and_full_raw_metrics(tmp_path):
    first = report("a" * 64)
    update_metric_history(tmp_path, 1, first)
    update_metric_history(tmp_path, 2, {"status": "failed", "version": "b" * 64, "scores": None, "error": "invalid MJCF"})
    third = report("c" * 64, chamfer=0., center=0., density_error=0.)
    history = update_metric_history(tmp_path, 3, third)
    assert len(history["checkpoints"]) == 3
    assert history["checkpoints"][0]["reconstruction"] == first["scores"]["reconstruction"]
    assert history["checkpoints"][1]["reconstruction"] is None
    assert history["checkpoints"][1]["unavailable_reason"] == "invalid MJCF"
    assert history["checkpoints"][2]["version"] == "c" * 64
    assert history["plot_summary"]["chamfer_distance_m"]["unavailable_checkpoints"] == 1
    assert history["plot_summary"]["density"]["unavailable_primitive_checkpoint_pairs"] == 1
    assert json.loads((tmp_path / "metric-history.json").read_text()) == history
    with Image.open(tmp_path / "metric-history.png") as png:
        assert png.format == "PNG"
        assert png.width >= 1200 and png.height >= 1000
    # Replacing an index is idempotent: it cannot create a second checkpoint.
    update_metric_history(tmp_path, 3, third)
    assert len(json.loads((tmp_path / "metric-history.json").read_text())["checkpoints"]) == 3


def test_disabled_or_failed_scores_are_unavailable_not_zero(tmp_path):
    history = update_metric_history(tmp_path, 1, {"status": "ok", "version": "disabled", "scores": {"mode": "disabled"}})
    assert history["checkpoints"][0]["reconstruction"] is None
    for key in ("chamfer_distance_m", "center_of_mass_error_m", "density"):
        assert history["plot_summary"][key]["unavailable_checkpoints"] == 1
    assert history["plot_summary"]["density"]["primitive_count"] == 0


def test_hinge_history_independent_of_reconstruction(tmp_path):
    hinge={'status':'ok','source_joint':'hinge','candidate_joint':'renamed',
        'source_damping':.12,'predicted_damping':.24,'damping_absolute_error':.12,'damping_relative_error':1.,
        'source_frictionloss':.05,'predicted_frictionloss':.08,'frictionloss_absolute_error':.03,'frictionloss_relative_error':.6}
    first={'status':'ok','version':'hinge-one','scores':{'hinges':{'hinge':hinge}}}
    update_metric_history(tmp_path,1,first)
    update_metric_history(tmp_path,2,{'status':'failed','version':'failed','scores':None})
    last=dict(hinge,damping_absolute_error=0.,frictionloss_absolute_error=0.)
    history=update_metric_history(tmp_path,3,{'status':'ok','version':'hinge-exact','scores':{'hinges':{'hinge':last}}})
    assert history['checkpoints'][0]['reconstruction'] is None
    assert history['checkpoints'][0]['hinges']['hinge']==hinge
    curves=checkpoint_series(history['checkpoints'])['hinges']['hinge']
    assert curves['damping_absolute_error'][0]==.12
    assert math.isnan(curves['damping_absolute_error'][1])
    assert curves['damping_absolute_error'][2]==0
    assert history['hinge_plot_summary']['frictionloss']['unavailable_hinge_checkpoint_pairs']==1
    with Image.open(tmp_path/'hinge-history.png') as png:
        assert png.format=='PNG'
        assert png.width>=1000
    assert json.loads((tmp_path/'metric-history.json').read_text())['checkpoints'][0]['hinges']['hinge']==hinge
