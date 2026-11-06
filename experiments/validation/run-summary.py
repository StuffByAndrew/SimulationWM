"""Researcher-only wave-1 summary: per-run execution, cost and private metrics (first, best, final submission).
Usage (Slurm test node: canary matching compiles MuJoCo scenes):
  .venv/bin/python experiments/validation/run-summary.py GLOB OUTNAME, e.g. 'w1-*-20261102-003' 003.
Reads recorded artifacts only; the cost ledger pool comes from each run's private config."""
import json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATTERN, OUTNAME = sys.argv[1], sys.argv[2]
RUNS = Path(sys.argv[3]).resolve() / "runs" if len(sys.argv) > 3 else ROOT / "experiments/runs"  # optional EXPERIMENTS_ROOT


def metrics(checkpoint):
    """Flatten one evaluated checkpoint into scalar errors (None when unavailable; never fabricated)."""
    rec, out = checkpoint.get("reconstruction") or {}, {}
    out["chamfer_m"] = rec.get("chamfer_distance_m")
    out["com_err_m"] = rec.get("center_of_mass_error_m")
    density = (rec.get("density") or {}).get("per_primitive")
    out["density_mean_rel_err"] = (sum(p["relative_error"] for p in density) / len(density)) if density and all(
        p.get("relative_error") is not None for p in density) else None
    hinge = next(iter((checkpoint.get("hinges") or {}).values()), None) or {}
    for key in ("damping", "frictionloss", "stiffness"):
        out[key + "_rel_err"] = hinge.get(key + "_relative_error")
        out[key + "_abs_err"] = hinge.get(key + "_absolute_error")
        out[key + "_pred"] = hinge.get("predicted_" + key)
        out[key + "_true"] = hinge.get("source_" + key)
    out["reset_rgb_mse"] = (checkpoint.get("reset") or {}).get("rgb_mse")
    return out


def canary_distance(run, submission):
    """Blind runs: each source object geom's geometry-matched visible candidate geom (the evaluator's geometry-only
    primitive assignment, never color) and its rendered RGB distance to the canary and to the original color."""
    import tempfile, tomllib
    from sim2sim.evaluation import extract_submission
    from sim2sim.perturbation import _simconfig
    from sim2sim.primitive_metrics import _primitives, match_primitives
    from sim2sim.simulation import Simulation
    colors = ((json.loads((run / "experiment.json").read_text()).get("perturbation") or {}).get("colors") or {}).get("geoms")
    if not colors or submission is None:
        return None
    config = json.loads((run / "config-private.json").read_text())
    evaluation = config["evaluation"]
    simconfig = _simconfig(config, tomllib.loads((Path(evaluation["source_dir"]) / evaluation.get("manifest", "manifest.toml")).read_text()))
    source = Simulation(config["source"]["scene"], simconfig)
    camera = {"pos": source._data.cam_xpos[source._camera_id].tolist(),
              "quat": [float(v) for v in source._model.cam_quat[source._camera_id]], "fovy": float(source._model.cam_fovy[source._camera_id])}
    with tempfile.TemporaryDirectory() as temp:
        extract_submission(submission.read_bytes(), Path(temp) / "project", evaluation.get("entrypoint", "scene.xml"))
        candidate = Simulation(Path(temp) / "project" / evaluation.get("entrypoint", "scene.xml"),
                               {k: v for k, v in dict(simconfig, trusted_camera=camera).items() if k != "graspable_bodies"})
    for sim in (source, candidate):
        sim.action({"op": "reset"}, render=False)
    m = candidate._model
    rgba = lambda g: m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 and list(m.geom_rgba[g]) == [.5, .5, .5, 1] else m.geom_rgba[g]
    sp = _primitives(source, geom_names=list(colors))
    cp = [p for p in _primitives(candidate, exclude_robot=source.robot == "panda") if rgba(p["id"])[3] > 0]  # visible geoms only
    match = evaluation.get("reconstruction") or {}
    rows, cols, cost = match_primitives(sp, cp, 0, match.get("matching_same_type", False), match.get("matching_max_distance_m"))
    pairs = {sp[i]["name"]: j for i, j in zip(rows, cols)}
    out = {}
    for i, p in enumerate(sp):
        c, j = colors[p["name"]], pairs.get(p["name"])
        rgb = None if j is None else [float(v) for v in rgba(cp[j]["id"])[:3]]
        out[p["name"]] = {"canary": c["value"][:3], "original": c["original"][:3], "candidate_geom": None if j is None else cp[j]["name"],
                          "candidate_rgb": rgb, "geometry_matching_cost_m": None if j is None else float(cost[i, j]),
                          "distance_to_canary": None if rgb is None else math.dist(rgb, c["value"][:3]),
                          "distance_to_original": None if rgb is None else math.dist(rgb, c["original"][:3])}
    source.close(); candidate.close()
    return out


rows = []
for run in sorted(RUNS.glob(PATTERN)):
    status = json.loads((run / "status.json").read_text())
    if status["status"] == "prepared":
        continue  # superseded or not yet started: no execution or ledger
    term = json.loads((run / "termination.json").read_text()) if (run / "termination.json").exists() else {}
    pool = json.loads((run / "config-private.json").read_text())["inference"]["cost_pool"]
    ledger = json.loads((ROOT / "experiments/cost-ledger" / pool / (run.name + ".json")).read_text())
    notes = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()]
    history = json.loads((run / "metric-history.json").read_text())["checkpoints"] if (run / "metric-history.json").exists() else []
    scored = [c for c in history if c.get("status") == "ok"]
    subs = sorted(run.glob("submission-*.zip"))
    row = {"run": run.name, "factor": json.loads((run / "experiment.json").read_text()).get("physical_parameters", {}).get("factor"),
           "status": status["status"], "reason": term.get("reason"), "wall_min": round(term.get("wall_elapsed", 0) / 60, 1),
           "budget_used": round(term.get("wall_elapsed", 0) / status["wall_limit_seconds"], 3), "submissions": term.get("submissions"),
           "candidate_loads": len(list(run.glob("candidate-change-*"))), "ticks": term.get("total_ticks"),
           "cost_usd": round(ledger["cost_usd"], 2), "calls": ledger["calls"],
           "early_finish_intercepts": sum("Intercepted an early finish" in str(n.get("text")) for n in notes),
           "budget_stop": any("budget exhausted" in str(n.get("text")) for n in notes),
           "first_submission_min": next((round(c["wall_elapsed"] / 60, 1) for c in history if "wall_elapsed" in c), None),
           "first": metrics(scored[0]) if scored else None, "final": metrics(scored[-1]) if scored else None,
           "canary": canary_distance(run, subs[-1] if subs else None)}
    numeric = [metrics(c) for c in scored]
    row["best"] = {k: min((m[k] for m in numeric if m[k] is not None), default=None) for k in (numeric[0] if numeric else {})
                   if k.endswith(("_err", "_m", "_mse"))}
    rows.append(row)

out = ROOT / f"experiments/validation/results-{OUTNAME}.json"
out.write_text(json.dumps(rows, indent=1))
for r in rows:
    print(r["run"], r["status"], r["reason"], f'{r["wall_min"]}min', f'used={r["budget_used"]}', f'subs={r["submissions"]}',
          f'${r["cost_usd"]}', f'intercepts={r["early_finish_intercepts"]}', "\n   final:", r["final"], "\n   best:", r["best"],
          ("\n   canary: " + json.dumps(r["canary"])) if r["canary"] else "")
print("wrote", out)
