"""Spring-powered positioning fixture: private mechanism, held-out release evaluation and the researcher view.

A fixture scene (experiments/validation/fixture-scenes.py) has a carriage on a rail (slide joint ``carriage_x``, 0 at the
ruler's zero mark, metres), a spring plunger (``plunger_x``), two dials the robot slides (``compression_dial``,
``pressure_dial``; 0 to 0.1 m of travel, read publicly as 0 to 10 scale divisions) and a brake pad whose holder
(``pad_z``) a fixed tendon spring presses onto the carriage's blade; the spring's hidden preload nut (``pad_preload``,
latched by ``latch_preload``) is moved to the pressure dial's position by each cocking cycle, so the dials carry no load
(MuJoCo's soft dry friction would let a loaded dial creep). The custom
numerics ``fixture_compression`` (spring compression = c0 + c1 * compression dial travel, m) and ``fixture_reset``
(settle ticks, pad lift range in m, pad hinge angle noise in rad) are private scene values.

``reset_carriage`` is the fixture's cocking cycle: it retracts and latches the plunger at the dialled compression with
the carriage against it, sets the pad spring preload from the pressure dial, lifts the pad by a random height with randomly perturbed hinge angles, and lets it settle onto
the blade for the settle ticks. ``release_carriage`` opens both latches. The perturbation stream comes from a private
seed; evaluation draws its own trials from the same distribution. Pad state never enters a public observation.
"""
from __future__ import annotations

import json
import time

import mujoco
import numpy as np

JOINTS = ("carriage_x", "plunger_x", "compression_dial", "pressure_dial", "pad_z", "pad_preload")
LATCHES = ("latch_plunger", "latch_carriage")
DIAL_M = .01  # dial travel per public scale division


def _numeric(model, name):
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    assert index >= 0, f"fixture scene lacks custom numeric {name}"
    start = model.numeric_adr[index]
    return model.numeric_data[start:start + model.numeric_size[index]].copy()


def is_fixture(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "fixture_compression") >= 0


class Mechanism:
    def __init__(self, sim, seed=None):
        model = sim._model
        self.sim = sim
        ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINTS}
        assert min(ids.values()) >= 0, "fixture scene lacks a named fixture joint"
        self.qpos = {name: int(model.jnt_qposadr[i]) for name, i in ids.items()}
        self.dof = {name: int(model.jnt_dofadr[i]) for name, i in ids.items()}
        self.latches = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name) for name in LATCHES]
        self.preload = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "latch_preload")
        assert min(self.latches) >= 0 and self.preload >= 0, "fixture scene lacks its latch equalities"
        self.eq_data0 = model.eq_data.copy()  # authored latch targets; a world reset restores them
        self.compression = _numeric(model, "fixture_compression")
        settle, self.lift_lo, self.lift_hi, self.hinge_sigma = _numeric(model, "fixture_reset")
        self.settle_ticks = int(settle)
        holder = model.jnt_bodyid[ids["pad_z"]]
        pad_bodies = {b for b in range(model.nbody) if b == holder or self._descends(model, b, holder)}
        hinges = [j for j in range(model.njnt) if model.jnt_bodyid[j] in pad_bodies and j != ids["pad_z"]]
        assert all(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE for j in hinges)
        self.hinge_qpos = model.jnt_qposadr[hinges]
        self.pad_dofs = np.append(model.jnt_dofadr[hinges], self.dof["pad_z"]).astype(int)
        self.rng = None if seed is None else np.random.default_rng(seed)

    @staticmethod
    def _descends(model, body, ancestor):
        while body:
            body = int(model.body_parentid[body])
            if body == ancestor:
                return True
        return False

    def readings(self):
        """Public instrument readings: both dial positions in scale divisions and whether the carriage is cocked."""
        q = self.sim._data.qpos
        return {"compression_dial": float(q[self.qpos["compression_dial"]] / DIAL_M), "pressure_dial": float(q[self.qpos["pressure_dial"]] / DIAL_M),
                "carriage_cocked": bool(self.sim._data.eq_active[self.latches[0]])}

    def set_dials(self, compression, pressure):
        """Evaluator only: put both dials directly at these scale readings (at rest)."""
        data = self.sim._data
        for name, value in (("compression_dial", compression), ("pressure_dial", pressure)):
            assert 0 <= value <= 10, value
            data.qpos[self.qpos[name]], data.qvel[self.dof[name]] = value * DIAL_M, 0
        mujoco.mj_forward(self.sim._model, data)

    def reset_carriage(self, rng):
        """Cock at the dialled compression, re-seat the pad from a random lifted pose, then settle (advances ticks)."""
        model, data = self.sim._model, self.sim._data
        x = -(self.compression[0] + self.compression[1] * data.qpos[self.qpos["compression_dial"]])
        for name in ("carriage_x", "plunger_x"):
            data.qpos[self.qpos[name]], data.qvel[self.dof[name]] = x, 0
        for latch in self.latches:
            model.eq_data[latch, 0] = x
            data.eq_active[latch] = True
        preload = data.qpos[self.qpos["pressure_dial"]]  # nut travel = dial travel; latch targets are relative to qpos0
        data.qpos[self.qpos["pad_preload"]], data.qvel[self.dof["pad_preload"]] = preload, 0
        model.eq_data[self.preload, 0] = preload - model.qpos0[self.qpos["pad_preload"]]
        data.qpos[self.qpos["pad_z"]] = rng.uniform(self.lift_lo, self.lift_hi)
        data.qpos[self.hinge_qpos] = model.qpos_spring[self.hinge_qpos] + rng.normal(0, self.hinge_sigma, len(self.hinge_qpos))
        data.qvel[self.pad_dofs] = 0
        mujoco.mj_forward(model, data)
        self.sim._advance(self.settle_ticks)

    def restore(self):
        """World reset: the authored latch targets (the reset model state also returns the nut to its written place)."""
        self.sim._model.eq_data[:] = self.eq_data0

    def release(self):
        if not self.sim._data.eq_active[self.latches[0]]:
            raise ValueError("the carriage is not cocked: reset_carriage first")
        self.sim._data.eq_active[self.latches] = False

    def position_mm(self):
        """Private: the carriage index on the ruler, mm."""
        return 1000 * float(self.sim._data.qpos[self.qpos["carriage_x"]])

    def speed(self):
        return abs(float(self.sim._data.qvel[self.dof["carriage_x"]]))


STILL_SPEED, STILL_TICKS = 1e-3, 50  # a release ends once the carriage moves slower than 1 mm/s for 50 ticks (0.1 s)


def release_trial(sim, compression, pressure, rng, max_ticks, every):
    """One evaluator release from a fresh world: dials set directly, reset_carriage with this trial's perturbation, release,
    run until the carriage rests. Returns the stop (ruler mm) and the carriage trajectory sampled every `every` ticks."""
    mechanism = sim.fixture
    sim.action({"op": "reset"}, render=False)
    mechanism.set_dials(compression, pressure)
    mechanism.reset_carriage(rng)
    mechanism.release()
    dt = sim._model.opt.timestep
    times, positions, still, tick = [0.], [mechanism.position_mm()], 0, 0
    while tick < max_ticks and (still < STILL_TICKS or tick < STILL_TICKS):
        sim._advance(1)
        tick += 1
        still = still + 1 if mechanism.speed() < STILL_SPEED else 0
        if tick % every == 0:
            times.append(round(tick * dt, 4))
            positions.append(round(mechanism.position_mm(), 3))
    return {"stop_mm": mechanism.position_mm(), "ticks": tick, "settled": still >= STILL_TICKS, "t_s": times, "x_mm": positions}


# Public predictor contract for the fixture metric (task["submission"]["predictor"]); scale readings and ruler mm.
PREDICTOR_FUNCTIONS = {
    "predict": "predict(compression, pressure, n): compression and pressure are dial readings (0 to 10 scale divisions); return a "
               "list of n numbers, your predicted ruler readings in mm at which the carriage index comes to rest in n separate "
               "releases at those dial settings, each release made from reset_carriage",
    "settings": "settings(): return {target name: {\"compression\": dial reading, \"pressure\": dial reading}} with the settings "
                "you choose for every marked target region",
}


def parse_settings(value, names):
    """The agent's chosen dial settings per target, validated on the host: {name: (compression, pressure)}."""
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"settings() must return an object with exactly the keys {sorted(names)}")
    out = {}
    for name in names:
        entry = value[name]
        if not isinstance(entry, dict) or set(entry) != {"compression", "pressure"}:
            raise ValueError(f"settings()[{name!r}] must be {{compression, pressure}}")
        reading = [entry["compression"], entry["pressure"]]
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not 0 <= x <= 10 for x in reading):
            raise ValueError(f"settings()[{name!r}] dial readings must be numbers from 0 to 10")
        out[name] = (float(reading[0]), float(reading[1]))
    return out


def collect_predictions(table, call):
    """Host side (predictor.calls): the predictor's chosen settings and its predictions at every held-out setting
    and at each chosen setting. `call(function, **kwargs)` returns {"ok", "value" | "error"}."""
    names = [target["name"] for target in table["targets"]]
    n = int(table["samples"])
    out = {"settings": call("settings"), "chosen": {},
           "grid": [call("predict", compression=u, pressure=v, n=n) for u, v in table["settings"]]}
    if out["settings"]["ok"]:
        try:
            chosen = parse_settings(out["settings"]["value"], names)
        except ValueError as exc:
            out["settings"] = {"ok": False, "error": str(exc), "value": out["settings"]["value"]}
        else:
            out["chosen"] = {name: {"setting": list(setting), **call("predict", compression=setting[0], pressure=setting[1], n=n)}
                             for name, setting in chosen.items()}
    return out


def evaluate_submission(job):
    """runtime.PREDICTOR_METRICS entry: the program answers the calls of collect_predictions in its call-mode predictor
    container (predictor.py), then the private evaluator container scores those answers against fresh source releases."""
    from . import predictor
    evaluation = job.evaluation
    with job.container("call") as proc:
        predictions = predictor.calls(proc, job.data, evaluation["entrypoint"], lambda call: collect_predictions(job.manifest["fixture"], call),
                                      min(time.monotonic() + float(evaluation.get("predictor_timeout_seconds", 900)), job.supervisor.deadline),
                                      job.logs / "predictor-calls.json")
    return job.evaluate_scene({**evaluation, "predictions": predictions})


def _samples(reply, n):
    """Validated predicted stopping positions (mm) of one query, or an error string."""
    if not reply.get("ok"):
        return reply.get("error", "predictor call failed")
    value = reply["value"]
    if not isinstance(value, list) or len(value) != n or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        return f"predict must return a list of {n} numbers"
    value = np.asarray(value, dtype=float)
    return value if np.isfinite(value).all() else "predict returned non-finite numbers"


def crps(samples, observations):
    """Mean sample CRPS (energy form) of a predictive sample set against each observation, in the observations' unit."""
    x = np.sort(np.asarray(samples, dtype=float))
    m = len(x)
    spread = 2 * np.sum(x * (2 * np.arange(m) - m + 1)) / m ** 2  # E|X - X'|
    return float(np.mean([np.mean(np.abs(x - y)) for y in observations]) - spread / 2)


def _hit(values, target):
    values = np.asarray(values, dtype=float)
    return (values >= target["lo_mm"]) & (values <= target["hi_mm"])


def _wilson(k, n, z=1.96):
    p = k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


def evaluate(sim, table, predictions, artifacts):
    """Researcher-only fixture scores: repeated real releases at the held-out settings and at the agent's chosen settings,
    scored against the isolated predictor's answers. Raw trials, trajectories and predictions go to artifacts/."""
    seed, n, targets = int(table["seed"]), int(table["samples"]), table["targets"]
    run = lambda u, v, stream, repeats: [release_trial(sim, u, v, np.random.default_rng([seed, stream, r]), int(table["max_release_ticks"]),
                                                       int(table["trajectory_every"])) for r in range(repeats)]
    grid = []
    for index, ((u, v), reply) in enumerate(zip(table["settings"], predictions["grid"])):
        samples = _samples(reply, n)
        grid.append({"compression": u, "pressure": v, "trials": run(u, v, index, int(table["repeats"])),
                     **({"predicted_mm": samples.tolist()} if isinstance(samples, np.ndarray) else {"prediction_error": samples})})
    truth = [np.array([trial["stop_mm"] for trial in setting["trials"]]) for setting in grid]
    pooled = np.concatenate(truth)
    failed = [setting for setting in grid if "prediction_error" in setting]
    prediction = {"status": "failed" if failed else "ok", "failed_queries": len(failed), "queries": len(grid),
                  "error": failed[0]["prediction_error"] if failed else None,
                  "crps_mm": None if failed else float(np.mean([crps(s["predicted_mm"], y) for s, y in zip(grid, truth)])),
                  "oracle_crps_mm": float(np.mean([np.abs(y[:, None] - y[None]).sum() / (len(y) * (len(y) - 1)) / 2 for y in truth])),
                  "mean_only_crps_mm": float(np.mean([np.mean(np.abs(y - y.mean())) for y in truth])),
                  "pooled_crps_mm": float(np.mean([crps(pooled, y) for y in truth])), "brier": {}}
    prediction["crps_skill_vs_pooled"] = None if failed else 1 - prediction["crps_mm"] / prediction["pooled_crps_mm"]
    for target in targets:
        hits = [_hit(y, target) for y in truth]
        prediction["brier"][target["name"]] = {
            "predicted": None if failed else float(np.mean([np.mean((_hit(s["predicted_mm"], target).mean() - h) ** 2) for s, h in zip(grid, hits)])),
            "mean_only": float(np.mean([np.mean((float(_hit([y.mean()], target)[0]) - h) ** 2) for y, h in zip(truth, hits)])),
            "oracle": float(np.mean([h.mean() * (1 - h.mean()) for h in hits]))}
    chosen, scores = {}, {}
    for stream, target in enumerate(targets, start=len(grid)):
        name, reply = target["name"], predictions["chosen"].get(target["name"])
        centre = (target["lo_mm"] + target["hi_mm"]) / 2
        nearest = int(np.argmin([abs(y.mean() - centre) for y in truth]))
        score = {"lo_mm": target["lo_mm"], "hi_mm": target["hi_mm"], "chosen": None, "measured_hit_fraction": None, "hits": None,
                 "trials": None, "wilson95": None, "predicted_hit_fraction": None,
                 "grid_best_hit_fraction": float(max(_hit(y, target).mean() for y in truth)),
                 "mean_choice_hit_fraction": float(_hit(truth[nearest], target).mean()),
                 "mean_choice_setting": [grid[nearest]["compression"], grid[nearest]["pressure"]]}
        if reply is None:
            score["error"] = predictions["settings"].get("error", "no chosen setting")
        else:
            u, v = reply["setting"]
            trials = run(u, v, stream, int(table["chosen_repeats"]))
            hits = int(_hit([t["stop_mm"] for t in trials], target).sum())
            samples = _samples(reply, n)
            score.update(chosen={"compression": u, "pressure": v}, hits=hits, trials=len(trials), measured_hit_fraction=hits / len(trials),
                         wilson95=_wilson(hits, len(trials)),
                         predicted_hit_fraction=float(_hit(samples, target).mean()) if isinstance(samples, np.ndarray) else None)
            chosen[name] = {"compression": u, "pressure": v, "trials": trials,
                            **({"predicted_mm": samples.tolist()} if isinstance(samples, np.ndarray) else {"prediction_error": samples})}
        scores[name] = score
    data = {"targets": targets, "samples": n, "grid": grid, "chosen": chosen, "scores": {"prediction": prediction, "targets": scores},
            "settings_reply": predictions["settings"]}
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "trials.json").write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    from .fixture_view import write_view
    view = write_view(data, artifacts / "view.html")
    return {"status": "ok", "prediction": prediction, "targets": scores,
            "view": {"path": f"artifacts/{artifacts.parent.name}/{artifacts.name}/view.html", "sha256": view}}
