"""Explicit development bridge; native checks run only on Slurm test nodes."""
import json
import os
from pathlib import Path
import threading
import time
from unittest.mock import patch

import pytest

from sim2sim import dev_rpc
from sim2sim.devrunner import run_development, native_process
from sim2sim.runtime import RuntimeFailure, WallClock, run


def test_development_is_explicit_and_production_never_falls_back(tmp_path):
    config = {"sandbox": {}, "budget": {"wall_seconds": 10}, "output_dir": str(tmp_path/"run"),
              "execution_mode": "development_shared_filesystem"}
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeFailure, match="Slurm test"):
            run_development(config, tmp_path/"public")
    with patch("sim2sim.runtime.check_rootless", side_effect=RuntimeFailure("PRODUCTION-GUARD")), patch("sim2sim.devrunner.native_process") as launch:
        with pytest.raises(RuntimeFailure, match="PRODUCTION-GUARD"):
            run(config)
        launch.assert_not_called()
    with pytest.raises(RuntimeFailure, match="exact gpt-6-astra"):
        run_development(config, tmp_path/"public", model="gpt-6-sol")


def test_native_worker_environment_does_not_inherit_credentials_or_source(tmp_path):
    clock = WallClock(10)
    config = {"source": {"scene": "/PRIVATE/SCENE.xml"}, "development_worker": {}}
    with patch.dict(os.environ, {"OPENAI_API_KEY": "SECRET-KEY", "AWS_SECRET_ACCESS_KEY": "AWS-KEY", "SLURM_JOB_ID": "123", "SLURM_JOB_PARTITION": "test"}), \
         patch("sim2sim.devrunner.GroupProcess") as launch:
        native_process("candidate", tmp_path/"worker", config, clock)
    args, kwargs = launch.call_args
    assert "PRIVATE" not in json.dumps(args) + json.dumps(kwargs["env"])
    assert "SECRET-KEY" not in json.dumps(kwargs["env"]) and "AWS-KEY" not in json.dumps(kwargs["env"])
    assert kwargs["start_new_session"] is True
    assert kwargs["env"]["SLURM_JOB_PARTITION"] == "test"
    clock.close()


SCENE = '''<mujoco><option timestep=".002"/><worldbody>
<camera name="fixed" pos="1 -2 1" xyaxes=".894 .447 0 -.18 .36 .915"/>
<light pos="0 0 2"/><geom type="plane" size="2 2 .1"/>
<body name="secret_reference_object" pos="0 0 .3"><freejoint/><geom name="object" type="box" size=".1 .1 .1" mass="2.3"/></body>
</worldbody></mujoco>'''


def configuration(tmp_path, wall=60):
    source = tmp_path/"source"
    source.mkdir()
    (source/"scene.xml").write_text(SCENE)
    return {"sandbox": {}, "output_dir": str(tmp_path/"private"), "record_video": False,
            "task": "Reconstruct the unknown scene using permitted observations and interactions.",
            "source": {"scene": str(source/"scene.xml")},
            "simulation": {"camera":"fixed","width":64,"height":64,"calibration":False},
            "budget": {"wall_seconds":wall,"active_seconds":wall,"physics_ticks":30,"submissions":2},
            "evaluation": {"source_dir":str(source),"camera":"fixed","width":64,"height":64}}


native = pytest.mark.skipif(not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") not in ("test", "sapphire"),
                            reason="native development bridge requires a Slurm test or sapphire allocation")


@native
@pytest.mark.parametrize("wall_only", [False, True])
def test_native_mailbox_pipeline_independent_worlds_private_evaluation(tmp_path, wall_only):
    from sim2sim.experiments import prepare_experiment, start_experiment
    original = configuration(tmp_path)
    preset = tmp_path/"prepared.toml"
    preset.write_text(f'''task = "Reconstruct the unknown scene using permitted observations and interactions."
record_video = false
plot_checkpoints = false
evaluate_candidate_changes = {str(wall_only).lower()}
[source]
scene = {json.dumps(original["source"]["scene"])}
[sandbox]
[budget]
wall_seconds = 60
wall_only = {str(wall_only).lower()}
active_seconds = 60
physics_ticks = {0 if wall_only else 30}
submissions = {0 if wall_only else 2}
[simulation]
camera = "fixed"
width = 64
height = 64
calibration = false
[evaluation]
camera = "fixed"
width = 64
height = 64
[experiment]
name = "Private prepared native regression"
environment = "trusted-fixture-private-name"
execution_mode = "development_shared_filesystem"
model = "gpt-6-astra"
evaluation_metrics = ["appearance", "reconstruction"]
[experiment.physical_parameters]
mass = "Researcher-only mass description"
''')
    store = tmp_path/"experiments"
    prepared = prepare_experiment(preset, store, "native-prepared")
    directory = Path(prepared["run_dir"])
    config = json.loads((directory/"config-private.json").read_text())
    public = directory/"public"
    result, errors = [], []
    def controller():
        try:
            result.append(start_experiment(prepared["id"], store, development=True, refresh_dashboard=False))
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=controller, daemon=True)
    with patch("sim2sim.runtime.InferenceBroker.call", side_effect=AssertionError("No host API calls in development mode")):
        thread.start()
        deadline = time.monotonic()+40
        while not (public/"task.json").exists() and not errors and not result and time.monotonic()<deadline:
            time.sleep(.025)
        assert not errors, errors
        assert not result, result
        task = json.loads((public/"task.json").read_text())
        encoded = json.dumps(task)
        assert "secret_reference_object" not in encoded and 'mass=\"2.3\"' not in encoded
        assert str(tmp_path/"source") not in encoded
        assert task["execution_mode"] == "development_shared_filesystem"
        assert task["agent_model"] == "gpt-6-astra" and task["model_usage"]["tokens_available"] is False
        assert task["deadline_unix"] > time.time()
        if wall_only:
            assert all(task["budgets"][key] is None for key in ("physics_ticks","submissions","active_seconds"))
        own = public/"project"
        (own/"scene.xml").write_text(SCENE.replace("secret_reference_object", "own_hypothesis").replace('mass="2.3"','mass="1.0"'))
        def call(request, project=None):
            response = dev_rpc.call(public, request, timeout=30, project=project)
            assert response["ok"], response
            return response
        call({"op":"progress","origin":"model","text":"Testing my own hypothesis against the observed dynamics."})
        assert call({"op":"observe","ticks":2})["observation"]["total_ticks"] == 2
        loaded = call({"op":"load_reconstruction"}, "project")
        assert "scores" not in loaded
        assert json.loads((public/"task.json").read_text())["deadline_unix"] == task["deadline_unix"]
        assert call({"op":"observe","world":"reconstruction","ticks":3})["observation"]["total_ticks"] == 3
        assert call({"op":"reset","world":"reconstruction"})["observation"]["ticks"] == 0
        assert call({"op":"observe"})["observation"]["total_ticks"] == 2
        submission = call({"op":"submit"}, "project")
        assert set(submission).issuperset({"ok","submission","status"}) and "scores" not in submission
        assert submission["status"] == "ok"
        call({"op":"finish"})
        thread.join(timeout=15)
    assert not thread.is_alive() and not errors, errors
    assert result[0]["status"] == "completed", result
    assert result[0]["result"]["world_ticks"] == {"reference":2,"reconstruction":3}
    from sim2sim.audit import read_audit
    evaluations = [event for _, event in read_audit(directory) if event.get("kind") == "evaluation"]
    changes = [event for event in evaluations if event.get("checkpoint_kind") == "candidate_change"]
    assert len(changes) == int(wall_only)
    if wall_only:
        assert changes[0]["change_index"] == 1 and "submission" not in changes[0]
        assert changes[0]["report"]["status"] == "ok"
    assert len([event for event in evaluations if "submission" in event]) == 1
    assert (public/"stopped.json").exists()
    assert not (public/"source").exists() and not (public/"evaluation.json").exists()
    private_report = json.loads((Path(config["output_dir"])/"version-001/evaluation.json").read_text())
    assert private_report["status"] == "ok" and private_report["scores"]
    assert json.loads((Path(config["output_dir"])/"manifest.json").read_text())["execution_mode"] == "development_shared_filesystem"


@native
def test_development_wall_stops_waiting_controller(tmp_path):
    config = configuration(tmp_path, wall=3)
    public = Path(config["output_dir"])/"public"
    assert run_development(config, public)["reason"] == "wall_deadline"
    status = json.loads((Path(config["output_dir"])/"termination.json").read_text())
    assert status["hard_wall_expired"] and status["status"] == "finished"
    assert (public/"stopped.json").exists()


def test_prepared_guard_allows_only_regular_nfs_silly_renames(tmp_path):
    from sim2sim.runtime import validate_prepared_output
    for name in ("experiment.json", "status.json", "config-private.json"):
        (tmp_path/name).write_text("{}")
    (tmp_path/"source").mkdir()
    transient = tmp_path/".nfs0000000a6bb929da00005fe4"
    transient.write_text("old open metadata")
    validate_prepared_output(tmp_path)
    assert transient.read_text() == "old open metadata"
    transient.unlink()
    transient.symlink_to(tmp_path/"status.json")
    with pytest.raises(RuntimeFailure, match="unexpected"):
        validate_prepared_output(tmp_path)
    transient.unlink()
    for name in (".nfsother", ".nfs", ".nfsABCDEF", "events.jsonl", "status.json.tmp"):
        path = tmp_path/name
        path.write_text("must not be accepted")
        with pytest.raises(RuntimeFailure, match="unexpected"):
            validate_prepared_output(tmp_path)
        path.unlink()
    transient.mkdir()
    with pytest.raises(RuntimeFailure, match="unexpected"):
        validate_prepared_output(tmp_path)
