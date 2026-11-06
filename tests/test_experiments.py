"""Pure configuration/private-file checks; no native model or model API calls."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from sim2sim.experiments import load_experiment, prepare_experiment, start_experiment
from sim2sim.runtime import SandboxFailure


@pytest.fixture
def specification(tmp_path):
    source=tmp_path/'authored';source.mkdir()
    (source/'scene.xml').write_text('<mujoco><worldbody><body name="object"><freejoint/><geom type="sphere" size=".1" density="100"/></body></worldbody></mujoco>')
    (source/'manifest.toml').write_text('name="fixture"\n')
    config=tmp_path/'experiment.toml'
    config.write_text(f'''task="Infer the observed scene."
[experiment]
name="Private fixture"
environment="fixture"
model_constraints=["Use primitives."]
evaluation_metrics=["appearance","reconstruction"]
[experiment.physical_parameters]
density="PRIVATE_PARAMETER_CANARY_8e6a: the authored density is 100 kg/m³."
[source]
scene="{source/'scene.xml'}"
[budget]
wall_seconds=1800
active_seconds=1800
physics_ticks=100
submissions=2
[sandbox]
image="trusted/image"
[evaluation]
source_dir="{source}"
''')
    return config,source


def test_private_source_edit_is_actual_and_never_forwarded_metadata(specification,tmp_path):
    config,source=specification;root=tmp_path/'experiments'
    prepared=prepare_experiment(config,root,'chosen')
    directory=Path(prepared['run_dir']);scene=directory/'source/scene.xml'
    scene.write_text(scene.read_text().replace('density="100"','density="750"'))
    with patch('sim2sim.runtime.run',return_value={'total_ticks':2}) as run:
        result=start_experiment('chosen',root,refresh_dashboard=False)
    assert result['status']=='completed'
    native=run.call_args.args[0]
    assert 'PRIVATE_PARAMETER_CANARY' not in json.dumps(native)
    assert native['evaluation']['metrics']==['appearance','reconstruction']
    assert run.call_args.kwargs=={'prepared_output':True}
    assert 'density="100"' in (source/'scene.xml').read_text()
    assert 'density="750"' in scene.read_text()
    metadata=json.loads((directory/'experiment.json').read_text())
    assert metadata['source_changed_after_prepare']
    assert metadata['resolved_source_sha256']==hashlib.sha256(scene.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='only a prepared'):
        start_experiment('chosen',root,refresh_dashboard=False)


def test_blocked_attempt_is_archived_without_agent_claim(specification,tmp_path):
    config,_=specification;root=tmp_path/'experiments'
    prepare_experiment(config,root,'blocked')
    with patch('sim2sim.runtime.run',side_effect=SandboxFailure('resource verification failed')):
        result=start_experiment('blocked',root,refresh_dashboard=False)
    assert result['status']=='blocked' and result['agent_launched'] is False
    directory=root/'runs/blocked'
    boundary=json.loads((directory/'private-boundary.json').read_text())
    assert boundary['execution_evidence']['image_boundary_verified'] is False
    assert boundary['static_configuration_check']['private_description_keys_found_in_task']==[]
    assert (directory/'preflight.json').is_file()


@pytest.mark.parametrize('mutation',[
    ('wall_seconds=1800','wall_seconds=-1'),
    ('evaluation_metrics=["appearance","reconstruction"]','evaluation_metrics=["made_up"]'),
    ('density="PRIVATE_PARAMETER_CANARY_8e6a: the authored density is 100 kg/m³."','density=100'),
])
def test_invalid_research_configuration_fails_before_run(specification,mutation):
    config,_=specification;config.write_text(config.read_text().replace(*mutation))
    with pytest.raises(ValueError):load_experiment(config)


def test_leaked_private_description_refuses_launch(specification,tmp_path):
    config,_=specification;root=tmp_path/'experiments';prepare_experiment(config,root,'leak')
    path=root/'runs/leak/config-private.json';native=json.loads(path.read_text())
    native['task']='PRIVATE_PARAMETER_CANARY_8e6a: the authored density is 100 kg/m³.'
    path.write_text(json.dumps(native))
    with patch('sim2sim.runtime.run') as run:
        result=start_experiment('leak',root,refresh_dashboard=False)
    run.assert_not_called();assert result['status']=='failed'


def test_source_snapshot_failure_has_durable_failed_status(specification,tmp_path):
    config,_=specification;root=tmp_path/'experiments'
    with patch('sim2sim.experiments.shutil.copytree',side_effect=OSError('controlled source copy failure')):
        with pytest.raises(OSError):prepare_experiment(config,root,'copy-failed')
    status=json.loads((root/'runs/copy-failed/status.json').read_text())
    assert status['status']=='failed' and status['stage']=='source_snapshot'
    assert status['agent_launched'] is False


def test_post_launch_failure_never_claims_no_model_calls(specification,tmp_path):
    config,_=specification;root=tmp_path/'experiments';prepare_experiment(config,root,'partial')
    path=root/'runs/partial/config-private.json';native=json.loads(path.read_text())
    native['inference']={'endpoint':'https://example.invalid','model':'gpt-6-astra'};path.write_text(json.dumps(native))
    def fail_after_audit(*args,**kwargs):
        (root/'runs/partial/audit').mkdir();raise RuntimeError('controlled post-launch failure')
    with patch('sim2sim.runtime.run',side_effect=fail_after_audit):
        result=start_experiment('partial',root,refresh_dashboard=False)
    assert result['agent_launched'] is None and result['model_called'] is None


def test_development_requires_explicit_mode_and_routes_without_api(specification,tmp_path):
    import sys,types
    from unittest.mock import Mock
    config,_=specification
    config.write_text(config.read_text().replace('environment="fixture"','environment="fixture"\nexecution_mode="development_shared_filesystem"\nmodel="gpt-6-astra"'))
    root=tmp_path/'experiments';prepare_experiment(config,root,'development')
    with pytest.raises(ValueError,match='explicit --development'):
        start_experiment('development',root,refresh_dashboard=False)
    module=types.ModuleType('sim2sim.devrunner');module.run_development=Mock(return_value={'total_ticks':2})
    with patch.dict(sys.modules,{'sim2sim.devrunner':module}),patch('sim2sim.runtime.run') as production:
        result=start_experiment('development',root,refresh_dashboard=False,development=True)
    production.assert_not_called()
    native,public=module.run_development.call_args.args
    assert 'inference' not in native and 'PRIVATE_PARAMETER_CANARY' not in json.dumps(native)
    assert public==root/'runs/development/public'
    assert module.run_development.call_args.kwargs=={'prepared_output':True,'model':'gpt-6-astra'}
    assert result['execution_mode']=='development_shared_filesystem'
    assert result['agent_launched'] is None and result['model_called'] is None


def test_included_source_edit_changes_tree_hash_even_when_entrypoint_unchanged(specification,tmp_path):
    config,source=specification;(source/'parts.xml').write_text('<mujocoinclude/>')
    root=tmp_path/'experiments';prepare_experiment(config,root,'included')
    (root/'runs/included/source/parts.xml').write_text('<mujocoinclude><!-- requested density edit --></mujocoinclude>')
    with patch('sim2sim.runtime.run',return_value={}):start_experiment('included',root,refresh_dashboard=False)
    metadata=json.loads((root/'runs/included/experiment.json').read_text())
    assert metadata['source_sha256']==metadata['resolved_source_sha256']
    assert metadata['source_changed_after_prepare']
