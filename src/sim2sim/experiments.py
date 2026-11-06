"""Private experiment configuration and durable run-status orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import tomllib
import uuid

from .evaluation import selected_metrics
from . import perturbation as perturbations
from .runtime import PUBLIC_PRIOR, enabled_tools, extract_bundle, public_prior_files


@dataclass(frozen=True)
class Experiment:
    name: str
    environment: str
    physical_parameters: dict[str, str]
    model_constraints: tuple[str, ...]
    evaluation_metrics: tuple[str, ...]
    execution_mode: str = "strict_oci"
    model: str | None = None
    perturbation: dict | None = None
    continuation: dict | None = None


SAFE_ID = r'[A-Za-z0-9][A-Za-z0-9._-]{0,95}'
ENVIRONMENTS = Path(__file__).resolve().parents[2] / 'environments'
CONTINUATION_NOTE = ('A previous session on this same scene has ended. One of its submitted projects and its public progress notes '
                     f'are available read-only under {PUBLIC_PRIOR} (project/ and progress-notes.txt); you may copy and build on them.')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _text(value, field, maximum=4096):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f'{field} must be nonempty text of at most {maximum} characters')
    return value


def _texts(value, field):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f'{field} must be an array of strings')
    return tuple(_text(item, field, 1024) for item in value)


def load_experiment(path):
    path = Path(path).resolve()
    with path.open('rb') as handle:
        config = tomllib.load(handle)
    metadata = config.pop('experiment', None)
    perturbation = config.pop('perturbation', None)
    continuation = config.pop('continuation', None)
    if perturbation is not None:
        perturbations.validate(perturbation)
    if not isinstance(metadata, dict):
        raise ValueError('configuration requires a private [experiment] table')
    parameters = metadata.get('physical_parameters', {})
    if not isinstance(parameters, dict) or len(parameters) > 64:
        raise ValueError('experiment.physical_parameters must be a text-keyed table with at most 64 entries')
    parameters = {_text(key, 'physical parameter key', 128): _text(value, 'physical parameter description')
                  for key, value in parameters.items()}
    metrics = _texts(metadata.get('evaluation_metrics', ['appearance', 'dynamics', 'reconstruction']),
                     'experiment.evaluation_metrics')
    selected_metrics({'metrics': list(metrics)})
    if not metrics:
        raise ValueError('choose at least one evaluation metric family')
    mode = metadata.get('execution_mode', 'strict_oci')
    if mode not in ('strict_oci', 'development_shared_filesystem'):
        raise ValueError('unsupported experiment.execution_mode')
    model = metadata.get('model', config.get('inference', {}).get('model'))
    if mode == 'development_shared_filesystem':
        _text(model, 'experiment.model', 100)
    spec = Experiment(_text(metadata.get('name'), 'experiment.name', 200),
                      _text(metadata.get('environment'), 'experiment.environment', 200), parameters,
                      _texts(metadata.get('model_constraints', []), 'experiment.model_constraints'), metrics, mode, model,
                      perturbation, continuation)
    if enabled_tools(config) and mode != 'strict_oci':
        raise ValueError('[tools] requires strict_oci')
    budget = config.get('budget', {})
    wall_only = budget.get('wall_only', False)
    if not isinstance(wall_only, bool):
        raise ValueError('budget.wall_only must be boolean')
    for field in (('wall_seconds',) if wall_only else ('wall_seconds', 'active_seconds')):
        value = budget.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'budget.{field} must be a positive finite number')
    for field in (() if wall_only else ('physics_ticks', 'submissions')):
        value = budget.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'budget.{field} must be a nonnegative integer')
    source = Path(_text(config.get('source', {}).get('scene'), 'source.scene')).resolve()
    if not source.is_file():
        raise ValueError('source.scene does not exist')
    config['source']['scene'] = str(source)
    evaluation = config.setdefault('evaluation', {})
    if 'metrics' in evaluation and set(evaluation['metrics']) != set(metrics):
        raise ValueError('evaluation.metrics conflicts with experiment.evaluation_metrics')
    evaluation['metrics'] = list(metrics)
    evaluation.setdefault('source_dir', str(source.parent))
    evaluation['source_dir'] = str(Path(evaluation['source_dir']).resolve())
    goal = _text(config.get('task', 'Reconstruct the observed reset scene and its physical behavior as a self-contained MuJoCo scene.xml project. Use permitted observations and interactions, submit your best reconstruction, then finish.'), 'task')
    if spec.model_constraints:
        goal += '\nModel constraints:\n' + '\n'.join('- ' + item for item in spec.model_constraints)
    config['task'] = goal
    return spec, config, {'config_path': str(path), 'config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _tree_hash(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('source tree must not contain symlinks')
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode() + b'\0')
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _clock(seconds):
    seconds = int(seconds)
    return '%d:%02d:%02d' % (seconds // 3600, seconds // 60 % 60, seconds % 60)


SPEND = re.compile(rb"\$\s?\d|\bUSD\b|dollar|cost budget|inference budget|budget exhausted|budget_stop|max_cost|cost pool", re.I)


def stage_continuation(table, spec, config, directory, relative_scene):
    """Validate a finished strict parent and stage only agent-authored public carry-over under directory/prior."""
    if not isinstance(table, dict) or set(table) != {'parent', 'submission'}:
        raise ValueError('[continuation] requires exactly parent and submission')
    parent_id = table['parent']
    if not isinstance(parent_id, str) or not re.fullmatch(SAFE_ID, parent_id):
        raise ValueError('continuation.parent must be a run id')
    parent = directory.parent / parent_id
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError('continuation parent run does not exist')
    status = json.loads((parent / 'status.json').read_text())
    metadata = json.loads((parent / 'experiment.json').read_text())
    parent_config = json.loads((parent / 'config-private.json').read_text())
    if spec.execution_mode != 'strict_oci' or metadata.get('execution_mode') != 'strict_oci':
        raise ValueError('continuation requires strict_oci for both the parent and the new run')
    if status.get('status') != 'completed' or status['result']['status'] != 'finished':
        raise ValueError('continuation parent must have completed (agent finish or wall deadline)')
    count = status['result']['submissions']
    number = count if table['submission'] == 'last' else table['submission']
    if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= count:
        raise ValueError(f'continuation.submission must be "last" or an accepted submission 1..{count}')
    archive = (parent / ('submission-%03d.zip' % number)).read_bytes()
    digest = hashlib.sha256(archive).hexdigest()
    if json.loads((parent / ('version-%03d' % number) / 'evaluation.json').read_text())['version'] != digest:
        raise ValueError('continuation submission archive does not match the parent accepted version')
    if _tree_hash(directory / 'source') != metadata['resolved_source_tree_sha256'] or \
            parent_config['evaluation']['source_entrypoint'] != relative_scene.as_posix():
        raise ValueError('continuation must use the parent resolved source tree and entrypoint')
    if parent_config.get('simulation') != config.get('simulation'):
        raise ValueError('continuation must keep the parent simulation and observation settings')
    if parent_config['task'] != config['task']:
        raise ValueError('continuation must keep the parent public task')
    # Only the agent's own model-origin notes: adapter notes can carry host-side events (e.g. budget suspension).
    notes = ['Public progress notes of the previous session (its own model-origin notes), as [elapsed wall time, origin] text.']
    for line in (parent / 'progress.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event['source'] == 'agent' and event['origin'] == 'model':
            notes.append('[%s %s] %s' % (_clock(event['wall_elapsed']), event['origin'], event['text']))
    prior = directory / 'prior'
    extract_bundle(archive, prior / 'project')
    (prior / 'progress-notes.txt').write_text('\n'.join(notes) + '\n')
    files = public_prior_files(prior)
    # Single plain words (blocks, spring, panda) are legitimate physical descriptions; identifiers are not.
    private = {parent_id, metadata['environment'], *(path.name for path in ENVIRONMENTS.iterdir() if path.is_dir())}
    private = sorted(name for name in private if not name.isalpha())
    leaked = [name for name in private for relative, data in files.items() if name.lower().encode() in relative.lower().encode() + b'\0' + data.lower()]
    if leaked:
        raise ValueError('continuation carry-over contains private identifiers: ' + ', '.join(sorted(set(leaked))))
    if any(SPEND.search(data) for data in files.values()):
        raise ValueError('continuation carry-over mentions inference spend; agents must never see cost information')
    config['task'] += '\n' + CONTINUATION_NOTE
    hashes = {relative: hashlib.sha256(data).hexdigest() for relative, data in files.items()}
    config['continuation'] = {'public_dir': str(prior), 'files_sha256': hashes}
    return {'parent': parent_id, 'submission': number, 'submission_sha256': digest, 'parent_status': status['status'],
            'parent_result': status['result']['status'], 'parent_reason': status['result']['reason'],
            'parent_source_tree_sha256': metadata['resolved_source_tree_sha256'], 'public_path': PUBLIC_PRIOR,
            'public_files_sha256': hashes}


def prepare_experiment(config_path, root='experiments', experiment_id=None):
    """Snapshot a private source that can be edited before starting the run."""
    spec, config, origin = load_experiment(config_path)
    identifier = experiment_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    if not re.fullmatch(SAFE_ID, identifier):
        raise ValueError('experiment id must be a safe single directory name')
    directory = Path(root).resolve() / 'runs' / identifier
    source_root = Path(config['evaluation']['source_dir']).resolve()
    original_scene = Path(config['source']['scene'])
    relative_scene = original_scene.relative_to(source_root)
    if source_root == directory or source_root in directory.parents:
        raise ValueError('source directory must not contain the experiment output directory')
    if any(path.is_symlink() for path in source_root.rglob('*')):
        raise ValueError('private source snapshots require regular files, not symlinks')
    directory.mkdir(parents=True, exist_ok=False)
    _write(directory / 'status.json', {'schema_version': 1, 'id': identifier, 'status': 'preparing', 'started_at': utc_now()})
    stage, record, continuation, parameters = 'source_snapshot', None, None, dict(spec.physical_parameters)
    try:
        shutil.copytree(source_root, directory / 'source')
        if spec.perturbation is not None:
            stage = 'perturbation'
            record = perturbations.apply(directory / 'source' / relative_scene,
                                         directory / 'source' / config['evaluation'].get('manifest', 'manifest.toml'),
                                         spec.perturbation, original_scene, config)
            for family in ('colors', 'physics', 'pose'):
                if family in record:
                    assert f'perturbation.{family}' not in parameters, 'reserved physical parameter key'
                    parameters[f'perturbation.{family}'] = json.dumps(record[family], separators=(',', ':'))
        if spec.continuation is not None:
            stage = 'continuation'
            continuation = stage_continuation(spec.continuation, spec, config, directory, relative_scene)
    except Exception as exc:
        _write(directory / 'status.json', {'schema_version': 1, 'id': identifier, 'status': 'failed',
            'stage': stage, 'error': str(exc), 'finished_at': utc_now(), 'agent_launched': False})
        raise
    config['source']['scene'] = str(directory / 'source' / relative_scene)
    config['evaluation']['source_dir'] = str(directory / 'source')
    config['evaluation']['source_entrypoint'] = relative_scene.as_posix()
    config['output_dir'] = str(directory)
    metadata = {'schema_version': 1, 'id': identifier, 'name': spec.name, 'environment': spec.environment,
                'created_at': utc_now(), 'physical_parameters': parameters,
                'model_constraints': list(spec.model_constraints), 'evaluation_metrics': list(spec.evaluation_metrics),
                'model': spec.model, 'reasoning_effort': config.get('reasoning_effort'), 'execution_mode': spec.execution_mode,
                'source_tree_sha256': _tree_hash(directory / 'source'), **origin,
                **({'perturbation': record} if record else {}), **({'continuation': continuation} if continuation else {})}
    _write(directory / 'experiment.json', metadata)
    _write(directory / 'config-private.json', config)
    _write(directory / 'status.json', {'schema_version': 1, 'id': identifier, 'status': 'prepared', 'prepared_at': utc_now()})
    return {'id': identifier, 'run_dir': str(directory), 'status': 'prepared'}


def start_experiment(experiment_id, root='experiments', *, refresh_dashboard=True, development=False):
    """Start a prepared private source; edits are reflected in its recorded hash."""
    from . import runtime
    if not re.fullmatch(SAFE_ID, experiment_id):
        raise ValueError('experiment id must be a safe single directory name')
    directory = Path(root).resolve() / 'runs' / experiment_id
    if directory.is_symlink():
        raise ValueError('experiment directory must not be a symlink')
    previous = json.loads((directory / 'status.json').read_text())
    if previous.get('status') != 'prepared':
        raise ValueError('only a prepared experiment can be started; use a new id for another run')
    config = json.loads((directory / 'config-private.json').read_text())
    metadata = json.loads((directory / 'experiment.json').read_text())
    parameters = metadata.get('physical_parameters', {})
    if not isinstance(parameters, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in parameters.items()):
        raise ValueError('private physical parameter descriptions must remain text keyed/text valued')
    mode = metadata.get('execution_mode', 'strict_oci')
    if mode not in ('strict_oci', 'development_shared_filesystem'):
        raise ValueError('unsupported execution mode in prepared metadata')
    if development != (mode == 'development_shared_filesystem'):
        raise ValueError('development_shared_filesystem requires explicit --development and a development configuration')
    source = Path(config['source']['scene']).resolve()
    source.relative_to((directory / 'source').resolve())
    metadata['resolved_source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata['resolved_source_tree_sha256'] = _tree_hash(directory / 'source')
    metadata['source_changed_after_prepare'] = metadata['resolved_source_tree_sha256'] != metadata['source_tree_sha256']
    if 'continuation' in metadata and metadata['resolved_source_tree_sha256'] != metadata['continuation']['parent_source_tree_sha256']:
        raise ValueError('a continuation source must stay identical to its parent resolved source tree')
    metadata['started_at'] = utc_now()
    _write(directory / 'experiment.json', metadata)
    status = {'schema_version': 1, 'id': experiment_id, 'status': 'running', 'started_at': utc_now(),
              'wall_limit_seconds': config['budget']['wall_seconds'], 'execution_mode': mode,
              'agent_launched': None if development else False, 'model_called': None if development else False}
    _write(directory / 'status.json', status)
    inference = config.get('inference', {})
    preflight = {'schema_version': 1, 'started_at': utc_now(), 'host_credential_present':
                 bool(os.environ.get(inference.get('api_key_env', 'OPENAI_API_KEY'))) if inference.get('endpoint') else None}
    leaked = [key for key, value in metadata.get('physical_parameters', {}).items()
              if isinstance(value, str) and value and value in config['task']]
    boundary = {'schema_version': 1,
                'static_configuration_check': {'private_description_keys_found_in_task': leaked,
                    'experiment_metadata_passed_to_runtime': False,
                    'agent_task': config['task'], 'agent_model': metadata.get('model'), 'execution_mode': mode},
                'execution_evidence': {'agent_launch_confirmed': False, 'image_boundary_verified': False},
                'runtime_module': str(Path(runtime.__file__).resolve()),
                'runtime_source_sha256': hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest(),
                'prepared_output_entries_before_runtime': sorted(path.name for path in directory.iterdir())}

    clock = time.monotonic()
    try:
        if leaked:
            raise ValueError('private parameter descriptions occur in the public task; revise task/model constraints')
        if 'perturbation' in metadata:
            status['perturbation_verification'] = perturbations.verify(directory)
        if development:
            from .devrunner import run_development
            result = run_development(config, directory / 'public', prepared_output=True, model=metadata['model'])
        else:
            result = runtime.run(config, prepared_output=True)
        status.update(status='completed', result=result, agent_launched=None if development else True,
                      model_called=None if development or inference.get('endpoint') else False,
                      evidence_note='Completed runtime; consult verified inference boundary records for model calls.')
    except Exception as exc:
        message = str(exc)
        blocked = isinstance(exc, runtime.SandboxFailure) or any(token in message.lower() for token in ('resource verification', 'rootless', 'credential'))
        status.update(status='blocked' if blocked else 'failed', reason=type(exc).__name__, error=message,
                      agent_launched=None if (directory / 'audit').exists() else False,
                      model_called=None if (directory / 'audit').exists() and (development or inference.get('endpoint')) else False,
                      evidence_note='Runtime audit exists; execution details require verified records.' if (directory / 'audit').exists() else 'No runtime audit: failure occurred before agent/model execution.')
    finally:
        status.update(finished_at=utc_now(), wall_elapsed=time.monotonic() - clock)
        preflight['finished_at'] = utc_now()
        boundary['execution_evidence'].update(audit_present=(directory / 'audit').is_dir(),
            runtime_termination_present=(directory / 'termination.json').is_file(),
            agent_launch_confirmed=status.get('agent_launched'),
            preflight_error=status.get('error'))
        _write(directory / 'status.json', status)
        _write(directory / 'preflight.json', preflight)
        _write(directory / 'private-boundary.json', boundary)
        if refresh_dashboard:
            try:
                from .dashboard import build_dashboard
                build_dashboard(root)
            except Exception as exc:
                _write(directory / 'dashboard-error.json', {'error': str(exc), 'at': utc_now()})
    return {'run_dir': str(directory), **status}


def run_experiment(config_path, root='experiments', experiment_id=None, *, refresh_dashboard=True, development=False):
    prepared = prepare_experiment(config_path, root, experiment_id)
    return start_experiment(prepared['id'], root, refresh_dashboard=refresh_dashboard, development=development)
