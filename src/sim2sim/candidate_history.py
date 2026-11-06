"""Private candidate-load evaluation, with immutable retrospective provenance."""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
import uuid
from pathlib import Path

from .audit import read_audit, _atomic
from .runtime import RuntimeFailure, WallClock, decode_submission


def recover_changes(records, metadata=None):
    """Consume verified audit records; keep original successful load timestamps."""
    requests, changes, unavailable = {}, [], []
    source_digest = None
    intents = {}
    seen_versions = set()
    for _, event in records:
        if event.get('kind') == 'run_start':
            source_digest = event.get('manifest', {}).get('source_bundle_sha256')
            if metadata is not None:
                metadata.update(event.get('manifest', {}))
        action = event.get('action_id')
        if event.get('kind') == 'candidate_load_intent':
            intents[action] = event
        if event.get('kind') == 'boundary' and event.get('direction') == 'request':
            wire = event.get('wire_b64')
            if wire:
                request = json.loads(base64.b64decode(wire, validate=True))
            else:
                request = event.get('payload', {})
            if isinstance(request, dict) and request.get('op') in ('load_reconstruction', 'submit'):
                requests[action] = request
        if event.get('kind') != 'action_end' or event.get('operation') not in ('load_reconstruction', 'submit'):
            continue
        submitted = event.get('operation') == 'submit'
        row = dict(checkpoint_kind='submitted_candidate' if submitted else 'candidate_change', change_index=len(changes)+1, action_id=action,
                   wall_elapsed=event.get('wall_elapsed'), version=event.get('candidate_version'))
        if event.get('outcome') != 'ok':
            unavailable.append(dict(action_id=action, wall_elapsed=event.get('wall_elapsed'), reason='original_submission_failed' if submitted else 'original_load_failed'))
            requests.pop(action, None)
            continue
        try:
            request = requests.pop(action)
            archive = decode_submission(request.get('project_b64'), 8*1024*1024)
            digest = hashlib.sha256(archive).hexdigest()
            if submitted:
                row['version'] = digest
                if digest in seen_versions:
                    continue
            if digest != row['version']:
                raise ValueError('candidate archive digest differs from successful load')
            row.update(archive=archive, entrypoint=intents.get(action, {}).get('entrypoint', (metadata or {}).get('evaluation', {}).get('entrypoint', 'scene.xml')))
            seen_versions.add(digest)
        except (KeyError, ValueError, RuntimeFailure) as exc:
            row.update(recovery_error=str(exc))
        changes.append(row)
    return changes, unavailable, source_digest


def source_entrypoint(config):
    """Recorded source scene path inside the source snapshot bundle."""
    evaluation = config['evaluation']
    source_dir = Path(evaluation.get('source_dir', Path(config['source']['scene']).parent))
    return evaluation.get('source_entrypoint', Path(config['source']['scene']).relative_to(source_dir).as_posix())


def _save(path, value):
    _atomic(path, json.dumps(value, indent=2, allow_nan=False).encode())


def require_compute_allocation():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') not in ('test', 'sapphire'):
        raise RuntimeFailure('Candidate evaluation requires a Slurm test or sapphire allocation')


def trusted_evaluation(run, config, digest, timeout):
    """The recorded evaluation configuration and hash-verified source snapshot of a run, as at its original evaluation."""
    source = (run/'source-snapshot.zip').read_bytes()
    if not digest or hashlib.sha256(source).hexdigest() != digest:
        raise RuntimeFailure('Recorded source snapshot hash mismatch')
    evaluation = dict(config['evaluation'])
    evaluation['source_config'] = config.get('resolved_source_config', config.get('simulation', {}))
    metrics = evaluation.get('metrics')
    if metrics is not None:
        evaluation['metrics'] = list(dict.fromkeys([*metrics, 'image_mse']))
    evaluation['timeout_seconds'] = timeout
    evaluation['source_entrypoint'] = source_entrypoint(config)
    return evaluation, source


def evaluate_archive(config, evaluation, source, archive, entrypoint, worker_root, timeout):
    """One bounded native development evaluation of a trusted archive; failures become explicit failed reports."""
    from .devrunner import DevelopmentBackend
    worker_root.mkdir(parents=True, exist_ok=True)
    (worker_root/'project.zip').write_bytes(archive)
    clock = WallClock(timeout)
    backend = DevelopmentBackend(dict(config, output_dir=str(worker_root)), worker_root/'unused-public', clock)
    try:
        return backend.evaluate(archive, dict(evaluation, entrypoint=entrypoint), {}, worker_root/('result-'+uuid.uuid4().hex), source, clock)
    except Exception as exc:
        return {'status': 'failed', 'scores': None, 'error_type': type(exc).__name__, 'error': str(exc)}
    finally:
        clock.close()


def backfill(run_dir, *, timeout=600):
    """Verify once, then use bounded native workers on a test compute allocation."""
    require_compute_allocation()
    run = Path(run_dir).resolve()
    path = run/'candidate-change-history.json'
    if not (run/'audit/manifest.json').is_file():
        history = {'schema_version': 1, 'checkpoints': [], 'unavailable': [{'reason': 'verified_audit_unavailable'}]}
        _save(path, history)
        return history
    config = {}
    changes, unavailable, digest = recover_changes(read_audit(run), config)
    if not changes:
        history = {'schema_version': 1, 'checkpoints': [], 'unavailable': unavailable + [{'reason': 'no_recorded_successful_candidate_loads'}]}
        _save(path, history)
        return history
    evaluation, source = trusted_evaluation(run, config, digest, timeout)
    path = run/'candidate-change-history.json'
    history = json.loads(path.read_text()) if path.exists() else {'schema_version': 1, 'checkpoints': []}
    history['unavailable'] = unavailable
    history['limitations'] = ['Only recorded successful loads and saved submission archives can be recovered; edits never loaded or saved are unavailable.']
    done = {(r['change_index'], r['version']) for r in history['checkpoints']}
    for row in changes:
        archive = row.pop('archive', None)
        if (row['change_index'], row['version']) in done:
            continue
        row.update(origin='retrospective')
        if archive is None:
            report = {'status': 'unavailable', 'scores': None, 'error': row.pop('recovery_error')}
        else:
            report = evaluate_archive(config, evaluation, source, archive, row['entrypoint'],
                                      run/'candidate-changes'/('change-%04d' % row['change_index']), timeout)
        row.update(evaluated_at=datetime.now(timezone.utc).isoformat(), status=report['status'], report=report)
        history['checkpoints'].append(row)
        _save(path, history)
        print(json.dumps({'run': run.name, 'change_index': row['change_index'], 'status': row['status']}), flush=True)
    _save(path, history)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    for run in args.runs:
        try:
            backfill(run, timeout=args.timeout)
        except Exception as exc:
            # Verification failure is a private explicit gap, never permission to
            # evaluate unverified source or candidate bytes. Keep other runs moving.
            destination = Path(run)/'candidate-change-history.json'
            history = json.loads(destination.read_text()) if destination.exists() else {'schema_version': 1, 'checkpoints': []}
            history['unavailable'] = [{'reason': 'recovery_failed', 'error_type': type(exc).__name__, 'error': str(exc)}]
            _save(destination, history)
            print(json.dumps({'run': str(run), 'status': 'unavailable', 'error': str(exc)}), flush=True)

if __name__ == '__main__':
    main()
