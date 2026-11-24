"""Continuation of a finished strict run: parent validation, public carry-over scrubbing and the read-only prior mount.

The OCI check is opt-in on a Slurm test node with the trusted image loaded:
SIM2SIM_OCI_TEST=1 SIM2SIM_PROCESS_LIMIT_BACKEND=slurm_step sbatch -c 4 --mem=8G --export=ALL scripts/test.slurm tests/test_continuation.py
"""
import hashlib
import io
import json
import os
from pathlib import Path
import time
import uuid
import zipfile
from unittest.mock import patch
import pytest
from sim2sim.experiments import CONTINUATION_NOTE, prepare_experiment, start_experiment

PARENT = 'w9-fixture-base-20261102-001'
CONFIG = '''task="{task}"
[experiment]
name="Continuation fixture"
environment="damped_lever"
execution_mode="{mode}"
model="test-model"
model_constraints=["Use primitives."]
evaluation_metrics=["appearance"]
[experiment.physical_parameters]
density="PRIVATE_PARAMETER_CANARY"
[source]
scene="{source}/scene.xml"
[budget]
wall_only=true
wall_seconds={wall}
[sandbox]
image="trusted/image"
[simulation]
rgb=true
depth={depth}
[evaluation]
source_dir="{source}"
{extra}'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tree(root):
    return {path.relative_to(root).as_posix(): sha(path.read_bytes()) for path in sorted(root.rglob('*')) if path.is_file()}


def finished_parent(tmp_path, notes=('I see a lever.',), status='completed'):
    source = tmp_path / 'authored'
    source.mkdir()
    (source / 'scene.xml').write_text('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
    (source / 'manifest.toml').write_text('name="PRIVATE_MANIFEST_CANARY"\n')
    root = tmp_path / 'experiments'

    def config(name, wall=1800, depth='false', mode='strict_oci', task='Infer the observed scene.', submission='"last"', parent=PARENT):
        extra = f'[continuation]\nparent="{parent}"\nsubmission={submission}\n' if name != 'parent' else ''
        path = tmp_path / (name + '.toml')
        path.write_text(CONFIG.format(task=task, mode=mode, source=source, wall=wall, depth=depth, extra=extra))
        return path

    parent = Path(prepare_experiment(config('parent'), root, PARENT)['run_dir'])
    metadata = json.loads((parent / 'experiment.json').read_text())
    metadata['resolved_source_tree_sha256'] = metadata['source_tree_sha256']
    (parent / 'experiment.json').write_text(json.dumps(metadata))
    (parent / 'status.json').write_text(json.dumps({'status': status, 'result': {'status': 'finished', 'reason': 'wall_deadline', 'submissions': 2}}))
    for index, body in ((1, b'<mujoco/>'), (2, b'<mujoco><!-- v2 --></mujoco>')):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('scene.xml', body)
            archive.writestr('parts/a.xml', b'<mujoco/>')
        (parent / f'submission-00{index}.zip').write_bytes(buffer.getvalue())
        (parent / f'version-00{index}').mkdir()
        (parent / f'version-00{index}/evaluation.json').write_text(json.dumps({'status': 'ok', 'version': sha(buffer.getvalue()), 'scores': {'SCORE_CANARY': 1}}))
    (parent / 'metric-history.json').write_text('{"SCORE_CANARY": 1}')
    events = [('agent', 'adapter', 'Sandbox agent started.'), *(('agent', 'model', text) for text in notes),
              ('agent', 'agent', 'CLIENT_NOTE_CANARY'), ('provider_summary', 'provider', 'PROVIDER_SUMMARY_CANARY')]
    (parent / 'progress.jsonl').write_text(''.join(json.dumps({'wall_elapsed': 3725.2, 'source': source_, 'origin': origin, 'text': text}) + '\n'
                                                   for source_, origin, text in events))
    return root, config, parent


def test_continuation_stages_only_public_agent_authored_carry_over(tmp_path):
    root, config, parent = finished_parent(tmp_path)
    before = tree(parent)
    child = Path(prepare_experiment(config('child', wall=7200), root, 'child-001')['run_dir'])
    assert tree(parent) == before
    prior = child / 'prior'
    files = {path.relative_to(prior).as_posix(): path.read_bytes() for path in prior.rglob('*') if path.is_file()}
    assert set(files) == {'project/scene.xml', 'project/parts/a.xml', 'progress-notes.txt'}
    assert files['project/scene.xml'] == b'<mujoco><!-- v2 --></mujoco>'
    notes = files['progress-notes.txt'].decode()
    assert '[1:02:05 model] I see a lever.\n' in notes and 'adapter' not in notes.split('\n', 1)[1]
    visible = json.dumps(sorted(files)) + b''.join(files.values()).decode()
    for private in ('CANARY', PARENT, 'damped_lever', 'Continuation fixture', 'density'):
        assert private not in visible
    metadata = json.loads((child / 'experiment.json').read_text())
    hashes = {relative: sha(data) for relative, data in files.items()}
    assert metadata['continuation'] == {
        'parent': PARENT, 'submission': 2, 'submission_sha256': sha((parent / 'submission-002.zip').read_bytes()),
        'parent_status': 'completed', 'parent_result': 'finished', 'parent_reason': 'wall_deadline',
        'parent_source_tree_sha256': metadata['source_tree_sha256'], 'public_path': '/opt/public/prior', 'public_files_sha256': hashes}
    private = json.loads((child / 'config-private.json').read_text())
    assert private['task'] == json.loads((parent / 'config-private.json').read_text())['task'] + '\n' + CONTINUATION_NOTE
    assert private['budget']['wall_seconds'] == 7200
    assert private['continuation'] == {'public_dir': str(prior), 'files_sha256': hashes}
    with patch('sim2sim.runtime.run', return_value={}) as run:
        assert start_experiment('child-001', root, refresh_dashboard=False)['status'] == 'completed'
    assert run.call_args.args[0]['continuation'] == private['continuation']

    first = Path(prepare_experiment(config('first', submission=1), root, 'child-002')['run_dir'])
    assert (first / 'prior/project/scene.xml').read_bytes() == b'<mujoco/>'
    (first / 'source/scene.xml').write_text('<mujoco/>')
    with pytest.raises(ValueError, match='identical to its parent'):
        start_experiment('child-002', root, refresh_dashboard=False)
    assert json.loads((first / 'status.json').read_text())['status'] == 'prepared'


@pytest.mark.parametrize('case,message', [
    ('running', 'must have completed'), ('range', r'1\.\.2'), ('physics', 'simulation and observation'),
    ('task', 'public task'), ('development', 'strict_oci'), ('missing', 'does not exist'), ('source', 'resolved source tree'),
    ('tampered', 'accepted version'), ('run_id_note', 'private identifiers'), ('environment_note', 'private identifiers'), ('spend_note', 'inference spend')])
def test_continuation_refusals(tmp_path, case, message):
    notes = {'run_id_note': (f'Continuing {PARENT}.',), 'environment_note': ('This is the Damped_Lever scene.',),
             'spend_note': ('The inference budget exhausted at $12.',)}.get(case, ('Fine.',))
    root, config, parent = finished_parent(tmp_path, notes, 'running' if case == 'running' else 'completed')
    options = {'range': {'submission': 3}, 'physics': {'depth': 'true'}, 'task': {'task': 'Another task.'},
               'development': {'mode': 'development_shared_filesystem'}, 'missing': {'parent': 'absent-run'}}.get(case, {})
    if case == 'source':
        (tmp_path / 'authored/scene.xml').write_text('<mujoco/>')
    if case == 'tampered':
        (parent / 'submission-002.zip').write_bytes(b'PK\x05\x06' + bytes(18))
    with pytest.raises(ValueError, match=message):
        prepare_experiment(config('child', **options), root, 'child-001')
    status = json.loads((root / 'runs/child-001/status.json').read_text())
    assert (status['status'], status['stage'], status['agent_launched']) == ('failed', 'continuation', False)
    assert not (root / 'runs/child-001/experiment.json').exists()


@pytest.mark.skipif(os.environ.get('SIM2SIM_OCI_TEST') != '1', reason='requires explicit rootless OCI integration opt-in')
def test_prior_volume_is_read_only_and_readable_by_sandbox_user(tmp_path):
    import subprocess
    from sim2sim.runtime import (AGENT_IMAGE, PUBLIC_PRIOR, Transport, WallClock, launch_container, podman_env, public_prior_files,
                                 stage_public_volume, stop_container)
    root, config, parent = finished_parent(tmp_path)
    prior = Path(prepare_experiment(config('child', wall=7200), root, 'child-001')['run_dir']) / 'prior'
    files = public_prior_files(prior)
    code = r'''import hashlib, json, os, subprocess
from pathlib import Path
root = Path("/opt/public/prior")
try:
    (root / "x").write_text("x"); writable = True
except OSError:
    writable = False
copied = subprocess.run(["bash", "-c", "cp -r /opt/public/prior/project /workspace/project && echo edit >> /workspace/project/scene.xml && cat /opt/public/prior/progress-notes.txt"],
                        capture_output=True, text=True)
print(json.dumps({"files": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()},
                  "owners": sorted({p.stat().st_uid for p in root.rglob("*")}), "uid": os.getuid(), "writable": writable,
                  "copy": [copied.returncode, copied.stdout, copied.stderr], "mounts": Path("/proc/self/mountinfo").read_text()}), flush=True)
input()
'''
    supervisor = WallClock(300)
    volume = stage_public_volume(files, supervisor)
    name = 'sim2sim-prior-' + uuid.uuid4().hex
    sandbox = {'process_limit_backend': os.environ.get('SIM2SIM_PROCESS_LIMIT_BACKEND', 'cgroup'), 'memory': '1g', 'cpus': 1, 'pids': 64}
    proc = launch_container(os.environ.get('SIM2SIM_AGENT_IMAGE', AGENT_IMAGE), ['python', '-c', code], sandbox, name,
                            Path(os.devnull), supervisor, mounts=((volume, PUBLIC_PRIOR),))
    try:
        result = Transport(proc, 1 << 20).receive(time.monotonic() + 60)
        assert result['files'] == {relative: sha(data) for relative, data in files.items()}
        assert set(result['files']) == {'project/scene.xml', 'project/parts/a.xml', 'progress-notes.txt'}
        assert result['uid'] == 65532 and 65532 not in result['owners'] and not result['writable']
        assert result['copy'][0] == 0 and 'I see a lever.' in result['copy'][1], result['copy']
        mount = next(line for line in result['mounts'].splitlines() if line.split()[4] == PUBLIC_PRIOR)
        assert 'ro' in mount.split()[5].split(','), mount
        for private in ('CANARY', PARENT, 'damped_lever', str(tmp_path)):
            assert private not in json.dumps(result['files']) + result['copy'][1]
    finally:
        stop_container(proc, name)
        supervisor.close()
        subprocess.run(['podman', 'volume', 'rm', '--force', volume], capture_output=True, timeout=15, env=podman_env())
