import base64
import json
import os

import pytest

from sim2sim.audit import AuditFailure, AuditLog, _digest, blob_reader, read_audit
from sim2sim.compact import compact


def _run(tmp_path, seal=True):
    run = tmp_path / 'run-001'
    audit = AuditLog(run, max_bytes=64 * 1024 * 1024)
    frame = base64.b64encode(b'\x89PNG' + os.urandom(3000)).decode()
    for tick in range(40):
        audit.record({'kind': 'tick', 'world': 'reference', 'state': {'qpos': [tick * .1, 1.0], 'time': tick},
                      'observation': {'rgb': frame if tick % 10 else base64.b64encode(os.urandom(500)).decode(),
                                      'depth': base64.b64encode(bytes(200)).decode()}})
    audit.record({'kind': 'boundary', 'direction': 'request', 'wire_b64': base64.b64encode(b'{"op":"observe"}' * 50).decode()})
    if seal:
        audit.close()
    orphan = os.urandom(100)   # a blob a killed run wrote before its index line: kept, not dropped
    (run / 'audit' / 'blobs' / _digest(orphan)).write_bytes(orphan)
    (run / 'status.json').write_text(json.dumps({'status': 'completed'}))
    return run, _digest(orphan), orphan


def test_compaction_keeps_every_event_and_blob_and_removes_loose_files(tmp_path):
    run, orphan_digest, orphan = _run(tmp_path)
    before = [(pointer, event) for pointer, event in read_audit(run)]
    loose = len(os.listdir(run / 'audit' / 'blobs'))
    summary = compact(run)
    assert summary['status'] == 'compacted' and summary['loose_files_removed'] == loose
    assert os.listdir(run / 'audit' / 'blobs') == []
    assert sorted(os.listdir(run / 'audit' / 'pack')) == ['depth.pack', 'events.pack', 'index.tsv.gz', 'other.pack', 'rgb.pack', 'state.pack']
    assert [(pointer, event) for pointer, event in read_audit(run)] == before   # budget check holds again after removal
    read, close = blob_reader(run)
    assert read(orphan_digest) == orphan
    close()
    assert compact(run)['loose_files_removed'] == 0   # idempotent: a compacted run re-verifies and removes nothing


def test_compaction_refuses_live_or_recent_unsealed_runs(tmp_path):
    run, _, _ = _run(tmp_path, seal=False)
    with pytest.raises(AuditFailure, match='last 2 h'):
        compact(run)
    with pytest.raises(AuditFailure, match='live Slurm job'):
        compact(run, live='scripts/experiment-strict.slurm run-001')
    assert len(os.listdir(run / 'audit' / 'blobs')) > 10   # nothing removed
