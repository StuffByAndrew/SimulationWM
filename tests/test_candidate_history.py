import base64
import hashlib
import json
from sim2sim.candidate_history import recover_changes


def records(outcome='ok', version=None):
    archive = b'archived-project'
    request = {'op': 'load_reconstruction', 'project_b64': base64.b64encode(archive).decode(), 'entrypoint': 'main.xml'}
    return [(None, {'kind': 'boundary', 'direction': 'request', 'action_id': 7,
                    'wire_b64': base64.b64encode(json.dumps(request).encode()).decode()}),
            (None, {'kind': 'action_end', 'operation': 'load_reconstruction', 'action_id': 7,
                    'outcome': outcome, 'candidate_version': version or hashlib.sha256(archive).hexdigest(), 'wall_elapsed': 12.5})]


def test_recovery_preserves_original_clock_and_archive():
    changes, unavailable, _ = recover_changes(records())
    assert changes[0]['archive'] == b'archived-project'
    assert changes[0]['wall_elapsed'] == 12.5
    assert changes[0]['entrypoint'] == 'scene.xml'  # protocol resolves its own entrypoint
    assert unavailable == []


def test_failed_load_and_archive_mismatch_are_gaps():
    changes, unavailable, _ = recover_changes(records('error'))
    assert not changes and unavailable[0]['reason'] == 'original_load_failed'
    changes, _, _ = recover_changes(records(version='wrong'))
    assert 'archive' not in changes[0]
    assert 'recovery_error' in changes[0]


def test_verified_manifest_and_resolved_entrypoint_override_request():
    metadata = {}
    events = [(None, {'kind': 'run_start', 'manifest': {'source_bundle_sha256': 'trusted', 'evaluation': {'metrics': ['chamfer']}}}),
              (None, {'kind': 'candidate_load_intent', 'action_id': 7, 'entrypoint': 'resolved.xml'})] + records()
    changes, _, digest = recover_changes(events, metadata)
    assert digest == 'trusted' and metadata['evaluation']['metrics'] == ['chamfer']
    assert changes[0]['entrypoint'] == 'resolved.xml'


def test_submission_without_load_is_recovered_once():
    events = records()
    events[-1][1]['operation'] = 'submit'
    events[0][1]['wire_b64'] = base64.b64encode(json.dumps({'op': 'submit', 'project_b64': base64.b64encode(b'archived-project').decode()}).encode()).decode()
    changes, _, _ = recover_changes(events + events)
    assert len(changes) == 1
    assert changes[0]['checkpoint_kind'] == 'submitted_candidate'
