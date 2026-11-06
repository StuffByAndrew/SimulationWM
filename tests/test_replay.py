"""Pure file/JSON replay checks; no physics, renderer, model or network calls."""
import base64
import hashlib
import json
from pathlib import Path
import pytest
from sim2sim.audit import AuditFailure,AuditLog,read_audit
from sim2sim.replay import replay_run

# One-pixel PNG kept as opaque bytes: replay must never decode/re-render it.
PNG=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')


def tick(world,ticks,episode=0):
    return dict(kind='tick',world=world,episode=episode,state={'qpos':[1.,2.],'qvel':[0.]},
        observation=dict(rgb=base64.b64encode(PNG).decode(),depth=base64.b64encode(PNG).decode(),ticks=ticks,time=ticks*.002))


def test_lossless_wires_states_and_two_world_visuals(tmp_path):
    run=tmp_path/'run';audit=AuditLog(run,max_bytes=100000)
    raw=b'{ "op": "inference", "messages": [{"content":"full prompt"}]}\r\n'
    request=dict(kind='boundary',direction='request',world='reference',wire_b64=base64.b64encode(raw).decode())
    token=audit.reserve(90000)
    audit.record(request,reservation=token)
    audit.record(tick('reference',1),reservation=token)
    audit.record(tick('candidate',2),reservation=token)
    audit.record(tick('reference',0,episode=1),reservation=token)
    audit.record(dict(kind='action_end',world='candidate',outcome='wall_timeout',executed_ticks=2),reservation=token)
    audit.release(token);audit.close()
    events=[e for _,e in read_audit(run)]
    assert events[0]==request
    assert events[1]==tick('reference',1)
    output=tmp_path/'replay';manifest=replay_run(run,output)
    assert manifest['worlds']==['candidate','reference']
    assert manifest['frames']==3
    assert (output/'wire/00000001.bin').read_bytes()==raw
    frames=json.loads((output/'frames.json').read_text())
    for frame in frames:
        assert (output/frame['rgb']).read_bytes()==PNG
        assert frame['rgb_sha256']==hashlib.sha256(PNG).hexdigest()
    assert frames[-1]['episode']==1
    timeline=json.loads((output/'timeline.json').read_text())
    assert timeline[-1]['event']['outcome']=='wall_timeout'
    assert json.loads((output/timeline[1]['state_artifacts'][0]['path']).read_text())==events[1]['state']
    assert (output/'index.html').exists()


def test_reservation_fails_before_record_or_mutation(tmp_path):
    audit=AuditLog(tmp_path,max_bytes=4096)
    before=(tmp_path/'audit/events.jsonl').read_bytes()
    with pytest.raises(AuditFailure): audit.reserve(4096)
    token=audit.reserve(10)
    with pytest.raises(AuditFailure): audit.record(tick('reference',1),reservation=token)
    assert (tmp_path/'audit/events.jsonl').read_bytes()==before
    assert not list((tmp_path/'audit/blobs').iterdir())
    token.release()
    token=audit.reserve(3500)
    audit.record({'kind':'action_begin','request':{'op':'move'}},reservation=token)
    assert token.remaining<3500
    audit.record({'kind':'action_end','outcome':'error'},reservation=token)
    token.release()
    assert len(list(read_audit(tmp_path)))==2


@pytest.mark.parametrize('damage',['blob','truncated_index','anchor','symlink'])
def test_replay_rejects_tamper_before_output(tmp_path,damage):
    run=tmp_path/'run';audit=AuditLog(run,max_bytes=100000)
    audit.record(tick('reference',1));audit.close()
    if damage=='blob':
        path=next((run/'audit/blobs').iterdir());data=path.read_bytes();path.write_bytes(bytes([data[0]^1])+data[1:])
    elif damage=='truncated_index':
        path=run/'audit/events.jsonl';path.write_bytes(path.read_bytes().rstrip(b'\n'))
    elif damage=='anchor':
        path=run/'audit/manifest.json';value=json.loads(path.read_text());value['event_count']=2;path.write_text(json.dumps(value))
    else:
        path=next((run/'audit/blobs').iterdir());outside=tmp_path/'outside';outside.write_bytes(path.read_bytes());path.unlink();path.symlink_to(outside)
    output=tmp_path/'replay'
    with pytest.raises(AuditFailure): replay_run(run,output)
    assert not output.exists()


def test_invalid_wire_and_json_marker_like_data_preserved(tmp_path):
    audit=AuditLog(tmp_path,max_bytes=100000)
    raw=b'\xffnot json\x00\n'
    event={'kind':'boundary','direction':'request','wire_b64':base64.b64encode(raw).decode(),
        'request':{'$audit_blob':'literal user text','encoding':'base64','size':42}}
    audit.record(event);audit.close()
    assert list(read_audit(tmp_path))[0][1]==event


def test_pending_action_and_terminal_partial_state_preserved(tmp_path):
    audit=AuditLog(tmp_path,max_bytes=100000)
    audit.record({'kind':'action_begin','world':'candidate','request_seq':8,'resolved_action':{'op':'move','ticks':100}})
    audit.record(tick('candidate',3))
    audit.close()
    manifest=replay_run(tmp_path,tmp_path/'playback')
    assert manifest['events']==2 and manifest['frames']==1
    timeline=json.loads((tmp_path/'playback/timeline.json').read_text())
    assert timeline[0]['event']['resolved_action']['ticks']==100
    assert timeline[1]['event']['observation']['ticks']==3


def test_submission_and_image_url_externalize_losslessly(tmp_path):
    audit=AuditLog(tmp_path,max_bytes=100000)
    encoded=base64.b64encode(PNG).decode()
    event={'project_b64':encoded,'submission_b64':encoded,'source_b64':encoded,'artifacts_b64':encoded,
        'payload':{'image_url':{'url':'data:image/png;base64,'+encoded}}}
    pointer=audit.record(event);audit.close()
    envelope=json.loads((tmp_path/'audit/blobs'/pointer['event_sha256']).read_text())
    assert len(envelope['attachments'])==5
    assert len({a['sha256'] for a in envelope['attachments']})==1
    assert list(read_audit(tmp_path))[0][1]==event


def test_video_and_declared_frame_state_hashes(tmp_path):
    run=tmp_path/'run';audit=AuditLog(run,max_bytes=100000)
    event=tick('candidate',1)
    event['state_sha256']=hashlib.sha256(json.dumps(event['state'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    event['rgb_sha256']=hashlib.sha256(PNG).hexdigest()
    audit.record(event)
    video=run/'reconstruction/interaction-000.mp4';video.parent.mkdir();video.write_bytes(b'opaque recorded video bytes')
    audit.record({'kind':'video','world':'candidate','path':'reconstruction/interaction-000.mp4',
        'sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'size':video.stat().st_size})
    audit.close()
    output=tmp_path/'replay';manifest=replay_run(run,output)
    assert (output/'videos/reconstruction/interaction-000.mp4').read_bytes()==video.read_bytes()
    assert manifest['videos'][0]['world']=='candidate'


@pytest.mark.parametrize('kind',['video_hash','video_traversal','state_hash'])
def test_declared_artifact_errors_rejected(tmp_path,kind):
    run=tmp_path/'run';audit=AuditLog(run,max_bytes=100000)
    if kind=='state_hash':
        event=tick('reference',1);event['state_sha256']='0'*64
    else:
        (run/'interaction.mp4').write_bytes(b'video')
        event={'kind':'video','world':'reference','path':'../escape.mp4' if kind=='video_traversal' else 'interaction.mp4',
            'sha256':'0'*64,'size':5}
    audit.record(event);audit.close()
    with pytest.raises(AuditFailure): replay_run(run,tmp_path/'playback')
    assert not (tmp_path/'playback').exists()


def test_viewer_compact_action_text_safe_and_timing_labeled(tmp_path):
    import re
    run=tmp_path/'run';audit=AuditLog(run,max_bytes=100000)
    malicious='</script><img src=x onerror=alert(1)>'
    audit.record({'kind':'boundary','direction':'request','world':'reference',
        'wire_b64':base64.b64encode(b'{"op":"move","ticks":1}\n').decode()})
    audit.record({'kind':'action_begin','world':'reference','request_seq':1,
        'resolved_action':{'op':malicious,'ticks':1,'target':'reference'},'before_ticks':0})
    audit.record(tick('reference',1))
    audit.record({'kind':'action_end','world':'reference','outcome':'ok','before_ticks':0,'after_ticks':1})
    audit.close();output=tmp_path/'replay';replay_run(run,output)
    document=(output/'index.html').read_text()
    assert malicious not in document
    assert 'innerHTML' not in document
    assert 'row.textContent=' in document and 'p.meta.textContent=' in document
    assert 'fixed 10 frames/second' in document
    assert 'original encoded frame rate' in document
    assert 'timeline.json' in document and 'wire/' in document
    data=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',document,re.S).group(1))
    assert data['actions'][0]['resolved_action']['op']==malicious
    assert data['frames'][0]['action']['outcome']=='ok'
    assert data['frames'][0]['action']['before_ticks']==0
    assert data['frames'][0]['action']['after_ticks']==1
    assert 'wire_b64' not in re.search(r'<script type="application/json" id="data">(.*?)</script>',document,re.S).group(1)


def test_replay_streams_two_passes_without_retaining_hydrated_events(tmp_path,monkeypatch):
    import importlib
    module=importlib.import_module('sim2sim.replay')
    counts={'passes':0,'live':0,'peak':0}
    class Event(dict):
        def __init__(self,index):
            super().__init__(kind='note',world='reference',payload=str(index)+'x'*4096)
            counts['live']+=1;counts['peak']=max(counts['peak'],counts['live'])
        def __del__(self): counts['live']-=1
    def events(_):
        counts['passes']+=1
        for index in range(30):
            yield {'seq':index+1,'hash':'0'*64},Event(index)
    monkeypatch.setattr(module,'read_audit',events)
    result=module.replay_run(tmp_path/'run',tmp_path/'playback')
    assert counts['passes']==2
    assert counts['peak']<=4
    assert result['events']==30
    timeline=json.loads((tmp_path/'playback/timeline.json').read_text())
    assert timeline[-1]['event']['payload']=='29'+'x'*4096
    assert not (tmp_path/'playback/timeline.json.tmp').exists()


def test_source_snapshot_verified_and_copied_without_parsing(tmp_path):
    run=tmp_path/'run';audit=AuditLog(run,max_bytes=100000)
    raw=b'opaque source zip bytes, deliberately not parsed'
    (run/'source-snapshot.zip').write_bytes(raw)
    audit.record({'kind':'run_start','source_snapshot':'source-snapshot.zip',
        'manifest':{'source_bundle_sha256':hashlib.sha256(raw).hexdigest()}})
    audit.close();output=tmp_path/'replay';manifest=replay_run(run,output)
    assert (output/'snapshots/source-snapshot.zip').read_bytes()==raw
    assert manifest['source_snapshots'][0]['sha256']==hashlib.sha256(raw).hexdigest()
    (run/'source-snapshot.zip').write_bytes(b'corrupt')
    with pytest.raises(AuditFailure): replay_run(run,tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()
