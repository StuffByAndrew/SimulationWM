"""Pure public mailbox checks: no model, physics, container or network execution."""
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile
import pytest
from sim2sim.dev_transport import MailboxTransport
from sim2sim.dev_rpc import call,zip_project
from sim2sim.runtime import RuntimeFailure

PNG=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')


def task():return {'type':'task','task':'Public visual reconstruction only','observation':{'rgb':base64.b64encode(PNG).decode(),'ticks':0}}


def test_mailbox_exact_boundary_roundtrip_and_public_images(tmp_path):
    received=[];sent=[]
    transport=MailboxTransport(tmp_path,4096,on_receive=lambda raw,complete:received.append((raw,complete)),on_send=lambda raw,complete:sent.append((raw,complete)))
    transport.send(task(),time.monotonic()+2)
    result={}
    thread=threading.Thread(target=lambda:result.update(call(tmp_path,{'op':'observe'},timeout=2)))
    thread.start();request=transport.receive(time.monotonic()+2)
    assert request=={'op':'observe'}
    transport.send({'ok':True,'world':'reference','observation':{'rgb':base64.b64encode(PNG).decode(),'ticks':0}},time.monotonic()+2)
    thread.join(3);assert not thread.is_alive()
    assert received==[(b'{"op":"observe"}\n',True)]
    assert json.loads(sent[0][0])==task()
    assert result['ok'] and 'rgb' not in result['observation']
    assert Path(result['observation']['rgb_path']).read_bytes()==PNG
    assert (tmp_path/'rpc.py').is_file()
    assert 'not a hard' in (tmp_path/'MODE.txt').read_text()
    transport.close({'status':'finished'})
    assert json.loads((tmp_path/'stopped.json').read_text())['status']=='finished'


def test_copied_helper_is_standalone_and_task_compact(tmp_path):
    transport=MailboxTransport(tmp_path,4096);transport.send(task(),time.monotonic()+2)
    result=subprocess.run([sys.executable,str(tmp_path/'rpc.py'),'--task'],capture_output=True,text=True,check=True)
    value=json.loads(result.stdout)
    assert value['task']==task()['task']
    assert Path(value['observation']['rgb_path']).read_bytes()==PNG
    assert 'rgb' not in value['observation']
    assert 'private' not in value


@pytest.mark.parametrize('raw',[b'not json\n',b'[]\n',b'{"op":"observe"}',b'{"op":"observe"}\n{}\n',b'{"op":NaN}\n'])
def test_invalid_requests_logged_before_failure(tmp_path,raw):
    captured=[];transport=MailboxTransport(tmp_path,4096,on_receive=lambda data,complete:captured.append(data))
    (tmp_path/'requests/000001.json').write_bytes(raw)
    with pytest.raises(RuntimeFailure):transport.receive(time.monotonic()+1)
    assert captured==[raw]
    assert transport.pending is None


def test_oversize_bounded_and_symlink_refused(tmp_path):
    captured=[];transport=MailboxTransport(tmp_path,10,on_receive=lambda raw,complete:captured.append((raw,complete)))
    path=tmp_path/'requests/000001.json';path.write_bytes(b'x'*1000)
    with pytest.raises(RuntimeFailure):transport.receive(time.monotonic()+1)
    assert len(captured[0][0])==11 and captured[0][1] is False
    path.unlink();path.symlink_to(tmp_path/'private')
    with pytest.raises(RuntimeFailure):transport.receive(time.monotonic()+1)


def test_pending_request_recovery_and_stop_deadline(tmp_path):
    stopped=threading.Event();transport=MailboxTransport(tmp_path,4096,stop_event=stopped)
    transport.send(task(),time.monotonic()+1)
    with pytest.raises(TimeoutError):call(tmp_path,{'op':'observe'},timeout=.01)
    with pytest.raises(RuntimeError,match='Prior request'):call(tmp_path,{'op':'reset'},timeout=.01)
    assert transport.receive(time.monotonic()+1)=={'op':'observe'}
    transport.send({'ok':True},time.monotonic()+1)
    assert call(tmp_path,wait_seq=1,timeout=.1)['ok']
    stopped.set()
    with pytest.raises(RuntimeFailure):transport.receive(time.monotonic()+1)


def test_project_zip_only_curated_own_files(tmp_path):
    (tmp_path/'project').mkdir();(tmp_path/'project/scene.xml').write_text('<mujoco/>')
    data=base64.b64decode(zip_project(tmp_path,'project'))
    with zipfile.ZipFile(io.BytesIO(data)) as archive:assert archive.namelist()==['scene.xml']
    outside=tmp_path/'private';outside.mkdir();(outside/'scene.xml').write_text('private source')
    with pytest.raises(ValueError):zip_project(tmp_path,outside)
    (tmp_path/'project/leak.xml').symlink_to(outside/'scene.xml')
    with pytest.raises(ValueError):zip_project(tmp_path,tmp_path/'project')


def test_public_script_and_transport_do_not_add_private_configuration(tmp_path):
    transport=MailboxTransport(tmp_path,4096)
    transport.send(task(),time.monotonic()+1)
    value=json.loads((tmp_path/'task.json').read_text())
    assert set(value)=={'type','task','observation'}
    assert all(key not in value for key in ('source_dir','source','scores','api_key','manifest','physical_parameters'))


def test_replaced_mailbox_directory_symlink_refused(tmp_path):
    root=tmp_path/'public';transport=MailboxTransport(root,4096)
    outside=tmp_path/'outside';outside.mkdir();(outside/'000001.json').write_text('{"op":"observe"}\n')
    (root/'requests').rmdir();(root/'requests').symlink_to(outside,target_is_directory=True)
    with pytest.raises(RuntimeFailure,match='directory symlinks'):transport.receive(time.monotonic()+1)


def test_completed_response_recoverable_after_backend_stops(tmp_path):
    transport=MailboxTransport(tmp_path,4096);transport.send(task(),time.monotonic()+1)
    (tmp_path/'requests/000001.json').write_text('{"op":"finish"}\n')
    assert transport.receive(time.monotonic()+1)['op']=='finish'
    transport.send({'ok':True,'status':'finished'},time.monotonic()+1);transport.close()
    assert call(tmp_path,wait_seq=1)['status']=='finished'
    with pytest.raises(RuntimeError,match='stopped'):call(tmp_path,{'op':'observe'})


def test_concurrent_close_waits_for_durable_stopped_marker(tmp_path,monkeypatch):
    import sim2sim.dev_transport as module
    transport=MailboxTransport(tmp_path,4096)
    entered=threading.Event();release=threading.Event();second_done=threading.Event()
    atomic=module._atomic
    def delayed(path,data):
        entered.set();assert release.wait(2);atomic(path,data)
    monkeypatch.setattr(module,'_atomic',delayed)
    first=threading.Thread(target=lambda:transport.close({'status':'stopped'}))
    second=threading.Thread(target=lambda:(transport.close(),second_done.set()))
    first.start();assert entered.wait(1);second.start()
    assert not second_done.wait(.03)
    assert not (tmp_path/'stopped.json').exists()
    release.set();first.join(2);second.join(2)
    assert second_done.is_set()
    assert json.loads((tmp_path/'stopped.json').read_text())=={'status':'stopped'}
