"""Explicit development-only public mailbox; shared filesystem is not isolation."""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
import stat
import threading
import time
from .audit import _atomic


def _failure(message):
    from .runtime import RuntimeFailure
    return RuntimeFailure(message)


class MailboxTransport:
    """One outstanding public request, same audit hooks as the OCI pipe transport."""
    def __init__(self,public_dir,max_bytes,on_receive=None,on_send=None,stop_event=None):
        self.root=Path(public_dir).resolve();self.max_bytes=int(max_bytes)
        self.on_receive=on_receive;self.on_send=on_send;self.stop_event=stop_event
        self.next_seq=1;self.pending=None;self.task_sent=False;self.closed=False;self.audited=set();self.close_lock=threading.Lock()
        self.root.mkdir(parents=True,exist_ok=True)
        for name in ('requests','responses','observations','project'):
            path=self.root/name
            if path.is_symlink(): raise _failure('Development mailbox symlinks are forbidden')
            path.mkdir(exist_ok=True)
        _atomic(self.root/'rpc.py',Path(__file__).with_name('dev_rpc.py').read_bytes())
        _atomic(self.root/'MODE.txt',b'DEVELOPMENT SHARED FILESYSTEM: prompt-scoped public context only; this is not a hard filesystem or tool security boundary.\n')

    def _check(self,deadline):
        if self.root.is_symlink() or self.root.resolve()!=self.root or any((self.root/name).is_symlink() for name in ('requests','responses','observations','project')):
            raise _failure('Development mailbox directory symlinks are forbidden')
        if self.closed or (self.stop_event is not None and self.stop_event.is_set()):
            raise _failure('Development mailbox stopped')
        if time.monotonic()>=deadline: raise _failure('Development mailbox wall-time budget exhausted')

    def _images(self,payload,label):
        observation=payload.get('observation')
        if not isinstance(observation,dict): return
        world=payload.get('world','reference')
        world='reconstruction' if world in ('reconstruction','candidate') else 'reference'
        paths={}
        for key in ('rgb','depth','wrist_rgb','wrist_depth'):
            if isinstance(observation.get(key),str):
                try: raw=base64.b64decode(observation[key],validate=True)
                except ValueError: continue
                if not raw.startswith(b'\x89PNG\r\n\x1a\n'): continue
                relative=f'observations/{world}-{label}-{key}.png'
                _atomic(self.root/relative,raw);paths[key]=relative
        _atomic(self.root/'observations/latest.json',json.dumps(dict(world=world,paths=paths,ticks=observation.get('ticks'))).encode())

    def send(self,payload,deadline):
        self._check(deadline)
        raw=json.dumps(payload,separators=(',',':'),allow_nan=False).encode()+b'\n'
        if self.on_send: self.on_send(raw,True)
        if not self.task_sent:
            if payload.get('type')!='task': raise _failure('Initial development payload must be a task')
            self._images(payload,'task');self._check(deadline);_atomic(self.root/'task.json',raw);self.task_sent=True
        else:
            if self.pending is None: raise _failure('Development response has no pending request')
            label=f'{self.pending:06d}';self._images(payload,label);self._check(deadline)
            _atomic(self.root/'responses'/f'{label}.json',raw)
            self.pending=None;self.next_seq+=1

    def _raw(self,path):
        if path.is_symlink(): raise _failure('Development request symlinks are forbidden')
        flags=os.O_RDONLY | getattr(os,'O_NOFOLLOW',0) | getattr(os,'O_NONBLOCK',0)
        fd=os.open(path,flags)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode): raise _failure('Development requests must be regular files')
            with os.fdopen(fd,'rb',closefd=False) as handle: return handle.read(self.max_bytes+1)
        finally: os.close(fd)

    def receive(self,deadline):
        if self.pending is not None: raise _failure('Send the pending development response before receiving another request')
        path=self.root/'requests'/f'{self.next_seq:06d}.json'
        while True:
            self._check(deadline)
            if path.exists() or path.is_symlink(): break
            time.sleep(min(.025,max(0,deadline-time.monotonic())))
        raw=self._raw(path);complete=len(raw)<=self.max_bytes and raw.endswith(b'\n') and raw.count(b'\n')==1
        if self.on_receive: self.on_receive(raw,complete)
        self.audited.add(self.next_seq)
        if len(raw)>self.max_bytes: raise _failure('Development protocol message exceeds byte limit')
        if not complete: raise _failure('Development request must be one newline-terminated JSON record')
        try:
            value=json.loads(raw);json.dumps(value,allow_nan=False)
        except (ValueError,UnicodeError) as exc: raise _failure('Development agent emitted invalid JSON') from exc
        if not isinstance(value,dict): raise _failure('Development request must be a JSON object')
        self.pending=self.next_seq
        return value

    def audit_pending(self):
        path=self.root/'requests'/f'{self.next_seq:06d}.json'
        if self.next_seq not in self.audited and path.is_file() and not path.is_symlink():
            raw=self._raw(path)
            if self.on_receive: self.on_receive(raw,False)
            self.audited.add(self.next_seq)

    def close(self,status=None):
        with self.close_lock:
            if self.closed:return
            self.closed=True
            # Caller supplies only public status; never pass private source errors.
            _atomic(self.root/'stopped.json',json.dumps(status or {'status':'stopped'},allow_nan=False).encode()+b'\n')
