"""Bounded, durable researcher-only boundary/state audit with content-addressed blobs."""
from __future__ import annotations
import base64
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import stat
import threading


class AuditFailure(RuntimeError):
    pass


def _json(value):
    return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _safe_file(root, relative):
    path=Path(relative)
    if path.is_absolute() or not path.parts or any(p in ('..','.') for p in path.parts) or '\\' in str(relative):
        raise AuditFailure('unsafe audit path')
    current=root
    for part in path.parts:
        current=current/part
        if current.is_symlink(): raise AuditFailure('audit symlinks are forbidden')
    if not current.resolve().is_relative_to(root.resolve()): raise AuditFailure('audit path escapes run')
    return current


def _atomic(path,data):
    temp=path.with_name(f'{path.name}.{os.getpid()}.tmp')  # per process: parallel dashboard workers share assets
    if path.is_symlink() or temp.is_symlink(): raise AuditFailure('atomic output symlinks forbidden')
    with temp.open('wb') as handle:
        handle.write(data);handle.flush();os.fsync(handle.fileno())
    temp.replace(path)
    directory=os.open(path.parent,os.O_RDONLY | getattr(os,'O_DIRECTORY',0))
    try: os.fsync(directory)
    finally: os.close(directory)


class Reservation:
    def __init__(self,audit,size):
        self.audit=audit;self.remaining=size;self.closed=False
    def release(self): self.audit.release(self)
    def __enter__(self): return self
    def __exit__(self,*args): self.release()


class AuditLog:
    def __init__(self,run_dir,max_bytes=512*1024*1024):
        self.root=Path(run_dir)/'audit'
        self.limit=int(max_bytes)
        if self.limit<4096: raise AuditFailure('audit budget must be at least 4096 bytes')
        if self.root.is_symlink(): raise AuditFailure('audit symlink forbidden')
        self.root.mkdir(parents=True,exist_ok=True)
        (self.root/'blobs').mkdir(exist_ok=True)
        if (self.root/'manifest.json').exists() or (self.root/'events.jsonl').exists():
            raise AuditFailure('audit already exists; use a new run directory')
        self.count=0;self.last='0'*64;self.held=0;self.lock=threading.RLock();self.closed=False
        self.manifest=_json(self._manifest())
        _atomic(self.root/'manifest.json',self.manifest)
        (self.root/'events.jsonl').write_bytes(b'')
        self.used=len(self.manifest)

    def _manifest(self):
        return dict(schema_version=1,max_bytes=self.limit,event_count=self.count,last_hash=self.last,sealed=self.closed)

    def reserve(self,size):
        size=int(size)
        with self.lock:
            if self.closed or size<0 or self.used+self.held+size>self.limit:
                raise AuditFailure('audit byte budget exhausted before action')
            self.held+=size
            return Reservation(self,size)

    def release(self,token):
        with self.lock:
            if token.audit is not self: raise AuditFailure('foreign audit reservation')
            if not token.closed:
                self.held-=token.remaining;token.remaining=0;token.closed=True

    def _externalize(self,event):
        blobs={};attachments=[]
        def visit(value,path):
            key=path[-1] if path else None
            encoding=None;data=None;prefix=None
            if key in ('wire_b64','rgb','depth','wrist_rgb','wrist_depth','project_b64','submission_b64','source_b64','artifacts_b64','data_b64') and isinstance(value,str):
                try:
                    data=base64.b64decode(value,validate=True)
                    if base64.b64encode(data).decode('ascii')==value: encoding='base64'
                except (ValueError,TypeError): pass
            elif key in ('url','image_url') and isinstance(value,str) and value.startswith('data:') and ';base64,' in value:
                prefix,encoded=value.split(',',1)
                try:
                    data=base64.b64decode(encoded,validate=True)
                    if base64.b64encode(data).decode('ascii')==encoded: encoding='data_url'
                except (ValueError,TypeError): pass
            elif key in ('state','beforestate','afterstate','before_state','after_state') and isinstance(value,(dict,list)):
                data=_json(value);encoding='json'
            if encoding:
                digest=_digest(data);blobs[digest]=data
                item=dict(path=path,sha256=digest,size=len(data),encoding=encoding)
                if encoding=='data_url': item['prefix']=prefix+','
                attachments.append(item)
                return None
            if isinstance(value,dict): return {k:visit(v,path+[k]) for k,v in value.items()}
            if isinstance(value,list): return [visit(v,path+[i]) for i,v in enumerate(value)]
            return value
        payload=visit(event,[])
        envelope=_json(dict(payload=payload,attachments=attachments))
        digest=_digest(envelope);blobs[digest]=envelope
        return digest,envelope,blobs

    def record(self,event,reservation=None):
        """Persist a complete event; caller reserves every possible outcome before mutation."""
        if not isinstance(event,dict): raise AuditFailure('audit event must be an object')
        digest,envelope,blobs=self._externalize(event)
        with self.lock:
            if self.closed: raise AuditFailure('audit closed')
            core=dict(seq=self.count+1,previous=self.last,event_sha256=digest,event_size=len(envelope))
            pointer=dict(core,hash=_digest(_json(core)))
            line=_json(pointer)+b'\n'
            nextmanifest=_json(dict(schema_version=1,max_bytes=self.limit,event_count=self.count+1,last_hash=pointer['hash'],sealed=False))
            missing={h:b for h,b in blobs.items() if not (self.root/'blobs'/h).exists()}
            cost=sum(map(len,missing.values()))+len(line)+len(nextmanifest)-len(self.manifest)
            own=reservation is None
            token=self.reserve(cost) if own else reservation
            if token.audit is not self or token.closed or token.remaining<cost:
                raise AuditFailure('event exceeds pre-action audit reservation')
            try:
                for h,data in missing.items(): _atomic(self.root/'blobs'/h,data)
                with (self.root/'events.jsonl').open('ab') as handle:
                    handle.write(line);handle.flush();os.fsync(handle.fileno())
                _atomic(self.root/'manifest.json',nextmanifest)
            except OSError as exc:
                self.closed=True
                raise AuditFailure(f'audit persistence failed: {exc}') from exc
            self.used+=cost;self.held-=cost;token.remaining-=cost
            self.count+=1;self.last=pointer['hash'];self.manifest=nextmanifest
            if own: self.release(token)
            return pointer

    write=record
    def flush(self): pass  # Every record already flushes and fsyncs.
    def close(self):
        with self.lock:
            if self.closed: return
            self.closed=True
            manifest=_json(self._manifest())
            _atomic(self.root/'manifest.json',manifest)
            self.used+=len(manifest)-len(self.manifest);self.manifest=manifest


def read_audit(run_dir, *, snapshot=False, manifest_out=None, storage_out=None):
    """Verify a complete log, or the fixed initial-manifest prefix for an active snapshot."""
    root=Path(run_dir).resolve()/'audit'
    if root.is_symlink(): raise AuditFailure('audit symlinks forbidden')
    blob_directory=None
    try:
        manifest=json.loads(_safe_file(root,'manifest.json').read_bytes())
        if manifest_out is not None: manifest_out.update(manifest)
        if manifest['schema_version']!=1: raise AuditFailure('unsupported audit schema')
        if 'sealed' in manifest and not isinstance(manifest['sealed'],bool): raise AuditFailure('invalid audit seal')
        if isinstance(manifest['event_count'],bool) or not isinstance(manifest['event_count'],int) or manifest['event_count']<0:
            raise AuditFailure('invalid audit event count')
        maximum=int(manifest['max_bytes'])
        actual=0;regular_files=0;allocated=0
        # One non-following stat per entry. Repeated pathlib queries incur many
        # extra metadata RPCs on the shared experiment filesystem.
        directories=[root]
        while directories:
            with os.scandir(directories.pop()) as entries:
                for entry in entries:
                    try:
                        info=entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        if snapshot and entry.name.endswith('.tmp'): continue
                        raise
                    if stat.S_ISLNK(info.st_mode): raise AuditFailure('audit symlinks forbidden')
                    if stat.S_ISDIR(info.st_mode): directories.append(entry.path)
                    elif stat.S_ISREG(info.st_mode) and not (snapshot and entry.name.endswith('.tmp')):
                        actual+=info.st_size;regular_files+=1;allocated+=info.st_blocks*512
        if actual>maximum: raise AuditFailure('audit files exceed declared budget')
        if storage_out is not None:
            storage_out.update(logical_bytes=actual,allocated_file_bytes=allocated,regular_file_count=regular_files)
        previous='0'*64;count=0
        # Pin the verified directory. Relative no-follow opens avoid resolving
        # the full NFS ancestry for every content-addressed blob, and cannot
        # follow a replaced parent symlink out of this snapshot directory.
        blob_directory=os.open(root/'blobs',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        # A snapshot reuses exact bytes already hash-verified in this traversal.
        # Never cache across calls: a later refresh must detect changed evidence.
        verified=OrderedDict();verified_bytes=0;cache_limit=16*1024*1024
        def blob(digest,size):
            nonlocal verified_bytes
            if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                raise AuditFailure('invalid blob digest')
            if digest in verified:
                data=verified[digest]
                if len(data)!=size: raise AuditFailure('blob size mismatch')
                verified.move_to_end(digest)
                return data
            descriptor=os.open(digest,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=blob_directory)
            with os.fdopen(descriptor,'rb') as handle:
                info=os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size!=size or size>128*1024*1024:
                    raise AuditFailure('blob size mismatch or excessive blob')
                data=handle.read(size+1)
            if len(data)!=size: raise AuditFailure('blob size changed while reading')
            if _digest(data)!=digest: raise AuditFailure('blob hash mismatch')
            if len(data)<=cache_limit:
                while verified and verified_bytes+len(data)>cache_limit:
                    _,old=verified.popitem(last=False);verified_bytes-=len(old)
                verified[digest]=data;verified_bytes+=len(data)
            return data
        with _safe_file(root,'events.jsonl').open('rb') as handle:
            while not snapshot or manifest.get('sealed',False) or count<manifest['event_count']:
                line=handle.readline(4097)
                if not line: break
                if not line.endswith(b'\n') or len(line)>4096: raise AuditFailure('truncated or excessive audit index')
                pointer=json.loads(line);count+=1
                core={k:pointer[k] for k in ('seq','previous','event_sha256','event_size')}
                if pointer['seq']!=count or pointer['previous']!=previous or pointer['hash']!=_digest(_json(core)):
                    raise AuditFailure('audit sequence/hash chain mismatch')
                envelope=json.loads(blob(pointer['event_sha256'],pointer['event_size']))
                event=envelope['payload']
                for item in envelope['attachments']:
                    data=blob(item['sha256'],item['size'])
                    if item['encoding']=='base64': value=base64.b64encode(data).decode('ascii')
                    elif item['encoding']=='json': value=json.loads(data)
                    elif item['encoding']=='data_url': value=item['prefix']+base64.b64encode(data).decode('ascii')
                    else: raise AuditFailure('unknown audit attachment encoding')
                    target=event;parts=item['path']
                    for part in parts[:-1]: target=target[part]
                    if target[parts[-1]] is not None: raise AuditFailure('audit attachment target is not empty')
                    target[parts[-1]]=value
                previous=pointer['hash']
                yield pointer,event
        if count!=manifest['event_count'] or previous!=manifest['last_hash']:
            raise AuditFailure('audit manifest anchor mismatch')
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
        if isinstance(exc,AuditFailure): raise
        raise AuditFailure(f'invalid audit: {exc}') from exc
    finally:
        if blob_directory is not None: os.close(blob_directory)
