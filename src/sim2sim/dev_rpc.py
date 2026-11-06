"""Copied into a development public folder; standard-library-only public RPC helper."""
from __future__ import annotations
import argparse
import fcntl
import base64
import io
import json
import math
import os
from pathlib import Path
import time
import zipfile


def atomic(path,raw):
    if path.is_symlink(): raise ValueError('Public mailbox symlinks are forbidden')
    temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    if temp.is_symlink(): raise ValueError('Public temporary symlink forbidden')
    with temp.open('wb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
    temp.replace(path)


def zip_project(root,project):
    supplied=Path(project)
    if not supplied.is_absolute():supplied=root/supplied
    if (root/'project').is_symlink() or supplied.is_symlink():raise ValueError('Project symlinks are forbidden')
    project=supplied.resolve()
    if not project.is_relative_to((root/'project').resolve()) or not project.is_dir():
        raise ValueError('Project must be an authored directory inside PUBLIC/project')
    data=io.BytesIO();total=0;count=0
    with zipfile.ZipFile(data,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(project.rglob('*')):
            if path.is_symlink():raise ValueError('Project symlinks are forbidden')
            if not path.is_file():continue
            count+=1;total+=path.stat().st_size
            if count>256 or total>32*1024*1024:raise ValueError('Project exceeds file/expanded byte limits')
            archive.write(path,path.relative_to(project).as_posix())
    if len(data.getvalue())>8*1024*1024:raise ValueError('ZIP exceeds 8MiB')
    return base64.b64encode(data.getvalue()).decode()


def compact(root,response,label):
    if (root/'observations').is_symlink():raise ValueError('Public observation directory symlink forbidden')
    response=json.loads(json.dumps(response))
    observation=response.get('observation')
    if isinstance(observation,dict):
        world='reconstruction' if response.get('world') in ('reconstruction','candidate') else 'reference'
        for key in ('rgb','depth','wrist_rgb','wrist_depth'):
            encoded=observation.pop(key,None)
            if not isinstance(encoded,str):continue
            raw=base64.b64decode(encoded,validate=True)
            if not raw.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('Invalid public PNG')
            path=root/'observations'/f'{world}-{label}-{key}.png'
            atomic(path,raw);observation[key+'_path']=str(path)
    return response


def call(root,request=None,timeout=300,project=None,wait_seq=None):
    root=Path(root).resolve()
    if not math.isfinite(timeout) or timeout<=0:raise ValueError('RPC timeout must be positive and finite')
    if wait_seq is not None and int(wait_seq)<1:raise ValueError('Wait sequence must be positive')
    deadline=time.monotonic()+timeout
    if wait_seq is None and (root/'stopped.json').exists():raise RuntimeError('Development backend stopped: '+(root/'stopped.json').read_text()[:1000])
    lock=root/'rpc.lock'
    if lock.is_symlink():raise ValueError('Public lock symlink forbidden')
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd);raise RuntimeError('Another public RPC client is active; wait for its response')
    try:
        pending=root/'pending.json'
        if wait_seq is not None: seq=int(wait_seq)
        else:
            if pending.exists():
                old=json.loads(pending.read_text())['seq']
                if not (root/'responses'/f'{old:06d}.json').exists():raise RuntimeError(f'Prior request {old} is pending; use --wait {old}')
            if not isinstance(request,dict):raise ValueError('Request must be a JSON object')
            if project is not None:
                if request.get('op') not in ('load_reconstruction','submit'):raise ValueError('--project is only for load_reconstruction/submit')
                request['project_b64']=zip_project(root,project)
            entries=[int(p.stem) for p in (root/'requests').glob('*.json') if p.stem.isdigit()]
            seq=max(entries,default=0)+1
            raw=json.dumps(request,separators=(',',':'),allow_nan=False).encode()+b'\n'
            if len(raw)>16*1024*1024:raise ValueError('Request exceeds 16MiB public helper limit')
            atomic(pending,json.dumps({'seq':seq}).encode())
            atomic(root/'requests'/f'{seq:06d}.json',raw)
        response_path=root/'responses'/f'{seq:06d}.json'
        while not response_path.exists():
            if (root/'stopped.json').exists():raise RuntimeError('Development backend stopped before response')
            if time.monotonic()>=deadline:raise TimeoutError(f'Request {seq} still pending; recover with --wait {seq}')
            time.sleep(.025)
        if response_path.is_symlink():raise ValueError('Response symlink forbidden')
        response=json.loads(response_path.read_text())
        if pending.exists() and json.loads(pending.read_text()).get('seq')==seq:pending.unlink()
        result=compact(root,response,f'{seq:06d}')
        result['public_raw_response_path']=str(response_path)
        return result
    finally:
        fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)


def main():
    parser=argparse.ArgumentParser(description='Prompt-scoped development RPC; no hard filesystem isolation')
    parser.add_argument('request',nargs='?');parser.add_argument('--task',action='store_true');parser.add_argument('--timeout',type=float,default=300)
    parser.add_argument('--project');parser.add_argument('--wait',type=int)
    args=parser.parse_args();root=Path(__file__).resolve().parent
    if args.task:
        result=compact(root,json.loads((root/'task.json').read_text()),'task');result['public_raw_task_path']=str(root/'task.json')
    else:result=call(root,json.loads(args.request) if args.request else None,args.timeout,args.project,args.wait)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError,TimeoutError) as exc:
        print(json.dumps({'ok':False,'error':str(exc)}));raise SystemExit(1)
