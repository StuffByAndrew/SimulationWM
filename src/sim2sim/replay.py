"""Exact recorded visual playback; never reruns models, physics or submissions."""
from __future__ import annotations
import base64
import json
import hashlib
import shutil
import re
import os
from pathlib import Path
from .audit import AuditFailure, read_audit, _atomic, _json, _digest, _safe_file

PNG_SIGNATURE=b'\x89PNG\r\n\x1a\n'


def _observations(value,path=()):
    if isinstance(value,dict):
        images={}
        for key in ('rgb','depth','wrist_rgb','wrist_depth'):
            if isinstance(value.get(key),str):
                try:
                    data=base64.b64decode(value[key],validate=True)
                    if data.startswith(PNG_SIGNATURE): images[key]=data
                except ValueError: pass
        if images: yield path,value,images
        for key,child in value.items():
            if key not in ('request','state','beforestate','afterstate','before_state','after_state'):
                yield from _observations(child,path+(key,))
    elif isinstance(value,list):
        for index,child in enumerate(value): yield from _observations(child,path+(index,))


def _declared_hashes(value):
    if isinstance(value,dict):
        for key in ('state','beforestate','afterstate','before_state','after_state','rgb','depth','wrist_rgb','wrist_depth'):
            declared=value.get(key+'_sha256')
            if declared is None: continue
            payload=value.get(key)
            if payload is None and key in ('rgb','depth','wrist_rgb','wrist_depth') and isinstance(value.get('observation'),dict):
                payload=value['observation'].get(key)
            if payload is None: raise AuditFailure('declared hash has no associated '+key)
            if key in ('rgb','depth','wrist_rgb','wrist_depth'):
                try: data=base64.b64decode(payload,validate=True)
                except (ValueError,TypeError) as exc: raise AuditFailure('invalid declared image payload') from exc
            else:
                data=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
            if _digest(data)!=declared: raise AuditFailure('declared '+key+' hash mismatch')
        for child in value.values(): _declared_hashes(child)
    elif isinstance(value,list):
        for child in value: _declared_hashes(child)


def _video_sources(run,records,snapshots=None):
    videos=[]
    for pointer,event in records:
        _declared_hashes(event)
        if event.get('kind')=='run_start' and event.get('source_snapshot') and event.get('manifest',{}).get('source_bundle_sha256'):
            relative=event['source_snapshot'];source=_safe_file(run,relative)
            if source.stat().st_size>64*1024*1024: raise AuditFailure('source snapshot exceeds limit')
            digest=hashlib.sha256()
            with source.open('rb') as handle:
                for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
            if digest.hexdigest()!=event['manifest']['source_bundle_sha256']: raise AuditFailure('source snapshot hash mismatch')
            if snapshots is not None: snapshots.append(dict(source_path=relative,path='snapshots/source-snapshot.zip',sha256=digest.hexdigest(),size=source.stat().st_size))
        if event.get('kind')!='video': continue
        relative=event.get('path')
        if not isinstance(relative,str) or not relative.endswith('.mp4'): raise AuditFailure('invalid video path')
        source=_safe_file(run,relative)
        if source.stat().st_size!=event.get('size'): raise AuditFailure('recorded video size mismatch')
        digest=hashlib.sha256()
        with source.open('rb') as handle:
            for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
        if digest.hexdigest()!=event.get('sha256'): raise AuditFailure('recorded video hash mismatch')
        segment=re.search(r'-(\d+)\.mp4$',relative)
        episode=event.get('episode',int(segment.group(1)) if segment else None)
        videos.append(dict(seq=pointer['seq'],world=event.get('world'),episode=episode,source_path=relative,
            path='videos/'+relative,sha256=event['sha256'],size=event['size']))
    return videos


def replay_run(run_dir,output_dir):
    """Verify the complete audit before writing self-contained visuals and timeline.

    Returns the replay manifest. Original JSON and wire bytes remain available in
    timeline/wire artifacts; PNGs are copied byte-for-byte without image decoding.
    """
    run=Path(run_dir).resolve();output=Path(output_dir).resolve()
    if output==run or output.is_relative_to(run/'audit'):
        raise AuditFailure('replay destination must not overwrite the source audit')
    # Exhaust generator first: a truncated final anchor cannot produce a valid replay.
    snapshots=[]
    try:
        videos=_video_sources(run,read_audit(run),snapshots)
    except OSError as exc: raise AuditFailure(f'invalid recorded video: {exc}') from exc
    if output.exists() and any(output.iterdir()):
        raise AuditFailure('replay destination must be empty')
    output.mkdir(parents=True,exist_ok=True)
    for subdir in ('frames','states','wire'):
        path=output/subdir
        if path.is_symlink(): raise AuditFailure('replay destination symlinks forbidden')
        path.mkdir(exist_ok=True)
    for video in videos:
        target=output/video['path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(_safe_file(run,video['source_path']),target)
    for snapshot in snapshots:
        target=output/snapshot['path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(_safe_file(run,snapshot['source_path']),target)
    frames=[];episodes={};lastticks={};actions=[];active_actions={};wire_paths={};wires=[];event_count=0
    timeline_temp=output/'timeline.json.tmp'
    try:
        with timeline_temp.open('wb') as timeline_handle:
            timeline_handle.write(b'[')
            for pointer,event in read_audit(run):
                seq=pointer['seq'];world=event.get('world','harness')
                if not isinstance(world,str): world='harness'
                if event.get('kind')=='action_begin':
                    resolved=event.get('resolved_action',{})
                    compact={key:value for key,value in resolved.items() if key in ('op','action','target','ticks','pixel','displacement','args') and len(_json(value))<=2048} if isinstance(resolved,dict) else {}
                    action=dict(seq=seq,world=world,request_seq=event.get('request_seq'),action_id=event.get('action_id'),
                        resolved_action=compact,request_wire=wire_paths.get(event.get('request_seq')),outcome='pending',before_ticks=event.get('beforeticks',event.get('before_ticks')))
                    actions.append(action);active_actions[world]=action
                elif event.get('kind')=='action_end' and world in active_actions:
                    action=active_actions[world]
                    action.update(outcome=event.get('outcome','complete'),after_ticks=event.get('afterticks',event.get('after_ticks')),
                        executed_ticks=event.get('executed_ticks'),end_seq=seq)
                state_paths=[]
                for key in ('state','beforestate','afterstate','before_state','after_state'):
                    if isinstance(event.get(key),(dict,list)):
                        relative=f'states/{seq:08d}-{key}.json'
                        data=_json(event[key]);_atomic(output/relative,data)
                        state_paths.append(dict(path=relative,sha256=_digest(data),kind=key))
                observation_events=[] if event.get('kind')=='boundary' and event.get('direction')=='request' else [event]
                wire_path=None
                if isinstance(event.get('wire_b64'),str):
                    try: wire=base64.b64decode(event['wire_b64'],validate=True)
                    except ValueError as exc: raise AuditFailure('invalid hydrated boundary wire') from exc
                    wire_path=f'wire/{seq:08d}.bin';_atomic(output/wire_path,wire)
                    wire_paths[seq]=wire_path
                    wires.append(dict(seq=seq,world=world,direction=event.get('direction'),path=wire_path))
                    if event.get('direction')=='response':
                        try: observation_events.append(json.loads(wire))
                        except (ValueError,UnicodeDecodeError): pass
                found=[]
                for value in observation_events:
                    for path,obs,images in _observations(value):
                        ticks=obs.get('ticks',event.get('afterticks',event.get('after_ticks')))
                        if isinstance(ticks,(int,float)) and not isinstance(ticks,bool):
                            if world in lastticks and ticks<lastticks[world]: episodes[world]=episodes.get(world,0)+1
                            lastticks[world]=ticks
                        episode=event.get('episode',episodes.get(world,0))
                        entry=dict(seq=seq,world=world,episode=episode,ticks=ticks,time=obs.get('time'),observation_path=list(path),
                            action=active_actions.get(world))
                        for kind,data in images.items():
                            relative=f'frames/{seq:08d}-{len(found):03d}-{kind}.png'
                            _atomic(output/relative,data)
                            entry[kind]=relative;entry[kind+'_sha256']=_digest(data)
                        frames.append(entry);found.append(entry)
                row=dict(seq=seq,audit_hash=pointer['hash'],event=event,frames=found,state_artifacts=state_paths,wire_path=wire_path)
                if event_count: timeline_handle.write(b',')
                timeline_handle.write(_json(row));event_count+=1
            timeline_handle.write(b']');timeline_handle.flush();os.fsync(timeline_handle.fileno())
        timeline_temp.replace(output/'timeline.json')
        directory=os.open(output,os.O_RDONLY | getattr(os,'O_DIRECTORY',0))
        try: os.fsync(directory)
        finally: os.close(directory)
    except Exception:
        timeline_temp.unlink(missing_ok=True)
        raise
    manifest=dict(schema_version=1,mode='exact_recorded_visuals',events=event_count,frames=len(frames),
        worlds=sorted(set(f['world'] for f in frames)),videos=videos,source_snapshots=snapshots,limitation='No physics, model, evaluation or external calls are rerun; unrecorded intermediate frames are unavailable.')
    _atomic(output/'frames.json',_json(frames))
    _atomic(output/'replay.json',_json(manifest))
    # JSON loaded via a non-executable script tag also works when opened using file://.
    payload=json.dumps(dict(frames=frames,actions=actions,videos=videos,wires=wires),ensure_ascii=True).replace('<','\\u003c').replace('&','\\u0026')
    document='''<!doctype html><meta charset="utf-8"><title>Recorded simulation replay</title>
<style>body{font:16px system-ui;margin:2rem;background:#17212b;color:#eef3f8}#worlds{display:flex;flex-wrap:wrap;gap:1rem}.world{flex:1;min-width:320px}img{max-width:100%;background:#111}input{width:80%}pre{white-space:pre-wrap}</style>
<h1>Recorded visual replay</h1><p>Exact saved frames for both worlds. No models or physics are rerun. Missing intermediate frames remain unavailable.</p>
<p>The frame viewer steps at a fixed 10 frames/second for inspection; it does not reproduce simulation or wall-clock timing. Verified videos below retain their original encoded frame rate.</p>
<p><a href="timeline.json">Full event timeline, prompts and submissions</a> · <a href="frames.json">Frame/state metadata</a> · <a href="replay.json">Verified video manifest</a></p>
<button id="play">Step frames at 10 fps</button> <input id="position" type="range" min="0" value="0"> <span id="label"></span><div id="worlds"></div><h2>Actions and exact wire records</h2><div id="actions"></div><details><summary>Exact boundary records, including inference and submissions</summary><div id="wire"></div></details><h2>Original recorded videos</h2><div id="videos"></div>
<script type="application/json" id="data">'''+payload+'''</script><script>
const data=JSON.parse(document.getElementById('data').textContent),frames=data.frames, slider=document.getElementById('position');
for(const action of data.actions){const row=document.createElement('p');row.textContent='Event '+action.seq+' | '+action.world+' | '+JSON.stringify(action.resolved_action)+' | '+action.outcome+' | ticks '+action.before_ticks+' → '+action.after_ticks;
if(action.request_wire){const link=document.createElement('a');link.href=action.request_wire;link.textContent=' Exact request wire';row.append(link);}document.getElementById('actions').append(row);}
for(const item of data.wires){const row=document.createElement('p'),link=document.createElement('a');link.href=item.path;link.textContent='Event '+item.seq+' | '+item.world+' | '+item.direction+' | exact wire';row.append(link);document.getElementById('wire').append(row);}
for(const item of data.videos){const label=document.createElement('p');label.textContent=item.world+' | episode '+(item.episode===null?'see file segment':item.episode)+' | '+item.source_path;const video=document.createElement('video');video.controls=true;video.preload='metadata';video.src=item.path;video.style.maxWidth='100%';document.getElementById('videos').append(label,video);}
slider.max=Math.max(0,frames.length-1);const worlds=[...new Set(frames.map(f=>f.world))],panels=Object.create(null);
for(const world of worlds){const p=document.createElement('section');p.className='world';const h=document.createElement('h2');h.textContent=world;
const image=document.createElement('img'),meta=document.createElement('pre');p.append(h,image,meta);document.getElementById('worlds').append(p);panels[world]={image,meta};}
function draw(){const index=Number(slider.value),latest=Object.create(null);for(let i=0;i<=index&&i<frames.length;i++)latest[frames[i].world]=frames[i];
for(const world of worlds){const f=latest[world],p=panels[world];if(f){p.image.src=f.rgb||f.depth;p.image.hidden=false;p.meta.textContent='Event '+f.seq+' | episode '+f.episode+' | ticks '+f.ticks+' | time '+f.time+'\\nAction: '+(f.action?JSON.stringify(f.action.resolved_action)+' | '+f.action.outcome+' | ticks '+f.action.before_ticks+' → '+f.action.after_ticks:'no action recorded yet');}else{p.image.hidden=true;p.meta.textContent='No recorded frame yet';}}
document.getElementById('label').textContent=frames.length?(index+1)+' / '+frames.length:'No recorded frames';}
slider.oninput=draw;let timer=null;document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;return;}timer=setInterval(()=>{if(Number(slider.value)>=Number(slider.max)){clearInterval(timer);timer=null;return;}slider.value=Number(slider.value)+1;draw();},100);};draw();
</script>'''
    _atomic(output/'index.html',document.encode('utf-8'))
    return manifest
