"""Read-only offline experiment dashboard built from verified audit snapshots."""
from __future__ import annotations
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from .audit import AuditFailure,read_audit,_atomic,_json,_digest,_safe_file
from .replay import _observations,_declared_hashes
from .dashboard_view import HTML,SCRIPT


def _metadata(path):
    if not path.is_file() or path.is_symlink() or path.stat().st_size>2*1024*1024: return {}
    try:
        value=json.loads(path.read_text())
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError): return {}


def _wall(event):
    value=event.get('wall_elapsed')
    return float(value) if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0 else None


def _world(value):
    if value in ('reference','source'): return 'reference'
    if value in ('candidate','reconstruction','reconstructed'): return 'candidate'
    return None


def _scalars(value,path=()):
    """Stable primitive/probe labels; no seeds, indices, flags or implementation counts."""
    if isinstance(value,bool): return
    if isinstance(value,(int,float)):
        if math.isfinite(value): yield '.'.join(path),float(value)
    elif value is None:
        if path: yield '.'.join(path),None
    elif isinstance(value,dict):
        for key,child in value.items():
            if key in ('seed','surface_seed','samples','surface_samples','source_hinge_count','candidate_hinge_count') or key.endswith('_count'): continue
            yield from _scalars(child,path+(str(key),))
    elif isinstance(value,list):
        for index,child in enumerate(value):
            label=(child.get('source_name') or child.get('name')) if isinstance(child,dict) else None
            yield from _scalars(child,path+(str(label or index),))


def _unit(name):
    if name=='reset.rgb_mse':return 'squared normalized RGB'
    field=name.split('.')[-1]
    if field.isdigit() and len(name.split('.'))>1: field=name.split('.')[-2]
    if name.startswith('parameters.') or 'relative_error' in name or field in ('coverage','probe_grasp_failure_rate'): return 'ratio'
    if field.endswith('_mae_rad_s'): return 'rad/s'
    if field.endswith('_mae_rad'): return 'rad'
    if field.endswith('_mae_nm'): return 'N·m'
    if field.endswith('_mae_m_s'): return 'm/s'
    if field.endswith('_ticks') or field.endswith('_tick_difference'): return 'ticks'
    if 'kg_m3' in field: return 'kg/m³'
    if field.endswith('_kg'): return 'kg'
    if 'frictionloss' in field: return 'N·m'
    if field.endswith('_mu') or '_mu_' in field: return 'friction coefficient'
    if 'damping' in field and name.startswith('hinges.'): return 'N·m·s/rad'
    if 'stiffness' in field and name.startswith('hinges.'): return 'N·m/rad'
    if field.endswith('_m') or field=='depth_error': return 'm'
    if field=='image_trajectory_error': return 'image/depth mixed units'
    if field in ('rgb_error','silhouette_error'): return 'ratio'
    return 'value'


def _asset(output,data,suffix,verified=None):
    digest=_digest(data);relative=f'assets/{digest}{suffix}'
    path=output/relative
    if verified is not None and relative in verified: return relative
    if path.is_symlink(): raise AuditFailure('dashboard cached asset symlink forbidden')
    if not path.is_file() or path.stat().st_size!=len(data) or _digest(path.read_bytes())!=digest:
        _atomic(path,data)
    if verified is not None: verified[relative]=len(data)
    return relative


def _metric_group(name):
    """Group comparable curves; retain every historical metric in the checklist."""
    if name=='reset.rgb_mse':return 'Full-image RGB MSE',True
    if name in ('reconstruction.chamfer_distance_m','reconstruction.center_of_mass_error_m'):
        return 'Geometry and center of mass error', True
    if name.startswith('arm_probes.'):
        field=name.rsplit('.',1)[-1]
        groups={
            'joint_position_mae_rad':('Panda joint position error',True),
            'joint_velocity_mae_rad_s':('Panda joint velocity error',False),
            'actuator_effort_mae_nm':('Panda actuator effort error',True),
            'finger_position_mae_m':('Panda finger position error',False),
            'finger_velocity_mae_m_s':('Panda finger velocity error',False),
            'gripper_width_mae_m':('Panda gripper width error',False),
            'source_target_contact_ticks':('Panda target contact ticks',False),
            'candidate_target_contact_ticks':('Panda target contact ticks',False),
            'target_contact_tick_difference':('Panda target contact difference',True),
            'rgb_error':('Panda probe RGB error',False),
        }
        if field in groups:return groups[field]
    if name.startswith('reconstruction.density.per_primitive.'):
        field=name.rsplit('.',1)[-1]
        return {'absolute_error_kg_m3':('Density error by primitive',True),
                'relative_error':('Relative density error by primitive',False),
                'source_density_kg_m3':('Density measurements by primitive',False),
                'predicted_density_kg_m3':('Density measurements by primitive',False),
                'geometry_matching_cost_m':('Primitive matching distance',False)}.get(field,(name,False))
    if name.startswith('reconstruction.friction.per_primitive.'):
        field=name.rsplit('.',1)[-1]
        return {'support_sliding_mu_absolute_error':('Effective support sliding friction error by primitive',True),
                'sliding_mu_absolute_error':('Geom sliding friction error by primitive',False),
                'source_support_sliding_mu':('Effective support sliding friction by primitive',False),
                'predicted_support_sliding_mu':('Effective support sliding friction by primitive',False)}.get(field,(name,False))
    if name.startswith('hinges.'):
        for key,label in (('damping','Viscous hinge friction'),('frictionloss','Dry hinge friction'),('stiffness','Hinge stiffness')):
            if key in name:
                suffix='relative error' if 'relative_error' in name else 'error' if 'absolute_error' in name else 'measurements'
                return label+' '+suffix, suffix=='error'
    if 'center_of_mass' in name:
        return ('Center of mass vectors',False)
    return name,False


def _metric_values(scores):
    values=dict(_scalars(scores))
    if not isinstance(scores,dict): return values
    reconstruction=scores.get('reconstruction') or {}
    for name in (reconstruction.get('density') or {}).get('unmatched_source',[]):
        for field in ('source_density_kg_m3','predicted_density_kg_m3','absolute_error_kg_m3','relative_error'):
            values.setdefault(f'reconstruction.density.per_primitive.{name}.{field}',None)
    for name,record in (scores.get('hinges') or {}).items():
        if record.get('status')!='ok':
            for parameter in ('damping','frictionloss','stiffness'):
                for field in (f'source_{parameter}',f'predicted_{parameter}',f'{parameter}_absolute_error',f'{parameter}_relative_error'):
                    values[f'hinges.{name}.{field}']=None
    return values


def _execution_mode(metadata,manifest=None):
    if metadata.get('fixture') is True or metadata.get('execution_mode')=='trusted_fixture':
        return dict(kind='trusted_fixture',label='Trusted fixture',detail='Recorded development fixture; no model or hard-isolation claim.')
    manifest=manifest or {}
    value=metadata.get('execution_mode') or metadata.get('mode') or manifest.get('execution_mode')
    if value=='development_shared_filesystem':
        return dict(kind=value,label='Development · shared FS',detail='Prompt-scoped development execution; shared filesystem/tools are not a hard security boundary.')
    if value in ('strict_oci','oci'):
        return dict(kind='strict_oci',label='Strict OCI mode',detail='Declared OCI execution mode; actual enforcement and launch evidence must be checked in the run preflight/audit.')
    return dict(kind='unrecorded',label='Mode unrecorded',detail='No explicit execution-mode provenance was recorded; isolation is not inferred.')


def _model_context(task,metadata,origin):
    """Only allowlisted public fields; never copy private configuration or source."""
    keys=('task','actions','action_schema','worlds','public_tool_config','observation_contract',
          'public_robot_path','budgets','submission','execution_mode','agent_model')
    return dict(origin=origin,orchestrator_prompt=metadata.get('orchestrator_prompt'),researcher_notes=metadata.get('orchestration_notes'),continuation=metadata.get('continuation'),public_task={key:task[key] for key in keys if key in task},
        model_constraints=metadata.get('model_constraints',[]),
        visibility='Permitted observations, public action protocol, own reconstruction and supplied public fixture only. Target-object source XML and parameters, private probe artifacts and evaluation scores are withheld from the public protocol. Any supplied public robot fixture is available to the model.',
        boundary='Shared filesystem development uses a prompt convention; filesystem and tool isolation are not enforced.' if metadata.get('execution_mode')=='development_shared_filesystem' else 'Execution-mode declaration alone does not establish enforced isolation.',
        missing=('Full external model/tool transcript is not recorded here. '+('Exact orchestrator prompt is recorded above. ' if isinstance(metadata.get('orchestrator_prompt'),str) else 'The original orchestrator prompt is not recorded here. ')+'The recorded public task below is known; it is not a reconstruction of missing records.'))


def _candidate_evaluations(run):
    sidecar=_metadata(run/'candidate-change-history.json')
    if sidecar.get('schema_version')!=1:return [],[]
    result=[];unavailable=list(sidecar.get('unavailable',[]))
    for checkpoint in sidecar.get('checkpoints',[]):
        if not isinstance(checkpoint,dict):continue
        wall=_wall(checkpoint)
        if wall is None:continue
        report=checkpoint.get('report') or {}
        if report.get('status')!='ok':
            unavailable.append(dict(change_index=checkpoint.get('change_index'),version=checkpoint.get('version'),
                wall_elapsed=wall,evaluated_at=checkpoint.get('evaluated_at'),reason='evaluation_'+str(report.get('status','unavailable')),
                error=report.get('error') or report.get('errors')))
        result.append(dict(wall_elapsed=wall,values=_metric_values(report.get('scores')) if report.get('status')=='ok' else {},
            submission=None,version=checkpoint.get('version'),status=checkpoint.get('status',report.get('status','unavailable')),
            seq=checkpoint.get('change_index'),change_index=checkpoint.get('change_index'),checkpoint_kind=checkpoint.get('checkpoint_kind'),origin='retrospective_candidate_change',evaluated_at=checkpoint.get('evaluated_at')))
    return sorted(result,key=lambda item:item['wall_elapsed']),unavailable


def _run_snapshot(run,output,identifier,verified_assets=None):
    metadata=_metadata(run/'experiment.json');status_meta=_metadata(run/'status.json')
    experiment=metadata.get('experiment') if isinstance(metadata.get('experiment'),dict) else {}
    name=metadata.get('name') or experiment.get('name') or run.name
    status=status_meta.get('status',status_meta.get('state','empty'));reason=status_meta.get('error') or status_meta.get('reason')
    data=dict(id=identifier,name=str(name),directory=run.name,status=status,reason=reason,execution_mode=_execution_mode(metadata),
        started_at=status_meta.get('started_at'),model_context=_model_context(_metadata(run/'public/task.json'),metadata,'public/task.json (recorded public artifact; not audit-verified)'),
        frames={'reference':[],'candidate':[]},videos=[],progress=[],actions=[],metrics=[],duration_wall=_wall(status_meta) or 0.,
        snapshot=dict(verified=False,complete=False,event_count=0),warnings=[])
    if not (run/'audit/manifest.json').is_file():
        data['warnings'].append('No recorded audit, camera frames or model progress are available.')
        return data
    evaluations=[];action_starts={};untimed=0;termination=False;start_unix=None;last_pointer=None;snapshot_manifest={};audit_storage={}
    try:
        # Fix the initial manifest anchor inside read_audit(snapshot=True); do not
        # require or imply that the actively growing run has a final anchor.
        for pointer,event in read_audit(run,snapshot=True,manifest_out=snapshot_manifest,storage_out=audit_storage):
            last_pointer=pointer
            _declared_hashes(event)
            payload=event.get('payload')
            if isinstance(payload,dict) and payload.get('type')=='task':
                data['model_context']=_model_context(payload,metadata,'verified audit public task')
            wall=_wall(event);world=_world(event.get('world'))
            action_key=(world,event.get('action_id'))
            if wall is not None: data['duration_wall']=max(data['duration_wall'],wall)
            if event.get('kind')=='run_start':
                manifest=event.get('manifest',{})
                data['execution_mode']=_execution_mode(metadata,manifest)
                start_unix=manifest.get('started_at_unix')
            if event.get('kind')=='termination':
                termination=True;data['status']=event.get('status','terminated');data['reason']=event.get('reason')
            if event.get('kind')=='progress' and event.get('source') in ('agent','provider_summary') and isinstance(event.get('text'),str):
                if wall is not None:
                    data['progress'].append(dict(wall_elapsed=wall,text=event['text'],source=event['source'],origin=event.get('origin','agent'),model=event.get('model'),seq=pointer['seq']))
                else: untimed+=1
            if event.get('kind')=='action_begin':
                resolved=event.get('resolved_action') or {}
                if resolved.get('op') in ('observe','wait','grasp','move','release','reset','arm_target','load_reconstruction','submit','finish') and world:
                    action=dict(action_id=event.get('action_id'),seq=pointer['seq'],world=world,operation=resolved.get('op'),
                        category='meta' if resolved.get('op') in ('reset','load_reconstruction','submit','finish') else 'world',
                        parameters={key:resolved[key] for key in ('pixel','displacement','joint_targets','gripper_width_m','entrypoint','camera') if key in resolved},
                        wall_elapsed=wall,completed_wall_elapsed=None,requested_ticks=resolved.get('ticks'),executed_ticks=None,
                        charged_ticks=None,outcome='completion unrecorded')
                    assert action_key not in action_starts, 'Overlapping actions with the same world and correlation ID'
                    action['correlation']='recorded action ID' if event.get('action_id') is not None else 'serial world order; no action ID recorded'
                    action_starts[action_key]=action
                    if wall is not None:data['actions'].append(action)
                    else:untimed+=1
            if event.get('kind')=='action_end' and action_key in action_starts:
                action=action_starts.pop(action_key)
                executed=event.get('executed_ticks')
                if isinstance(executed,(int,float)) and executed<0:executed=None
                action.update(completed_wall_elapsed=wall,executed_ticks=executed,
                    charged_ticks=event.get('charged_ticks'),inflight_ticks_unconfirmed=event.get('inflight_ticks_unconfirmed'),
                    outcome=event.get('outcome','unknown'))
            if event.get('kind')=='evaluation':
                report=event.get('report',{})
                if wall is not None:
                    values=_metric_values(report.get('scores')) if report.get('status')=='ok' else {}
                    evaluations.append(dict(wall_elapsed=wall,values=values,submission=event.get('submission'),
                        version=report.get('version'),status=report.get('status','failed'),seq=pointer['seq'],
                        origin=event.get('checkpoint_kind','submission'),change_index=event.get('change_index'),
                        checkpoint_kind=event.get('checkpoint_kind','submission'),evaluated_at=event.get('evaluated_at')))
                else: untimed+=1
            if event.get('kind')=='video':
                relative=event.get('path')
                if not isinstance(relative,str) or not relative.endswith('.mp4'): raise AuditFailure('invalid dashboard video path')
                source=_safe_file(run,relative)
                if source.stat().st_size!=event.get('size'): raise AuditFailure('dashboard video size mismatch')
                digest=hashlib.sha256()
                with source.open('rb') as handle:
                    for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
                if digest.hexdigest()!=event.get('sha256'): raise AuditFailure('dashboard video hash mismatch')
                target=f"assets/{event['sha256']}.mp4"
                destination=output/target
                if destination.is_symlink(): raise AuditFailure('dashboard cached video symlink forbidden')
                cached=hashlib.sha256()
                if destination.is_file():
                    with destination.open('rb') as handle:
                        for chunk in iter(lambda:handle.read(1024*1024),b''): cached.update(chunk)
                if not destination.is_file() or cached.hexdigest()!=event['sha256']:
                    temporary=destination.with_suffix(f'.mp4.{os.getpid()}.tmp')
                    if temporary.is_symlink(): raise AuditFailure('dashboard temporary video symlink forbidden')
                    shutil.copyfile(source,temporary);temporary.replace(destination)
                data['videos'].append(dict(world=world,path=target,original_path=relative,episode=event.get('episode'),wall_elapsed=wall))
                if verified_assets is not None: verified_assets[target]=event['size']
            if world and event.get('kind') not in ('boundary','worker_boundary','action_begin','action_end'):
                for _,observation,images in _observations(event):
                    if wall is None: untimed+=1;continue
                    frame=dict(wall_elapsed=wall,seq=pointer['seq'],episode=event.get('episode'),
                        ticks=observation.get('ticks'),simulation_time=observation.get('time'))
                    state=event.get('state') or event.get('after_state') or {}
                    frame['grasp_overlay']=state.get('grasp_overlay')
                    # The player displays RGB when available, with depth as a
                    # fallback. Keep other exact modalities in the source audit
                    # instead of duplicating unused images in the view cache.
                    displayed={'rgb':images['rgb']} if 'rgb' in images else images
                    for key,raw in displayed.items(): frame[key]=_asset(output,raw,'.png',verified_assets)
                    previous=data['frames'][world][-1] if data['frames'][world] else None
                    if previous and all(previous.get(k)==frame.get(k) for k in ('episode','ticks','rgb','depth','grasp_overlay')):
                        continue
                    data['frames'][world].append(frame)
        data['snapshot']=dict(verified=True,complete=bool(termination and snapshot_manifest.get('sealed')),sealed=bool(snapshot_manifest.get('sealed')),event_count=last_pointer['seq'] if last_pointer else 0,
            last_hash=last_pointer['hash'] if last_pointer else '0'*64,captured_at_unix=time.time(),audit_storage=audit_storage)
        if not termination:
            data['status']=status_meta.get('status',status_meta.get('state','active'))
            if isinstance(start_unix,(int,float)) and math.isfinite(start_unix):
                data['duration_wall']=max(data['duration_wall'],time.time()-start_unix)
        if untimed: data['warnings'].append(f'{untimed} recorded items lack wall timestamps and are excluded from synchronized playback.')
        if any(action['action_id'] is None for action in data['actions']):
            data['warnings'].append('Legacy actions have no recorded IDs. Completions are paired by serial world order; overlapping anonymous actions are rejected.')
        retrospective,unavailable=_candidate_evaluations(run)
        data['candidate_change_history']=dict(origin='retrospective sidecar; not part of sealed audit',unavailable=unavailable,checkpoints=retrospective)
        for series,label in (([p for p in evaluations if p.get('origin')!='candidate_change'],'submission'),
                             ([p for p in evaluations if p.get('origin')=='candidate_change'],'candidate_change'),
                             (retrospective,'retrospective_candidate_change')):
            names=sorted({name for checkpoint in series for name in checkpoint['values']})
            for name in names:
                data['metrics'].append(dict(name=name if label=='submission' else name+(' (candidate changes)' if label=='candidate_change' else ' (retrospective candidate changes)'),metric_name=name,
                    origin=label,unit=_unit(name),group=_metric_group(name)[0],default_visible=_metric_group(name)[1],points=[
                    dict(checkpoint,values=None,value=checkpoint['values'].get(name),origin=checkpoint.get('origin',label)) for checkpoint in series]))
        for frames in data['frames'].values(): frames.sort(key=lambda f:(f['wall_elapsed'],f['seq']))
        data['progress'].sort(key=lambda item:(item['wall_elapsed'],item['seq']))
    except (AuditFailure,OSError,ValueError,TypeError) as exc:
        data.update(status='invalid audit snapshot',reason=str(exc),snapshot=dict(verified=False,complete=False,event_count=0),
            frames={'reference':[],'candidate':[]},metrics=[],progress=[],actions=[],videos=[])
    return data


class _Serial:
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def map(self,function,items): return map(function,items)


def _snapshot_job(job):
    run,output,identifier=job;assets={}
    return _run_snapshot(run,output,identifier,assets),assets


def build_dashboard(root='experiments',output=None,workers=1):
    """Discover compatible run directories; read-only snapshot, no native execution."""
    roots=[Path(r).resolve() for r in (root if isinstance(root,(list,tuple)) else [root])]  # several roots, e.g. home + lab storage
    output=Path(output).resolve() if output is not None else roots[0]/'dashboard'
    if any(output.is_relative_to(r/'runs') for r in roots): raise AuditFailure('dashboard output must not overwrite experiment runs')
    run_dirs=sorted((run for r in roots if (r/'runs').is_dir() for run in (r/'runs').iterdir()),key=lambda run:run.name)
    if len({run.name for run in run_dirs})!=len(run_dirs): raise AuditFailure('duplicate run directory names across experiment roots')
    if output.is_symlink(): raise AuditFailure('dashboard destination symlink forbidden')
    output.mkdir(parents=True,exist_ok=True);(output/'assets').mkdir(exist_ok=True)
    if (output/'assets').is_symlink(): raise AuditFailure('dashboard assets symlink forbidden')
    catalog=[];verified_assets={}
    jobs=[(run,output,hashlib.sha256(run.name.encode()).hexdigest()[:16]) for run in run_dirs if run.is_dir() and not run.is_symlink()]
    from concurrent.futures import ProcessPoolExecutor  # audit reads are NFS-latency bound; runs are independent
    with ProcessPoolExecutor(workers) if workers>1 else _Serial() as pool:
        snapshots=list(pool.map(_snapshot_job,jobs))
    for (run,_,identifier),(data,assets) in zip(jobs,snapshots):
        verified_assets.update(assets)
        relative=f'runs/{identifier}.js';(output/'runs').mkdir(exist_ok=True)
        if (output/'runs').is_symlink(): raise AuditFailure('dashboard runs symlink forbidden')
        encoded=json.dumps(data,ensure_ascii=True,allow_nan=False).replace('<','\\u003c')
        _atomic(output/relative,('window.__dashboardRunData('+encoded+');').encode())
        catalog.append(dict(id=identifier,name=data['name'],directory=data['directory'],status=data['status'],
            started_at=data['started_at'],execution_mode=data['execution_mode'],data_path=relative,duration_wall=data['duration_wall']))
    encoded=json.dumps(catalog,ensure_ascii=True,allow_nan=False).replace('<','\\u003c')
    _atomic(output/'catalog.js',('window.__dashboardCatalog='+encoded+';').encode())
    _atomic(output/'dashboard.js',SCRIPT.encode());_atomic(output/'index.html',HTML.encode())
    manifest=dict(schema_version=1,runs=catalog,generated_at_unix=time.time(),
                  verified_assets=dict(count=len(verified_assets),logical_bytes=sum(verified_assets.values())))
    _atomic(output/'dashboard.json',_json(manifest))
    return manifest
