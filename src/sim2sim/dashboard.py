"""Read-only offline experiment dashboard built from verified audit snapshots."""
from __future__ import annotations
import base64
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
from .audit import AuditFailure,read_audit,_atomic,_json,_digest,_safe_file
from .replay import _observations,_declared_hashes
from .dashboard_view import HTML,SCRIPT

REPO=Path(__file__).resolve().parents[2]
ENVIRONMENTS=REPO/'environments'
_VS=re.compile(r'\s*\(vs\.?\s+([^)]*)\)')


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


class UnknownRun(KeyError):
    """No run directory in any root has this identifier (moved away, renamed or removed)."""


class CacheFailure(RuntimeError):
    """The dashboard cache (not a run or its audit) could not be written, e.g. its storage quota is full."""


def _cache_write(path,data):
    try: _atomic(path,data)
    except OSError as exc: raise CacheFailure(f'dashboard cache write failed at {path}: {exc.strerror or exc}') from exc


def _asset(output,data,suffix,verified=None):
    digest=_digest(data);relative=f'assets/{digest}{suffix}'
    path=output/relative
    if verified is not None and relative in verified: return relative
    if path.is_symlink(): raise AuditFailure('dashboard cached asset symlink forbidden')
    if not path.is_file() or path.stat().st_size!=len(data) or _digest(path.read_bytes())!=digest:
        _cache_write(path,data)
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


def _run_snapshot(run,output,identifier,verified_assets=None,progress=None):
    """``progress(verified_events, total_events)`` is called as the audit is verified."""
    metadata=_metadata(run/'experiment.json');status_meta=_metadata(run/'status.json')
    experiment=metadata.get('experiment') if isinstance(metadata.get('experiment'),dict) else {}
    name=metadata.get('name') or experiment.get('name') or run.name
    status=status_meta.get('status',status_meta.get('state','empty'));reason=status_meta.get('error') or status_meta.get('reason')
    data=dict(id=identifier,name=str(name),directory=run.name,status=status,reason=reason,execution_mode=_execution_mode(metadata),
        started_at=status_meta.get('started_at'),model_context=_model_context(_metadata(run/'public/task.json'),metadata,'public/task.json (recorded public artifact; not audit-verified)'),
        frames={'reference':[],'candidate':[]},videos=[],links=[],progress=[],actions=[],metrics=[],duration_wall=_wall(status_meta) or 0.,
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
            if progress: progress(pointer['seq'],snapshot_manifest['event_count'])
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
                if resolved.get('op') in ('observe','wait','grasp','move','release','reset','arm_target','hand_target','load_reconstruction','submit','finish') and world:
                    action=dict(action_id=event.get('action_id'),seq=pointer['seq'],world=world,operation=resolved.get('op'),
                        category='meta' if resolved.get('op') in ('reset','load_reconstruction','submit','finish') else 'world',
                        parameters={key:resolved[key] for key in ('pixel','displacement','joint_targets','gripper_width_m','actuator_targets','entrypoint','camera') if key in resolved},
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
                # Fixture evaluations carry a researcher view (fixture_view.py) whose hash the audited report records.
                view=((report.get('scores') or {}).get('fixture') or {}).get('view') if report.get('status')=='ok' else None
                if view and event.get('submission'):
                    page=_safe_file(run,f"version-{int(event['submission']):03d}/evaluation/{view['path']}").read_bytes()
                    if _digest(page)!=view['sha256']: raise AuditFailure('dashboard fixture view hash mismatch')
                    data['links'].append(dict(path=_asset(output,page,'.html',verified_assets),label=f"submission {event['submission']} · fixture evaluation view (predicted vs measured stops, settings map, trial replay)"))
            if event.get('kind')=='video':
                relative=event.get('path')
                if not isinstance(relative,str) or not relative.endswith('.mp4'): raise AuditFailure('invalid dashboard video path')
                video=_safe_file(run,relative).read_bytes()  # recorded MP4s are small (largest 74 KB on 2026-11-24)
                if len(video)!=event.get('size') or _digest(video)!=event.get('sha256'): raise AuditFailure('dashboard video size or hash mismatch')
                data['videos'].append(dict(world=world,path=_asset(output,video,'.mp4',verified_assets),original_path=relative,episode=event.get('episode'),wall_elapsed=wall))
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
        # Researcher views written after the run (src/sim2sim/die.py, src/sim2sim/tabletop.py); not audit-verified.
        for page,label in (('die-view.html','Die evaluation view'),('touch-view.html','Touch contact coverage map and grab result')):
            if (run/page).is_file():
                data['links'].append(dict(label=label+' (researcher-only; not audit-verified)',path=_asset(output,(run/page).read_bytes(),'.html',verified_assets)))
        for frames in data['frames'].values(): frames.sort(key=lambda f:(f['wall_elapsed'],f['seq']))
        data['progress'].sort(key=lambda item:(item['wall_elapsed'],item['seq']))
    except (AuditFailure,OSError,ValueError,TypeError) as exc:
        data.update(status='invalid audit snapshot',reason=str(exc),snapshot=dict(verified=False,complete=False,event_count=0),
            frames={'reference':[],'candidate':[]},metrics=[],progress=[],actions=[],videos=[],links=[])
    return data


def environment_scenes(root=ENVIRONMENTS):
    """Environment scene paths, e.g. ``blocks`` or ``panda/damped_lever``, point-grasp scenes first."""
    root=Path(root)
    return [scene.parent.relative_to(root).as_posix() for pattern in ('*/scene.xml','*/*/scene.xml') for scene in sorted(root.glob(pattern))]


def _assignments(path,names=None):
    """Top-level assignments of a repository Python file, evaluated without running anything else in it (no loops, calls
    at statement level or file writes). The wave generator writes configs when run, so it is never imported."""
    import ast
    tree=ast.parse(Path(path).read_text())
    # with names=None also keep function definitions and statement-level ``NAME.update(...)`` calls: later assignments read keys they add
    update=lambda node:(isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                        and node.value.func.attr=='update' and isinstance(node.value.func.value,ast.Name))
    body=[node for node in tree.body if (isinstance(node,ast.Assign) and (names is None or any(isinstance(t,ast.Name) and t.id in names for t in node.targets)))
          or (names is None and (isinstance(node,ast.FunctionDef) or update(node)))]
    namespace={'Path':Path,'__file__':str(path)}
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def agent_prompts(repo=REPO):
    """System prompt and, per environment, the base task text the agent under investigation receives: the current (stop-when-done)
    task, the model constraints for its robot family, and a level statement only when every configured stop-when-done run of
    that environment appends the same sentence. Per-experiment additions (named parameters, stated masses, tool hints) differ
    across runs of an environment and are left out."""
    g=_assignments(repo/'experiments/configs/generate-waves.py')
    system=_assignments(repo/'examples/sandbox_agent.py',{'SYSTEM'})['SYSTEM']
    base=g['TASK_CONVERGE']
    runs=[condition for _,_,_,conditions in g['WAVES'].values() for condition in conditions.values()]

    def prompt(path):
        configured={o.get('task',g['TASK']) for env,_,o in runs if env==path}
        configured={task for task in configured if task.startswith(base)}
        level=next(iter(configured))[len(base):] if len(configured)==1 else ''
        family=path.split('/')[0]
        constraints={'panda':g['PANDA_CONSTRAINTS'],'panda_small':g['PANDA_CONSTRAINTS'],'shadow_hand':g['HAND_CONSTRAINTS'],'panda_shadow':g['TOUCH_CONSTRAINTS']}.get(family,g['CONSTRAINTS'])
        return dict(task=base+level+'\nModel constraints:\n'+'\n'.join('- '+c for c in constraints),level=level.strip())
    return system,{path:prompt(path) for path in environment_scenes(repo/'environments')}


def _environment(value):
    """Recorded environment name -> (robot label, scene, environment path)."""
    for prefix,robot in (('panda_small_','Panda (small)'),('panda_shadow_','Panda + Shadow hand'),('panda_','Panda'),('shadow_hand_','Shadow hand')):
        if value.startswith(prefix):
            return robot,value[len(prefix):],prefix[:-1]+'/'+value[len(prefix):]
    return 'Point grasp',value,value


def _condition(directory):
    """Run directory without its date/attempt suffix: the condition that attempts repeat."""
    return re.sub(r'-\d{8}-\d{3}$','',directory)


def _tags(metadata,status,config,robot,scene,series,factor,root):
    """Filterable tag categories from recorded metadata and the private resolved config."""
    simulation=config.get('simulation') or {};physical=metadata.get('physical_parameters') or {}
    task=str(config.get('task') or '');inference=config.get('inference') or {}
    model=metadata.get('model') or inference.get('model');effort=metadata.get('reasoning_effort') or inference.get('reasoning_effort')
    tags={'Scene':[scene],'Robot':[robot],'Series':[series],'Status':[status],'Mode':[_execution_mode(metadata)['label']],
        'Observations':[label for flag,label in ((simulation.get('rgb'),'RGB'),(simulation.get('depth'),'Depth'),
            (simulation.get('wrist_camera'),'Wrist camera'),(simulation and not simulation.get('rgb'),'Proprioception only')) if flag],
        'Interaction':[label for flag,label in ((simulation.get('wrench'),'Wrench sensing'),(simulation and not simulation.get('wrench'),'No wrench'),
            (simulation.get('reference_interaction') is False,'Observe-only reference')) if flag],
        'Variant':[label for key,label in (('perturbation.physics','Hard physics'),('perturbation.colors','Recolored')) if key in physical] or ['Authored'],
        'Tools':sorted(name for name,enabled in (config.get('tools') or {}).items() if enabled) or (['None'] if config else []),
        'Prompt':[label for flag,label in (('finishing early earns nothing' in task,'Full-budget prompt'),('Call finish when you believe' in task,'Converge prompt'),
            (re.search(r'\btask\b',factor),'Task hint')) if flag],
        'Model':[str(model)+(' · '+str(effort) if effort else '')] if model else [],
        'Storage':[root.name]}
    return {key:[str(value) for value in values] for key,values in tags.items() if values}


def _summary(run,root,identifier):
    """Catalog entry from small metadata files only; the audit is read when the run is opened."""
    metadata=_metadata(run/'experiment.json');status_meta=_metadata(run/'status.json');config=_metadata(run/'config-private.json')
    experiment=metadata.get('experiment') if isinstance(metadata.get('experiment'),dict) else {}
    name=str(metadata.get('name') or experiment.get('name') or run.name)
    robot,scene,path=_environment(str(metadata.get('environment') or 'unrecorded'))
    factor=str((metadata.get('physical_parameters') or {}).get('factor') or name.split(' · ')[-1])
    status=str(status_meta.get('status',status_meta.get('state','unknown')))
    return dict(id=identifier,name=name,directory=run.name,condition=_condition(run.name),status=status,
        started_at=status_meta.get('started_at') or metadata.get('started_at'),created_at=metadata.get('created_at'),
        execution_mode=_execution_mode(metadata),environment=path,scene=scene,robot=robot,factor=factor,root=str(root),
        tags=_tags(metadata,status,config,robot,scene,name.split(' · ')[0],factor,root),comparisons=[],data_path=f'runs/{identifier}.js')


def _comparisons(summaries):
    """Ablation pairs: an explicit ``(vs A, B)`` factor annotation, else the sibling ``<prefix>-base`` condition."""
    conditions={}
    for item in summaries: conditions.setdefault(item['condition'],[]).append(item)
    pairs={}
    for condition,items in conditions.items():
        factor=items[0]['factor'];match=_VS.search(factor)
        baselines=[ref.strip() for ref in match.group(1).split(',')] if match else [re.sub(r'-[^-]+$','-base',condition)]
        for baseline in baselines:
            if baseline in conditions and baseline!=condition:
                key=f'{condition} vs {baseline}';members=items+conditions[baseline]
                pairs[key]=dict(id=key,name=items[0]['scene']+' · '+_VS.sub('',factor).strip(),ablation=condition,baseline=baseline,
                    runs=[item['directory'] for item in members])
                for item in members: item['comparisons'].append(key)
    return sorted(pairs.values(),key=lambda pair:pair['name'])


def _discover(roots):
    runs=sorted(((root,run) for root in roots if (root/'runs').is_dir() for run in (root/'runs').iterdir()
                 if run.is_dir() and not run.is_symlink()),key=lambda item:item[1].name)
    names=[run.name for _,run in runs]
    duplicates=sorted({name for name in names if names.count(name)>1})
    if duplicates: raise AuditFailure('duplicate run directory names across experiment roots: '+', '.join(duplicates))
    return [(root,run,hashlib.sha256(run.name.encode()).hexdigest()[:16]) for root,run in runs]


def _encode(value):
    return json.dumps(value,ensure_ascii=True,allow_nan=False).replace('<','\\u003c')


def _catalog_script(runs,comparisons):
    # built: last modification of the scene's own files (scene, manifest, assets), the date it was last (re)built
    built=lambda path:datetime.fromtimestamp(max(f.stat().st_mtime for f in (ENVIRONMENTS/path).iterdir() if f.is_file()),timezone.utc).isoformat()
    system,prompts=agent_prompts()
    environments=[dict(path=path,robot=_environment(path.replace('/','_'))[0],scene=path.split('/')[-1],built=built(path),**prompts[path]) for path in environment_scenes()]
    return (f'window.__dashboardCatalog={_encode(runs)};window.__dashboardComparisons={_encode(comparisons)};'
            f'window.__dashboardEnvironments={_encode(environments)};window.__dashboardAgentSystem={_encode(system)};').encode()


def _record_key(run):
    """Every audit append rewrites the manifest, so unchanged stats mean an unchanged verified snapshot."""
    return [[item.stat().st_mtime_ns,item.stat().st_size] if item.is_file() else None for item in
            (run/'audit/manifest.json',run/'status.json',run/'experiment.json',run/'candidate-change-history.json')]


def _write_run(output,identifier,data,key):
    (output/'runs').mkdir(exist_ok=True)
    if (output/'runs').is_symlink(): raise AuditFailure('dashboard runs symlink forbidden')
    _cache_write(output/'runs'/f'{identifier}.js',('window.__dashboardRunData('+_encode(data)+');').encode())
    _cache_write(output/'runs'/f'{identifier}.key.json',json.dumps(key).encode())


def _prepare_output(roots,output):
    output=Path(output).resolve() if output is not None else roots[0]/'dashboard'
    if any(output.is_relative_to(r/'runs') for r in roots): raise AuditFailure('dashboard output must not overwrite experiment runs')
    if output.is_symlink(): raise AuditFailure('dashboard destination symlink forbidden')
    output.mkdir(parents=True,exist_ok=True);(output/'assets').mkdir(exist_ok=True)
    if (output/'assets').is_symlink(): raise AuditFailure('dashboard assets symlink forbidden')
    _cache_write(output/'dashboard.js',SCRIPT.encode());_cache_write(output/'index.html',HTML.encode())
    return output


class LiveDashboard:
    """Current catalog of every root on each request; a run's verified snapshot is rebuilt when its records change.

    Assets this server has verified stay trusted for its lifetime; ``sim2sim dashboard`` re-verifies everything.
    """
    def __init__(self,roots,output=None):
        self.roots=[Path(r).resolve() for r in roots]
        self.output=_prepare_output(self.roots,output)
        self.verified,self.guard,self.locks,self.building={},threading.Lock(),{},{}

    def progress(self,identifier):
        """Verified and total audit events of a snapshot being built now, else None."""
        return self.building.get(identifier)

    def catalog_script(self):
        summaries=[_summary(run,root,identifier) for root,run,identifier in _discover(self.roots)]
        return _catalog_script(summaries,_comparisons(summaries))

    def run_script(self,identifier):
        runs=[run for _,run,key in _discover(self.roots) if key==identifier]
        if not runs: raise UnknownRun('unknown run snapshot '+identifier)
        path,key_path=(self.output/'runs'/f'{identifier}{suffix}' for suffix in ('.js','.key.json'))
        with self.guard: lock=self.locks.setdefault(identifier,threading.Lock())
        with lock:
            key=_record_key(runs[0])
            if not (path.is_file() and key_path.is_file() and json.loads(key_path.read_text())==key):
                self.building[identifier]=dict(events=0,total=None)
                try: data=_run_snapshot(runs[0],self.output,identifier,self.verified,lambda done,total:self.building.__setitem__(identifier,dict(events=done,total=total)))
                finally: self.building.pop(identifier)
                _write_run(self.output,identifier,data,key)
            return path.read_bytes()


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
    runs=_discover(roots);output=_prepare_output(roots,output)
    summaries=[_summary(run,r,identifier) for r,run,identifier in runs];comparisons=_comparisons(summaries)
    jobs=[(run,output,identifier) for _,run,identifier in runs];verified_assets={}
    keys=[_record_key(run) for run,_,_ in jobs]  # before reading, so a later append marks the snapshot stale
    from concurrent.futures import ProcessPoolExecutor  # audit reads are NFS-latency bound; runs are independent
    with ProcessPoolExecutor(workers) if workers>1 else _Serial() as pool:
        snapshots=list(pool.map(_snapshot_job,jobs))
    for (run,_,identifier),key,summary,(data,assets) in zip(jobs,keys,summaries,snapshots):
        verified_assets.update(assets)
        _write_run(output,identifier,data,key)
        summary.update(status=data['status'],execution_mode=data['execution_mode'],duration_wall=data['duration_wall'])
        summary['tags']['Status']=[str(data['status'])]  # the verified audit's termination status, as listed
    _cache_write(output/'catalog.js',_catalog_script(summaries,comparisons))
    manifest=dict(schema_version=2,runs=summaries,comparisons=comparisons,generated_at_unix=time.time(),
                  verified_assets=dict(count=len(verified_assets),logical_bytes=sum(verified_assets.values())))
    _cache_write(output/'dashboard.json',_json(manifest))
    return manifest
