"""Pure dashboard snapshot checks; browser interaction runs only on Slurm test nodes."""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import os
import pytest
from sim2sim.audit import AuditLog,read_audit,AuditFailure
from sim2sim.dashboard import build_dashboard,_unit,_metric_group

PNG=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')


def fixture_run(root,name='trusted-fixture',close=True):
    run=root/'runs'/name;run.mkdir(parents=True)
    (run/'experiment.json').write_text(json.dumps({'name':'TRUSTED TEST FIXTURE · synthetic unit-test timestamps'}))
    audit=AuditLog(run,max_bytes=300000)
    audit.record({'kind':'run_start','wall_elapsed':0,'manifest':{}})
    for world,tick,wall in [('reference',0,0),('candidate',0,.5),('reference',1,1),('candidate',1,3)]:
        audit.record({'kind':'checkpoint','world':world,'wall_elapsed':wall,'episode':0,
            'observation':{'rgb':base64.b64encode(PNG).decode(),'ticks':tick,'time':tick*.002}})
    for index,wall in enumerate([.2,1.2,2.2]):
        audit.record({'kind':'progress','source':'agent','text':f'UNIT TEST NOTE {index}; not model output',
            'wall_elapsed':wall})
    for submission,wall,value in [(1,1.5,.2),(2,2.5,None),(3,4,0.)]:
        report={'status':'ok' if value is not None else 'failed','version':str(submission),'scores':
            {'reconstruction':{'chamfer_distance_m':value,'source_mass_kg':1.,'density':{'per_primitive':[
            {'source_name':'sphere','absolute_error_kg_m3':value}]}}} if value is not None else None}
        audit.record({'kind':'evaluation','submission':submission,'wall_elapsed':wall,'report':report})
    for action_id,world,wall,outcome in [(10,'reference',.6,'ok'),(11,'reconstruction',.9,'failed')]:
        audit.record({'kind':'action_begin','action_id':action_id,'world':world,'wall_elapsed':wall,
                      'resolved_action':{'op':'arm_target','ticks':10}})
        audit.record({'kind':'action_end','action_id':action_id,'world':world,'wall_elapsed':wall+.1,
                      'executed_ticks':3 if outcome=='failed' else 10,'charged_ticks':10,'outcome':outcome})
    for action_id,operation,wall in [(12,'reset',2.7),(13,'submit',3.1),(14,'finish',4.6)]:
        audit.record({'kind':'action_begin','action_id':action_id,'world':'reference','wall_elapsed':wall,
                      'resolved_action':{'op':operation,'ticks':0}})
        audit.record({'kind':'action_end','action_id':action_id,'world':'reference','wall_elapsed':wall+.1,
                      'executed_ticks':0,'charged_ticks':0,'outcome':'ok'})
    if close:
        audit.record({'kind':'termination','status':'completed','wall_elapsed':5});audit.close()
    return run,audit


def data_for(root):
    catalog=build_dashboard(root)
    script=(root/'dashboard'/catalog['runs'][0]['data_path']).read_text()
    return catalog,json.loads(script.removeprefix('window.__dashboardRunData(').removesuffix(');'))


def test_dashboard_two_world_wall_frames_metrics_and_progress(tmp_path):
    fixture_run(tmp_path)
    catalog,data=data_for(tmp_path)
    assert catalog['runs'][0]['name'].startswith('TRUSTED TEST FIXTURE')
    assert data['snapshot']['verified'] and data['snapshot']['complete']
    assert [f['wall_elapsed'] for f in data['frames']['reference']]==[0,1]
    assert [f['wall_elapsed'] for f in data['frames']['candidate']]==[.5,3]
    metric=next(m for m in data['metrics'] if m['name'].endswith('chamfer_distance_m'))
    assert [p['value'] for p in metric['points']]==[.2,None,0.]
    assert [p['wall_elapsed'] for p in metric['points']]==[1.5,2.5,4]
    density=next(m for m in data['metrics'] if 'sphere' in m['name'])
    assert density['unit']=='kg/m³'
    assert len(data['progress'])==3
    for frame in data['frames']['reference']:
        assert (tmp_path/'dashboard'/frame['rgb']).read_bytes()==PNG
    assert (tmp_path/'dashboard/index.html').is_file()


def test_active_snapshot_pins_initial_anchor_and_is_not_complete(tmp_path):
    run,audit=fixture_run(tmp_path,close=False)
    manifest={};records=read_audit(run,snapshot=True,manifest_out=manifest)
    first=next(records)
    audit.record({'kind':'late_append','wall_elapsed':6})
    prefix=[first,*records]
    assert len(prefix)==manifest['event_count']
    _,data=data_for(tmp_path)
    assert data['snapshot']['verified'] and not data['snapshot']['complete']
    assert data['status']=='active'
    audit.close()


def test_blocked_empty_and_corrupted_runs_stay_visible(tmp_path):
    blocked=tmp_path/'runs/blocked';blocked.mkdir(parents=True)
    (blocked/'status.json').write_text(json.dumps({'status':'blocked','reason':'SandboxFailure','error':'CPU/memory isolation unavailable','wall_elapsed':2.5}))
    bad,_=fixture_run(tmp_path,'corrupted')
    blob=next((bad/'audit/blobs').iterdir());blob.write_bytes(b'corrupt')
    catalog=build_dashboard(tmp_path)
    assert {item['status'] for item in catalog['runs']}=={'blocked','invalid audit snapshot'}
    for item in catalog['runs']:
        script=(tmp_path/'dashboard'/item['data_path']).read_text()
        data=json.loads(script.removeprefix('window.__dashboardRunData(').removesuffix(');'))
        assert data['frames']=={'reference':[],'candidate':[]}
        assert not data['progress']
        if data['status']=='blocked':
            assert data['reason']=='CPU/memory isolation unavailable'
            assert data['duration_wall']==2.5


def test_several_roots_share_one_dashboard(tmp_path):
    fixture_run(tmp_path/'home','run-a');fixture_run(tmp_path/'lab','run-b')
    runs=build_dashboard([tmp_path/'home',tmp_path/'lab'],tmp_path/'view',workers=2)['runs']
    assert [item['directory'] for item in runs]==['run-a','run-b']
    fixture_run(tmp_path/'lab','run-a')
    with pytest.raises(AuditFailure,match='duplicate'): build_dashboard([tmp_path/'home',tmp_path/'lab'],tmp_path/'view')


def test_catalog_identifies_attempts_with_the_same_display_name(tmp_path):
    fixture_run(tmp_path,'attempt-001')
    active,audit=fixture_run(tmp_path,'attempt-002',close=False)
    audit.close()
    (active/'status.json').write_text(json.dumps({'status':'running','started_at':'2026-10-28T23:00:00Z'}))
    runs=build_dashboard(tmp_path)['runs']
    assert {item['directory'] for item in runs}=={'attempt-001','attempt-002'}
    assert len({item['name'] for item in runs})==1
    assert next(item for item in runs if item['directory']=='attempt-002')['started_at']=='2026-10-28T23:00:00Z'


def test_cache_corruption_is_repaired_and_symlinks_refused(tmp_path):
    fixture_run(tmp_path);_,data=data_for(tmp_path)
    frame=tmp_path/'dashboard'/data['frames']['reference'][0]['rgb']
    frame.write_bytes(b'wrong cached data');data_for(tmp_path)
    assert frame.read_bytes()==PNG
    frame.unlink();frame.symlink_to(tmp_path/'outside')
    _,data=data_for(tmp_path)
    assert data['status']=='invalid audit snapshot'
    assert not (tmp_path/'outside').exists()


def test_units_and_non_summary_metadata_are_not_mislabeled():
    assert _unit('reconstruction.source_mass_kg')=='kg'
    assert _unit('reconstruction.source_center_of_mass_m.0')=='m'
    assert _unit('hinges.hinge.damping_absolute_error')=='N·m·s/rad'
    assert _unit('parameters.bodies.lever.hinge_damping')=='ratio'
    assert _unit('arm_probes.red_contact.joint_position_mae_rad')=='rad'
    assert _unit('arm_probes.red_contact.actuator_effort_mae_nm')=='N·m'
    assert _unit('arm_probes.red_contact.target_contact_tick_difference')=='ticks'
    assert _metric_group('arm_probes.red_contact.joint_position_mae_rad')==('Panda joint position error',True)


def test_notes_are_only_explicit_agent_or_provider_summaries(tmp_path):
    run,audit=fixture_run(tmp_path,close=False)
    audit.record({'kind':'inference','reasoning':'not archived as progress','wall_elapsed':4.5})
    audit.record({'kind':'progress','source':'hidden_chain','text':'must not appear','wall_elapsed':4.6})
    audit.close();_,data=data_for(tmp_path)
    assert all('must not appear' not in p['text'] for p in data['progress'])
    assert len(data['progress'])==3


BROWSER_CHECK=r'''<!doctype html><meta charset="utf-8"><iframe id="app" src="index.html" style="width:1250px;height:1100px"></iframe><pre id="result">waiting</pre><script>
function check(value,message){if(!value)throw Error(message);}
function nearly(a,b){return Math.abs(a-b)<.025;}
let tries=0;const timer=setInterval(()=>{try{const w=document.getElementById('app').contentWindow,d=w.document,test=w.__dashboardTest;if(!test||!test.getRun()){if(++tries>100)throw Error('dashboard did not load');return;}clearInterval(timer);
const preferred=w.__dashboardCatalog.find(item=>item.directory==='active'||item.directory==='finished'),rows=()=>[...d.querySelectorAll('#run-list .run-row')];check(test.getRun().id===preferred.id,'active or finished attempt selected by default');check(rows().length===w.__dashboardCatalog.length&&rows().every(row=>row.textContent.includes(w.__dashboardCatalog.find(item=>item.id===row.dataset.id).directory)),'attempt ids visible in run list');check(d.querySelector('.run-row[aria-pressed=true]').dataset.id===preferred.id,'selected run highlighted');check(d.getElementById('run-title').textContent===preferred.name,'selected run title');
const search=d.getElementById('run-search');search.value=preferred.directory;search.dispatchEvent(new w.Event('input'));check(rows().length===1&&rows()[0].dataset.id===preferred.id,'search narrows the list');search.value='';search.dispatchEvent(new w.Event('input'));
d.querySelector('#tag-filters button[data-category="Status"][data-value="'+preferred.status+'"]').click();check(rows().length===w.__dashboardCatalog.filter(item=>item.status===preferred.status).length,'status tag filter');d.getElementById('clear-filters').click();check(rows().length===w.__dashboardCatalog.length,'filters cleared');const days=[...new Set(w.__dashboardCatalog.map(i=>String(i.started_at||i.created_at||'').slice(0,10)).filter(Boolean))].sort(),df=d.getElementById('date-from');if(days.length>1){df.value=df.max;df.dispatchEvent(new w.Event('input'));check(rows().length>0&&rows().every(r=>{const i=w.__dashboardCatalog.find(x=>x.id===r.dataset.id);return String(i.started_at||i.created_at).startsWith(days[days.length-1]);}),'date range filter');d.getElementById('clear-filters').click();check(rows().length===w.__dashboardCatalog.length,'date filter cleared');}check(d.getElementById('date-readout').textContent.length>0,'date range readout');
const sort=d.getElementById('run-sort');sort.value='name-asc';sort.dispatchEvent(new w.Event('change'));const first=rows()[0].dataset.id;sort.value='name-desc';sort.dispatchEvent(new w.Event('change'));check(rows()[0].dataset.id!==first,'name sort direction');sort.value='date-desc';sort.dispatchEvent(new w.Event('change'));
const main=d.getElementById('timeline');main.value='1.8';main.dispatchEvent(new w.Event('input',{bubbles:true}));check(nearly(test.getState().time,1.8),'main seek');check(d.getElementById('reference-timeline').value===main.value,'reference range linked');check(test.getState().referenceSeq!==test.getState().candidateSeq,'independent held frames');
const candidate=d.getElementById('candidate-timeline');candidate.value='3.5';candidate.dispatchEvent(new w.Event('input',{bubbles:true}));check(nearly(test.getState().time,3.5),'recording seek');check(d.getElementById('timeline').value===candidate.value,'recording->main link');
const graph=d.querySelector('#plots svg'),rect=graph.getBoundingClientRect();graph.dispatchEvent(new w.MouseEvent('click',{clientX:rect.left+rect.width*.5,clientY:rect.top+50,bubbles:true}));const seek=test.getState().time;check(seek>1&&seek<4,'plot seek');const cursor=d.querySelector('.cursor');check(Number(cursor.getAttribute('x1'))>100,'plotcursor follows');
const note=d.querySelector('#progress button');note.click();check(nearly(test.getState().time,.2),'explicit note click seeks');
const before=test.getState().time,panel=d.getElementById('progress');panel.style.height='50px';panel.scrollTop=50;check(panel.scrollTop>0,'real panel scrolling');panel.dispatchEvent(new w.Event('scroll',{bubbles:true}));check(test.getState().time===before,'scroll must not seek');
main.value='2.5';main.dispatchEvent(new w.Event('input',{bubbles:true}));check(d.querySelector('#progress button.selected').textContent.includes('NOTE 2'),'text selection follows timeline');
const logFilter=d.getElementById('log-filter');logFilter.value='meta';logFilter.dispatchEvent(new w.Event('change'));check(d.querySelectorAll('#progress button').length===3,'meta filter reset submit finish');check([...d.querySelectorAll('#progress button')].every(row=>row.dataset.kind==='meta'),'meta entries distinct');check([...d.querySelectorAll('#progress button')].filter(row=>row.textContent.includes('submit')||row.textContent.includes('finish')).every(row=>row.textContent.includes('Run meta action')),'submit finish are run-level meta');check([...d.querySelectorAll('#action-ticks button')].filter(marker=>marker.title.includes('submit')||marker.title.includes('finish')).every(marker=>marker.title.startsWith('Run meta action')),'run meta marker titles');logFilter.value='candidate';logFilter.dispatchEvent(new w.Event('change'));check(d.querySelectorAll('#progress button').length===1,'filter reconstruction actions');check(d.querySelector('#progress button').textContent.includes('executed 3'),'partial failure actual ticks');
const marker=d.querySelector('#action-ticks button'),rangeBounds=main.getBoundingClientRect(),markerBounds=marker.getBoundingClientRect();check(Math.abs((markerBounds.left+markerBounds.width/2)-(rangeBounds.left+8+(rangeBounds.width-16)*.6/test.getRun().duration_wall))<2,'action marker aligned with range thumb travel');marker.click();check(nearly(test.getState().time,.6),'marker seeks start');check(logFilter.value==='reference','marker reveals filtered action');check(d.querySelector('#progress button.selected').dataset.logKey==='action-'+marker.dataset.actionSeq,'marker selects exact action');
const actionRow=d.querySelector('#progress button');test.setTime(4);actionRow.click();check(nearly(test.getState().time,.6),'action row seeks marker');check(test.getState().selectedLogKey===actionRow.dataset.logKey,'action row exact selection');check(marker.getAttribute('aria-pressed')==='true','action row selects timeline marker');
logFilter.value='all';logFilter.dispatchEvent(new w.Event('change'));const speed=d.getElementById('playback-speed');check([...speed.options].map(o=>o.value).join(',')==='1,2,4,8,16','speed choices');test.setTime(0);speed.value='8';d.getElementById('play').click();test.step(1000);test.step(1100);check(nearly(test.getState().time,.8),'8x playback wall advancement');test.stop();speed.value='1';
const checks=[...d.querySelectorAll('#metric-choices input')];check(checks.length>0,'metric checklist');for(const c of checks){c.checked=false;c.dispatchEvent(new w.Event('change'));}check(d.querySelectorAll('#plots svg').length===0,'unchecked metrics hidden');checks[0].checked=true;checks[0].dispatchEvent(new w.Event('change'));check(d.querySelectorAll('#plots svg').length===1,'selected group plotted');check(d.querySelector('#plots h3').textContent==='Total Chamfer distance [m]','single metric title excludes hidden COM');
const recorded=test.getRun();recorded.metrics.push({name:'reset.rgb_mse (retrospective candidate changes)',origin:'retrospective_candidate_change',unit:'squared normalized RGB',default_visible:true,points:[{wall_elapsed:.7,value:.2,origin:'retrospective_candidate_change',change_index:1,checkpoint_kind:'submitted_candidate',version:'abcdef123456789',evaluated_at:'2026-11-03T00:00:00Z'}]});w.__dashboardRunData(recorded);const retroTitle=[...d.querySelectorAll('#plots circle title')].find(t=>t.textContent.includes('Retrospective saved submission-only'));check(!!retroTitle&&retroTitle.textContent.includes('evaluated 2026-11-03T00:00:00Z')&&!retroTitle.textContent.includes('checkpoint null'),'retrospective tooltip identity and evaluation time');const density=recorded.metrics.find(metric=>metric.name.includes('density.per_primitive.'));check(!!density,'density series available');recorded.metrics.push({...density,name:density.name.replace('sphere','cube'),points:density.points.map(point=>({...point,value:point.value===null?null:point.value+2}))});w.__dashboardRunData(recorded);
let densityPlot=[...d.querySelectorAll('#plots article')].find(article=>article.querySelectorAll('.legend input').length===2);check(!!densityPlot,'two lines share a plot');let lineChecks=[...densityPlot.querySelectorAll('.legend input')];check(densityPlot.querySelectorAll('.point').length===4,'both lines initially shown');
lineChecks[0].checked=false;lineChecks[0].dispatchEvent(new w.Event('change',{bubbles:true}));densityPlot=[...d.querySelectorAll('#plots article')].find(article=>article.querySelectorAll('.legend input').length===2);check(densityPlot.querySelectorAll('.point').length===2,'one line hidden without hiding plot');check(!densityPlot.querySelectorAll('.legend input')[0].checked,'line checkbox persists after redraw');
lineChecks=[...densityPlot.querySelectorAll('.legend input')];lineChecks[1].checked=false;lineChecks[1].dispatchEvent(new w.Event('change',{bubbles:true}));densityPlot=[...d.querySelectorAll('#plots article')].find(article=>article.querySelectorAll('.legend input').length===2);check(densityPlot.querySelectorAll('.point').length===0,'all lines may be hidden while plot remains');
lineChecks=[...densityPlot.querySelectorAll('.legend input')];lineChecks[0].checked=true;lineChecks[0].dispatchEvent(new w.Event('change',{bubbles:true}));densityPlot=[...d.querySelectorAll('#plots article')].find(article=>article.querySelectorAll('.legend input').length===2);check(densityPlot.querySelectorAll('.point').length===2,'one line can be reenabled');
check(d.getElementById('grasp-overlay').type==='checkbox','overlay toggle');
const originalFrame=recorded.frames.reference[0];recorded.frames.reference=[[1,1],[1,1],[2,2],null,null,[3,3],[4,4]].map((handle,index)=>({...originalFrame,seq:100+index,wall_elapsed:index*.05,grasp_overlay:{width:320,height:240,anchor:[10,10],handle}}));
w.__dashboardRunData(recorded);test.setTime(.45);check(d.querySelectorAll('#reference-overlay polyline').length===2,'trajectory must break across unavailable projections');check([...d.querySelectorAll('#reference-overlay polyline')].every(path=>path.getAttribute('points').split(' ').length===2),'stationary handle positions coalesced');
const overlayToggle=d.getElementById('grasp-overlay');overlayToggle.checked=false;overlayToggle.dispatchEvent(new w.Event('change'));check(d.querySelector('#reference-overlay svg')===null,'overlay disabled');overlayToggle.checked=true;overlayToggle.dispatchEvent(new w.Event('change'));check(d.querySelectorAll('#reference-overlay circle').length===2,'recorded grasp and handle markers');
recorded.frames.reference.push({...originalFrame,seq:107,wall_elapsed:.5,grasp_overlay:null},{...originalFrame,seq:108,wall_elapsed:.6,grasp_overlay:{width:320,height:240,anchor:[10,10],handle:[8,8]}});w.__dashboardRunData(recorded);test.setTime(.6);check(d.querySelectorAll('#reference-overlay polyline').length===0,'release starts a new trajectory');
check(d.getElementById('reference-frame').hidden===false&&d.getElementById('candidate-frame').hidden===false,'both recording panes');const runRows=[...d.querySelectorAll('#run-list .run-row')],target=runRows.find(r=>r.dataset.id!==preferred.id)||runRows[0],sel=w.getSelection(),pick=node=>{const r=d.createRange();r.selectNodeContents(node);sel.removeAllRanges();sel.addRange(r);};pick(target.querySelector('code'));d.getElementById('run-title').textContent='-';target.click();check(d.getElementById('run-title').textContent==='-','selecting text in a row does not open it');pick(d.getElementById('run-directory'));target.click();check(d.getElementById('run-title').textContent===w.__dashboardCatalog.find(item=>item.id===target.dataset.id).name,'a selection elsewhere does not block opening a run');sel.removeAllRanges();check(!d.getElementById('run-loading').hidden&&d.getElementById('content').classList.contains('stale'),'loading indicator while a run loads');w.__dashboardRunError({id:target.dataset.id,error:'cache quota full'});check(d.getElementById('run-loading').hidden&&!d.getElementById('run-error').hidden&&d.getElementById('run-error').textContent.includes('cache quota full'),'load failure replaces the indicator with its reason');document.getElementById('result').textContent='PASS: main, recording, per-line plot toggles, cursor, text click, scroll isolation, held frames';document.body.dataset.test='passed';
}catch(error){clearInterval(timer);document.getElementById('result').textContent='FAIL: '+error.message;document.body.dataset.test='failed';}},50);
</script>'''


@pytest.mark.parametrize('active_default',[True,False])
def test_dashboard_browser_controls(tmp_path,active_default):
    browser=shutil.which('chromium-browser') or shutil.which('chromium')
    if not browser: pytest.skip('Chromium unavailable; run headless browser check on Slurm test node')
    if not os.environ.get('SLURM_JOB_ID'): pytest.skip('Browser validation runs only on a Slurm test allocation')
    if active_default:
        fixture_run(tmp_path)
        active,audit=fixture_run(tmp_path,'active',close=False);audit.close()
        (active/'status.json').write_text(json.dumps({'status':'running','started_at':'2026-10-28T23:00:00Z'}))
    else:
        failed=tmp_path/'runs/aaa-failed';failed.mkdir(parents=True)
        (failed/'status.json').write_text(json.dumps({'status':'failed'}))
        finished,audit=fixture_run(tmp_path,'finished',close=False)
        audit.record({'kind':'termination','status':'finished','wall_elapsed':5});audit.close()
        (finished/'status.json').write_text(json.dumps({'status':'completed','started_at':'2026-10-28T23:00:00Z'}))
    build_dashboard(tmp_path)
    page=tmp_path/'dashboard/selftest.html';page.write_text(BROWSER_CHECK)
    screenshot=tmp_path/'dashboard-browser.png'
    result=subprocess.run([browser,'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
        '--allow-file-access-from-files','--disable-background-networking','--disable-sync','--no-first-run',
        '--virtual-time-budget=10000','--window-size=1440,1250',f'--screenshot={screenshot}','--dump-dom',f'--user-data-dir={tmp_path}/browser-profile',page.as_uri()],
        capture_output=True,text=True,timeout=45)
    artifact=os.environ.get('SIM2SIM_DASHBOARD_ARTIFACT_DIR')
    if artifact:
        destination=Path(artifact);destination.mkdir(parents=True,exist_ok=True)
        (destination/'browser-dom.html').write_text(result.stdout)
        (destination/'browser-stderr.txt').write_text(result.stderr)
        if screenshot.is_file(): shutil.copyfile(screenshot,destination/'dashboard-browser.png')
        shutil.copytree(tmp_path/'dashboard',destination/'fixture-dashboard',dirs_exist_ok=True)
    assert 'data-test="passed"' in result.stdout,result.stdout[-5000:]+'\n'+result.stderr[-2000:]


def test_progress_origin_distinguishes_model_and_adapter(tmp_path):
    run,audit=fixture_run(tmp_path,close=False)
    for origin in ('model','adapter'):
        audit.record({'kind':'progress','source':'agent','origin':origin,'text':origin+' explicit status',
            'wall_elapsed':4.5,'model':'test-model' if origin=='model' else None})
    audit.close();_,data=data_for(tmp_path)
    assert [note['origin'] for note in data['progress'][-2:]]==['model','adapter']
    assert data['progress'][-2]['model']=='test-model'


def test_sealed_snapshot_rejects_appended_tail(tmp_path):
    run,audit=fixture_run(tmp_path)
    index=run/'audit/events.jsonl'
    with index.open('ab') as handle: handle.write(index.read_bytes().splitlines(keepends=True)[0])
    with pytest.raises(AuditFailure): list(read_audit(run,snapshot=True))
    _,data=data_for(tmp_path)
    assert data['status']=='invalid audit snapshot'
    assert not data['snapshot']['complete']


@pytest.mark.parametrize('mode,label',[('development_shared_filesystem','Development · shared FS'),('strict_oci','Strict OCI mode'),(None,'Mode unrecorded')])
def test_execution_mode_provenance_visible_without_false_isolation_claims(tmp_path,mode,label):
    run,audit=fixture_run(tmp_path)
    metadata={'name':'Mode provenance test'}
    if mode: metadata['execution_mode']=mode
    (run/'experiment.json').write_text(json.dumps(metadata))
    catalog,data=data_for(tmp_path)
    assert data['execution_mode']['label']==label
    assert catalog['runs'][0]['execution_mode']==data['execution_mode']
    if mode=='development_shared_filesystem':
        assert 'not a hard security boundary' in data['execution_mode']['detail']
    if mode=='strict_oci':
        assert 'Declared OCI execution mode' in data['execution_mode']['detail']
    assert 'id="execution-mode"' in (tmp_path/'dashboard/index.html').read_text()


def test_trusted_fixture_provenance_has_no_model_or_isolation_claim(tmp_path):
    run,audit=fixture_run(tmp_path)
    (run/'experiment.json').write_text(json.dumps({'name':'TRUSTED FIXTURE','fixture':True,'execution_mode':'strict_oci'}))
    _,data=data_for(tmp_path)
    assert data['execution_mode']['kind']=='trusted_fixture'
    assert 'no model or hard-isolation claim' in data['execution_mode']['detail']


def test_unmatched_metrics_remain_available_as_gaps():
    from sim2sim.dashboard import _metric_values, _metric_group
    values=_metric_values({'reconstruction':{'density':{'unmatched_source':['ball']}},
                          'hinges':{'joint':{'status':'unavailable'}}})
    assert values['reconstruction.density.per_primitive.ball.absolute_error_kg_m3'] is None
    assert values['hinges.joint.frictionloss_absolute_error'] is None
    assert _metric_group('reconstruction.density.per_primitive.a.absolute_error_kg_m3') == \
           _metric_group('reconstruction.density.per_primitive.b.absolute_error_kg_m3')
    # Sampled curves are not checkpoint series (one force_profile run had 12k of them); summaries and xyz vectors stay.
    profile=_metric_values({'force_profile':{'rms_resistance_error_n':.5,'source':{'resistance_n':[1.,2.,3.,4.],'force_world_n':[[0.,0.,1.]]*4}},
                            'reconstruction':{'center_of_mass_error_vector_m':[.1,.2,.3]}})
    assert set(profile)=={'force_profile.rms_resistance_error_n',*(f'reconstruction.center_of_mass_error_vector_m.{i}' for i in range(3))}


def test_recorded_overlay_is_private_metadata_and_missing_is_not_invented(tmp_path):
    run,audit=fixture_run(tmp_path,close=False)
    overlay={'anchor':[12,13],'handle':[14,15],'width':320,'height':240}
    audit.record({'kind':'checkpoint','world':'reference','wall_elapsed':4.5,'episode':0,
        'state':{'grasp_overlay':overlay},'observation':{'rgb':base64.b64encode(PNG).decode(),'ticks':2}})
    audit.close();_,data=data_for(tmp_path)
    assert data['frames']['reference'][0]['grasp_overlay'] is None
    assert data['frames']['reference'][-1]['grasp_overlay']==overlay


def test_blob_verification_cache_is_bounded_to_one_snapshot(tmp_path):
    from unittest.mock import patch
    run,audit=fixture_run(tmp_path)
    blob=run/'audit/blobs'/hashlib.sha256(PNG).hexdigest()
    reads=[];original=os.open
    def read(path,*args,**kwargs):
        if path==blob.name and kwargs.get('dir_fd') is not None: reads.append(path)
        return original(path,*args,**kwargs)
    with patch('sim2sim.audit.os.open',read):
        list(read_audit(run,snapshot=True))
    assert len(reads)==1
    blob.write_bytes(b'corrupt')
    with pytest.raises(AuditFailure): list(read_audit(run,snapshot=True))


def test_public_context_and_retrospective_series(tmp_path):
    run,_=fixture_run(tmp_path)
    (run/'experiment.json').write_text(json.dumps({'orchestration_notes':'Known retrospective note'}))
    (run/'public').mkdir()
    (run/'public/task.json').write_text(json.dumps({'task':'<script>untrusted</script>',
        'actions':['observe','arm_target'],'observation_contract':{'modalities':{'rgb':False}},
        'public_robot_path':'robot/panda.xml','observation':{'rgb':'do not copy'}}))
    (run/'candidate-change-history.json').write_text(json.dumps({'schema_version':1,'checkpoints':[
        {'change_index':1,'version':'first','wall_elapsed':.4,'evaluated_at':'later','status':'ok',
         'report':{'status':'ok','scores':{'reset':{'rgb_mse':.25}}}},
        {'change_index':2,'version':'second','wall_elapsed':.8,'status':'unavailable','report':{}}],
        'unavailable':[{'action_id':3,'reason':'failed load'}]}))
    _,data=data_for(tmp_path)
    context=data['model_context']
    assert context['public_task']['task']=='<script>untrusted</script>'
    assert 'observation' not in context['public_task']
    assert 'not recorded' in context['missing']
    assert context['researcher_notes']=='Known retrospective note'
    assert 'private' in context['visibility']
    original=next(m for m in data['metrics'] if m['name']=='reconstruction.chamfer_distance_m')
    assert len(original['points'])==3
    metric=next(m for m in data['metrics'] if m['origin']=='retrospective_candidate_change')
    assert metric['name']=='reset.rgb_mse (retrospective candidate changes)'
    assert metric['unit']=='squared normalized RGB' and metric['default_visible']
    assert [(p['wall_elapsed'],p['value']) for p in metric['points']]==[(.4,.25),(.8,None)]
    assert data['candidate_change_history']['unavailable'][0]['action_id']==3
    script=(tmp_path/'dashboard/dashboard.js').read_text()
    assert "$('model-context-text').textContent=" in script


def test_world_action_markers_preserve_partial_execution(tmp_path):
    fixture_run(tmp_path)
    _,data=data_for(tmp_path)
    assert len(data['actions'])==5
    assert [a['operation'] for a in data['actions'] if a['category']=='meta']==['reset','submit','finish']
    action=data['actions'][1]
    assert action['world']=='candidate' and action['operation']=='arm_target'
    assert action['wall_elapsed']==.9 and action['completed_wall_elapsed']==1.
    assert action['outcome']=='failed' and action['executed_ticks']==3 and action['charged_ticks']==10
    assert 'before_state' not in action


def test_legacy_action_completion_preserves_actual_partial_ticks(tmp_path):
    run=tmp_path/'runs'/'legacy';run.mkdir(parents=True)
    audit=AuditLog(run,max_bytes=30000)
    audit.record({'kind':'action_begin','world':'reconstruction','wall_elapsed':1,
                  'resolved_action':{'op':'observe','ticks':100}})
    audit.record({'kind':'action_end','world':'reconstruction','wall_elapsed':2,
                  'executed_ticks':3,'outcome':'error'})
    audit.record({'kind':'termination','wall_elapsed':3,'status':'completed'});audit.close()
    _,data=data_for(tmp_path)
    action=data['actions'][0]
    assert action['action_id'] is None and action['executed_ticks']==3
    assert action['outcome']=='error' and action['completed_wall_elapsed']==2
    assert 'serial world order' in action['correlation']
    assert any('Legacy actions' in warning for warning in data['warnings'])


def metadata_run(root,name,factor,environment='damped_lever',simulation=None,physical=None,status='completed'):
    run=root/'runs'/name;run.mkdir(parents=True)
    (run/'experiment.json').write_text(json.dumps({'name':f'Wave 2 · {environment} · {factor}','environment':environment,'model':'m',
        'execution_mode':'strict_oci','created_at':'2026-11-03T05:02:00+00:00','physical_parameters':{'factor':factor,**(physical or {})}}))
    (run/'status.json').write_text(json.dumps({'status':status,'started_at':'2026-11-03T06:00:00+00:00'}))
    (run/'config-private.json').write_text(json.dumps({'task':'Keep going; finishing early earns nothing.','simulation':simulation or {'rgb':True,'wrench':True}}))
    return run


def test_tags_and_ablation_comparisons_come_from_recorded_metadata(tmp_path):
    from sim2sim.dashboard import _discover,_summary,_comparisons
    hard={'perturbation.physics':'{}'}
    metadata_run(tmp_path,'w2-lever-hard-base-20261103-001','hard variant baseline',physical=hard)
    metadata_run(tmp_path,'w2-lever-hard-nowrench-20261103-001','hard variant, grasp wrench disabled (vs w2-lever-hard-base)',
                 simulation={'rgb':True,'wrench':False},physical=hard)
    for attempt in ('001','002'): metadata_run(tmp_path,f'w1-assembly-base-20261102-{attempt}','baseline','random_assembly')
    metadata_run(tmp_path,'w1-assembly-depth-20261102-003','depth observations enabled','random_assembly',{'rgb':True,'depth':True,'wrench':True})
    metadata_run(tmp_path,'w5-panda-noaction-20261103-001','observe-only reference world (vs w3-not-recorded)','panda_damped_lever',
                 {'rgb':True,'reference_interaction':False},status='running')
    summaries=[_summary(run,root,identifier) for root,run,identifier in _discover([tmp_path])]
    pairs={pair['id']:pair for pair in _comparisons(summaries)}
    # Explicit "(vs ...)" annotations, else the sibling -base condition; unrecorded baselines create no pair.
    assert set(pairs)=={'w2-lever-hard-nowrench vs w2-lever-hard-base','w1-assembly-depth vs w1-assembly-base'}
    assert pairs['w2-lever-hard-nowrench vs w2-lever-hard-base']['name']=='damped_lever · hard variant, grasp wrench disabled'
    assert sorted(pairs['w1-assembly-depth vs w1-assembly-base']['runs'])==['w1-assembly-base-20261102-001','w1-assembly-base-20261102-002','w1-assembly-depth-20261102-003']
    runs={item['directory']:item for item in summaries}
    nowrench=runs['w2-lever-hard-nowrench-20261103-001']
    assert nowrench['comparisons']==['w2-lever-hard-nowrench vs w2-lever-hard-base']
    assert nowrench['tags']['Interaction']==['No wrench'] and nowrench['tags']['Variant']==['Hard physics']
    assert nowrench['tags']['Series']==['Wave 2'] and nowrench['tags']['Prompt']==['Full-budget prompt'] and nowrench['tags']['Status']==['completed']
    assert runs['w1-assembly-depth-20261102-003']['tags']['Observations']==['RGB','Depth']
    panda=runs['w5-panda-noaction-20261103-001']
    assert panda['environment']=='panda/damped_lever' and panda['tags']['Robot']==['Panda'] and panda['tags']['Scene']==['damped_lever']
    assert 'Observe-only reference' in panda['tags']['Interaction'] and panda['comparisons']==[]


def test_live_dashboard_lists_new_runs_and_reverifies_only_changed_records(tmp_path,monkeypatch):
    import sim2sim.dashboard as dashboard
    fixture_run(tmp_path,'run-a')
    live=dashboard.LiveDashboard([tmp_path])
    assert b'"run-a"' in live.catalog_script() and b'__dashboardEnvironments' in live.catalog_script()
    assert b'__dashboardAgentSystem' in live.catalog_script()
    calls=[];original=dashboard._run_snapshot
    monkeypatch.setattr(dashboard,'_run_snapshot',lambda run,*rest:calls.append(run.name) or original(run,*rest))
    identifier=hashlib.sha256(b'run-a').hexdigest()[:16]
    first=live.run_script(identifier)
    assert first.startswith(b'window.__dashboardRunData(') and live.run_script(identifier)==first and calls==['run-a']
    (tmp_path/'runs/run-a/status.json').write_text(json.dumps({'status':'completed'}))
    live.run_script(identifier);assert calls==['run-a','run-a']
    fixture_run(tmp_path,'run-b');assert b'"run-b"' in live.catalog_script()
    with pytest.raises(KeyError): live.run_script('0'*16)
    # A static build records the same keys, so a later live server reuses its verified snapshot.
    build_dashboard(tmp_path);calls.clear()
    assert dashboard.LiveDashboard([tmp_path]).run_script(identifier).startswith(b'window.__dashboardRunData(') and calls==[]


def test_environment_base_prompts_are_the_base_task_without_experiment_additions():
    from sim2sim.dashboard import REPO, _assignments, agent_prompts, environment_scenes
    g = _assignments(REPO / 'experiments/configs/generate-waves.py')
    system, prompts = agent_prompts()
    assert system.startswith('You are an autonomous research agent') and set(prompts) == set(environment_scenes())
    additions = [value for name, value in g.items() if name.endswith('_HINT') or name.startswith('MASS_') or name == 'ALL_ANGLES']
    for path, prompt in prompts.items():
        assert prompt['task'].startswith(g['TASK_CONVERGE']) and '\nModel constraints:\n- ' in prompt['task']
        assert ('Panda robot fixture' in prompt['task']) == (path.split('/')[0] in ('panda', 'panda_small'))
        assert not any(addition in prompt['task'] for addition in additions), path
        assert prompt['level'] == '' or prompt['task'].startswith(g['TASK_CONVERGE'] + ' ' + prompt['level'])
