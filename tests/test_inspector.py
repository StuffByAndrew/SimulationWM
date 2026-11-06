"""Interactive inspector checks. Physics and browser checks run only on Slurm test nodes."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from sim2sim.audit import AuditFailure, AuditLog
from sim2sim.inspector import (BOX, CAPSULE, CYLINDER, ELLIPSOID, PLANE, SPHERE, _handler,
                               primitive_mesh, recorded_versions, version_at)

ON_TEST_NODE = bool(os.environ.get('SLURM_JOB_ID')) and os.environ.get('SLURM_JOB_PARTITION') == 'test'
SOURCE = '''<mujoco><option timestep="0.002"/><worldbody>
<camera name="fixed" pos="0 -1.2 .6" xyaxes="1 0 0 0 .45 .9"/>
<geom name="floor" type="plane" size="1 1 .1" rgba=".3 .3 .3 1"/>
<geom type="capsule" fromto="-.4 .3 .02 -.2 .3 .02" size=".02"/>
<geom type="cylinder" pos=".4 .3 .05" size=".03 .05"/>
<body name="box" pos="0 0 .05"><freejoint/><geom type="box" size=".05 .05 .05" mass=".2" rgba=".8 .2 .2 1"/></body>
<body name="ball" pos=".25 0 .04"><freejoint/><geom type="ellipsoid" size=".04 .04 .03" rgba=".2 .2 .8 1"/></body>
</worldbody></mujoco>'''
CANDIDATE = '''<mujoco><worldbody><geom type="plane" size="1 1 .1"/>
<body name="guess" pos="0 0 .05"><freejoint/><geom type="box" size=".05 .05 .05" mass=".3"/></body>
</worldbody></mujoco>'''
PLUGIN = '<mujoco><extension><plugin plugin="mujoco.elasticity.cable"/></extension><worldbody/></mujoco>'


def _zip(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buffer.getvalue()


def inspector_run(root, name='inspect-fixture', tamper=False):
    run = root / 'runs' / name
    run.mkdir(parents=True)
    source = _zip({'scene/scene.xml': SOURCE, 'scene/manifest.toml': 'name = "fixture"\n'})
    (run / 'source-snapshot.zip').write_bytes(source + (b'x' if tamper else b''))
    audit = AuditLog(run, max_bytes=2_000_000)
    audit.record({'kind': 'run_start', 'wall_elapsed': 0, 'manifest': {
        'source_bundle_sha256': hashlib.sha256(source).hexdigest(), 'source': {'scene': '/private/scene/scene.xml'},
        'evaluation': {'source_dir': '/private', 'source_entrypoint': 'scene/scene.xml', 'camera': 'fixed'},
        'simulation': {'camera': 'fixed'}, 'resolved_source_config': {'graspable_bodies': ['box', 'ball'], 'width': 64}}})
    archives = []
    for action, wall, text in ((5, 1.5, CANDIDATE), (9, 3.0, PLUGIN)):
        archive = _zip({'scene.xml': text})
        archives.append(archive)
        request = {'op': 'load_reconstruction', 'project_b64': base64.b64encode(archive).decode()}
        audit.record({'kind': 'boundary', 'direction': 'request', 'action_id': action,
                      'wire_b64': base64.b64encode(json.dumps(request).encode()).decode()})
        audit.record({'kind': 'action_end', 'operation': 'load_reconstruction', 'action_id': action, 'outcome': 'ok',
                      'candidate_version': hashlib.sha256(archive).hexdigest(), 'wall_elapsed': wall})
    audit.record({'kind': 'action_end', 'operation': 'load_reconstruction', 'action_id': 11, 'outcome': 'error', 'wall_elapsed': 3.5})
    audit.record({'kind': 'termination', 'status': 'completed', 'wall_elapsed': 4.0})
    audit.close()
    (run / 'candidate-change-history.json').write_text(json.dumps({'schema_version': 1, 'checkpoints': [
        {'change_index': 1, 'version': hashlib.sha256(archives[0]).hexdigest(), 'report': {'status': 'ok', 'scores': {
            'reconstruction': {'chamfer_distance_m': .004, 'center_of_mass_error_m': None}}}}]}))
    return run


@pytest.mark.parametrize('kind,size', [(BOX, (.1, .2, .3)), (SPHERE, (.2, 0, 0)), (ELLIPSOID, (.1, .2, .3)),
                                       (CAPSULE, (.1, .4, 0)), (CYLINDER, (.1, .4, 0))])
def test_primitive_meshes_are_closed_outward_and_sized(kind, size):
    vertices, faces = primitive_mesh(kind, size)
    triangles = vertices[faces]
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert (np.einsum('ij,ij->i', normal, triangles.mean(axis=1)) >= -1e-12).all()
    expected = {BOX: size, SPHERE: (size[0],) * 3, ELLIPSOID: size,
                CAPSULE: (size[0], size[0], size[0] + size[1]), CYLINDER: (size[0], size[0], size[1])}[kind]
    assert np.allclose(np.abs(vertices).max(axis=0), expected)
    assert faces.min() >= 0 and faces.max() < len(vertices)


def test_infinite_plane_uses_finite_display_extent_and_unknown_types_are_skipped():
    vertices, faces = primitive_mesh(PLANE, (0, 2, .1), plane_extent=7)
    assert np.allclose(np.abs(vertices).max(axis=0), (7, 2, 0)) and faces.shape == (2, 3)
    assert primitive_mesh(1, (1, 1, 1)) is None  # height fields are not displayed


def test_version_scrubber_selects_latest_load_at_or_before_wall_time():
    versions = [{'wall_elapsed': 1.5}, {'wall_elapsed': 3.0}]
    assert version_at(versions, 0) is None
    assert version_at(versions, 1.5) == 0 and version_at(versions, 2.9) == 0
    assert version_at(versions, 3.0) == 1 and version_at(versions, 99) == 1


def test_recorded_versions_come_from_verified_audit_and_sidecar(tmp_path):
    recorded = recorded_versions(inspector_run(tmp_path))
    assert [v['wall_elapsed'] for v in recorded['versions']] == [1.5, 3.0]
    assert recorded['versions'][0]['archive'].startswith(b'PK')
    # Unavailable scores stay absent instead of becoming zero.
    assert recorded['versions'][0]['scores'] == {'reconstruction.chamfer_distance_m': .004}
    assert recorded['versions'][1]['scores'] == {}
    assert recorded['unavailable'][0]['reason'] == 'original_load_failed'
    assert recorded['source_entrypoint'] == 'scene/scene.xml'
    assert recorded['source_simulation'] == {'graspable_bodies': ['box', 'ball']}
    assert recorded['duration_wall'] == 4.0


def test_source_snapshot_mismatch_and_missing_audit_are_refused(tmp_path):
    with pytest.raises(AuditFailure, match='hash mismatch'):
        recorded_versions(inspector_run(tmp_path, tamper=True))
    empty = tmp_path / 'runs/empty'
    empty.mkdir()
    with pytest.raises(AuditFailure, match='no verified audit'):
        recorded_versions(empty)


class FakeApp:
    def __init__(self):
        self.calls = []

    def state(self):
        return {'calls': self.calls}

    def select(self, request):
        self.calls.append(('select', request))

    def control(self, op, world, value):
        if op == 'bad':
            raise ValueError('unknown inspector control')
        self.calls.append(('control', op, world, value))


def test_api_requires_json_and_serves_current_view_code(tmp_path):
    from sim2sim.dashboard_view import SCRIPT
    (tmp_path / 'catalog.js').write_text('window.__dashboardCatalog=[];')
    (tmp_path / 'dashboard.js').write_text('stale snapshot view')
    app = FakeApp()
    server = ThreadingHTTPServer(('127.0.0.1', 0), _handler(app, tmp_path))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_address[1]}'

    def post(route, body, content_type='application/json'):
        request = urllib.request.Request(base + route, data=json.dumps(body).encode(), headers={'Content-Type': content_type})
        return urllib.request.urlopen(request, timeout=5)
    try:
        assert urllib.request.urlopen(base + '/dashboard.js', timeout=5).read().decode() == SCRIPT
        assert '__dashboardCatalog' in urllib.request.urlopen(base + '/catalog.js', timeout=5).read().decode()
        with pytest.raises(urllib.error.HTTPError) as error:
            post('/api/inspector/control', {'op': 'reset'}, 'text/plain')
        assert error.value.code == 415
        with pytest.raises(urllib.error.HTTPError) as error:
            post('/api/inspector/control', {'op': 'bad'})
        assert error.value.code == 400
        assert json.load(post('/api/inspector/select', {'run': 'r', 'time': 2}))['calls'] == [['select', {'run': 'r', 'time': 2}]]
        post('/api/inspector/control', {'op': 'reset'})
        assert app.calls[-1] == ('control', 'reset', 'both', None)
    finally:
        server.shutdown()
        server.server_close()


def test_live_server_without_physics_serves_current_catalog_and_snapshots(tmp_path):
    from sim2sim.dashboard import LiveDashboard
    inspector_run(tmp_path)
    live = LiveDashboard([tmp_path])
    server = ThreadingHTTPServer(('127.0.0.1', 0), _handler(None, live.output, live))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        catalog = urllib.request.urlopen(base + '/catalog.js', timeout=5).read().decode()
        assert '"inspect-fixture"' in catalog
        identifier = hashlib.sha256(b'inspect-fixture').hexdigest()[:16]
        assert urllib.request.urlopen(f'{base}/runs/{identifier}.js', timeout=30).read().startswith(b'window.__dashboardRunData(')
        inspector_run(tmp_path, 'added-later')
        assert '"added-later"' in urllib.request.urlopen(base + '/catalog.js', timeout=5).read().decode()
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + '/api/inspector/state', timeout=5)
        assert error.value.code == 503
    finally:
        server.shutdown()
        server.server_close()


def test_dashboard_has_physics_view_with_version_scrubber():
    from sim2sim.dashboard_view import HTML, SCRIPT
    for marker in ('id="view-physics"', 'id="physics-reference"', 'id="physics-candidate"', 'id="version-timeline"',
                   'id="recordings-view"', 'id="physics-view" hidden', 'id="physics-couple"', 'id="physics-overlay"',
                   'id="environments-view" hidden', 'id="environment-frame"', 'id="environment-list"', 'id="environment-sort"'):
        assert marker in HTML
    assert "fetch('api/inspector/'+path" in SCRIPT and "'Content-Type':'application/json'" in SCRIPT


def _wait(predicate, timeout=60, message='condition'):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError('timed out waiting for ' + message)


@pytest.fixture
def live_app(tmp_path):
    if not ON_TEST_NODE:
        pytest.skip('Interactive physics runs only on a Slurm test allocation')
    from sim2sim.inspector import InspectorApp
    inspector_run(tmp_path)
    app = InspectorApp(tmp_path, ports=(0, 0, 0, 0), session_seconds=300, workspace=tmp_path / 'workspace')
    yield app
    app.close()


def test_live_coupled_grasp_moves_both_worlds_and_overlay_shows_both(live_app):
    app = live_app
    app.select({'run': 'inspect-fixture', 'time': 2.0})
    reference, candidate, overlay = (app.viewers[name] for name in ('reference', 'candidate', 'overlay'))
    _wait(lambda: reference.positions is not None and candidate.positions is not None, message='both worlds ready')
    _wait(lambda: all(overlay.layers[w]['positions'] is not None for w in ('reference', 'candidate')), message='overlay layers')
    assert 0 in overlay.layers['reference']['frames'] and 0 not in overlay.layers['candidate']['frames']  # no duplicate floor
    box = next(i for i, b in enumerate(reference.model['bodies']) if b['name'] == 'box')
    guess = next(i for i, b in enumerate(candidate.model['bodies']) if b['name'] == 'guess')
    app.control('couple', 'both', True)
    down = {'origin': [0, 0, 1], 'direction': [0, 0, -1]}
    app.control('grasp', 'reference', down)
    _wait(lambda: reference.summary.get('grasp') == 'box', message='reference part picked')
    assert all(viewer.gizmo is None for viewer in app.viewers.values()) and not reference.summary['playing']
    app.control('grasp', 'candidate', down)
    _wait(lambda: all(viewer.gizmo is not None for viewer in app.viewers.values()), message='coupled gizmos')
    _wait(lambda: reference.summary['playing'] and candidate.summary['playing'], message='coupled worlds play together')
    app.drag(overlay, overlay.gizmo_origin + np.array([0, 0, .3]))
    _wait(lambda: reference.positions[box][2] > .25 and candidate.positions[guess][2] > .25, timeout=30, message='both lifted')
    assert np.allclose(candidate.gizmo.position, candidate.gizmo_origin + [0, 0, .3])
    assert app.state()['coupled'] and app.state()['overlay_port'] == overlay.port
    app.control('release', 'reference')  # coupled playback controls act on both worlds
    _wait(lambda: reference.summary.get('grasp') is None and candidate.summary.get('grasp') is None, message='both released')
    _wait(lambda: all(viewer.gizmo is None for viewer in app.viewers.values()), message='gizmos removed')


def test_live_environment_viewer_loads_one_scene_and_drags_it(tmp_path):
    if not ON_TEST_NODE:
        pytest.skip('Interactive physics runs only on a Slurm test allocation')
    from sim2sim.inspector import ENVIRONMENT, InspectorApp
    for name in ('alpha', 'beta'):
        (tmp_path / 'environments' / name).mkdir(parents=True)
        (tmp_path / 'environments' / name / 'scene.xml').write_text(SOURCE)
    app = InspectorApp(tmp_path, ports=(0, 0, 0, 0), session_seconds=300, workspace=tmp_path / 'workspace',
                       environments=tmp_path / 'environments')
    try:
        viewer = app.viewers[ENVIRONMENT]
        assert app.state()['environment']['path'] is None and viewer.positions is None  # nothing loads until one is picked
        with pytest.raises(ValueError):
            app.select_environment('gamma')
        with pytest.raises(ValueError):
            app.select_environment('../alpha')
        app.select_environment('beta')
        _wait(lambda: viewer.positions is not None, timeout=120, message='environment loaded')
        box = next(i for i, b in enumerate(viewer.model['bodies']) if b['name'] == 'box')
        app.control('couple', 'both', True)  # the run worlds' coupling never captures the environment viewer
        app.control('grasp', ENVIRONMENT, {'origin': [0, 0, 1], 'direction': [0, 0, -1]})
        _wait(lambda: viewer.summary.get('grasp') == 'box' and viewer.gizmo is not None, message='environment grasp')
        assert viewer.summary['playing'] and all(app.viewers[name].gizmo is None for name in ('reference', 'candidate', 'overlay'))
        app.drag(viewer, viewer.gizmo_origin + np.array([0, 0, .3]))
        _wait(lambda: viewer.positions[box][2] > .25, timeout=30, message='box lifted')
        state = app.state()['environment']
        assert state['path'] == 'beta' and state['port'] == viewer.port and state['grasp'] == 'box'
        app.select_environment('alpha')  # switching replaces the world
        _wait(lambda: app.state()['environment']['path'] == 'alpha' and viewer.positions is not None and viewer.summary.get('grasp') is None,
              timeout=120, message='switched environment')
    finally:
        app.close()


def test_live_versions_grasp_and_worker_errors(live_app):
    app = live_app
    app.select({'run': 'inspect-fixture'})
    # Default selection is the latest version, whose archive has a forbidden plugin.
    _wait(lambda: app.state()['worlds']['reference']['status'] == 'ready' and app.state()['worlds']['candidate']['status'] == 'error',
          message='source ready and plugin rejected')
    assert app.state()['selected'] == 1
    assert 'plugin' in app.state()['worlds']['candidate']['message'].lower()
    app.select({'run': 'inspect-fixture', 'time': 0.5})
    _wait(lambda: app.state()['worlds']['candidate']['status'] == 'empty', message='pre-load empty state')
    app.select({'run': 'inspect-fixture', 'time': 2.0})
    viewer = app.viewers['candidate']
    _wait(lambda: viewer.summary.get('status') == 'ready' and viewer.positions is not None, message='candidate ready')
    body = next(i for i, b in enumerate(viewer.model['bodies']) if b['name'] == 'guess')
    assert viewer.model['bodies'][body]['graspable']
    resting = viewer.positions[body].copy()
    app.control('grasp', 'candidate', {'origin': [0, 0, 1], 'direction': [0, 0, -1]})
    _wait(lambda: viewer.gizmo is not None and viewer.summary.get('grasp') == 'guess', message='grasp acknowledgement')
    app.control('target', 'candidate', [0, 0, .35])
    _wait(lambda: viewer.positions[body][2] > resting[2] + .2, timeout=30, message='grasped body lifted')
    reference = app.viewers['reference']
    assert reference.summary.get('grasp') is None  # worlds are simulated independently
    app.control('reset', 'both')
    _wait(lambda: viewer.summary.get('grasp') is None and abs(viewer.positions[body][2] - .05) < 5e-3 and viewer.gizmo is None,
          message='reset clears grasp')
    app.control('grasp', 'candidate', {'origin': [.9, .9, 1], 'direction': [0, 0, -1]})
    _wait(lambda: 'No graspable' in str(viewer.summary.get('notice')), message='floor click refused')
    assert {g['type'] for g in reference.model['geoms']} >= {PLANE, BOX, ELLIPSOID, CAPSULE, CYLINDER}


def test_unchanged_audit_reuses_verified_recovery(live_app, tmp_path, monkeypatch):
    import sim2sim.inspector as inspector
    inspector_run(tmp_path, 'second')
    calls = []
    original = inspector.recorded_versions
    monkeypatch.setattr(inspector, 'recorded_versions', lambda path: calls.append(path.name) or original(path))
    for name in ('inspect-fixture', 'second', 'inspect-fixture'):
        live_app.select({'run': name, 'index': 0})
        _wait(lambda: live_app.state()['worlds']['candidate']['status'] == 'ready', message=name + ' ready')
    assert calls == ['inspect-fixture', 'second']
    (tmp_path / 'runs/second/candidate-change-history.json').write_text('{}')
    live_app.select({'run': 'second', 'index': 0})
    _wait(lambda: live_app.state()['worlds']['candidate']['status'] == 'ready', message='changed sidecar reloaded')
    assert calls[-1] == 'second' and len(calls) == 3


def test_dashboard_physics_tab_with_live_backend(live_app, tmp_path):
    browser = shutil.which('chromium-browser') or shutil.which('chromium')
    if not browser:
        pytest.skip('Chromium unavailable on this test node')
    from sim2sim.dashboard import build_dashboard
    build_dashboard(tmp_path)
    (tmp_path / 'dashboard/selftest.html').write_text(PHYSICS_CHECK)
    server = ThreadingHTTPServer(('127.0.0.1', 0), _handler(live_app, tmp_path / 'dashboard'))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        result = subprocess.run([browser, '--headless', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
                                 '--disable-background-networking', '--no-first-run', '--window-size=1440,1250',
                                 '--virtual-time-budget=30000', '--dump-dom', f'--user-data-dir={tmp_path}/profile',
                                 f'http://127.0.0.1:{server.server_address[1]}/selftest.html'],
                                capture_output=True, text=True, timeout=180)
    finally:
        server.shutdown()
        server.server_close()
    assert 'data-test="passed"' in result.stdout, result.stdout[-4000:] + '\n' + result.stderr[-2000:]
    assert live_app.run == 'inspect-fixture'
    state = live_app.state()
    assert 0 < state['expires_at_unix'] - state['server_time_unix'] <= 600  # Slurm job end or 300 s session


def test_static_dashboard_reports_missing_inspector(tmp_path):
    browser = shutil.which('chromium-browser') or shutil.which('chromium')
    if not browser or not ON_TEST_NODE:
        pytest.skip('Browser validation runs only on a Slurm test allocation with Chromium')
    from sim2sim.dashboard import build_dashboard, environment_scenes
    run = inspector_run(tmp_path)
    (run / 'experiment.json').write_text(json.dumps({'name': 'Static fixture', 'environment': environment_scenes()[0]}))
    build_dashboard(tmp_path)
    page = tmp_path / 'dashboard/selftest.html'
    page.write_text(STATIC_CHECK)
    result = subprocess.run([browser, '--headless', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
                             '--allow-file-access-from-files', '--no-first-run', '--virtual-time-budget=10000',
                             '--dump-dom', f'--user-data-dir={tmp_path}/profile', page.as_uri()],
                            capture_output=True, text=True, timeout=45)
    assert 'data-test="passed"' in result.stdout, result.stdout[-4000:] + '\n' + result.stderr[-2000:]


_PAGE = '''<!doctype html><body><iframe id="dashboard" src="index.html" style="width:1400px;height:1100px"></iframe><pre id="result"></pre>
<script>
const started=Date.now();let stage=0;
function done(ok,text){document.getElementById('result').textContent=text;document.body.dataset.test=ok?'passed':'failed';clearInterval(timer);document.getElementById('dashboard').src='about:blank';}  // stop the WebGL viewers once decided
const timer=setInterval(()=>{try{const w=document.getElementById('dashboard').contentWindow,d=w.document;
if(Date.now()-started>25000)return done(false,'FAIL: timeout at stage '+stage+' '+(d.getElementById('version-label')?.textContent||'')+' '+(d.getElementById('physics-unavailable')?.textContent||''));
if(!w.__dashboardTest||!w.__dashboardTest.getRun())return;
%s
}catch(error){done(false,'FAIL: '+error.message);}},100);
</script></body>'''
PHYSICS_CHECK = _PAGE % '''if(stage===0){d.getElementById('view-physics').click();if(!d.getElementById('recordings-view').hidden||d.getElementById('physics-view').hidden)return done(false,'FAIL: tab visibility');stage=1;return;}
const state=w.__dashboardPhysics;if(!state||state.loading||!state.versions.length)return;
if(stage===1){const frame=d.getElementById('physics-candidate');if(!frame.src.includes(':'+state.worlds.candidate.port+'/'))return done(false,'FAIL: iframe port');
if(d.querySelectorAll('#version-ticks button').length!==2)return done(false,'FAIL: version markers');d.querySelector('#version-ticks button').click();stage=2;return;}
if(stage===2){if(state.selected!==0||!d.getElementById('version-label').textContent.includes('Version 1 of 2'))return;if(!d.getElementById('version-label').textContent.includes('Chamfer 4.00 mm'))return done(false,'FAIL: score label');
const slider=d.getElementById('version-timeline');slider.value='0.5';slider.dispatchEvent(new w.Event('input'));slider.dispatchEvent(new w.Event('change'));stage=3;return;}
if(stage===3){if(state.selected!==null)return;if(!d.getElementById('version-label').textContent.includes('No reconstruction had been loaded'))return done(false,'FAIL: empty label');
d.getElementById('version-next').click();stage=4;return;}
if(stage===4){if(state.selected!==0)return;const life=d.getElementById('server-life');if(life.hidden||!/Server time left [0-9]/.test(life.textContent))return done(false,'FAIL: server time left '+life.textContent);if(!d.getElementById('overlay-window').hidden)return done(false,'FAIL: overlay shown uncoupled');d.getElementById('physics-couple').click();stage=5;return;}
if(stage===5){if(!state.coupled)return;if(d.getElementById('overlay-window').hidden||!d.getElementById('physics-overlay').src.includes(':'+state.overlay_port+'/'))return done(false,'FAIL: overlay pane');if(!d.getElementById('couple-status').textContent.includes('Pick the reference and reconstruction part'))return done(false,'FAIL: couple status '+d.getElementById('couple-status').textContent);
d.querySelector('[data-physics="reset"]').click();d.getElementById('view-recordings').click();if(d.getElementById('recordings-view').hidden)return done(false,'FAIL: back to recordings');done(true,'PASS: physics tab, live version scrubber, markers and coupled overlay');}'''
STATIC_CHECK = _PAGE % '''if(stage===0){d.getElementById('view-physics').click();stage=1;return;}
const banner=d.getElementById('physics-unavailable');if(banner.hidden)return;if(!d.getElementById('server-life').hidden)return done(false,'FAIL: static snapshot shows server life');if(!banner.textContent.includes('Slurm test'))return done(false,'FAIL: banner text');
if(stage===1){d.getElementById('view-environments').click();stage=2;return;}
if(d.getElementById('environment-unavailable').hidden)return;const rows=d.querySelectorAll('#environment-list .env-row');if(!rows.length||rows.length!==w.__dashboardEnvironments.length)return done(false,'FAIL: environment list '+rows.length);
if(!rows[0].textContent.includes('built 20'))return done(false,'FAIL: built date '+rows[0].textContent);
rows[0].click();if(!d.getElementById('environment-task').textContent.includes('Model constraints:'))return done(false,'FAIL: base prompt not shown');
for(const id of ['browser','run-heading','browse-toggle'])if(w.getComputedStyle(d.getElementById(id)).display!=='none')return done(false,'FAIL: run chrome shown in the environments view: '+id);
d.getElementById('view-recordings').click();if(d.getElementById('recordings-view').hidden||w.getComputedStyle(d.getElementById('browse-toggle')).display==='none')return done(false,'FAIL: run chrome not restored');
done(true,'PASS: static snapshot explains missing inspector; environments view lists environments without the run browser');'''


ROBOT_FIRST_SCENE = '''<mujoco><compiler angle="radian" autolimits="true" meshdir="robot/assets"/>
<include file="robot/panda.xml"/>
<worldbody><geom type="plane" size="2 2 .1"/>
<geom name="hidden_slab" type="box" pos=".4 .4 .3" size=".1 .1 .01" group="3"/>
<body name="block" pos=".4 .4 .05"><freejoint/><geom type="box" size=".05 .05 .05" mass=".2"/></body>
</worldbody></mujoco>'''


def test_robot_include_before_scene_worldbody_keeps_object_bodies(tmp_path):
    from sim2sim.inspector_worker import _non_robot_bodies
    (tmp_path / 'robot.xml').write_text('<mujoco><worldbody><body name="link0"><body name="link1"><joint name="joint1"/>'
                                        '</body></body></worldbody></mujoco>')
    scene = tmp_path / 'scene.xml'
    scene.write_text('<mujoco><include file="robot.xml"/><worldbody><body name="block"><freejoint/></body>'
                     '<body name="fixed"/></worldbody></mujoco>')
    assert _non_robot_bodies(scene) == ['block']


def test_live_panda_candidate_with_robot_include_first_is_graspable(tmp_path):
    if not ON_TEST_NODE:
        pytest.skip('Native MJCF loading runs only on a Slurm test allocation')
    from sim2sim.inspector_worker import load
    robot = Path(__file__).resolve().parents[1] / 'environments/panda/robot'
    files = {'scene.xml': ROBOT_FIRST_SCENE}
    files.update({f'robot/{path.relative_to(robot).as_posix()}': path.read_bytes()
                  for path in robot.rglob('*') if path.is_file()})
    archive = _zip(files)
    sim = load(tmp_path / 'worker', {'kind': 'candidate', 'archive_b64': base64.b64encode(archive).decode(),
                                     'entrypoint': 'scene.xml', 'simulation': {'robot': 'panda'}})
    import mujoco
    names = [mujoco.mj_id2name(sim._model, mujoco.mjtObj.mjOBJ_BODY, body) for body in sim._constraints]
    assert names == ['block']
    # Hidden group-3 geometry blocks an all-group ray but not a displayed-group click.
    from sim2sim.inspector_worker import Session, VISIBLE_GROUPS
    origin, down = np.array([.4, .4, 1.0]), np.array([0, 0, -1.0])
    assert sim._ray_hit(origin, down) is None
    hit = sim._ray_hit(origin, down, VISIBLE_GROUPS)
    assert hit is not None and np.allclose(hit[1], [.4, .4, .1])
    reply = Session(sim).command({'op': 'grasp', 'origin': origin.tolist(), 'direction': down.tolist()})
    assert reply['ok'] and reply['body'] == 'block'
