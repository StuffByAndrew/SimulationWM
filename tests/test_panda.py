"""Panda scenes and researcher-only scores; run on a Slurm test node."""
from __future__ import annotations

import io
import re
from pathlib import Path
import tomllib
import zipfile

import pytest

from sim2sim.evaluation import evaluate_submission
from sim2sim.simulation import Simulation


ROOT = Path(__file__).resolve().parents[1]
PANDA = ROOT / 'environments' / 'panda'
SMALL = ROOT / 'environments' / 'panda_small'
LEVELS = ('blocks', 'spring', 'damped_lever', 'tetrahedral_cluster',
          'random_assembly', 'primitive_boat', 'blocks_proprio', 'fixed_cube_proprio')


def _candidate(level, scene_edit=None, robot_edit=None, root=PANDA):
    scene=(root/level/'scene.xml').read_text().replace('../robot/assets','robot/assets').replace('../robot/panda.xml','robot/panda.xml')
    if scene_edit:
        scene=scene_edit(scene)
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('scene.xml',scene)
        for path in sorted((root/'robot').rglob('*')):
            if path.is_file():
                data=path.read_bytes()
                if robot_edit and path.name=='panda.xml':
                    data=robot_edit(data)
                archive.writestr('robot/'+path.relative_to(root/'robot').as_posix(),data)
    return buffer.getvalue()


@pytest.mark.parametrize('level',LEVELS)
def test_panda_scene_control_and_observation(level):
    blind=level.endswith('_proprio')
    sim=Simulation(PANDA/level/'scene.xml',dict(robot='panda',rgb=not blind,depth=False,
        calibration=not blind,wrench=False,width=96,height=72))
    try:
        observation=sim.observe()
        assert len(observation['proprioception']['joint_positions_rad'])==7
        assert ('rgb' in observation)==(not blind)
        assert ('calibration' in observation)==(not blind)
        assert 'depth' not in observation and 'wrench' not in observation
        home=[0,0,0,-1.57079,0,1.57079,-.7853]
        after=sim.action(dict(op='arm_target',joint_targets=home,gripper_width_m=.08,ticks=5))
        assert after['total_ticks']==5
        sim.action({'op':'reset'},render=False)
        assert sim.ticks==0 and sim.total_ticks==5
        if level=='fixed_cube_proprio':
            import mujoco
            body=mujoco.mj_name2id(sim._model,mujoco.mjtObj.mjOBJ_BODY,'fixed_cube')
            assert body>0 and sim._model.body_jntnum[body]==0
    finally:
        sim.close()


@pytest.mark.parametrize('level',LEVELS)
def test_panda_private_arm_probe_reaches_target(level):
    manifest=tomllib.loads((PANDA/level/'manifest.toml').read_text())
    sim=Simulation(PANDA/level/'scene.xml',dict(robot='panda',rgb=False,depth=False,
        calibration=False,width=64,height=48))
    try:
        import mujoco
        from sim2sim.evaluation import _panda_probe
        bodies=[mujoco.mj_name2id(sim._model,mujoco.mjtObj.mjOBJ_BODY,name) for name in manifest['object_bodies']]
        for probe in manifest['arm_probes']:
            target=probe.get('target_body')
            selected=[mujoco.mj_name2id(sim._model,mujoco.mjtObj.mjOBJ_BODY,target)] if target else bodies
            _,contacts,_=_panda_probe(sim,probe,selected,False)
            assert contacts>0, f"{level} arm probe {probe['name']} never contacts its target"
    finally:
        sim.close()


@pytest.mark.parametrize('level',('blocks','blocks_proprio','fixed_cube_proprio','damped_lever','tetrahedral_cluster'))
def test_panda_identity_evaluation(level,tmp_path):
    blind=level.endswith('_proprio')
    manifest=tomllib.loads((PANDA/level/'manifest.toml').read_text())
    selection={'blocks':['dynamics','chamfer','center_of_mass'],
               'blocks_proprio':['dynamics','chamfer','center_of_mass'],
               'fixed_cube_proprio':['dynamics','chamfer'],
               'damped_lever':['dynamics','hinges'],
               'tetrahedral_cluster':['dynamics','chamfer','density']}[level]
    report=evaluate_submission(_candidate(level),dict(source_dir=str(PANDA),
        source_entrypoint=f'{level}/scene.xml',manifest=f'{level}/manifest.toml',
        output_dir=str(tmp_path),metrics=selection,
        source_config=dict(robot='panda',rgb=not blind,depth=False,calibration=not blind,
                           width=96,height=72),camera='fixed',width=96,height=72))
    assert report['status']=='ok',report.get('error')
    scores=report['scores']
    assert all(item['joint_position_mae_rad']<1e-8 for item in scores['arm_probes'])
    assert all(('rgb_error' in item)==(not blind) for item in scores['arm_probes'])
    assert all(item['source_target_contact_ticks']>0 for item in scores['arm_probes'])
    if 'reconstruction' in scores:
        assert scores['reconstruction']['chamfer_distance_m']==pytest.approx(0,abs=1e-8)
    if 'hinges' in scores:
        assert all(item['status']=='ok' and item['damping_absolute_error']==0 for item in scores['hinges'].values())
    if 'density' in selection:
        assert scores['reconstruction']['density']['status']=='ok'
        assert all(item['absolute_error_kg_m3']==pytest.approx(0) for item in scores['reconstruction']['density']['per_primitive'])


def test_panda_fixed_cube_geometry_and_fixture_integrity(tmp_path):
    level='fixed_cube_proprio'
    config=dict(source_dir=str(PANDA),source_entrypoint=f'{level}/scene.xml',
        manifest=f'{level}/manifest.toml',output_dir=str(tmp_path),metrics=['chamfer'],
        source_config=dict(robot='panda',rgb=False,depth=False,calibration=False,width=64,height=48),
        camera='fixed',width=64,height=48)
    altered=_candidate(level,scene_edit=lambda s:s.replace('size=".06 .06 .06"','size=".08 .06 .06"'))
    report=evaluate_submission(altered,config)
    assert report['status']=='ok',report.get('error')
    assert report['scores']['reconstruction']['chamfer_distance_m']>0.001
    config['output_dir']=str(tmp_path/'tampered')
    bad=_candidate(level,robot_edit=lambda data:data.replace(b'gainprm="4500"',b'gainprm="3500"'))
    report=evaluate_submission(bad,config)
    assert report['status']=='failed'
    assert 'robot fixture' in report['error']


def test_panda_fixed_cube_contact_dynamics_distinguish_absent_target(tmp_path):
    level='fixed_cube_proprio'
    moved=_candidate(level,scene_edit=lambda s:s.replace('name="fixed_cube" pos="0 0 .46"',
                                                        'name="fixed_cube" pos="1.4 1.4 .46"'))
    report=evaluate_submission(moved,dict(source_dir=str(PANDA),
        source_entrypoint=f'{level}/scene.xml',manifest=f'{level}/manifest.toml',
        output_dir=str(tmp_path),metrics=['dynamics'],
        source_config=dict(robot='panda',rgb=False,depth=False,calibration=False,
                           width=64,height=48),camera='fixed',width=64,height=48))
    assert report['status']=='ok',report.get('error')
    probe=report['scores']['arm_probes'][0]
    assert probe['source_target_contact_ticks']>0
    assert probe['candidate_target_contact_ticks']==0
    assert probe['actuator_effort_mae_nm']>0.01


def test_panda_box_object_com_trajectory(tmp_path):
    level='box_marble_equal'
    probe=tomllib.loads((SMALL/level/'manifest.toml').read_text())['arm_probes'][0]
    def evaluate(name,edit=None):
        report=evaluate_submission(_candidate(level,scene_edit=edit,root=SMALL),dict(source_dir=str(SMALL),
            source_entrypoint=f'{level}/scene.xml',manifest=f'{level}/manifest.toml',output_dir=str(tmp_path/name),
            metrics=['dynamics'],arm_probes=[probe],camera='fixed',width=64,height=48,
            source_config=dict(robot='panda',rgb=False,depth=False,calibration=False,width=64,height=48)))
        assert report['status']=='ok',report.get('error')
        assert (tmp_path/name/'artifacts'/report['version']/'arm_probe_000'/'com.png').is_file()
        return report['scores']['arm_probes'][0]['object_com']
    same=evaluate('same')
    assert same['status']=='ok' and same['trajectory_max_error_m']==0 and same['mass_absolute_error_kg']==0
    double=lambda s: re.sub(r'(name="marble" type="sphere" size="[^"]+" density=")([0-9.]+)',
                            lambda m: m[1]+str(2*float(m[2])),s)
    heavier=evaluate('heavier',double)
    assert heavier['mass_absolute_error_kg']==pytest.approx(.04,rel=1e-6)
    assert heavier['trajectory_mean_error_m']>1e-3
