"""Primitive geometry/numerical checks; execute in the Slurm test allocation."""
from types import SimpleNamespace
import math
import numpy as np
import pytest
from sim2sim.primitive_metrics import volume, exposed_surface, reconstruction_metrics


@pytest.mark.parametrize('kind,size,expected', [
    ('sphere',[2,0,0],32*math.pi/3), ('ellipsoid',[2,3,4],32*math.pi),
    ('box',[2,3,4],192), ('cylinder',[2,3,0],24*math.pi),
    ('capsule',[2,3,0],24*math.pi+32*math.pi/3)])
def test_analytic_volumes(kind,size,expected):
    assert volume(kind,size)==pytest.approx(expected)


def test_internal_overlap_surface_excluded():
    outer=dict(kind='sphere',size=np.array([1.,0,0]),pos=np.zeros(3),rotation=np.eye(3))
    inner=dict(kind='box',size=np.array([.2,.2,.2]),pos=np.zeros(3),rotation=np.eye(3))
    points=exposed_surface([outer,inner],samples=512,seed=2)
    assert np.linalg.norm(points,axis=1)==pytest.approx(np.ones(512))


def make_sim(tmp_path,name,geoms,inertial='',pos='0 0 0'):
    import mujoco
    scene=tmp_path / f'{name}.xml'
    scene.write_text(f'<mujoco><default><geom density="600"/></default><worldbody>'
        f'<geom name="floor" type="plane" size="3 3 .1"/><body name="object" pos="{pos}"><freejoint/>{inertial}{geoms}</body></worldbody></mujoco>')
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    def action(_):
        mujoco.mj_resetData(model,data);mujoco.mj_forward(model,data)
    action(None)
    return SimpleNamespace(_model=model,_data=data,action=action),scene


def metrics(source,candidate):
    result=reconstruction_metrics(source[0],candidate[0],source[1],candidate[1],{'samples':512,'seed':12})
    if result.get('chamfer_status') == 'ok':
        assert result['density']['status']=='ok', result
    return result


def test_reordered_renamed_identical_geometry_zero(tmp_path):
    a=make_sim(tmp_path,'a','<geom name="a" type="box" size=".1 .2 .3"/><geom name="b" type="sphere" pos=".3 0 0" size=".1"/>')
    b=make_sim(tmp_path,'b','<geom name="renamed2" type="sphere" pos=".3 0 0" size=".1"/><geom name="renamed1" type="box" size=".1 .2 .3"/>')
    result=metrics(a,b)
    assert result['chamfer_distance_m']==pytest.approx(0,abs=1e-12)
    assert result['center_of_mass_error_m']==pytest.approx(0,abs=1e-12)
    assert result['density']['coverage']==1
    assert all(p['absolute_error_kg_m3']==0 for p in result['density']['per_primitive'])


def test_world_pose_error_counts_and_com_vector(tmp_path):
    geom='<geom name="part" type="box" size=".1 .2 .3"/>'
    result=metrics(make_sim(tmp_path,'a',geom),make_sim(tmp_path,'b',geom,pos='.4 0 0'))
    assert result['chamfer_distance_m']>.1
    assert result['center_of_mass_error_m']==pytest.approx(.4)
    assert result['center_of_mass_error_vector_m']==pytest.approx([.4,0,0])


def test_defaults_mass_override_and_heterogeneous_density(tmp_path):
    geoms='<geom name="default" type="box" size=".1 .1 .1"/><geom name="override" type="sphere" pos=".3 0 0" size=".1" mass=".5"/>'
    result=metrics(make_sim(tmp_path,'a',geoms),make_sim(tmp_path,'b',geoms.replace('mass=".5"','mass="1"')))
    records={p['source_name']:p for p in result['density']['per_primitive']}
    assert records['default']['source_density_kg_m3']==pytest.approx(600)
    assert records['override']['source_density_kg_m3']==pytest.approx(.5/(4*math.pi*.1**3/3))
    assert records['override']['absolute_error_kg_m3']==pytest.approx(records['override']['source_density_kg_m3'])
    assert result['density']['source_compiled_mass_consistent'] is True
    assert result['chamfer_distance_m']==pytest.approx(0,abs=1e-12)
    expected=.3/(4.8+1)-.3*.5/(4.8+.5)
    assert result['center_of_mass_error_vector_m']==pytest.approx([expected,0,0])
    assert result['center_of_mass_error_m']==pytest.approx(expected)


def test_explicit_inertial_mass_inconsistency_flag(tmp_path):
    geom='<geom name="part" type="sphere" size=".1" density="1000"/>'
    result=metrics(make_sim(tmp_path,'a',geom),make_sim(tmp_path,'b',geom,
        inertial='<inertial pos="0 0 0" mass="9" diaginertia="1 1 1"/>'))
    assert result['density']['candidate_compiled_mass_consistent'] is False
    record=result['density']['per_primitive'][0]
    assert record['status']=='unavailable' and record['predicted_density_kg_m3'] is None and record['absolute_error_kg_m3'] is None
    assert record['unavailable_reason']=='candidate: explicit body inertial overrides geom masses'


def test_missing_primitive_explicit_coverage(tmp_path):
    a=make_sim(tmp_path,'a','<geom name="a" type="sphere" size=".1"/><geom name="b" type="sphere" pos=".3 0 0" size=".1"/>')
    b=make_sim(tmp_path,'b','<geom name="a" type="sphere" size=".1"/>')
    result=metrics(a,b)
    assert result['density']['coverage']==.5
    assert result['density']['unmatched_source']==['b']
    assert len(result['density']['per_primitive'])==1


def test_density_does_not_force_distant_or_wrong_type_assignment(tmp_path):
    a=make_sim(tmp_path,'source','<geom name="ball" type="sphere" size=".1"/>')
    for name,geom in [('distant','<geom name="guess" type="sphere" size=".1" pos="1 0 0"/>'),
                      ('wrong_type','<geom name="guess" type="box" size=".1 .1 .1"/>')]:
        b=make_sim(tmp_path,name,geom)
        result=reconstruction_metrics(a[0],b[0],a[1],b[1],{'components':['density'],
            'matching_same_type':True,'matching_max_distance_m':.08})
        assert result['density']['per_primitive']==[]
        assert result['density']['unmatched_source']==['ball']
        assert 'center_of_mass_error_m' not in result and 'chamfer_distance_m' not in result


def test_mesh_candidate_counts_as_its_convex_hull(tmp_path):
    import mujoco
    source=make_sim(tmp_path,'source','<geom name="part" type="sphere" size=".1"/>')
    path=tmp_path/'mesh.xml'
    path.write_text('<mujoco><asset><mesh name="tetra" vertex="0 0 0 .1 0 0 0 .1 0 0 0 .1" face="0 2 1 0 1 3 0 3 2 1 2 3"/></asset>'
                    '<worldbody><body name="object"><freejoint/><geom name="part" type="mesh" mesh="tetra"/></body></worldbody></mujoco>')
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    def action(_):
        mujoco.mj_resetData(model,data);mujoco.mj_forward(model,data)
    action(None)
    result=metrics(source,(SimpleNamespace(_model=model,_data=data,action=action),path))
    assert result['chamfer_status']=='ok' and result['chamfer_distance_m']>0.01
    assert result['density']['per_primitive'][0]['predicted_density_kg_m3']==pytest.approx(1000)
    assert result['density']['candidate_compiled_mass_consistent'] is True  # hull volume = MuJoCo's mesh volume
    assert result['center_of_mass_status']=='ok'


def test_same_mass_explicit_inertial_shift_flagged(tmp_path):
    geom='<geom name="part" type="sphere" size=".1" density="1000"/>'
    mass=4*math.pi*.1**3*1000/3
    result=metrics(make_sim(tmp_path,'a',geom),make_sim(tmp_path,'b',geom,
        inertial=f'<inertial pos=".03 0 0" mass="{mass}" diaginertia=".01 .01 .01"/>'))
    assert result['density']['candidate_compiled_mass_consistent'] is True
    assert result['density']['candidate_compiled_center_of_mass_consistent'] is False
    assert result['center_of_mass_error_m']==pytest.approx(.03)


def test_included_class_defaults_and_compiler_mass_scaling(tmp_path):
    import mujoco
    (tmp_path/'parts.xml').write_text('<mujocoinclude><body name="object" childclass="heavy"><freejoint/>'
        '<geom name="part" type="sphere" size=".1"/></body></mujocoinclude>')
    path=tmp_path/'scaled.xml'
    path.write_text('<mujoco><compiler settotalmass="3"/><default><default class="heavy"><geom density="1250"/></default></default>'
        '<worldbody><include file="parts.xml"/></worldbody></mujoco>')
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    def action(_):
        mujoco.mj_resetData(model,data);mujoco.mj_forward(model,data)
    action(None)
    sim=SimpleNamespace(_model=model,_data=data,action=action)
    result=metrics((sim,path),(sim,path))
    assert result['density']['per_primitive'][0]['source_density_kg_m3']==pytest.approx(1250)
    assert result['density']['source_compiled_mass_consistent'] is False
    assert result['density']['source_compiled_center_of_mass_consistent'] is True


TETRA=[(0,-.173205081,0,1),(-.15,.08660254,0,2),(.15,.08660254,0,4),(0,0,.244948974,8)]


@pytest.mark.parametrize('inertial',[True,False])
def test_visual_collision_split_cluster_density_unavailable(tmp_path,inertial):
    """w1-tetra-base-20261102-003 submission-012: one body, contype-0 visual spheres at unused default density,
    massless invisible collision twins, optionally an explicit inertial carrying the mass."""
    source=make_sim(tmp_path,'source',''.join(f'<geom name="s{m}" type="sphere" pos="{x} {y} {z}" size=".16" mass="{m}"/>'
                                              for x,y,z,m in TETRA))
    com=np.sum([np.array(p[:3])*p[3] for p in TETRA],axis=0)/15
    split=''.join(f'<geom name="v{m}" type="sphere" pos="{x} {y} {z}" size=".1605" contype="0" conaffinity="0"/>'
                  f'<geom name="c{m}" type="sphere" pos="{x} {y} {z}" size=".1612" mass="0" rgba="0 0 0 0"/>' for x,y,z,m in TETRA)
    candidate=make_sim(tmp_path,'candidate',split,inertial=f'<inertial pos="{com[0]} {com[1]} {com[2]+.0003}" mass="15.007" '
                       'diaginertia=".5 .45 .32"/>' if inertial else '')
    result=metrics(source,candidate)
    records=result['density']['per_primitive']
    assert len(records)==4 and all(r['status']=='unavailable' and r['source_density_kg_m3']>0 and
        r['predicted_density_kg_m3'] is None and r['absolute_error_kg_m3'] is None and r['relative_error'] is None for r in records)
    reasons={r['unavailable_reason'] for r in records}
    assert reasons=={'candidate: explicit body inertial overrides geom masses'} if inertial else \
        reasons<={'candidate: geom contributes no mass to its body','candidate: visual-only geom with a massless collision twin'}
    if inertial:
        assert result['center_of_mass_error_m']==pytest.approx(3e-4,abs=1e-6)


def test_authored_colliding_densities_stay_available(tmp_path):
    geoms=''.join(f'<geom name="s{m}" type="sphere" pos="{x} {y} {z}" size=".16" mass="{m}"/>' for x,y,z,m in TETRA)
    visual='<geom name="decal" type="sphere" pos="0 0 .5" size=".02" contype="0" conaffinity="0"/>'
    result=metrics(make_sim(tmp_path,'a',geoms),make_sim(tmp_path,'b',geoms.replace('mass="8"','mass="9"')+visual))
    records={r['source_name']:r for r in result['density']['per_primitive']}
    assert all(r['status']=='ok' for r in records.values()) and len(records)==4
    assert records['s8']['absolute_error_kg_m3']==pytest.approx(1/(4*math.pi*.16**3/3))
