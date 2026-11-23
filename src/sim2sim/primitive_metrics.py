"""Private reset-frame metrics for unions of uniformly dense MuJoCo primitives and convex meshes.

A mesh geom counts as the convex hull of its compiled vertices, exactly the shape MuJoCo collides; its `size` here is
that scipy ConvexHull (geom-frame vertices)."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np

_TYPES = {2: 'sphere', 3: 'capsule', 4: 'ellipsoid', 5: 'cylinder', 6: 'box', 7: 'mesh'}


def volume(kind, size):
    if kind == 'mesh': return size.volume
    s = np.asarray(size)
    if kind == 'sphere': return 4*math.pi*s[0]**3/3
    if kind == 'ellipsoid': return 4*math.pi*np.prod(s[:3])/3
    if kind == 'box': return 8*np.prod(s[:3])
    if kind == 'cylinder': return 2*math.pi*s[0]**2*s[1]
    if kind == 'capsule': return 2*math.pi*s[0]**2*s[1]+4*math.pi*s[0]**3/3
    raise ValueError(f'unsupported primitive: {kind}')


def surface_area(kind, size):
    if kind == 'mesh': return size.area
    a,b,c = size
    if kind == 'sphere': return 4*math.pi*a*a
    if kind == 'box': return 8*(a*b+a*c+b*c)
    if kind == 'cylinder': return 4*math.pi*a*b+2*math.pi*a*a
    if kind == 'capsule': return 4*math.pi*a*b+4*math.pi*a*a
    # Integrate the exact sphere-to-ellipsoid surface Jacobian deterministically.
    z,w=np.polynomial.legendre.leggauss(64)
    phi=np.arange(128)*2*math.pi/128
    radial=np.sqrt(1-z*z)[:,None]
    jac=np.sqrt((radial*np.cos(phi)/a)**2+(radial*np.sin(phi)/b)**2+(z[:,None]/c)**2)
    return float(a*b*c*2*math.pi*np.sum(w*jac.mean(axis=1)))


def _sphere(rng, n):
    points = rng.normal(size=(n,3))
    return points / np.linalg.norm(points,axis=1)[:,None]


def sample_surface(kind, size, n, rng):
    """Uniform-area local surface samples (ellipsoid uses Jacobian rejection)."""
    if kind == 'mesh':
        triangles=size.points[size.simplices]
        area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
        t=triangles[rng.choice(len(triangles),size=n,p=area/area.sum())]
        u,v=rng.random((2,n,1)); flip=(u+v)>1; u,v=np.where(flip,1-u,u),np.where(flip,1-v,v)
        return t[:,0]+u*(t[:,1]-t[:,0])+v*(t[:,2]-t[:,0])
    s = np.asarray(size,dtype=float)
    if kind == 'sphere': return _sphere(rng,n)*s[0]
    if kind == 'ellipsoid':
        chunks=[]; remaining=n
        while remaining:
            u=_sphere(rng,max(remaining*2,32))
            weights=np.linalg.norm(u/s,axis=1)*np.min(s)
            accepted=u[rng.random(len(u)) < weights][:remaining]*s
            chunks.append(accepted); remaining-=len(accepted)
        return np.concatenate(chunks)
    if kind == 'box':
        weights=np.array([s[1]*s[2],s[0]*s[2],s[0]*s[1]])
        axis=rng.choice(3,size=n,p=weights/weights.sum())
        p=rng.uniform(-1,1,size=(n,3))*s
        p[np.arange(n),axis]=rng.choice([-1,1],size=n)*s[axis]
        return p
    if kind in ('cylinder','capsule'):
        r,h=s[:2]; angle=rng.uniform(0,2*math.pi,n)
        side_area=4*math.pi*r*h
        cap_area=(2 if kind=='cylinder' else 4)*math.pi*r*r
        side=rng.random(n)<side_area/(side_area+cap_area)
        p=np.column_stack([r*np.cos(angle),r*np.sin(angle),rng.uniform(-h,h,n)])
        count=int((~side).sum())
        if kind=='cylinder':
            rad=r*np.sqrt(rng.random(count)); theta=rng.uniform(0,2*math.pi,count)
            p[~side]=np.column_stack([rad*np.cos(theta),rad*np.sin(theta),rng.choice([-h,h],count)])
        else:
            sphere=_sphere(rng,count)*r
            sphere[:,2]+=np.where(sphere[:,2]>=0,h,-h)
            p[~side]=sphere
        return p
    raise ValueError(f'unsupported primitive: {kind}')


def _inside(points, primitive):
    p=(points-primitive['pos']) @ primitive['rotation']
    s=primitive['size']; kind=primitive['kind']; eps=1e-9
    if kind=='mesh': return np.all(p@s.equations[:,:3].T+s.equations[:,3]<-eps,axis=1)
    if kind=='sphere': return np.sum(p*p,axis=1)<(s[0]-eps)**2
    if kind=='ellipsoid': return np.sum((p/s)**2,axis=1)<1-eps
    if kind=='box': return np.all(np.abs(p)<s-eps,axis=1)
    if kind=='cylinder': return (np.sum(p[:,:2]**2,axis=1)<(s[0]-eps)**2)&(np.abs(p[:,2])<s[1]-eps)
    if kind=='capsule':
        p=p.copy();p[:,2]-=np.clip(p[:,2],-s[1],s[1])
        return np.sum(p*p,axis=1)<(s[0]-eps)**2
    raise ValueError('unsupported union primitive')


def exposed_surface(primitives, samples=2048, seed=0):
    """Area-weighted primitive proposals, rejecting strictly internal union points."""
    rng=np.random.default_rng(seed)
    weights=np.array([surface_area(p['kind'],p['size']) for p in primitives]);weights/=weights.sum()
    result=[];count=0
    for attempt in range(100):
        n=max(2*(samples-count),256)
        choice=rng.choice(len(primitives),n,p=weights)
        for i,p in enumerate(primitives):
            m=int(np.sum(choice==i))
            if not m: continue
            points=sample_surface(p['kind'],p['size'],m,rng) @ p['rotation'].T+p['pos']
            keep=np.ones(m,dtype=bool)
            for j,other in enumerate(primitives):
                if i!=j: keep &= ~_inside(points,other)
            visible=points[keep]
            result.append(visible); count+=len(visible)
        if count>=samples:
            # Shuffle before truncating so body/geom traversal order has no sampling bias.
            points=np.concatenate(result); return points[rng.permutation(len(points))[:samples]]
    raise ValueError('unable to sample exposed union surface')


def _robot_bodies(sim):
    """Body ids belonging to the public robot fixture subtree (Panda arm and fingers, or Shadow hand and forearm)."""
    import mujoco
    from .robots import ROBOTS
    root=ROBOTS[sim.robot][1]
    base=mujoco.mj_name2id(sim._model,mujoco.mjtObj.mjOBJ_BODY,root)
    if base<0: raise ValueError(f'{sim.robot} fixture is missing {root}')
    ids=set()
    for body in range(1,sim._model.nbody):
        parent=body
        while parent:
            if parent==base:
                ids.add(body);break
            parent=int(sim._model.body_parentid[parent])
    return ids


def _primitives(sim, *, geom_names=None, exclude_robot=False):
    primitives=[]
    selected=set(geom_names) if geom_names is not None else None
    if selected is not None and len(selected)!=len(geom_names):
        raise ValueError('duplicate target geom names')
    robot_bodies=_robot_bodies(sim) if exclude_robot else set()
    found=set()
    for i in range(sim._model.ngeom):
        body=int(sim._model.geom_bodyid[i])
        if body==0 or body in robot_bodies: continue
        kind=_TYPES.get(int(sim._model.geom_type[i]))
        name=sim._model.geom(i).name or f'geom_{i}'
        if selected is not None and name not in selected: continue
        found.add(name)
        if kind is None: raise ValueError(f'unsupported object geom {name}')
        if kind=='mesh':
            from scipy.spatial import ConvexHull
            mesh=int(sim._model.geom_dataid[i]);start=int(sim._model.mesh_vertadr[mesh])
            size=ConvexHull(sim._model.mesh_vert[start:start+int(sim._model.mesh_vertnum[mesh])].astype(float))
        else: size=sim._model.geom_size[i].copy()
        primitives.append(dict(id=i,name=name,kind=kind,size=size,extent=sim._model.geom_size[i].copy(),
            pos=sim._data.geom_xpos[i].copy(),rotation=sim._data.geom_xmat[i].reshape(3,3).copy()))
    if selected is not None and found!=selected:
        raise ValueError('missing target object geom: '+', '.join(sorted(selected-found)))
    if not primitives: raise ValueError('no object primitives')
    return sorted(primitives,key=lambda p: (p['kind'],*np.round(p['extent'],12),*np.round(p['pos'],12),*np.round(p['rotation'].ravel(),12)))


def _densities(scene_path, sim, primitives):
    """Parsed effective densities, and why a primitive's declared mass does not set its compiled body's mass
    distribution (None when it does): explicit body inertial, massless geom, or a visual-only geom with a
    massless collision twin (geometry cannot tell the twins apart)."""
    import mujoco
    spec=mujoco.MjSpec.from_file(str(Path(scene_path).resolve()))
    geoms=list(spec.geoms)
    byname={g.name:g for g in geoms if g.name}
    model,data=sim._model,sim._data
    lo,hi=spec.compiler.inertiagrouprange
    inertial_wins=spec.compiler.inertiafromgeom!=mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE
    def parsed(i):
        name=model.geom(i).name
        geom=byname[name] if name in byname else geoms[i]
        if int(geom.type)!=int(model.geom_type[i]): raise ValueError('compiled/spec geometry order mismatch')
        return geom
    def massless(i):
        geom=parsed(i)
        return not lo<=model.geom_group[i]<=hi or (geom.mass==0 if np.isfinite(geom.mass) else geom.density==0)
    def collides(i): return bool(model.geom_contype[i] or model.geom_conaffinity[i])
    out,reasons={},{}
    for p in primitives:
        i=p['id'];geom=parsed(i)
        mass=float(geom.mass); density=float(geom.density)
        if int(geom.typeinertia) == int(mujoco.mjtGeomInertia.mjINERTIA_SHELL):
            raise ValueError('shell inertia density is not volumetric')
        rho=mass/volume(p['kind'],p['size']) if np.isfinite(mass) and mass>=0 else density
        if not np.isfinite(rho) or rho<0: raise ValueError('invalid primitive density')
        out[p['name']]=rho
        twins=[j for j in range(model.ngeom) if j!=i and model.geom_bodyid[j]==model.geom_bodyid[i] and collides(j)
               and np.linalg.norm(data.geom_xpos[j]-data.geom_xpos[i])<model.geom_rbound[i]]
        reasons[p['name']]=('explicit body inertial overrides geom masses' if inertial_wins and geom.parent.explicitinertial else
            'geom contributes no mass to its body' if massless(i) else
            'visual-only geom with a massless collision twin' if not collides(i) and any(massless(j) for j in twins) else None)
    return out,reasons


def _com(sim, *, body_names=None, exclude_robot=False):
    import mujoco
    if body_names is not None:
        ids=[]
        for name in body_names:
            body=mujoco.mj_name2id(sim._model,mujoco.mjtObj.mjOBJ_BODY,name)
            if body<0: raise ValueError('missing target object body: '+name)
            ids.append(body)
    else:
        robot_bodies=_robot_bodies(sim) if exclude_robot else set()
        ids=[i for i in range(1,sim._model.nbody) if i not in robot_bodies and
             not sim._model.body(i).name.startswith('_grasp_target_')]
    mass=sim._model.body_mass[ids]
    if not len(ids) or float(mass.sum())<=0: raise ValueError('object has no positive compiled mass')
    return np.sum(sim._data.xipos[ids]*mass[:,None],axis=0)/mass.sum(),float(mass.sum())


def _friction(sim, primitives):
    """Compiled [sliding, torsional, rolling] mu and condim per primitive, and the effective sliding mu of its contact with
    the world geom straight below its center at reset (a ray excluding its own body) under MuJoCo's mixing rule: the
    higher priority geom wins, else the elementwise max; condim 1 is frictionless. Unavailable, with a reason, when an
    explicit <pair> overrides the geom, nothing is below, it rests on another object, or the geom below cannot touch it."""
    import mujoco
    m,d=sim._model,sim._data
    paired={int(g) for g in np.asarray(m.pair_geom1).tolist()+np.asarray(m.pair_geom2).tolist()}
    out={}
    for p in primitives:
        i=p['id'];hit=np.array([-1],dtype=np.int32)
        mujoco.mj_ray(m,d,d.geom_xpos[i],np.array([0.,0.,-1.]),None,1,int(m.geom_bodyid[i]),hit)
        f=int(hit[0])
        record=dict(mu=m.geom_friction[i].tolist(),condim=int(m.geom_condim[i]),support_sliding_mu=None,reason=None)
        if i in paired: record['reason']='explicit contact pair overrides geom friction'
        elif f<0: record['reason']='no geom below'
        elif m.geom_bodyid[f]!=0: record['reason']='rests on another object, not the world'
        elif not (m.geom_contype[i]&m.geom_conaffinity[f] or m.geom_contype[f]&m.geom_conaffinity[i]):
            record['reason']='the world geom below cannot collide with it'
        else:
            a,b=int(m.geom_priority[i]),int(m.geom_priority[f])
            mu=m.geom_friction[i if a>b else f] if a!=b else np.maximum(m.geom_friction[i],m.geom_friction[f])
            condim=int(m.geom_condim[i if a>b else f]) if a!=b else max(int(m.geom_condim[i]),int(m.geom_condim[f]))
            record.update(support_sliding_mu=0.0 if condim==1 else float(mu[0]),support_geom=m.geom(f).name or f'geom_{f}')
        out[p['name']]=record
    return out


def match_primitives(sp, cp, seed=0, same_type=False, max_distance=None):
    """Geometry-only one-to-one assignment (pairwise 128-point surface Chamfer, reset world frame): never names,
    color, density or mass. Returns matched source/candidate indices and the cost matrix."""
    from scipy.spatial import cKDTree
    from scipy.optimize import linear_sum_assignment
    ss=[sample_surface(p['kind'],p['size'],128,np.random.default_rng(seed))@p['rotation'].T+p['pos'] for p in sp]
    cs=[sample_surface(p['kind'],p['size'],128,np.random.default_rng(seed))@p['rotation'].T+p['pos'] for p in cp]
    cost=np.array([[(cKDTree(a).query(b)[0].mean()+cKDTree(b).query(a)[0].mean())/2 for b in cs] for a in ss])
    eligible=np.ones(cost.shape,dtype=bool)
    if same_type:
        eligible &= np.array([[a['kind']==b['kind'] for b in cp] for a in sp])
    if max_distance is not None:
        if not isinstance(max_distance,(int,float)) or not math.isfinite(max_distance) or max_distance<=0:
            raise ValueError('matching_max_distance_m must be positive and finite')
        eligible &= cost<=max_distance
    rows,cols=linear_sum_assignment(np.where(eligible,cost,1e12))
    valid=eligible[rows,cols]
    return rows[valid],cols[valid],cost


def reconstruction_metrics(source,candidate,source_path,candidate_path,config=None):
    """Evaluate independent reset scenes; input paths are trusted evaluator-owned paths."""
    from scipy.spatial import cKDTree
    config=dict(config or {}); samples=int(config.get('samples',2048)); seed=int(config.get('seed',0))
    panda_objects=bool(config.get('panda_objects',False))
    if not 128<=samples<=32768: raise ValueError('reconstruction samples must be 128..32768')
    components=set(config.get('components', ['chamfer','center_of_mass','density']))
    source.action({'op':'reset'}, render=False) if hasattr(source, '_renderer') else source.action({'op':'reset'})
    candidate.action({'op':'reset'}, render=False) if hasattr(candidate, '_renderer') else candidate.action({'op':'reset'})
    out={}
    if 'chamfer' in components: out['chamfer_distance_m']=None
    if 'center_of_mass' in components: out['center_of_mass_error_m']=None
    for key in components & {'density','friction'}: out[key]={'status':'unavailable','per_primitive':[]}
    amass=bmass=float('nan'); a=b=None
    if 'center_of_mass' in components:
        try:
            a,amass=_com(source,body_names=config.get('source_object_bodies') if panda_objects else None)
            b,bmass=_com(candidate,exclude_robot=panda_objects);error=b-a
            out.update(center_of_mass_error_m=float(np.linalg.norm(error)),center_of_mass_error_vector_m=error.tolist(),
                source_center_of_mass_m=a.tolist(),predicted_center_of_mass_m=b.tolist(),center_of_mass_status='ok')
        except ValueError as exc: out.update(center_of_mass_status='unavailable',center_of_mass_error=str(exc))
    if not components & {'chamfer','density','friction'}: return out
    try:
        sp=_primitives(source,geom_names=config.get('source_object_geoms') if panda_objects else None)
        for key in components & {'density','friction'}: out[key]['unmatched_source']=[p['name'] for p in sp]
        cp=_primitives(candidate,exclude_robot=panda_objects)
    except ValueError as exc:
        if 'chamfer' in components: out.update(chamfer_status='unavailable',chamfer_error=str(exc))
        for key in components & {'density','friction'}: out[key]['error']=str(exc)
        return out
    if 'chamfer' in components:
        try:
            sa=exposed_surface(sp,samples,seed);ca=exposed_surface(cp,samples,seed)
            out.update(chamfer_distance_m=float((cKDTree(sa).query(ca)[0].mean()+cKDTree(ca).query(sa)[0].mean())/2),
                chamfer_status='ok',surface_samples=samples,surface_seed=seed)
        except ValueError as exc:
            out.update(chamfer_status='unavailable',chamfer_error=str(exc))
    if not components & {'density','friction'}: return out
    try: rows,cols,cost=match_primitives(sp,cp,seed,config.get('matching_same_type',False),config.get('matching_max_distance_m'))
    except ValueError as exc:
        for key in components & {'density','friction'}: out[key]['error']=str(exc)
        return out
    unmatched=dict(coverage=len(rows)/len(sp),unmatched_source=[p['name'] for i,p in enumerate(sp) if i not in rows],
                   unmatched_candidate=[p['name'] for j,p in enumerate(cp) if j not in cols])
    if 'friction' in components:
        sf,cf=_friction(source,sp),_friction(candidate,cp);records=[]
        for i,j in zip(rows,cols):
            s,c=sf[sp[i]['name']],cf[cp[j]['name']]
            reason='; '.join(f'{side}: {r["reason"]}' for side,r in (('source',s),('candidate',c)) if r['reason'])
            error=None if reason else abs(c['support_sliding_mu']-s['support_sliding_mu'])
            records.append(dict(source_name=sp[i]['name'],candidate_name=cp[j]['name'],source_mu=s['mu'],predicted_mu=c['mu'],
                source_condim=s['condim'],predicted_condim=c['condim'],sliding_mu_absolute_error=abs(c['mu'][0]-s['mu'][0]),
                source_support_sliding_mu=s['support_sliding_mu'],predicted_support_sliding_mu=c['support_sliding_mu'],
                support_sliding_mu_absolute_error=error,geometry_matching_cost_m=float(cost[i,j]),status='unavailable' if reason else 'ok',
                **({'unavailable_reason':reason} if reason else {})))
        out['friction']=dict(status='ok',matching='automatic_geometry_only',per_primitive=records,**unmatched)
    if 'density' not in components: return out
    try:
        (sd,sreason),(cd,creason)=_densities(source_path,source,sp),_densities(candidate_path,candidate,cp)
        records=[]
        for i,j in zip(rows,cols):
            # A density that does not set its body's mass distribution is a gap with a reason, never a number.
            s,c=sp[i]['name'],cp[j]['name']
            sr=None if sreason[s] else sd[s];cr=None if creason[c] else cd[c]
            error=None if sr is None or cr is None else abs(cr-sr)
            reason='; '.join(f'{side}: {r}' for side,r in (('source',sreason[s]),('candidate',creason[c])) if r)
            records.append(dict(source_name=s,candidate_name=c,source_density_kg_m3=sr,
                predicted_density_kg_m3=cr,absolute_error_kg_m3=error,relative_error=error/sr if error is not None and sr else None,
                geometry_matching_cost_m=float(cost[i,j]),status='unavailable' if reason else 'ok',
                **({'unavailable_reason':reason} if reason else {})))
        inferred_source=sum(sd[p['name']]*volume(p['kind'],p['size']) for p in sp)
        inferred_candidate=sum(cd[p['name']]*volume(p['kind'],p['size']) for p in cp)
        inferred_source_com=sum(sd[p['name']]*volume(p['kind'],p['size'])*p['pos'] for p in sp)/inferred_source if inferred_source>0 else None
        inferred_candidate_com=sum(cd[p['name']]*volume(p['kind'],p['size'])*p['pos'] for p in cp)/inferred_candidate if inferred_candidate>0 else None
        out['density']=dict(status='ok',matching='automatic_geometry_only',per_primitive=records,**unmatched,
            source_compiled_center_of_mass_consistent=bool(a is not None and inferred_source_com is not None and np.allclose(inferred_source_com,a,rtol=1e-5,atol=1e-8)),
            candidate_compiled_center_of_mass_consistent=bool(b is not None and inferred_candidate_com is not None and np.allclose(inferred_candidate_com,b,rtol=1e-5,atol=1e-8)),
            source_compiled_mass_consistent=bool(np.isclose(inferred_source,amass,rtol=1e-5,atol=1e-8)),
            candidate_compiled_mass_consistent=bool(np.isclose(inferred_candidate,bmass,rtol=1e-5,atol=1e-8)))
        if 'center_of_mass' not in components:
            # No compiled COM/mass comparison was requested or performed.
            for key in list(out['density']):
                if key.endswith('_consistent'): del out['density'][key]
    except (ValueError,IndexError,KeyError) as exc: out['density'].update(error=str(exc))
    return out
