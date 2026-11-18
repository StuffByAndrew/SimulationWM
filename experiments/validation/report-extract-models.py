# Run on a compute node: srun -p sapphire -c 8 --mem=32G .venv/bin/python experiments/validation/report-extract-models.py WORK/models.json RUN_DIR...
"""Researcher-side: compile each run's private source and every submission ZIP with MuJoCo (XML compile + mj_forward only,
no plugins) and dump world-frame object geoms plus physical parameters. Usage: python extract_models.py OUT.json RUN_DIR..."""
import json, re, sys, tempfile, zipfile, xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

TYPES = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}


def robot_bodies(root):
    """Bodies of the public robot fixture (robot/*.xml: the Panda or the Shadow hand); they are instruments, not objects."""
    names = set()
    for path in Path(root).rglob("robot/*.xml"):
        names |= {b.get("name") for b in ET.parse(path).iter("body")}
    return names


def check_xml(root):
    for path in Path(root).rglob("*.xml"):
        text = path.read_text()
        assert not re.search(r"<\s*(plugin|extension)\b", text), path


def geom_masses(path, objects, m):
    """Per-geom mass: recompile with every other geom of the body massless. Bodies with explicit <inertial> are skipped
    (per-geom mass is undefined there)."""
    out = {}
    spec = mujoco.MjSpec.from_file(str(path))
    assert [b.name for b in spec.bodies] == [m.body(i).name for i in range(m.nbody)]
    for b in objects:
        ids = np.flatnonzero(m.geom_bodyid == b)
        if spec.bodies[b].explicitinertial or len(ids) == 0:
            continue
        assert len(spec.bodies[b].geoms) == len(ids)
        for k, g in enumerate(ids):
            s = mujoco.MjSpec.from_file(str(path))
            s.compiler.boundmass, s.compiler.boundinertia = 1e-9, 1e-12  # a massless geom alone must still compile
            for i, sg in enumerate(s.bodies[b].geoms):
                if i != k:
                    sg.mass = 0
            mass = float(s.compile().body_mass[b])
            out[int(g)] = 0.0 if mass <= 1e-9 else mass
    return out


def extract(path, root):
    check_xml(root)
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    robot = robot_bodies(root)
    objects = [b for b in range(1, m.nbody) if m.body(b).name not in robot and not m.body_mocapid[b] >= 0]
    masses = geom_masses(path, objects, m)
    geoms = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b not in objects:
            continue
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, d.geom_xmat[g])
        geom = dict(name=m.geom(g).name, body=m.body(b).name, type=TYPES[int(m.geom_type[g])], size=m.geom_size[g].tolist(),
                    pos=d.geom_xpos[g].tolist(), quat=quat.tolist(), rgba=m.geom_rgba[g].tolist(),
                    collide=bool(m.geom_contype[g] or m.geom_conaffinity[g]), mass=masses.get(g))
        if m.geom_type[g] == 7:
            mid = int(m.geom_dataid[g])
            va, vn, fa, fn = m.mesh_vertadr[mid], m.mesh_vertnum[mid], m.mesh_faceadr[mid], m.mesh_facenum[mid]
            geom["mesh"] = dict(vert=m.mesh_vert[va:va + vn].round(5).tolist(), face=m.mesh_face[fa:fa + fn].tolist())
            # mesh_vert is in the mesh frame; geom_xpos/xmat already include the mesh offset
        geoms.append(geom)
    bodies = [dict(name=m.body(b).name, mass=float(m.body_mass[b]), xipos=d.xipos[b].tolist(), root=int(m.body_rootid[b]) == b,
                   subtree_mass=float(m.body_subtreemass[b]), subtree_com=d.subtree_com[b].tolist()) for b in objects]
    def axis_inertia(j):  # inertia about the hinge axis of everything the hinge carries: its body and all bodies below it
        a, total = d.xaxis[j], 0.0
        for b in range(1, m.nbody):
            c = b
            while c not in (0, m.jnt_bodyid[j]):
                c = m.body_parentid[c]
            if c == 0:
                continue
            R, r = d.ximat[b].reshape(3, 3), d.xipos[b] - d.xanchor[j]
            total += a @ (R @ np.diag(m.body_inertia[b]) @ R.T + m.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))) @ a
        return float(total)
    joints = [dict(name=m.joint(j).name, type=int(m.jnt_type[j]), body=m.body(m.jnt_bodyid[j]).name,
                   stiffness=float(m.jnt_stiffness[j]), damping=float(m.dof_damping[m.jnt_dofadr[j]]),
                   frictionloss=float(m.dof_frictionloss[m.jnt_dofadr[j]]), armature=float(m.dof_armature[m.jnt_dofadr[j]]),
                   axis_inertia=axis_inertia(j) if m.jnt_type[j] == 3 else None, anchor=d.xanchor[j].tolist(), axis=d.xaxis[j].tolist())
              for j in range(m.njnt) if int(m.jnt_bodyid[j]) in objects and m.jnt_type[j] in (2, 3)]
    total = sum(m.body_mass[b] for b in objects)
    com = sum(m.body_mass[b] * d.xipos[b] for b in objects) / total if total > 0 else None
    return dict(geoms=geoms, bodies=bodies, joints=joints, total_mass=float(total), com=None if com is None else com.tolist(),
                gravity=m.opt.gravity.tolist(), timestep=float(m.opt.timestep))


def safe_unzip(zpath, dest):
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            p = Path(info.filename)
            assert not p.is_absolute() and ".." not in p.parts, info.filename
        z.extractall(dest)


def one(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rel = manifest["source"]["scene"].split("/source/", 1)[1]
    out = dict(run=run_dir.name, source=extract(run_dir / "source" / rel, run_dir / "source"), submissions=[])
    for z in sorted(run_dir.glob("submission-*.zip")):
        with tempfile.TemporaryDirectory() as tmp:
            safe_unzip(z, tmp)
            out["submissions"].append(dict(file=z.name, **extract(Path(tmp) / "scene.xml", tmp)))
    return out


if __name__ == "__main__":
    with ProcessPoolExecutor(8) as pool:
        result = {r["run"]: r for r in pool.map(one, sys.argv[2:])}
    Path(sys.argv[1]).write_text(json.dumps(result))
    for k, v in result.items():
        print(k, len(v["submissions"]), len(v["source"]["geoms"]))
