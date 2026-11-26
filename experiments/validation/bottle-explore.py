"""Development exploration of the bottle mechanism with raw MuJoCo weld grasps (Slurm test node only), before the harness weld
grasp exists: PYTHONPATH=src .venv/bin/python experiments/validation/bottle-explore.py [scene]
Each procedure: welds (mocap handle at the grasp point, object rigidly fixed) on the cap shell's top centre (g0) and the bottle
wall (g1); scripted handle translations and rotations about the grasp point; reports cap lift relative to the bottle, inner-cap
turn, weld forces, max weld violation and max speed."""
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

SCENE = sys.argv[1] if len(sys.argv) > 1 else "environments/bottle_contents/scene.xml"
root = ET.parse(SCENE).getroot()
world, eq = root.find("worldbody"), ET.SubElement(root, "equality")
for g in range(2):
    ET.SubElement(world, "body", name=f"h{g}", mocap="true")
    ET.SubElement(eq, "weld", name=f"w{g}", body1="cap_shell" if g == 0 else "bottle", body2=f"h{g}", active="false",
                  solref=".01 1", solimp=".9 .95 .001 .5 2")
M = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
D = mujoco.MjData(M)
dt = M.opt.timestep
B = {n: M.body(n).id for n in ("bottle", "cap", "cap_shell")}


def reset():
    mujoco.mj_resetData(M, D)
    mujoco.mj_forward(M, D)


def weld(g, body, point):
    e, mocap = M.equality(f"w{g}").id, M.body(f"h{g}").mocapid[0]
    b = B[body]
    D.mocap_pos[mocap] = point
    D.mocap_quat[mocap] = D.xquat[b]
    M.eq_data[e, :3] = 0
    M.eq_data[e, 3:6] = D.xmat[b].reshape(3, 3).T @ (point - D.xpos[b])
    M.eq_data[e, 6:10] = [1, 0, 0, 0]
    M.eq_data[e, 10] = 1
    D.eq_active[e] = 1
    mujoco.mj_forward(M, D)


def release(g):
    D.eq_active[M.equality(f"w{g}").id] = 0


def force(g):
    e = M.equality(f"w{g}").id
    rows = np.flatnonzero((D.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (D.efc_id == e))
    return -D.efc_force[rows[:3]] if len(rows) else np.zeros(3)


stats = {}


def run(seconds, g=None, dpos=(0, 0, 0), dyaw=0.):
    n = max(1, round(seconds / dt))
    for _ in range(n):
        if g is not None:
            mocap = M.body(f"h{g}").mocapid[0]
            D.mocap_pos[mocap] += np.asarray(dpos) / n
            q = np.zeros(4)
            mujoco.mju_axisAngle2Quat(q, [0, 0, 1], dyaw / n)
            mujoco.mju_mulQuat(D.mocap_quat[mocap], q, D.mocap_quat[mocap].copy())
        mujoco.mj_step(M, D)
        stats["vmax"] = max(stats.get("vmax", 0), float(np.abs(D.qvel).max()))
        viol = [np.abs(D.efc_pos[(D.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)]).max()] if D.nefc and np.any(D.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) else [0]
        stats["viol"] = max(stats.get("viol", 0), float(viol[0]))


def yaw(b):
    R = D.xmat[b].reshape(3, 3)
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def state(tag):
    rel = D.xpos[B["cap"]] - D.xpos[B["bottle"]]
    print(f"  {tag:28s} cap-bottle dz {1000 * (rel[2] - .094):7.2f} mm  lateral {1000 * np.linalg.norm(rel[:2]):5.2f} mm  "
          f"cap yaw {yaw(B['cap']) - yaw(B['bottle']):7.2f}  shell q {1000 * D.qpos[M.joint('shell_push').qposadr[0]]:6.2f} mm  "
          f"liner q {1000 * D.qpos[M.joint('liner').qposadr[0]]:6.2f} mm  bottle z {1000 * (D.xpos[B['bottle']][2] - .4):6.2f} mm  "
          f"bottle yaw {yaw(B['bottle']):6.1f}  F0 {np.round(force(0), 2)}  vmax {stats.get('vmax', 0):.3f} viol {1000 * stats.get('viol', 0):.2f} mm")


TOP = lambda: D.xpos[B["cap_shell"]] + D.xmat[B["cap_shell"]].reshape(3, 3) @ [0, 0, .0125]  # noqa: E731 shell top centre
SIDE = lambda: D.xpos[B["bottle"]] + [0, -.0225, .04]  # noqa: E731 bottle wall, front


def procedure(name, push, turn_deg, two=True, turn_s=.5):
    reset(); stats.clear()
    run(.3)
    state(f"{name}: rest")
    if two:
        weld(1, "bottle", SIDE())
    weld(0, "cap_shell", TOP())
    run(.3, 0, (0, 0, -push))
    state(f"{name}: pushed {1000 * push:.1f} mm")
    run(turn_s, 0, (0, 0, 0), np.deg2rad(turn_deg))
    state(f"{name}: turned {turn_deg}")
    run(.2, 0, (0, 0, push))
    if two:
        release(1)
    run(.5, 0, (0, 0, .05))
    state(f"{name}: lifted 50 mm")
    release(0)
    run(.5)
    state(f"{name}: released")


reset(); stats.clear()
q0 = D.qpos.copy()
run(2.0)
print("rest 2 s: max |dq| (first 7 bottle, cap)", np.abs(D.qpos - q0).max(), "vmax", stats["vmax"])
state("rest")
for push in (.002, .004, .006, .008, .010, .014, .020):
    procedure(f"two welds push {push}", push, 90)
procedure("two welds twist only", 0., 90)
procedure("two welds fast twist", 0., 90, turn_s=.04)
for push in (.008, .014, .020):
    procedure(f"one weld push {push}", push, 90, two=False)
for ang in (30, 45, 50, 55, 60, 70):
    procedure(f"two welds push .014 turn {ang}", .014, ang)
# bottle spin (inertia shortcut): weld the bottle and spin it fast
reset(); stats.clear(); run(.3)
weld(1, "bottle", D.xpos[B["bottle"]] + [0, 0, .04])
run(.04, 1, (0, 0, 0), np.deg2rad(90))
run(.3)
state("bottle spun 90 deg in 40 ms")
