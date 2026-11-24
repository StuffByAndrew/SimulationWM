"""Public robot fixture registry (no MuJoCo import, so host code can use it without the simulator)."""

# Public robot fixtures: include file under robot/, root body of the fixture subtree, and the actuation actions agents
# use in place of point grasps. arm_target drives the Franka joints (plus the parallel gripper on robots in GRIPPER);
# hand_target drives the Shadow hand actuators.
ROBOTS = {"panda": ("panda.xml", "link0", ("arm_target",)), "shadow_hand": ("shadow_hand.xml", "rh_forearm", ("hand_target",)),
          "panda_shadow": ("panda_shadow.xml", "link0", ("arm_target", "hand_target"))}
GRIPPER = ("panda",)
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
# Arm reset pose. panda_shadow: palm down with the hand's grasp_site 0.25 m above the table at (0, -0.10), fingers toward +x
# (the palm-down IK solution nearest the Panda home; experiments/validation/touch-check.py).
ARM_HOME = {"panda": (0, 0, 0, -1.57079, 0, 1.57079, -.7853),
            "panda_shadow": (0.462951, 0.380096, 0.451862, -2.080431, -1.306237, 1.158302, 0.163117)}
FINGERS = ("FF", "MF", "RF")
HAND_JOINTS = (["rh_WRJ2", "rh_WRJ1"] + [f"rh_{f}J{j}" for f in FINGERS for j in (4, 3, 2, 1)]
               + [f"rh_LFJ{j}" for j in (5, 4, 3, 2, 1)] + [f"rh_THJ{j}" for j in (5, 4, 3, 2, 1)])  # qpos order
HAND_ACTUATORS = (["rh_A_WRJ2", "rh_A_WRJ1"] + [f"rh_A_THJ{j}" for j in (5, 4, 3, 2, 1)]
                  + [f"rh_A_{f}J{j}" for f in FINGERS for j in (4, 3, 0)] + ["rh_A_LFJ5", "rh_A_LFJ4", "rh_A_LFJ3", "rh_A_LFJ0"])
# Reset pose: a palm-up cradle with slightly flexed fingers and the thumb raised beside the palm (unlisted joints 0).
HAND_HOME = {**{f"rh_{f}J{j}": q for f in FINGERS + ("LF",) for j, q in ((3, .25), (2, .15), (1, .15))},
             "rh_THJ5": .3, "rh_THJ4": .4}


def joints(robot):
    """Proprioception joint order: the arm joints, then the hand joints."""
    actions = ROBOTS[robot][2]
    return (ARM_JOINTS if "arm_target" in actions else ()) + (tuple(HAND_JOINTS) if "hand_target" in actions else ())


def actuators(robot):
    """actuator_effort_nm order: arm_target's joint actuators, then hand_target's actuators (the Panda gripper excluded)."""
    actions = ROBOTS[robot][2]
    return (ARM_ACTUATORS if "arm_target" in actions else ()) + (tuple(HAND_ACTUATORS) if "hand_target" in actions else ())
