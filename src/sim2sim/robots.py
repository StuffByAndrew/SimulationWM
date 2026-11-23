"""Public robot fixture registry (no MuJoCo import, so host code can use it without the simulator)."""

# Public robot fixtures: include file under robot/, root body of the fixture subtree, and the one actuation action agents
# use in place of point grasps.
ROBOTS = {"panda": ("panda.xml", "link0", "arm_target"), "shadow_hand": ("shadow_hand.xml", "rh_forearm", "hand_target")}
FINGERS = ("FF", "MF", "RF")
HAND_JOINTS = (["rh_WRJ2", "rh_WRJ1"] + [f"rh_{f}J{j}" for f in FINGERS for j in (4, 3, 2, 1)]
               + [f"rh_LFJ{j}" for j in (5, 4, 3, 2, 1)] + [f"rh_THJ{j}" for j in (5, 4, 3, 2, 1)])  # qpos order
HAND_ACTUATORS = (["rh_A_WRJ2", "rh_A_WRJ1"] + [f"rh_A_THJ{j}" for j in (5, 4, 3, 2, 1)]
                  + [f"rh_A_{f}J{j}" for f in FINGERS for j in (4, 3, 0)] + ["rh_A_LFJ5", "rh_A_LFJ4", "rh_A_LFJ3", "rh_A_LFJ0"])
# Reset pose: a palm-up cradle with slightly flexed fingers and the thumb raised beside the palm (unlisted joints 0).
HAND_HOME = {**{f"rh_{f}J{j}": q for f in FINGERS + ("LF",) for j, q in ((3, .25), (2, .15), (1, .15))},
             "rh_THJ5": .3, "rh_THJ4": .4}
