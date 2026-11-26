"""Write the strict-sandbox wave configs. Shared blocks are identical across conditions by construction;
each condition declares exactly one factor relative to its baseline. Run from the repository root:
python experiments/configs/generate-waves.py   (wave-1 output must stay byte-identical; prepared runs reference it)"""
from pathlib import Path

TASK = ("Reconstruct the observed scene and its physical behavior as a self-contained MuJoCo scene.xml project, using the "
        "permitted observations and interactions. This session lasts the full time budget and finishing early earns nothing. "
        "Hidden tests evaluate your submissions from unobserved viewpoints and under interactions you have not tried, so keep "
        "running discriminating comparison experiments in both worlds, refine, and resubmit whenever your reconstruction "
        "improves. Call finish only when little time remains.")
# From 2026-11-03 (user): do not tell agents to use up their time; they finish when they believe their model no longer
# differs from the reference world. New conditions use TASK_CONVERGE with the rebuilt harness image, and need baselines
# rerun under the same prompt (TASK and the old image are "use the full budget" prompts).
TASK_CONVERGE = ("Reconstruct the observed scene and its physical behavior as a self-contained MuJoCo scene.xml project, using "
                 "the permitted observations and interactions. Hidden tests evaluate your submissions from unobserved viewpoints "
                 "and under interactions you have not tried, so run discriminating comparison experiments in both worlds, refine, "
                 "and resubmit whenever your reconstruction improves. Call finish when you believe nothing about your "
                 "reconstruction differs from the reference world.")
HINGE_HINT = (" Make sure to estimate the hinge joint's viscous damping and dry friction loss accurately from experiments "
              "and include both in your model.")
ALL_ANGLES = (" Make sure your reconstruction's appearance and geometry match the scene from every viewing angle, including "
              "sides and parts that the camera cannot currently see.")
CONSTRAINTS = ["Use standard MuJoCo primitives and joints.",
               "Submit a self-contained scene.xml ZIP using only local assets, without native plugins."]
PANDA_CONSTRAINTS = ["Use standard MuJoCo primitives and joints for the target object; preserve the provided Panda robot fixture unchanged.",
                     CONSTRAINTS[1]]
HAND_CONSTRAINTS = ["Model the target object with standard MuJoCo primitives, joints and inline convex meshes (<mesh vertex=\"...\"/>, "
                    "each collided as the convex hull of its vertices); preserve the provided Shadow hand robot fixture unchanged.",
                    CONSTRAINTS[1]]
RECON = "\n[evaluation.reconstruction]\nsamples = 2048\nseed = 20261029\n"
ENVS = {  # name: (metrics, evaluation extras)
    "blocks": (["chamfer", "center_of_mass", "image_mse"], RECON),
    "spring": (["hinges", "image_mse"], 'hinge_matching = "single"\n'),
    "damped_lever": (["hinges", "image_mse"], 'hinge_matching = "single"\n'),
    "random_assembly": (["chamfer", "center_of_mass", "image_mse"], RECON),
    "primitive_boat": (["chamfer", "center_of_mass", "image_mse"], RECON),
    "tetrahedral_cluster": (["chamfer", "density", "image_mse"], RECON + "matching_same_type = true\nmatching_max_distance_m = 0.08\n"),
}
# id: (environment, factor description, overrides)
RUNS1 = {
    "smoke-blocks": ("blocks", "smoke test: baseline, 20 min, $15 cap", {"wall": 1200, "cost": 15.0}),
    "w1-blocks-base": ("blocks", "baseline", {}),
    "w1-spring-base": ("spring", "baseline", {}),
    "w1-lever-base": ("damped_lever", "baseline", {}),
    "w1-lever-hint": ("damped_lever", "task names hinge damping and friction loss", {"task": TASK + HINGE_HINT}),
    "w1-assembly-base": ("random_assembly", "baseline", {}),
    "w1-assembly-depth": ("random_assembly", "depth observations enabled", {"depth": True}),
    "w1-tetra-base": ("tetrahedral_cluster", "baseline", {}),
    "w1-panda-blind": ("panda/blocks_proprio", "Panda arm, proprioception only, random canary colors", {"colors": "random"}),
}
# Wave 2. Wave 1 saturated blocks/spring/lever (exact recovery of round authored values), so contrasts on those scenes
# use one shared seeded non-round hard variant with its own baseline; unsaturated scenes pair with wave-1 baselines.
HARD = {"physics": "nonround"}
RUNS2 = {
    "w2-lever-hard-base": ("damped_lever", "hard variant baseline", HARD),
    "w2-lever-hard-nowrench": ("damped_lever", "hard variant, grasp wrench disabled (vs w2-lever-hard-base)", {**HARD, "wrench": False}),
    "w2-lever-hard-hint": ("damped_lever", "hard variant, task names hinge damping and friction loss (vs w2-lever-hard-base)",
                           {**HARD, "task": TASK + HINGE_HINT}),
    "w2-blocks-hard-base": ("blocks", "hard variant baseline", HARD),
    "w2-blocks-hard-nowrench": ("blocks", "hard variant, grasp wrench disabled (vs w2-blocks-hard-base)", {**HARD, "wrench": False}),
    # center_of_mass added: per-geom density is undefined when the agent models the rigid cluster with a body <inertial>
    # (w1-tetra-base did, mass 15.007 vs 15 kg); compare w1-tetra-base by retrospective COM scoring.
    "w2-tetra-gray": ("tetrahedral_cluster", "all spheres uniform gray (vs w1-tetra-base)",
                      {"colors": "uniform_gray", "metrics": ["chamfer", "density", "center_of_mass", "image_mse"]}),
    "w2-assembly-allangles": ("random_assembly", "task asks for appearance from every angle (vs w1-assembly-base)", {"task": TASK + ALL_ANGLES}),
    "w2-panda-vision": ("panda/blocks", "Panda arm with RGB and calibration, same canary colors (vs w1-panda-blind, w1-blocks-base)",
                        {"vision": True, "colors": "random"}),
}
# Wave 3a: does the hinge wrench effect (w2-lever-hard-nowrench) generalize to a second hinge scene (stiffness + damping)?
RUNS3 = {
    "w3-spring-hard-base": ("spring", "hard variant baseline", HARD),
    "w3-spring-hard-nowrench": ("spring", "hard variant, grasp wrench disabled (vs w3-spring-hard-base)", {**HARD, "wrench": False}),
}
# Wave 3b (pool 2026-11-04): remaining list items plus contrasts made testable by wave 2.
RUNS4 = {
    "w3-boat-base": ("primitive_boat", "baseline", {}),
    "w3-boat-depth": ("primitive_boat", "depth observations enabled (vs w3-boat-base)", {"depth": True}),
    "w3-assembly-nowrench": ("random_assembly", "grasp wrench disabled (vs w1-assembly-base)", {"wrench": False}),
    "w3-tetra-nowrench": ("tetrahedral_cluster", "grasp wrench disabled (vs w1-tetra-base, w2-tetra-gray)", {"wrench": False}),
    "w3-ps-assembly-fixed": ("panda_small/random_assembly", "Panda (small) arm, fixed camera only", {"vision": True}),
    "w3-ps-assembly-wrist": ("panda_small/random_assembly", "Panda (small) arm, fixed + wrist camera (vs w3-ps-assembly-fixed)",
                             {"vision": True, "wrist": True}),
    "w3-panda-lever-hard": ("panda/damped_lever", "Panda arm with vision on the hard lever; joint efforts, no wrench "
                            "(vs w2-lever-hard-base, w2-lever-hard-nowrench)", {**HARD, "vision": True}),
    "w3-lever-hard-nowrench-hint": ("damped_lever", "hard variant, wrench disabled, task names hinge parameters "
                                    "(vs w2-lever-hard-nowrench)", {**HARD, "wrench": False, "task": TASK + HINGE_HINT}),
}
# Wave 3c (pool 2026-11-03 remainder): segmentation-tool ablation. BOTH runs use the tools image
# (SIM2SIM_TRUSTED_IMAGE_RECORD=experiments/images/trusted-image-tools.json) so torch presence is identical.
RUNS5 = {
    "w3-assembly-toolsimg-base": ("random_assembly", "tools image, no tools (baseline for w3-assembly-seg)", {}),
    "w3-assembly-seg": ("random_assembly", "segmentation tool available (vs w3-assembly-toolsimg-base)", {"tools": ("segmentation",)}),
}
# Wave 4 (pool 2026-11-04, after 3b): wave 3c showed an available tool goes unused, so the task now says when each
# tool helps. Both runs use the tools image. Outcomes: tool uptake (calls per run) and the usual metrics.
SEG_HINT = (" Use the optional segmentation tool on your camera images to separate the object's parts, and compare "
            "per-part outlines between the two worlds when fitting and checking your primitives.")
PLAN_HINT = (" Use the optional planning tool to compute collision-free arm motions (inverse kinematics and joint-space "
             "paths) when reaching, grasping and moving the object.")
RUNS6 = {
    "w4-assembly-seg-directed": ("random_assembly", "segmentation tool, task says when it helps (vs w3-assembly-seg, "
                                 "w3-assembly-toolsimg-base, w1-assembly-base)", {"tools": ("segmentation",), "task": TASK + SEG_HINT}),
    "w4-panda-lever-hard-planning": ("panda/damped_lever", "planning tool, task says when it helps (vs w3-panda-lever-hard)",
                                     {**HARD, "vision": True, "tools": ("planning",), "task": TASK + PLAN_HINT, "cost": 58.0}),
}
# Wave 5 (2026-11-03, daily cap lifted by the user): how informative are the agent's interactions with the real world?
# The reference world is no-interaction (no grasp/move/release/arm_target); gravity still acts and the reconstruction world
# keeps every action. Each run copies its baseline's config except this one factor.
NOACT = {"noaction": True, "cost": 58.0, "pool_usd": 2000.0}  # pool ceiling raised: user lifted the $500 cap for 2026-11-03
RUNS7 = {
    "w5-lever-hard-noaction": ("damped_lever", "observe-only reference world (vs w2-lever-hard-base)", {**HARD, **NOACT}),
    "w5-spring-hard-noaction": ("spring", "observe-only reference world (vs w3-spring-hard-base)", {**HARD, **NOACT}),
    "w5-blocks-hard-noaction": ("blocks", "observe-only reference world (vs w2-blocks-hard-base)", {**HARD, **NOACT}),
    "w5-assembly-noaction": ("random_assembly", "observe-only reference world (vs w1-assembly-base, w3 RGB baselines)", NOACT),
    "w5-tetra-noaction": ("tetrahedral_cluster", "observe-only reference world (vs w1-tetra-base)", NOACT),
    "w5-boat-noaction": ("primitive_boat", "observe-only reference world (vs w3-boat-base)", NOACT),
    "w5-panda-lever-hard-noaction": ("panda/damped_lever", "Panda arm, observe-only reference world (vs w3-panda-lever-hard)",
                                     {**HARD, "vision": True, **NOACT}),
}
# Wave 6 (pool 2026-11-04; first wave under TASK_CONVERGE and the rebuilt main image 44064383e99f): without force sensing
# (and when only watching) every hinge parameter came out off by one common factor, i.e. the absolute scale is unidentified.
# Does stating one true mass in the task let the agent fix the scale? Masses are the seeded hard-variant source values
# (seed 2026110208, physics nonround; prepare asserts them). New-prompt baselines are included; compare only within wave 6.
MASS_SPRING = " The hinged lever (orange arm and red ball, one rigid body) has a total mass of 0.37442 kg."
MASS_BLOCKS = " The red block has a mass of 0.3808 kg."
RUNS8 = {
    "w6-spring-hard-base": ("spring", "converge prompt, hard variant baseline", {**HARD, "task": TASK_CONVERGE}),
    "w6-spring-hard-nowrench": ("spring", "converge prompt, wrench disabled (vs w6-spring-hard-base)",
                                {**HARD, "wrench": False, "task": TASK_CONVERGE}),
    "w6-spring-hard-nowrench-mass": ("spring", "converge prompt, wrench disabled, task states the lever mass (vs w6-spring-hard-nowrench)",
                                     {**HARD, "wrench": False, "task": TASK_CONVERGE + MASS_SPRING}),
    "w6-blocks-hard-nowrench": ("blocks", "converge prompt, wrench disabled", {**HARD, "wrench": False, "task": TASK_CONVERGE}),
    "w6-blocks-hard-nowrench-mass": ("blocks", "converge prompt, wrench disabled, task states the red block mass (vs w6-blocks-hard-nowrench)",
                                     {**HARD, "wrench": False, "task": TASK_CONVERGE + MASS_BLOCKS}),
}
# Wave 7 (pool 2026-11-04 remainder, convergence prompt, image 44064383e99f): wave 6 follow-ups. The blocks baseline with
# force sensing under the new prompt; the stated-mass effect on a second hinge scene (lever: damping + friction loss);
# and no-interaction with and without a stated mass (is a video plus one mass enough for full identification?).
MASS_LEVER = " The hinged lever (orange arm and red ball, one rigid body) has a total mass of 1.3290 kg."
RUNS9 = {
    "w7-blocks-hard-base": ("blocks", "converge prompt, hard variant baseline (vs w6-blocks-hard-nowrench)", {**HARD, "task": TASK_CONVERGE}),
    "w7-lever-hard-nowrench": ("damped_lever", "converge prompt, wrench disabled", {**HARD, "wrench": False, "task": TASK_CONVERGE}),
    "w7-lever-hard-nowrench-mass": ("damped_lever", "converge prompt, wrench disabled, task states the lever mass (vs w7-lever-hard-nowrench)",
                                    {**HARD, "wrench": False, "task": TASK_CONVERGE + MASS_LEVER}),
    "w7-spring-hard-noaction": ("spring", "converge prompt, observe-only reference world (vs w6-spring-hard-base)",
                                {**HARD, "noaction": True, "task": TASK_CONVERGE}),
    "w7-spring-hard-noaction-mass": ("spring", "converge prompt, observe-only, task states the lever mass (vs w7-spring-hard-noaction)",
                                     {**HARD, "noaction": True, "task": TASK_CONVERGE + MASS_SPRING}),
}

# Wave 8 (pool 2026-11-04 remainder): both new-prompt no-force lever runs set friction loss to 0, while the old-prompt
# no-force run found it (x0.288, near its x0.245 common factor) at similar tokens. Is the new prompt missing friction even
# with force sensing, and does naming friction recover it without force?
RUNS10 = {
    "w8-lever-hard-base": ("damped_lever", "converge prompt, hard variant baseline (vs w7-lever-hard-nowrench)", {**HARD, "task": TASK_CONVERGE}),
    "w8-lever-hard-nowrench-hint": ("damped_lever", "converge prompt, wrench disabled, task names hinge parameters (vs w7-lever-hard-nowrench)",
                                    {**HARD, "wrench": False, "task": TASK_CONVERGE + HINGE_HINT}),
}

# Wave 9 (pool 2026-11-04, ceiling raised to $700 by the user at ~17 UTC): wave 7 stated mass fixed the lever scale but
# friction stayed 0; wave 8's hinge hint recovered friction near the common factor. Together they should give full
# identification without force sensing, and the no-interaction variant asks whether video alone is enough on a friction hinge.
DAY700 = {"pool_usd": 700.0}
RUNS11 = {
    "w9-lever-hard-nowrench-mass-hint": ("damped_lever", "converge prompt, wrench disabled, task states the lever mass and names "
                                         "hinge parameters (vs w7-lever-hard-nowrench-mass, w8-lever-hard-nowrench-hint)",
                                         {**HARD, **DAY700, "wrench": False, "task": TASK_CONVERGE + MASS_LEVER + HINGE_HINT}),
    "w9-lever-hard-noaction-mass-hint": ("damped_lever", "converge prompt, observe-only, task states the lever mass and names "
                                         "hinge parameters (vs w9-lever-hard-nowrench-mass-hint)",
                                         {**HARD, **DAY700, "noaction": True, "task": TASK_CONVERGE + MASS_LEVER + HINGE_HINT}),
}

# Wave 10 (pool 2026-11-05, $700/day; main image 5904fd183fa4, tools image rebuilt on it): first use of the 2026-11-04
# infrastructure, all under the convergence prompt. (a) New-prompt baselines for the reconstruction scenes, scored with the
# new contact-friction, expert-video and policy-transfer metrics. (b) Tactile-only vs proprioception-only vs vision on Panda
# blocks (random canary colors). (c) Planning tools v2 on the Panda hard lever; both arms on the tools image.
NEWM = ["friction", "expert", "policy_transfer"]
RUNS12 = {
    "w10-assembly-base": ("random_assembly", "converge prompt baseline, new metrics", {"task": TASK_CONVERGE, "metrics": ENVS["random_assembly"][0] + NEWM}),
    "w10-tetra-base": ("tetrahedral_cluster", "converge prompt baseline, new metrics", {"task": TASK_CONVERGE, "metrics": ENVS["tetrahedral_cluster"][0] + NEWM}),
    "w10-boat-base": ("primitive_boat", "converge prompt baseline, new metrics", {"task": TASK_CONVERGE, "metrics": ENVS["primitive_boat"][0] + NEWM}),
    "w10-panda-blocks-proprio": ("panda/blocks_proprio", "converge prompt, Panda proprioception only, random canary colors",
                                 {"task": TASK_CONVERGE, "colors": "random", "metrics": ["dynamics", "chamfer", "center_of_mass", "image_mse", "friction"]}),
    "w10-panda-blocks-tactile": ("panda/blocks_proprio", "converge prompt, Panda proprioception + finger tactile, random canary colors "
                                 "(vs w10-panda-blocks-proprio)", {"task": TASK_CONVERGE, "colors": "random", "tactile": True,
                                 "metrics": ["dynamics", "chamfer", "center_of_mass", "image_mse", "friction"]}),
    "w10-panda-blocks-vision": ("panda/blocks", "converge prompt, Panda with RGB and calibration, random canary colors "
                                "(vs w10-panda-blocks-proprio)", {"task": TASK_CONVERGE, "colors": "random", "vision": True,
                                "metrics": ["dynamics", "chamfer", "center_of_mass", "image_mse", "friction"]}),
    "w10-panda-lever-hard-toolsimg-base": ("panda/damped_lever", "converge prompt, Panda hard lever, tools image, no tools",
                                           {**HARD, "vision": True, "task": TASK_CONVERGE, "cost": 58.0}),
    "w10-panda-lever-hard-planning": ("panda/damped_lever", "converge prompt, planning tools v2, task says when they help "
                                      "(vs w10-panda-lever-hard-toolsimg-base)",
                                      {**HARD, "vision": True, "tools": ("planning",), "task": TASK_CONVERGE + PLAN_HINT, "cost": 58.0}),
}
# Wave 11 (user, 2026-11-06): the panda_small (Franka) tetrahedral cluster with a slightly damped linear spring lying on its
# side and a round hole in the table that it fits flush into (environments/panda_small/tetra_spring*, generated by
# experiments/validation/tetra-spring-scenes.py). Camera on and movable (movecam); no force sensing (no wrench, and effort = false hides the arm's
# joint torques). Three levels state the exact stiffness: round plates, square plates (diagonal = round diameter, so the hole
# fit is less obvious), and round plates with a plug of unstated mass starting in the hole. The fourth states only that the
# spring is ideal and adds a block of stated mass. Statements are appended to the convergence task (user wording "ideal",
# although the spring now has very slight damping, also by user request).
SPRING_K = " The spring between the two plates is an ideal linear spring with a stiffness of exactly 21.7 N/m."
SPRING_BLOCK = " The spring between the two plates is an ideal linear spring. The purple block has a mass of exactly 0.0235 kg."
# Ablation option "movecam" (user, 2026-11-07): [simulation] movable_camera = true adds the `camera` action (re-aim either
# world's observation camera; no physics time). The spring levels use it.
SPRING_LEVEL = {"vision": True, "effort": False, "movecam": True, "metrics": ["chamfer", "center_of_mass", "density", "image_mse"]}
RUNS13 = {"w11-tetra-spring-k": ("panda_small/tetra_spring", "Franka, round plates, stiffness stated, no force sensing",
                                 {**SPRING_LEVEL, "task": TASK_CONVERGE + SPRING_K}),
          "w11-tetra-spring-k-square": ("panda_small/tetra_spring_square", "Franka, square plates, stiffness stated, no force sensing "
                                        "(vs w11-tetra-spring-k)", {**SPRING_LEVEL, "task": TASK_CONVERGE + SPRING_K}),
          "w11-tetra-spring-k-plug": ("panda_small/tetra_spring_plug", "Franka, plug of unstated mass in the hole, stiffness stated, "
                                      "no force sensing (vs w11-tetra-spring-k)", {**SPRING_LEVEL, "task": TASK_CONVERGE + SPRING_K}),
          "w11-tetra-spring-block": ("panda_small/tetra_spring_block", "Franka, ideal spring stated, block of stated mass, no force sensing",
                                     {**SPRING_LEVEL, "task": TASK_CONVERGE + SPRING_BLOCK})}

WAVES = {"wave3c-20261103": ("Wave 3c", "sim2sim-20261103", 44.0, RUNS5), "wave1-20261102": ("Wave 1", "sim2sim-500", 58.0, RUNS1), "wave2-20261103": ("Wave 2", "sim2sim-20261103", 44.0, RUNS2),
         "wave3-20261103": ("Wave 3a", "sim2sim-20261103", 44.0, RUNS3), "wave3b-20261104": ("Wave 3b", "sim2sim-20261104", 58.0, RUNS4),
         "wave4-20261104": ("Wave 4", "sim2sim-20261104", 44.0, RUNS6), "wave5-20261103": ("Wave 5", "sim2sim-20261103", 58.0, RUNS7),
         "wave6-20261104": ("Wave 6", "sim2sim-20261104", 44.0, RUNS8),
         "wave7-20261104": ("Wave 7", "sim2sim-20261104", 44.0, RUNS9),
         "wave8-20261104": ("Wave 8", "sim2sim-20261104", 44.0, RUNS10),
         "wave9-20261104": ("Wave 9", "sim2sim-20261104", 44.0, RUNS11),
         "wave10-20261105": ("Wave 10", "sim2sim-20261105", 44.0, RUNS12),
         "wave11-20261106": ("Wave 11", "sim2sim-20261106", 44.0, RUNS13)}

# SAM 3D tool: a condition sets {"tools": ("sam3d",)} (combinable, e.g. ("segmentation", "sam3d"), which needs the tools
# image). It adds the `sam3d` controller action (image + mask -> camera-frame mesh) and needs the rebuilt main image (the
# harness exposes the action) plus a running researcher GPU service: sbatch scripts/sam3d-service.slurm, then launch with
# SIM2SIM_SAM3D_SPOOL exported (the run fails at start without a fresh service heartbeat). Its baseline is the same
# condition without the tool on the same image; a hint such as SAM3D_HINT below is a separate factor.
SAM3D_HINT = (" Use the optional sam3d action on a camera image and a mask of one part (for example from your own "
              "segmentation) to get a 3D shape estimate of that part, including its unseen sides, and check it against your "
              "observations before using it.")
# Wave 12 (2026-11-06): SAM 3D tool, first runs. Image experiments/images/trusted-image-sam3d.json; one GPU service and
# spool per run. The new-prompt baselines (w10-*-base) used image 5904fd183fa4, and the sam3d harness change is inert
# without [tools] sam3d. The directed condition adds SAM3D_HINT (a separate factor, vs the undirected tool run).
RUNS14 = {"w12-boat-sam3d": ("primitive_boat", "sam3d tool available (vs w10-boat-base)", {"tools": ("sam3d",), "task": TASK_CONVERGE, "metrics": ENVS["primitive_boat"][0] + NEWM}),
          "w12-boat-sam3d-directed": ("primitive_boat", "sam3d tool, task says when it helps (vs w12-boat-sam3d)",
                                      {"tools": ("sam3d",), "task": TASK_CONVERGE + SAM3D_HINT, "metrics": ENVS["primitive_boat"][0] + NEWM}),
          "w12-assembly-sam3d-directed": ("random_assembly", "sam3d tool, task says when it helps (vs w10-assembly-base)",
                                          {"tools": ("sam3d",), "task": TASK_CONVERGE + SAM3D_HINT, "metrics": ENVS["random_assembly"][0] + NEWM})}
WAVES["wave12-20261106"] = ("Wave 12", "sim2sim-20261106", 44.0, RUNS14)
# Wave 13 (2026-11-07): fixed-camera twins of two wave 11 spring levels; the only difference is movable_camera.
RUNS15 = {f"w13-tetra-spring-{k}-fixedcam": (RUNS13[f"w11-tetra-spring-{k}"][0], f"fixed camera (vs w11-tetra-spring-{k})",
                                             {**RUNS13[f"w11-tetra-spring-{k}"][2], "movecam": False}) for k in ("k", "block")}
WAVES["wave13-20261106"] = ("Wave 13", "sim2sim-20261106", 44.0, RUNS15)
# Wave 14 (2026-11-07): movable camera on the two geometry scenes; otherwise identical to the w10 baselines.
RUNS16 = {f"w14-{e}-movecam": (RUNS12[f"w10-{e}-base"][0], f"movable camera (vs w10-{e}-base)", {**RUNS12[f"w10-{e}-base"][2], "movecam": True})
          for e in ("boat", "assembly")}
WAVES["wave14-20261107"] = ("Wave 14", "sim2sim-20261107", 44.0, RUNS16)
# Wave 15 (2026-11-07): wave 12 agents never called sam3d; here the task requires it, to measure the tool's effect.
SAM3D_USE = (" Before modeling each part, call the sam3d action on a camera image with a mask of that part (for example from your "
             "own segmentation), and compare the returned 3D shape with your observations before using it.")
RUNS17 = {f"w15-{e}-sam3d-required": (env, f"sam3d tool, task requires calling it (vs w10-{e}-base, w12)",
                                      {"tools": ("sam3d",), "task": TASK_CONVERGE + SAM3D_USE, "metrics": ENVS[env][0] + NEWM})
          for e, env in (("boat", "primitive_boat"), ("assembly", "random_assembly"))}
WAVES["wave15-20261107"] = ("Wave 15", "sim2sim-20261107", 44.0, RUNS17)

# Wave 16 (user, 2026-11-08; launched 2026-11-09): occluded tunnel levels (environments/tunnel_* and environments/panda_small/tunnel_*, written by
# experiments/validation/tunnel-scenes.py): a stick in a closed tunnel whose far end hides a mechanism (linear spring with a hard
# stop; nonlinear profile; curved tunnel and stick; push-push latch; spring-damper). Each level x {point grasp + wrench, Franka +
# joint torques} x {sequence tool off, on}. The task never mentions the mechanism (plain TASK_CONVERGE). Every run shows a 6 h
# wall budget from the start and, at the $44 spend cap, holds (budget_hold_seconds) for the orchestrator's review: resuming
# raises the cap within the same run, so a continued run is the run it would have been with the higher cap from the start.
TUNNELS = ("tunnel_spring", "tunnel_nonlinear", "tunnel_curved", "tunnel_latch", "tunnel_damper")
TUNNEL_METRICS = ["chamfer", "center_of_mass", "image_mse", "force_profile"]
ENVS.update({level: (TUNNEL_METRICS, RECON) for level in TUNNELS})
TUNNEL = {"wall": 21600, "hold": 129600, "audit": 17179869184, "task": TASK_CONVERGE, "metrics": TUNNEL_METRICS}
RUNS18 = {f"w16-{level[7:]}-{arm}{'-seq' if seq else ''}": (env, f"{label}{', sequence tool' if seq else ''}",
                                                            {**TUNNEL, **extra, **({"tools": ("sequence",)} if seq else {})})
          for level in TUNNELS for arm, env, label, extra in (("grasp", level, "point grasp with wrench", {}),
                                                              ("franka", "panda_small/" + level, "Franka with joint torques", {"vision": True}))
          for seq in (False, True)}
WAVES["wave16-20261109"] = ("Wave 16", "sim2sim-20261109", 44.0, RUNS18)
# Wave 17 (2026-11-10): the wave-16 conditions whose final model had no mechanism (force_profile reads ~0 or only geometry
# artifacts in the plunger range) rerun with a hint that the tunnel hides one; nothing about its kind.
HINT = (" The closed housing hides a mechanism at its far end, out of view, that the visible parts can interact with; your "
        "reconstruction must reproduce its behavior too.")
UNFOUND = ("spring-grasp", "nonlinear-grasp", "nonlinear-grasp-seq", "curved-grasp", "curved-grasp-seq", "latch-grasp-seq",
           "damper-grasp", "damper-grasp-seq")
RUNS19 = {f"w17-{name}-hint": (RUNS18[f"w16-{name}"][0], RUNS18[f"w16-{name}"][1] + ", far-end hint",
                               {**RUNS18[f"w16-{name}"][2], "task": TASK_CONVERGE + HINT}) for name in UNFOUND}
WAVES["wave17-20261110"] = ("Wave 17", "sim2sim-20261110", 44.0, RUNS19)
# Wave 18 (user, 2026-11-13): sealed boxes with hidden contents (experiments/validation/box-scenes.py): a heavy marble, a
# marble of the box's mass, and an axially captive rolling cylinder. Franka with joint torques, movable camera, sequence tool
# (code-written action lists); the task never mentions the contents. Evaluation adds pick-and-throw arm probes with the
# composite object CoM trajectory.
BOXES = ("box_marble_heavy", "box_marble_equal", "box_cylinder")
BOX = {"wall": 21600, "hold": 129600, "audit": 17179869184, "task": TASK_CONVERGE, "vision": True, "movecam": True, "tools": ("sequence",)}
RUNS20 = {f"w18-{level[4:].replace('_', '-')}": ("panda_small/" + level, "Franka with joint torques, movable camera, sequence tool", BOX)
          for level in BOXES}
WAVES["wave18-20261113"] = ("Wave 18", "sim2sim-20261113", 44.0, RUNS20)
# Wave 19 (2026-11-14): the wave-18 marble levels, whose agents never identified a moving body inside (both final models
# scored worse composite-CoM paths than a solid 80 g box), rerun with a hint that the box has contents; nothing about their kind.
BOX_HINT = (" The closed box is not solid: it has hidden contents that can move inside it when the box moves; your reconstruction "
            "must reproduce them too.")
RUNS21 = {f"w19-{name}-hint": (RUNS20[f"w18-{name}"][0], RUNS20[f"w18-{name}"][1] + ", contents hint",
                               {**BOX, "task": TASK_CONVERGE + BOX_HINT}) for name in ("marble-heavy", "marble-equal")}
WAVES["wave19-20261114"] = ("Wave 19", "sim2sim-20261114", 44.0, RUNS21)


def lighting(env):
    """Ground-truth lighting sentence for a prompt (user, 2026-11-16): the scene's exact <light> elements and headlight,
    read from its scene.xml, so the agent need not fit appearance lighting. Nothing about geometry, materials or physics."""
    import xml.etree.ElementTree as ET
    root = ET.parse(f"environments/{env}/scene.xml").getroot()
    lights = " ".join(ET.tostring(light, encoding="unicode").strip() for light in root.iter("light"))
    assert lights, env
    headlight = root.find("visual/headlight")
    head = ("its headlight is " + ET.tostring(headlight, encoding="unicode").strip() if headlight is not None else
            "it sets no <headlight>, so MuJoCo's default headlight applies (ambient 0.1 0.1 0.1, diffuse 0.4 0.4 0.4, specular 0.5 0.5 0.5)")
    return f" The reference scene's lighting is exactly these MJCF lights: {lights}; {head}."


# Wave 20 (user, 2026-11-16). All: point grasp (no robot), RGB + depth, sequence tool (code-written action lists), plain
# TASK_CONVERGE ("find everything that differs"), 6 h visible budget with a 4 h orchestrator checkpoint and the $44 hold.
# (a) tunnel spring / latch with the stick starting outside the bore, in a trough (tunnel-scenes.py `_out`): wrench, fixed
# camera. (c) 1D mass on an ideal spring, frictionless, looks / does not look like one, without / with a damper
# (msd-scenes.py): wrench, movable camera. Lighting ablation: the basic level of each family also runs with the ground-truth
# lighting stated (lighting()), to measure time spent fitting lighting instead of physics.
W20 = {"wall": 21600, "hold": 129600, "checkpoint": 14400, "audit": 17179869184, "task": TASK_CONVERGE, "depth": True,
       "tools": ("sequence",)}
MSD_METRICS = ["chamfer", "center_of_mass", "image_mse", "parameters", "dynamics", "force_profile"]
ENVS.update({f"tunnel_{k}_out": (TUNNEL_METRICS, RECON) for k in ("spring", "latch")})
ENVS.update({f"msd_{k}_{v}": (MSD_METRICS, RECON) for k in ("spring", "damper") for v in ("visible", "hidden")})
RUNS22 = {**{f"w20-{level[7:]}-out": (f"tunnel_{level[7:]}_out", "stick starts outside the bore; point grasp with wrench, RGBD, sequence tool",
                                    {**W20, "metrics": TUNNEL_METRICS}) for level in ("tunnel_spring", "tunnel_latch")},
          **{f"w20-msd-{kind}-{look}": (f"msd_{kind}_{look}", f"1D mass on an ideal spring{' + damper' if kind == 'damper' else ''}, "
                                       f"{'looks like one' if look == 'visible' else 'does not look like one'}; point grasp with wrench, "
                                       "RGBD, movable camera, sequence tool", {**W20, "movecam": True, "metrics": MSD_METRICS})
             for kind in ("spring", "damper") for look in ("visible", "hidden")}}
RUNS22.update({f"{name}-lit": (env, factor + ", ground-truth lighting stated", {**o, "task": o["task"] + lighting(env)})
               for name, (env, factor, o) in list(RUNS22.items()) if name in ("w20-spring-out", "w20-msd-spring-visible")})
# (b) tetrahedral cluster / cube of unknown mass + ideal spring (+ block of stated mass), point grasp WITHOUT force sensing,
# RGBD, movable camera, sequence tool (tetra-spring-scenes.py point-grasp levels; k = 21.7 N/m, block 0.0235 kg). The prompt
# stresses the target's inertial properties; spring levels state k, block levels state the block mass instead.
TETRA_INERTIA = " The inertial properties (mass, centre of mass and inertia) of the tetrahedral cluster are important to identify."
CUBE_INERTIA = " The cube's mass is unknown; its inertial properties (mass, centre of mass and inertia) are important to identify."
IDEAL_K = " The spring between the two plates is an ideal linear spring (Hooke's law only, no damping) with a stiffness of exactly 21.7 N/m."
IDEAL_BLOCK = (" The spring between the two plates is an ideal linear spring (Hooke's law only, no damping). The purple block has a mass "
               "of exactly 0.0235 kg.")
TETRA_METRICS = ["chamfer", "center_of_mass", "density", "image_mse", "dynamics", "expert"]
ENVS.update({f"{t}_spring{b}": (TETRA_METRICS, RECON) for t in ("tetra", "cube") for b in ("", "_block")})
# A plain assignment (not only an update) so the dashboard's _assignments, which runs assignments only, sees these entries.
SCALE_RUNS = {f"w20-{t}-spring{b.replace('_', '-')}": (f"{t}_spring{b}", f"{'tetrahedral cluster' if t == 'tetra' else 'cube of unknown mass'} "
               f"+ ideal spring{' + block of stated mass' if b else ', k stated'}; point grasp without force, RGBD, movable camera, sequence tool",
               {**W20, "movecam": True, "wrench": False, "metrics": TETRA_METRICS, "eval_timeout": 600,
                "task": TASK_CONVERGE + (TETRA_INERTIA if t == "tetra" else CUBE_INERTIA) + (IDEAL_BLOCK if b else IDEAL_K)})
               for t in ("tetra", "cube") for b in ("", "_block")}
RUNS22.update(SCALE_RUNS)
RUNS22.update({f"{name}-lit": (RUNS22[name][0], RUNS22[name][1] + ", ground-truth lighting stated",
                               {**RUNS22[name][2], "task": RUNS22[name][2]["task"] + lighting(RUNS22[name][0])})
               for name in ("w20-tetra-spring", "w20-cube-spring")})
WAVES["wave20-20261116"] = ("Wave 20", "sim2sim-20261116", 44.0, RUNS22)
# Wave 21 (user, 2026-11-16): a palm-up Shadow hand (environments/shadow_hand, experiments/validation/hand-scenes.py) cradles
# one object at reset: a uniform box, the same box with an off-centre COM, a smooth convex mesh, a box with two sharp spikes.
# Touch only (hand_target, joint proprioception and efforts, fingertip and palm taxels; no images, no point grasp), code
# allowed (sequence tool); the task says reset restores the object if it falls out of the hand. 6 h with the 4 h checkpoint.
HAND_LEVELS = ("hand_prism", "hand_prism_offset_com", "hand_smooth_mesh", "hand_spiky")
HAND_RESET = (" The object starts resting in the robot hand. If it falls out of the hand, use reset to restore that world's "
              "initial state.")
HAND = {**W20, "depth": False, "tactile": True, "task": TASK_CONVERGE + HAND_RESET, "metrics": ["dynamics", "chamfer", "center_of_mass"]}
RUNS23 = {f"w21-{level.replace('_', '-')}": ("shadow_hand/" + level, "Shadow hand, proprioception + fingertip/palm tactile, no images, "
                                             "sequence tool", HAND) for level in HAND_LEVELS}
WAVES["wave21-20261116"] = ("Wave 21", "sim2sim-20261116", 44.0, RUNS23)
# Wave 22 (user, 2026-11-16): cloth (experiments/validation/cloth-scenes.py; RoboSimStudio-style triangle-mesh sheet as a
# MuJoCo 2D flex): base square, quadrant stiffnesses (plain, quadrant colors, diagonal colors), irregular with holes and cuts,
# center hole with a stiff or a soft rim. Point grasp with wrench (attaches to cloth vertices), RGBD, movable camera, sequence
# tool. Metric `cloth` (stiffness map, drops over a bar and a cone with side-by-side videos, rest shape) + image_mse.
CLOTH_CONSTRAINTS = ["Model the cloth as a MuJoCo 2D flex (for example <flexcomp dim=\"2\" dof=\"full\"> or explicit vertex bodies with "
                     "three slide joints each) and everything else with standard MuJoCo primitives and joints.", CONSTRAINTS[1]]
CLOTH_LEVELS = ("cloth_base", "cloth_quadrants", "cloth_quadrants_colored", "cloth_quadrants_diagonal_colors", "cloth_holes",
                "cloth_center_hole_stiff_rim", "cloth_center_hole_soft_rim")
CLOTH = {**W20, "movecam": True, "metrics": ["image_mse", "cloth"], "eval_timeout": 600, "constraints": CLOTH_CONSTRAINTS}
ENVS.update({level: (CLOTH["metrics"], "") for level in CLOTH_LEVELS})
RUNS24 = {f"w22-{level.replace('_', '-')}": (level, "cloth; point grasp with wrench, RGBD, movable camera, sequence tool", CLOTH)
          for level in CLOTH_LEVELS}
RUNS24.update({"w22-cloth-base-lit": ("cloth_base", RUNS24["w22-cloth-base"][1] + ", ground-truth lighting stated",
                                      {**CLOTH, "task": CLOTH["task"] + lighting("cloth_base")})})
WAVES["wave22-20261116"] = ("Wave 22", "sim2sim-20261116", 44.0, RUNS24)
# Wave 23 (user, 2026-11-23; planned, not launched): calibrate a spring-powered positioning fixture
# (experiments/validation/fixture-scenes.py): a carriage launched along a rail by a spring, braked by a pad in a closed
# caliper; the Franka slides a compression dial and a pad-pressure dial. Pads: rigid shoe (repeatable), rubber flap on a
# rocking holder (settle-dependent spread, stick-slip), two-faced flap that tucks under or folds back (two stopping ranges).
# The submission is a predictor program (src/sim2sim/predictor.py runs it in isolation) plus chosen settings per target;
# metric `fixture` (CRPS at held-out settings, measured vs predicted hit rates at the chosen settings). Gemini models
# (AGENTS.md): unlimited spend, provider outages held outside the run; 6 h visible wall with the 4 h checkpoint hold.
FIXTURE_LEVELS = ("fixture_rigid", "fixture_rubber", "fixture_bimodal")
FIXTURE_TASK = ("Calibrate the spring-powered positioning fixture on the table. A small carriage rides a rail. reset_carriage cocks "
                "its spring launcher at the compression set on the yellow compression dial and re-seats the brake pad, whose pressing "
                "force is set by the blue pressure dial; release_carriage releases the carriage, which slides along the rail until the "
                "brake stops it. Both dials are sliders on tick scales reading 0 to 10 (the observation's fixture field reports their "
                "exact readings); move them with the robot arm by pushing their pegs along their tracks. The ruler beside the rail "
                "reads the carriage position in mm at the carriage's red index line: zero is the red tick, major ticks every 50 mm and "
                "minor ticks every 10 mm, up to 250 mm. Two coloured bands on the ruler mark target regions: target (green, 70 to "
                "100 mm) and narrow_target (orange, 18 to 22 mm). Characterize the fixture and find dial settings that stop the "
                "carriage inside each target region. Submit a ZIP project whose predictor.py defines predict(compression, pressure, n) "
                "and settings() as the submission's predictor contract specifies; it may use other files in the project, such as your "
                "own MuJoCo model of the fixture. Hidden tests run predict at other dial settings and compare it with repeated releases "
                "there, and release the carriage repeatedly at your chosen settings to count how often it stops in each target region. "
                "Call finish when you believe your predictor matches the fixture.")
FIXTURE_CONSTRAINTS = ["Submit a ZIP project with predictor.py at its root, written in Python with numpy, scipy and mujoco available; "
                       "it may read other local files of the project but no native code.",
                       "Any MuJoCo model you load into the reconstruction world must include the provided Panda robot fixture unchanged."]
# PROVIDER_HOLD (2026-11-24): on 2026-11-23 Google's 503 "high demand" outages used up the 4 h provider-hold allowance
# (14400 s) of runs launched that day, which then died. Held time is outside the run, so the allowance does not change what
# the agent gets; unstarted waves and reruns get 36 h.
PROVIDER_HOLD = 129600
FIXTURE = {"wall": 21600, "hold": 129600, "checkpoint": 14400, "provider_hold": PROVIDER_HOLD, "audit": 17179869184, "task": FIXTURE_TASK,
           "vision": True, "depth": True, "movecam": True, "tools": ("sequence",), "metrics": ["fixture"], "entrypoint": "predictor.py",
           "candidate_changes": False, "eval_timeout": 1800, "extra": "", "constraints": FIXTURE_CONSTRAINTS, "cost": 1e6, "pool_usd": 1e6}
RUNS25 = {f"w23-{level.replace('_', '-')}-{model.removeprefix('gemini-4-')}": (
    "panda_small/" + level, f"positioning fixture, {level[8:]} pad; Franka with joint torques, RGBD, movable camera, sequence tool; {model}",
    {**FIXTURE, "model": model}) for level in FIXTURE_LEVELS for model in ("gemini-4-neon", "gemini-4-hydrogen")}
WAVES["wave23-20261123"] = ("Wave 23", "gemini-20261123", 1e6, RUNS25)

# Wave 24 (user, 2026-11-23; planned, not launched): reconstruct and predict a biased d12 die (experiments/validation/die-scenes.py,
# environments/[panda_small/]die_biased). Generic real-time world (o["realtime"] = frame rate: [simulation] realtime = true,
# frame_rate) and the die's launcher protocols; the submission is a predictor program scored by the `die` metric on fresh
# throws streamed to it in real time. RGB-D, movable camera, sequence tool + controller channel (code acts on the live world).
# Franka with joint efforts, or point grasp without wrench. Control condition, one factor: frame_rate 100 instead of 25
# (more precise observations; throws, protocols, evaluation trials and baselines identical). Gemini neon and hydrogen; 6 h
# with the 4 h checkpoint, provider holds; accounting-only caps (AGENTS.md "Gemini models").
TASK_DIE = ("Investigate the die resting in the tray and build a model that predicts which face ends up uppermost when the launcher "
            "throws it with each of its protocols (task launcher_protocols; your throw action runs them). Submit the model as a "
            "predictor program (task die.predictor): during evaluation it watches new throws through the fixed camera in real time and "
            "must give its confidence allocation across the faces before the deadline after release. The world runs in real time "
            "(task realtime). Resubmit whenever your predictor improves, and call finish when you believe it cannot be improved further.")
DIE_CONSTRAINTS = ["Submit a ZIP whose entry point predict.py follows the predictor protocol in the task (die.predictor); it may use "
                   "the Python standard library, numpy, scipy, mujoco and files you include."]
DIE = {"wall": 21600, "hold": 129600, "checkpoint": 14400, "provider_hold": PROVIDER_HOLD, "audit": 34359738368, "task": TASK_DIE, "vision": True,
       "depth": True, "movecam": True, "wrench": False, "tools": ("sequence", "channel"), "metrics": ["die"], "entrypoint": "predict.py", "extra": "",
       "constraints": DIE_CONSTRAINTS, "cost": 1e6, "pool_usd": 1e6}
ENVS["die_biased"] = (["die"], "")
RUNS26 = {f"w24-die-{arm}-{model.removeprefix('gemini-4-')}{'-fps100' if rate == 100 else ''}":
          (env, f"biased die; {label}; real time at {rate} frames/s{' (control: more precise observations)' if rate == 100 else ''}",
           {**DIE, "realtime": rate, "model": model})
          for model in ("gemini-4-neon", "gemini-4-hydrogen") for rate in (25, 100)
          for arm, env, label in (("grasp", "die_biased", "point grasp without force sensing"), ("franka", "panda_small/die_biased", "Franka with joint efforts"))}
WAVES["wave24-20261123"] = ("Wave 24", "gemini-20261123", 1e6, RUNS26)
# Wave 25 (analyst session's touch-harness wave; user 2026-11-23: "Yes, but on Neon"; the controls also repeat the original hand
# levels on Neon): the four w21 touch-only hand levels with the local reconstruction-world copy (tools.local_controller) and
# pausable bash (no command time limit; code is paused while the model thinks), against the exact w21 config on Neon.
NEON = {"model": "gemini-4-neon", "provider_hold": PROVIDER_HOLD, "cost": 1e6, "pool_usd": 1e6}
# Wave 29 (user, 2026-11-24): the spring-as-a-scale levels (w20 point-grasp tetra/cube + spring [+ block], w11 Panda tetra +
# spring) with one appended sentence asking for the scene's important low-rank structure; the only factor vs the originals.
# Neon baselines (original prompt) where none existed: w20 tetra-spring, cube-spring, tetra-spring-block (cube-spring-block's
# is w26-cube-spring-block-mjcf; w11's is gemini-20261120 w11-tetra-spring-k-neon, whose recipe the w11 run copies: no 4 h checkpoint).
LOWRANK_PROMPT = (" First identify the important low-rank structure of this scene: the few physical properties that govern most of "
                  "its behavior. Estimate those first, and make sure your model gets them right.")
SCALES = (("tetra", ""), ("cube", ""), ("tetra", "-block"), ("cube", "-block"))
RUNS29 = {**{f"w29-{t}-spring{b}-lowrank-prompt": (SCALE_RUNS[f"w20-{t}-spring{b}"][0], f"{t} + spring{b}, w20 harness; low-rank prompt; neon",
                                                   {**SCALE_RUNS[f"w20-{t}-spring{b}"][2], **NEON, "task": SCALE_RUNS[f"w20-{t}-spring{b}"][2]["task"] + LOWRANK_PROMPT})
             for t, b in SCALES},
          **{f"w29-{t}-spring{b}-neon": (SCALE_RUNS[f"w20-{t}-spring{b}"][0], f"{t} + spring{b}, w20 harness; original prompt; neon",
                                         {**SCALE_RUNS[f"w20-{t}-spring{b}"][2], **NEON}) for t, b in SCALES[:3]}}
_env, _factor, _o = WAVES["wave11-20261106"][3]["w11-tetra-spring-k"]
RUNS29["w29-w11-tetra-spring-k-lowrank-prompt"] = (_env, "Panda tetra + spring, w11 harness; low-rank prompt; neon",
    {**{k: v for k, v in _o.items() if k != "checkpoint"}, **NEON, "hold": 129600, "task": _o.get("task", TASK) + LOWRANK_PROMPT})
WAVES["wave29-20261124"] = ("Wave 29", "gemini-20261124", 1e6, RUNS29)
# Second Gemini key (user, 2026-11-24): GEMINI_API_KEY_BYCOST is billed by cost, with a LIFETIME cap of $15000. Every run on it
# uses the one pool "gemini-bycost" (never renamed or reset), so the pool is the lifetime cap; the per-run cap is a safety net.
# Use it for extra capacity when the first key's rate limits would be hit. Never resume a hold above the pool's $15000.
BYCOST = {"key_env": ("GEMINI_API_KEY_BYCOST",), "pool": "gemini-bycost", "pool_usd": 15000.0, "cost": 1000.0}
RUNS27 = {f"w25-{level.replace('_', '-')}{suffix}": ("shadow_hand/" + level, f"Shadow hand, touch only, {label}; gemini-4-neon", {**HAND, **NEON, **o})
          for level in HAND_LEVELS for suffix, label, o in (
              ("", "sequence tool + local controller copy, pausable bash", {"tools": ("sequence", "local_controller"), "pausable": True}),
              ("-control", "sequence tool (w21 harness)", {}))}
WAVES["wave25-20261123"] = ("Wave 25", "gemini-20261123", 1e6, RUNS27)
# Wave 26 (user, 2026-11-23): low-rank identification instead of MuJoCo reconstruction. Cube + ideal spring + block of stated mass,
# point grasp without force sensing (w20-cube-spring-block's harness and observations); the agent submits parameters.json (cube mass,
# spring stiffness, cube COM and inertia), scored by metric `lowrank` (manifest [lowrank]). Twin: the w20 full-MJCF config on Neon.
TASK_LOWRANK = ("Identify the physical parameters of this scene that matter, using the permitted observations and interactions. Instead "
                "of a MuJoCo scene, submit a ZIP containing parameters.json at its root, a JSON object with exactly these keys: \"mass\": "
                "the cube's mass in kg; \"stiffness\": the spring's stiffness in N/m; \"com\": the cube's centre of mass as [x, y, z] "
                "world coordinates in m, in the scene's initial state; \"inertia\": the cube's 3x3 inertia tensor in kg m^2 about its "
                "centre of mass, in world axes, in the scene's initial state. Hidden tests compare these values with the reference "
                "world's. Resubmit whenever your estimates improve, and call finish when you believe they cannot be improved further.")
LOWRANK_CONSTRAINTS = ["Submit a ZIP whose parameters.json holds exactly the keys mass, stiffness, com and inertia, as numbers in SI units."]
CUBE_MJCF = SCALE_RUNS["w20-cube-spring-block"]
RUNS28 = {"w26-cube-spring-block-lowrank": (CUBE_MJCF[0], "cube + ideal spring + block of stated mass; point grasp without force; parameter answer",
                                             {**CUBE_MJCF[2], **NEON, "provider_hold": 14400, "task": TASK_LOWRANK + IDEAL_BLOCK, "metrics": ["lowrank"], "extra": "",
                                              "entrypoint": "parameters.json", "candidate_changes": False, "constraints": LOWRANK_CONSTRAINTS}),
          "w26-cube-spring-block-mjcf": (CUBE_MJCF[0], "cube + ideal spring + block of stated mass; point grasp without force; full MJCF (w20 config)",
                                          {**CUBE_MJCF[2], **NEON, "provider_hold": 14400})}
WAVES["wave26-20261123"] = ("Wave 26", "gemini-20261123", 1e6, RUNS28)
WAVES["wave26-20261124"] = ("Wave 26", "gemini-20261123", 1e6, {name: (env, factor, {**o, "provider_hold": PROVIDER_HOLD})
                                                                for name, (env, factor, o) in RUNS28.items()})
# Wave 27 (user, 2026-11-24; planned, not launched): touch-only tabletop levels (docs/level-designs/touch-tabletop-20261124.md;
# experiments/validation/touch-scenes.py, environments/panda_shadow): the Franka arm with the Shadow hand in place of its gripper
# (robot "panda_shadow": arm_target + hand_target), joint proprioception and efforts plus fingertip/palm taxels, no images, no
# wrench. L1 three abstract objects matched to L2's mug, banana and hammer; L2 unstated vs stated categories; L3 the spiky ball
# in a bin (identification by parameters.json, grab at finish). Each level with the w21 harness (sequence tool) and the w25 one
# (+ local controller copy, pausable bash). Gemini Neon, 4 h wall, no checkpoint hold, provider holds, accounting-only caps.
TOUCH_CONSTRAINTS = ["Model the objects with standard MuJoCo primitives, joints and inline convex meshes (<mesh vertex=\"...\"/>, "
                     "each collided as the convex hull of its vertices); preserve the provided Franka arm and Shadow hand robot "
                     "fixture unchanged.", CONSTRAINTS[1]]
NAME_OBJECTS = " Name each object's body in your scene.xml after what the object is."
OBJECTS_STATED = " The objects are a mug, a banana and a hammer."
SPIKY_TASK = (" One of the balls in the bin is spiky. Find it and report its initial position in every submission: put parameters.json "
              "at the root of the submitted ZIP, beside scene.xml, holding a JSON object whose key \"spiky_ball_xyz\" is [x, y, z], the "
              "spiky ball's initial centre in world coordinates in metres. Only this file is used to judge which ball you identified. "
              "Model the scene in scene.xml as usual. Before you call finish, pick up the spiky ball in the reference world and hold it "
              "in the hand above the bin rim.")
TABLE_METRICS = ["objects", "chamfer", "center_of_mass", "density"]
TOUCH = {"wall": 14400, "hold": 129600, "audit": 17179869184, "depth": False, "tactile": True, "constraints": TOUCH_CONSTRAINTS, **NEON}
RUNS_W27 = {f"w27-{name}{suffix}": ("panda_shadow/" + level, f"Franka + Shadow hand, touch only, {label}; {harness}; gemini-4-neon",
                                  {**TOUCH, "task": TASK_CONVERGE + task, "metrics": metrics, **o})
          for name, level, label, task, metrics in (
              ("shapes", "table_shapes", "three abstract shapes (L1)", NAME_OBJECTS, TABLE_METRICS),
              ("objects-unstated", "table_objects", "mug, banana, hammer unstated (L2)", NAME_OBJECTS, TABLE_METRICS),
              ("objects-stated", "table_objects", "mug, banana, hammer stated (L2)", NAME_OBJECTS + OBJECTS_STATED, TABLE_METRICS),
              ("bin-spiky", "bin_spiky", "spiky ball in a bin (L3)", SPIKY_TASK, ["objects", "identification"]))
          for suffix, harness, o in (("", "sequence tool (w21 harness)", {"tools": ("sequence",)}),
                                     ("-touch", "sequence tool + local controller copy, pausable bash (w25 harness)",
                                      {"tools": ("sequence", "local_controller"), "pausable": True}))}
WAVES["wave27-20261124"] = ("Wave 27", "gemini-20261124", 1e6, RUNS_W27)


def q(value):
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


# Google's native API reports no cost; the host prices tokens at these list prices (USD per 1M tokens: input, cached input,
# output incl. thinking), from https://aistudio.google.com/docs/pricing (Standard tier), read 2026-11-19.
GEMINI_PRICES = {"gemini-4-neon": (3.0, 0.15, 15.0), "gemini-4-hydrogen": (1.5, 0.075, 7.5),
                 "gemini-4-carbon": (3.0, 0.15, 15.0)}  # Carbon: Neon's prices as a placeholder (accounting only; spend is unlimited)


# Self-hosted open-weights models (api local_openai, zero cost): the private record of the researcher's GPU service.
# Their raw reasoning is no provider summary, so their route has no reasoning_summary line (the broker refuses one).
LOCAL_SERVICES = {"inkling-2-mini": "/n/netscratch/qianqian_lab/Lab/aejxie/inkling/service/service.json"}  # scripts/inkling-service.slurm


def route(model, key_env="GEMINI_API_KEY"):
    """[inference] lines (before and after the reasoning effort) that select the model; the rest is model-independent."""
    summary = 'reasoning_summary = "auto"\n'
    if model == "gpt-6-astra":
        return ('endpoint = "https://api.aqinference.com/v1/responses"\nprovider = "openai"\nmodel = "gpt-6-astra"\n',
                summary + 'metadata = { project = "amish-research", feature = "sim2sim-agent-eval" }\napi_key_env = "INFERENCE_API_KEY"\n')
    if model in LOCAL_SERVICES:
        return f'api = "local_openai"\nendpoint = "{LOCAL_SERVICES[model]}"\nmodel = "{model}"\n', ""
    price = GEMINI_PRICES[model]
    return (f'api = "gemini"\nendpoint = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"\nmodel = "{model}"\n',
            summary + f'api_key_env = "{key_env}"\nprice_usd_per_mtok = {{ input = {price[0]}, cached_input = {price[1]}, output = {price[2]} }}\n')


ROBOT_FAMILIES = {"panda": "panda", "panda_small": "panda", "shadow_hand": "shadow_hand", "panda_shadow": "panda_shadow"}


def config(wave, pool, cap, env, factor, o):
    robot = ROBOT_FAMILIES.get(env.split("/")[0])
    panda = robot == "panda"
    hand = robot in ("shadow_hand", "panda_shadow")
    hinged = panda and "hinges" in ENVS.get(env.split("/")[1], ([], ""))[0]
    metrics, extra = ((["dynamics", "hinges", "image_mse"], ENVS[env.split("/")[1]][1]) if hinged else
                      (["dynamics", "chamfer", "center_of_mass", "image_mse"], RECON)) if panda else (["dynamics", "chamfer", "center_of_mass"], RECON) if hand else ENVS[env]
    metrics, extra = o.get("metrics", metrics), o.get("extra", extra)
    source_dir, entry = ("environments/" + env.split("/")[0], env.split("/", 1)[1] + "/") if panda or hand else ("environments/" + env, "")
    vision = o.get("vision", not panda and not hand)
    wrench = o.get("wrench", vision and not panda)
    return f"""# Generated by generate.py — researcher-only; never mount in the agent image.
evaluate_candidate_changes = {str(o.get("candidate_changes", True)).lower()}
record_video = false
plot_checkpoints = false
task = {q(o.get("task", TASK))}

[experiment]
name = {q(f"{wave} · {env} · {factor}")}
environment = {q(env.replace("/", "_"))}
execution_mode = "strict_oci"
model = {q(o.get("model", "gpt-6-astra"))}
model_constraints = [{", ".join(q(c) for c in o.get("constraints", PANDA_CONSTRAINTS if panda else HAND_CONSTRAINTS if hand else CONSTRAINTS))}]
evaluation_metrics = [{", ".join(q(m) for m in metrics)}]

[experiment.physical_parameters]
scene = "Original authored MJCF physical parameters of {env}."
factor = {q(factor)}
{'''
[perturbation]
seed = 2026110208
''' + "".join(f'{k} = "{o[k]}"' + chr(10) for k in ("colors", "physics", "pose") if k in o) if {"colors", "physics", "pose"} & set(o) else ""}
[source]
scene = "{source_dir}/{entry}scene.xml"

[sandbox]
image = "localhost/sim2sim-agent:latest"
evaluator_image = "localhost/sim2sim:latest"
command = ["/usr/local/bin/python", "/opt/agent/sandbox_agent.py"{', "--bash-mode", "pausable"' if o.get("pausable") else ""}]
process_limit_backend = "slurm_step"
memory = "8g"
cpus = 4
pids = 256
workspace = "1g"

[sandbox.worker]
memory = "4g"
cpus = 2
pids = 128

[budget]
wall_only = true
wall_seconds = {o.get("wall", 14400)}
submission_bytes = 8388608
max_message_bytes = 16777216
audit_bytes = {o.get("audit", 8589934592)}
event_log_bytes = 67108864

[simulation]
{f'robot = "{robot}"' + chr(10) if robot else ""}rgb = {str(vision).lower()}
depth = {str(o.get("depth", False)).lower()}
calibration = {str(vision).lower()}
wrench = {str(wrench).lower()}
{f"realtime = true{chr(10)}frame_rate = {o['realtime']}{chr(10)}" if o.get("realtime") else ""}{"tactile = true" + chr(10) if o.get("tactile") else ""}{"effort = false" + chr(10) if o.get("effort") is False else ""}{"movable_camera = true" + chr(10) if o.get("movecam") else ""}width = 320
height = 240
camera = "fixed"
{'wrist_camera = "wrist"' + chr(10) if o.get("wrist") else ""}{'reference_interaction = false' + chr(10) if o.get("noaction") else ""}{'resets = false' + chr(10) if o.get("noreset") else ""}default_ticks = 20
{f"grasp_slip_m = {o['slip']}{chr(10)}" if o.get("slip") else ""}{'grasp = "weld"' + chr(10) if o.get("weld") else ""}{f"grasp_slip_rad = {o['slip_rad']}{chr(10)}" if o.get("slip_rad") else ""}
[evaluation]
manifest = "{entry}manifest.toml"
source_dir = "{source_dir}"
source_entrypoint = "{entry}scene.xml"
entrypoint = "{o.get("entrypoint", "scene.xml")}"
camera = "fixed"
width = 320
height = 240
timeout_seconds = {o.get("eval_timeout", 180)}
{extra}
{(chr(10) + "[tools]" + chr(10) + "".join(f"{k} = true" + chr(10) for k in o["tools"])) if o.get("tools") else ""}[inference]
{route(o.get("model", "gpt-6-astra"), *o.get("key_env", ()))[0]}reasoning_effort = "medium"
{route(o.get("model", "gpt-6-astra"), *o.get("key_env", ()))[1]}max_calls = 2000
max_cost_usd = {o.get("cost", cap)}
cost_pool = "{o.get("pool", pool)}"
cost_pool_usd = {o.get("pool_usd", 500.0 if pool <= "sim2sim-20261104" or pool == "sim2sim-500" else 700.0 if pool <= "sim2sim-20261105" else 2000.0 if pool in ("sim2sim-20261116", "sim2sim-20261117") else 1000.0)}
cost_ledger_dir = "experiments/cost-ledger"
call_reserve_usd = 2.0
max_output_tokens = 16384
max_request_bytes = 12582912
max_response_bytes = 1048576
timeout_seconds = {o.get("call_timeout", 300)}
{f"budget_hold_seconds = {o['hold']}" + chr(10) if o.get("hold") else ""}{f"checkpoint_wall_seconds = {o['checkpoint']}" + chr(10) if o.get("checkpoint") else ""}{f"provider_hold_seconds = {o['provider_hold']}" + chr(10) if o.get("provider_hold") else ""}"""


for directory, (wave, pool, cap, runs) in WAVES.items():
    out = Path(__file__).parent / directory
    out.mkdir(exist_ok=True)
    for run_id, (env, factor, overrides) in runs.items():
        (out / f"{run_id}.toml").write_text(config(wave, pool, cap, env, factor, overrides))
    print(directory, sorted(p.name for p in out.glob("*.toml")))

# Replicates (2026-11-04, user: up to n=3 per condition for the important findings). Each replicate config equals its
# original except the cost-pool lines; run IDs are <original>-rN-<pool date>-001. Same prepared source (prepare is seeded).
FORCE = [("wave8-20261104", "w8-lever-hard-base"), ("wave7-20261104", "w7-lever-hard-nowrench"),
         ("wave6-20261104", "w6-spring-hard-base"), ("wave6-20261104", "w6-spring-hard-nowrench"),
         ("wave7-20261104", "w7-blocks-hard-base"), ("wave6-20261104", "w6-blocks-hard-nowrench")]
ANCHOR = [("wave6-20261104", "w6-spring-hard-nowrench-mass"), ("wave8-20261104", "w8-lever-hard-nowrench-hint"),
          ("wave9-20261104", "w9-lever-hard-nowrench-mass-hint"), ("wave9-20261104", "w9-lever-hard-noaction-mass-hint")]
CAMERA = [("wave11-20261106", "w11-tetra-spring-k"), ("wave11-20261106", "w11-tetra-spring-block"),
          ("wave13-20261106", "w13-tetra-spring-k-fixedcam"), ("wave13-20261106", "w13-tetra-spring-block-fixedcam")]
REPLICATES = {"rep-20261104": ("sim2sim-20261104", 2, FORCE[:3]), "rep-20261105": ("sim2sim-20261105", 2, FORCE[3:] + ANCHOR),
              "rep-20261106": ("sim2sim-20261106", 3, FORCE + ANCHOR),
              "rep-20261107": ("sim2sim-20261107", 2, CAMERA), "rep3-20261107": ("sim2sim-20261107", 3, CAMERA)}
# 2026-11-16 (user: use the $2000 day fully, with repeats where needed): second replicates of the lighting-ablation pairs,
# the box levels whose agents found the contents, the cloth color-cue levels and the touch-only hand levels.
LIGHT = [("wave20-20261116", f"w20-{n}{s}") for n in ("spring-out", "tetra-spring", "cube-spring", "msd-spring-visible") for s in ("", "-lit")]
LIGHT += [("wave22-20261116", f"w22-cloth-base{s}") for s in ("", "-lit")]
FOUND = [("wave19-20261114", "w19-marble-heavy-hint"), ("wave19-20261114", "w19-marble-equal-hint"), ("wave18-20261113", "w18-cylinder")]
CUES = [("wave22-20261116", f"w22-cloth-quadrants{s}") for s in ("", "-colored", "-diagonal-colors")]
TOUCH = [("wave21-20261116", f"w21-{level.replace('_', '-')}") for level in HAND_LEVELS]
REPLICATES["rep-20261116"] = ("sim2sim-20261116", 2, LIGHT + FOUND + CUES + TOUCH)
# 2026-11-17 ($200/ET-day hard cap): 12 of the 2026-11-16 replicates hung at their first candidate-change evaluation on the
# two nodes that each ran three strict jobs (six runs); redo the tetra and cube lighting pairs first, two jobs only.
REPLICATES["rep-20261117"] = ("sim2sim-20261117", 2, [("wave20-20261116", f"w20-{n}{s}") for n in ("tetra-spring", "cube-spring") for s in ("", "-lit")]
                                + [("wave21-20261116", f"w21-{n}") for n in ("hand-prism", "hand-prism-offset-com")])  # cheap (~$10/6 h): fill the day
# 2026-11-18 ($200/day; the hand pair above still draws on pool sim2sim-20261117's last $39, so this pool is capped at $161):
# complete the spring-out and msd lighting pairs and redo the hinted heavy-marble box (all lost to the node-packing hang).
REPLICATES["rep-20261118"] = ("sim2sim-20261118", 2, [("wave20-20261116", "w20-spring-out"), ("wave20-20261116", "w20-msd-spring-visible-lit"),
                                                     ("wave19-20261114", "w19-marble-heavy-hint"), ("wave19-20261114", "w19-marble-equal-hint"),
                                                     ("wave21-20261116", "w21-hand-smooth-mesh")])  # last two fill the day under the same $161 pool
POOL_USD = {"sim2sim-20261118": 161.0}
for directory, (pool, n, originals) in REPLICATES.items():
    out = Path(__file__).parent / directory
    out.mkdir(exist_ok=True)
    for wave_dir, run_id in originals:
        wave, _, cap, runs = WAVES[wave_dir]
        env, factor, overrides = runs[run_id]
        (out / f"{run_id}-r{n}.toml").write_text(config(wave, pool, cap, env, factor, {**overrides, "pool_usd": 700.0 if pool <= "sim2sim-20261106" else 2000.0 if pool == "sim2sim-20261116" else POOL_USD.get(pool, 200.0) if pool >= "sim2sim-20261117" else 1000.0}))
    print(directory, sorted(p.name for p in out.glob("*.toml")))

# 2026-11-19 (user): Gemini 4 Neon and Hydrogen against the existing gpt-6-astra runs on five hard levels (cloth identification,
# Panda spring weighing, touch-only hand shape, hidden box contents, latch force profile). Each config equals its astra original
# except the model/inference route and the cost lines, so harness, task, observations, tools, wall budget and output-token
# limit match. No 4 h checkpoint hold: each run gets its original's full wall budget. Per-run caps are a safety net at roughly
# astra's largest extended spend ($120) at each model's price; the pool holds this batch to $200.
HARD = [("wave22-20261116", "w22-cloth-holes"), ("wave11-20261106", "w11-tetra-spring-k"), ("wave21-20261116", "w21-hand-smooth-mesh"),
        ("wave19-20261114", "w19-marble-heavy-hint"), ("wave20-20261116", "w20-latch-out")]
# 2026-11-20 (user): Gemini spend is unlimited (key limits 200 RPM, 2M TPM, 100K RPD); compare fairly anyway (AGENTS.md).
# Google's latency spiked (2 s to > 290 s for one tiny request) and killed hand-hydrogen-001 after four timeouts, so batch
# gemini-20261120 adds provider_hold_seconds: a provider outage's failed attempts and retry backoff are held outside the run
# (containers paused, clocks stopped); successful calls' latency counts as astra's did. Caps are accounting-only.
for directory, pool, extra in (("gemini-20261119", "gemini-20261119", {}), ("gemini-20261120", "gemini-20261120", {"provider_hold": 14400})):
    out = Path(__file__).parent / directory
    out.mkdir(exist_ok=True)
    for model, cap in ((("gemini-4-neon", 40.0), ("gemini-4-hydrogen", 20.0)) if not extra else (("gemini-4-neon", 1e6), ("gemini-4-hydrogen", 1e6))):
        for wave_dir, run_id in HARD:
            wave, _, _, runs = WAVES[wave_dir]
            env, factor, overrides = runs[run_id]
            overrides = {k: v for k, v in overrides.items() if k != "checkpoint"} | {"model": model, "cost": cap, "pool_usd": 200.0 if not extra else 1e6,
                                                                                   "hold": 129600, **extra}
            (out / f"{run_id}-{model.removeprefix('gemini-4-')}.toml").write_text(config(wave, pool, cap, env, factor, overrides))
    print(out.name, sorted(p.name for p in out.glob("*.toml")))
# 2026-11-23 (user: Neon is the default; evaluate Carbon on the same five hard levels in parallel): the gemini-20261120 recipe
# (provider holds, accounting-only caps) with gemini-4-carbon.
# gemini-carbon-20261124: the same with the 36 h provider-hold allowance, for reruns of runs that outages killed.
# inkling-20261124 (user, 2026-11-24): Thinking Machines' open-weights Inkling 2 Mini, self-hosted at zero cost (api local_openai,
# scripts/inkling-service.slurm on a Kempner A100), with the gemini-carbon-20261124 recipe; its cost lines are accounting-only
# (the broker charges $0) and its own pool counts calls and tokens. Its per-call timeout is 1800 s instead of 300 s: on the
# A100 service one stream decodes ~26 tokens/s (~14 with five runs), so a valid 16384-token reply needs up to ~20 min. Successful
# calls' latency still counts as run time, and a timed-out attempt is a held provider outage, so this grants no extra budget.
for directory, model, pool, allowance in (("gemini-carbon-20261123", "gemini-4-carbon", "gemini-carbon-20261123", 14400),
                                          ("gemini-carbon-20261124", "gemini-4-carbon", "gemini-carbon-20261123", PROVIDER_HOLD),
                                          ("inkling-20261124", "inkling-2-mini", "inkling-20261124", PROVIDER_HOLD)):
    out = Path(__file__).parent / directory
    out.mkdir(exist_ok=True)
    for wave_dir, run_id in HARD:
        wave, _, _, runs = WAVES[wave_dir]
        env, factor, overrides = runs[run_id]
        overrides = {k: v for k, v in overrides.items() if k != "checkpoint"} | {"model": model, "cost": 1e6, "pool_usd": 1e6,
                                                                               "hold": 129600, "provider_hold": allowance,
                                                                               **({"call_timeout": 1800} if model in LOCAL_SERVICES else {})}
        (out / f"{run_id}-{model.removeprefix('gemini-4-').removesuffix('-2-mini')}.toml").write_text(config(wave, pool, 1e6, env, factor, overrides))
    print(out.name, sorted(p.name for p in out.glob("*.toml")))

# Wave 28 (user-approved design docs/level-designs/gear-train-depth-20261124.md; planned): explanation depth on the gear train
# (experiments/validation/gear-scenes.py, environments/gear_train). A neutral base task (not TASK_CONVERGE); the only factor is
# one appended sentence stating the model's downstream use. Point grasp with wrench, RGB-D, movable camera, sequence tool,
# lock-step, reference interaction, no hints or perturbation; gemini-4-neon, n = 1, 4 h wall, no checkpoint hold. Metric
# gear_train (src/sim2sim/gear_train.py): structure/depth class, sim_cost, probe accuracy, G-cheap and G-deep scores.
TASK_GEAR = ("Build a MuJoCo model of this mechanism. Name the crank body `crank` and the dial hand body `hand`. Submit it as a "
             "self-contained scene.xml project, using the permitted observations and interactions; resubmit whenever your model "
             "improves, and call finish when you are done.")
GEAR_USES = {"recon": "",
             "cheap": (" Your model will be used by a planner that runs many thousands of simulations of how the hand moves when the "
                       "crank turns; it is scored on prediction accuracy and simulation speed."),
             "deep": (" Your model will be used to predict how the crank and hand move after individual gears are taken out of the "
                      "train; we remove gears from your model by their axle position, and it is scored on those predictions.")}
ENVS["gear_train"] = (["gear_train"], "")
# The level's public weld grasp and slips (simulation.grasp = "weld", grasp_slip_m and grasp_slip_rad = gear-scenes.py GRASP;
# gear-check.py asserts they agree; user 2026-11-25: new point-grasp levels use the weld): an over-pull or over-twist would
# otherwise strain the grasp until MuJoCo's inertia-scaled constraint softness tells the gear train from its lumped equivalent.
GEAR_W28 = {**NEON, "wall": 14400, "hold": 129600, "audit": 17179869184, "vision": True, "depth": True, "movecam": True, "tools": ("sequence",),
            "metrics": ["gear_train"], "eval_timeout": 600, "pool": "gemini-20261124", "weld": True, "slip": 0.002, "slip_rad": 0.05}
RUNS_W28 = {f"w28-gear-{name}": ("gear_train", f"gear train, stated use: {name}; weld grasp with wrench, RGBD, movable camera, sequence tool; "
                                               "gemini-4-neon", {**GEAR_W28, "task": TASK_GEAR + use})
          for name, use in GEAR_USES.items()}
WAVES["wave28-20261124"] = ("Wave 28", "gemini-20261124", 1e6, RUNS_W28)
out = Path(__file__).parent / "wave28-20261124"
out.mkdir(exist_ok=True)
for run_id, (env, factor, overrides) in RUNS_W28.items():
    (out / f"{run_id}.toml").write_text(config("Wave 28", "gemini-20261124", 1e6, env, factor, overrides))
print(out.name, sorted(p.name for p in out.glob("*.toml")))

# Wave 31 (user, 2026-11-25, top priority): resets vs no resets. Levels whose completed Neon runs reset often (gemini-20261120:
# latch-out 27, hand smooth-mesh 21, cloth-holes 17, tetra-spring-k 11, marble-heavy-hint 9; w26 cube-spring-block mjcf 24).
# Each no-reset config is its Neon baseline's config with [simulation] resets = false (option noreset) (the reset action is not offered) and,
# for the hand, the task's reset sentence removed; the provider-hold allowance is the current 36 h (held time is off the
# run clock). Baselines: those Neon runs, which kept resets.
out = Path(__file__).parent / "wave31-20261125"
out.mkdir(exist_ok=True)
for wave_dir, run_id in HARD:
    wave, _, _, runs = WAVES[wave_dir]
    env, factor, overrides = runs[run_id]
    overrides = {k: v for k, v in overrides.items() if k != "checkpoint"} | {"model": "gemini-4-neon", "cost": 1e6, "pool_usd": 1e6,
                                                                           "hold": 129600, "provider_hold": PROVIDER_HOLD, "noreset": True}
    overrides["task"] = overrides.get("task", TASK).replace(HAND_RESET, " The object starts resting in the robot hand.")
    (out / f"w31-{run_id[4:]}-noreset.toml").write_text(config("Wave 31", "gemini-20261125", 1e6, env, factor + "; no resets; neon", overrides))
env, factor, overrides = RUNS28["w26-cube-spring-block-mjcf"]
(out / "w31-cube-spring-block-mjcf-noreset.toml").write_text(config("Wave 31", "gemini-20261125", 1e6, env, factor + "; no resets; neon",
                                                                    {**overrides, "provider_hold": PROVIDER_HOLD, "noreset": True}))
print(out.name, sorted(p.name for p in out.glob("*.toml")))
