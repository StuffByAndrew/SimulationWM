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


def q(value):
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def config(wave, pool, cap, env, factor, o):
    panda = env.split("/")[0] in ("panda", "panda_small")
    hinged = panda and "hinges" in ENVS.get(env.split("/")[1], ([], ""))[0]
    metrics, extra = ((["dynamics", "hinges", "image_mse"], ENVS[env.split("/")[1]][1]) if hinged else
                      (["dynamics", "chamfer", "center_of_mass", "image_mse"], RECON)) if panda else ENVS[env]
    metrics = o.get("metrics", metrics)
    source_dir, entry = ("environments/" + env.split("/")[0], env.split("/", 1)[1] + "/") if panda else ("environments/" + env, "")
    vision = o.get("vision", not panda)
    wrench = o.get("wrench", vision and not panda)
    return f"""# Generated by generate.py — researcher-only; never mount in the agent image.
evaluate_candidate_changes = true
record_video = false
plot_checkpoints = false
task = {q(o.get("task", TASK))}

[experiment]
name = {q(f"{wave} · {env} · {factor}")}
environment = {q(env.replace("/", "_"))}
execution_mode = "strict_oci"
model = "gpt-6-astra"
model_constraints = [{", ".join(q(c) for c in (PANDA_CONSTRAINTS if panda else CONSTRAINTS))}]
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
image = "localhost/sim2sim:latest"
evaluator_image = "localhost/sim2sim:latest"
command = ["/usr/local/bin/python", "/opt/agent/sandbox_agent.py"]
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
{'robot = "panda"' + chr(10) if panda else ""}rgb = {str(vision).lower()}
depth = {str(o.get("depth", False)).lower()}
calibration = {str(vision).lower()}
wrench = {str(wrench).lower()}
{"tactile = true" + chr(10) if o.get("tactile") else ""}{"effort = false" + chr(10) if o.get("effort") is False else ""}{"movable_camera = true" + chr(10) if o.get("movecam") else ""}width = 320
height = 240
camera = "fixed"
{'wrist_camera = "wrist"' + chr(10) if o.get("wrist") else ""}{'reference_interaction = false' + chr(10) if o.get("noaction") else ""}default_ticks = 20

[evaluation]
manifest = "{entry}manifest.toml"
source_dir = "{source_dir}"
source_entrypoint = "{entry}scene.xml"
entrypoint = "scene.xml"
camera = "fixed"
width = 320
height = 240
timeout_seconds = 180
{extra}
{(chr(10) + "[tools]" + chr(10) + "".join(f"{k} = true" + chr(10) for k in o["tools"])) if o.get("tools") else ""}[inference]
endpoint = "https://api.aqinference.com/v1/responses"
provider = "openai"
model = "gpt-6-astra"
reasoning_effort = "medium"
reasoning_summary = "auto"
metadata = {{ project = "amish-research", feature = "sim2sim-agent-eval" }}
api_key_env = "INFERENCE_API_KEY"
max_calls = 2000
max_cost_usd = {o.get("cost", cap)}
cost_pool = "{pool}"
cost_pool_usd = {o.get("pool_usd", 500.0 if pool <= "sim2sim-20261104" or pool == "sim2sim-500" else 700.0 if pool <= "sim2sim-20261105" else 1000.0)}
cost_ledger_dir = "experiments/cost-ledger"
call_reserve_usd = 2.0
max_output_tokens = 16384
max_request_bytes = 12582912
max_response_bytes = 1048576
timeout_seconds = 300
{f"budget_hold_seconds = {o['hold']}" + chr(10) if o.get("hold") else ""}"""


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
for directory, (pool, n, originals) in REPLICATES.items():
    out = Path(__file__).parent / directory
    out.mkdir(exist_ok=True)
    for wave_dir, run_id in originals:
        wave, _, cap, runs = WAVES[wave_dir]
        env, factor, overrides = runs[run_id]
        (out / f"{run_id}-r{n}.toml").write_text(config(wave, pool, cap, env, factor, {**overrides, "pool_usd": 700.0 if pool <= "sim2sim-20261106" else 1000.0}))
    print(directory, sorted(p.name for p in out.glob("*.toml")))
