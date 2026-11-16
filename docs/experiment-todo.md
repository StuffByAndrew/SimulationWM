# Experiment TODO (handoff)

What remains to run or build, so the next agent can continue. Dated log of what has
run: [experiment-log.md](experiment-log.md). Current plan and pre-registered
hypotheses: [experiments/plan-20261102.md](../experiments/plan-20261102.md).
Keep this list current: strike items when a log entry covers them.

## Standing constraints
- Strict OCI sandbox only for comparable results. Development-mode runs are not
  comparable: no filesystem boundary and a different harness.
- Exactly `gpt-6-astra`, medium reasoning.
- One run per condition, except replicates to n=3 for the key findings (user, 2026-11-05).
- Slurm `test` or `sapphire` (CPU) only, never GPU partitions. Pack 2–3 runs per `scripts/experiment-strict.slurm` job.
- Cost: one pool per ET day, $1000 since 2026-11-06 (user; $700 from 2026-11-04, $500 before) (`cost_pool = "sim2sim-YYYYMMDD"`), and per-run caps (usually $44). Agents never see
  spend. The orchestrator picks and launches the next wave without asking, within the day's pool.
- Only the declared factor may differ between paired runs. Since wave 6 the task is `TASK_CONVERGE` (finish when nothing
  differs); compare new-prompt runs only with new-prompt baselines.
- Run on home (hot storage, `--root experiments`); keep ≥10 GB free.

## Wave 16 tunnel levels (user, 2026-11-08): standing review rules
- [ ] When the runs end or hold, review each automatically (audit progress notes, submitted models, `force_profile`):
  1. If an agent never probes or discovers that something is at the far end of the tunnel, rerun that condition with a
     hint in the task that the tunnel hides a mechanism (the original prompt never mentions it). The user's message read
     "do try to or discover"; it is interpreted as "do not", since hinting runs that already found it adds nothing.
  2. If runs reach their spend cap (budget hold) while still working on identifying the mechanism, resume all held wave-16
     runs with a raised cap, up to the 6 h wall budget they had from the start (`hold-decision.json`). Stop runs that are no
     longer working on it. Keep the $1000/ET-day rule for money spent after resuming.
  3. For reports, plot each run's ground-truth vs reconstructed force-position profile
     (`experiments/validation/plot-force-profiles.py`), the latch included (loading and unloading strokes are separate).
- The original task must stay complete: the agent should finish only when nothing differs between its model and the world.
- Status 2026-11-10:
  - [x] Holds reviewed: 7 resumed, 7 stopped (see the log).
  - [x] latch-franka and curved-franka finished at 6 h.
  - [x] In-run `force_profile` missed most candidates and was replaced by a pixel grasp:
    - agents build in shifted frames;
    - Franka candidates that include the robot first hid their bodies from `graspable_bodies`, fixed in `simulation.py`.
  - [x] Rescored and plotted waves 16–17 (206 submissions, job `59840356`).
  - [ ] Rebuild the image before any further tunnel wave: the probe changed again after image `b986371a2e38`.
  - [x] Wave 17 hinted reruns done; all 8 found the mechanism.
  - [x] Review and rescore `w17-nonlinear-grasp-hint-20261110-002` (job `59831896`; watcher running), which replaces the -001 run that was cut short.
  - [ ] Write the tunnel results report (by ablation: hint, sequence tool, Franka vs point grasp; with force-profile plots).
  - [ ] Scene fixes for any later tunnel wave:
    - the plunger hard stop is soft (an agent drove it to 39–60 mm against a 30 mm stroke; curved to 0.158 rad against 0.12);
    - a strong pull drags the "captive" collar out through the lip.

## Waiting on the user (do not start without an answer)
- [x] ~~Spring levels review~~ Launched 2026-11-06 23:20 UTC (wave 11, four levels, movable camera).
- [ ] **Lab-storage copy.** Copy finished runs from home to lab only after the user states lab quota headroom (details
  under "Comparisons at equal budget").
- [ ] **Results report for waves 7–9.** The 2026-11-04 report covers wave 6 and earlier. Rebuild with
  `experiments/validation/build-results-report.py` plus a new spec file when the user wants it.
- [x] ~~SAM-3D weights~~ Fetched and lock-pinned 2026-11-06; wave 12 (boat undirected/directed, assembly directed) launched.
- [ ] **Netscratch → lab:** replicates live on netscratch, which purges after 90 days. Copy them to lab after quota headroom is confirmed.
- [ ] Retro-score earlier final submissions with `friction`/`expert`/`policy_transfer` (needs a metrics override in
  `candidate_history` or `baselines`-style tool). This would show whether scale errors (lever ×0.33) matter downstream.

## Config-only ablations (no new infrastructure)
- [x] `movecam` (movable observation camera, 2026-11-06) vs fixed camera: run on a geometry scene (boat or assembly)
  and on a spring level, now that the option exists. Spring levels: wave 13 done (2026-11-07). Fixed camera was worse in
  both: Chamfer 0.5→3.6 and 0.7→1.3 mm, block-level stiffness ×1.01→×1.38. Done at n=3 (2026-11-07): movable camera better geometry 5/6 pairs and reliable
  block-level stiffness (3/3 within 15% vs 0/3); cluster COM unaffected. Boat/assembly:
  no detectable effect (n=1, within noise).
- [ ] Franka load sensing without effort: the position controller's joint sag acts as a scale (w11 agents used it). To
  ablate force information fully, also hide or quantize joint positions under load, or raise the controller gains.
  Needs a user decision before any change.
- [x] ~~Wrench (force-torque at the grasp) off vs on~~ Done: lever, spring, blocks, assembly, tetra (waves 2–3, 6–8).
  - damped_lever (hinge damping and friction loss)
  - random_assembly and blocks (COM)
- [ ] Prompt: "match appearance from all angles": random_assembly done (w2); primitive_boat still open
  (Chamfer, image_mse).
- [ ] Prompt: explicitly name the hinge parameters. Done on damped_lever (w1, w2, w3, w8); spring still open.
- [x] ~~Depth vs RGB~~ Done: random_assembly (w1), primitive_boat (w3, no gain).
- [x] ~~Tetrahedral cluster, colored vs `[perturbation] colors = "uniform_gray"`~~ Done (w2)
  (tests spatial memory).
- [x] ~~Panda arm vs ideal point grasp~~ Done on the hard lever (w3).
- [x] ~~Multiview: Panda `wrist_camera` plus fixed camera, vs fixed only~~ Done on panda_small assembly (w3).
- [ ] Vision vs proprio-only on more scenes. Every blind run must use
  `colors = "random"` canaries, and colors must be excluded from blind scoring.
- [x] ~~primitive_boat sandboxed baseline~~ Done (w3).

- [x] ~~Saturation~~ Hard variants used from wave 2 on: wave 1 recovered round-valued blocks/spring/lever exactly. Use `physics = "nonround"` hard
  variants for further contrasts on those scenes.
- [x] ~~Bug: `environments/panda/tetrahedral_cluster` starts with the Panda hand ~50 mm inside the top sphere at
  reset.~~ Fixed 2026-11-03 (cluster +0.2 m in y; Slurm 57300325: ≤1 mm reset penetration, nonround prepares).
- [x] ~~Wave 3 candidates~~ Run except boat all-angles: boat base vs all-angles, assembly without wrench, multiview (`panda_small` fixed vs
  fixed + wrist), boat with depth.

- [x] ~~Metric bug: per-primitive `density` reads geom densities only.~~ Fixed 2026-11-03: such densities are
  now gaps with `unavailable_reason` (explicit inertial, massless geom, visual-only geom with massless collision
  twin). Still open: deriving per-primitive masses from COM and total mass, and retro-scoring wave-1/2 tetra. A candidate that models a rigid
  cluster with a body `<inertial>` and massless collision geoms scores ~540% error even when correct.
  `w1-tetra-base-20261102-003` final: mass 15.007 vs 15 kg, COM error 0.3 mm. Mark density *unavailable*
  in that case, and derive per-primitive masses from COM and total mass where they are identifiable.
  This needs a trusted-image rebuild, because evaluation runs in the image. Retro-score wave-1 and
  wave-2 tetra submissions with `center_of_mass`.

- [x] ~~Canary colors: draw them far from common palette colors (red, blue, green, yellow, orange, gray),
  not only from the source. Score blind runs per object, matched by geometry, instead of the
  nearest color anywhere in the scene (`wave1-summary.py`).~~ Done 2026-11-03: `perturbation.PALETTE`
  (0.35 RGB) and geometry-matched per-object canary distances in `experiments/validation/run-summary.py`.

- [ ] Information-limited or hidden-physics scenes. Simulation-based fitting recovers any parameter whose
  signal is observable, so harder benchmarks must limit the signal (no wrench, occlusion, proprio-only on
  hinges) or add hidden dynamics.
- [ ] Assembly is noisy at n=1: across three baseline runs, final COM error ranges 0.004–0.028 m (7×) and
  Chamfer 0.014–0.019 m (wave 3c log entry). Assembly contrasts (depth, all-angles, segmentation) need ≥3 runs per arm
  before any claim. The hinge-wrench effect (75% vs exact; 1.97× common factor) is far outside the spread.
- [ ] After wave 3b: replicate the strongest effect (hinge wrench) on more hinge scenes and seeds, once
  the user allows replicates.

- [x] Hinge stiffness is now in the `hinges` metric (2026-11-03). It takes effect after the next image rebuild
  (`sbatch scripts/build-image.slurm`); rebuild before any spring run. `springref` is still unscored.
  w3-spring-hard stiffness was checked statically.

## Needs new infrastructure
- [x] Segmentation-model tool inside the image (MobileSAM, CPU, offline; tools image `ea7d5678…`). Wave 3c:
  the agent never invoked it. Next: a condition whose task says when segmentation helps (e.g., per-part
  masks for primitive fitting), and record tool calls per run as an outcome. The planning tool will need the same.
- [x] SAM-3D tool. SAM 3D Objects needs an NVIDIA GPU with ≥32 GB VRAM and gated Hugging Face checkpoints. GPU approved
  2026-11-05: a `kempner_gpu_priority` service (A100-40GB or RTX Pro 6000, never H100/H200), relayed by the controller, so
  agent containers stay CPU-only and offline. Merged 2026-11-05 (log 06:30); weights fetched 2026-11-06.
  Wave 12 (2026-11-07): 0 calls in 3 runs, even when the task said when it helps. Next, if wanted: a stronger
  instruction (e.g. "call sam3d on each part before modeling it") to measure the tool's effect, separately from whether agents choose it. Wave 15 (2026-11-07): with use required, 10 and 11 calls, all ok; Chamfer 14.7 mm (boat) and 12.6 mm (assembly) vs 10.0–19.7 mm across the no-call runs: no detectable effect at n=1. Not worth replicates unless meshes become a submission format.
- [x] Robot planning tools (2026-11-04): `tools/planning/plan.py` gains `grasp` (antipodal parallel-jaw candidates
  with pregrasp/approach), `linear` (Cartesian straight line) and `fk` (pose, contacts, clearance), adapted from
  RoboSimStudio (Apache-2.0; `tools/planning/NOTICE`). Tool files are mounted per run, so no image rebuild is needed. The
  planning task text changed, so a new planning condition is not comparable with `w4-panda-lever-hard-planning`.
- [x] Tactile-only observation mode (2026-11-04): `[simulation] tactile = true` (Panda) adds 3×3 finger-pad taxels plus
  whole-finger normal force; tactile-only = `rgb = depth = calibration = false` (`generate-waves.py` option `tactile`).
  Needs the main and tools images rebuilt before a strict run.
- [x] Friction-coefficient metric (2026-11-04): selector `friction`, geometry-matched per primitive, geom μ and the
  effective μ against the world support. Evaluator change: rebuild the images first.
- [x] Expert-manipulation video metric (2026-11-04): selector `expert`, `[[expert_trajectories]]` in the six
  point-grasp manifests (none for Panda yet); side-by-side MP4 per trajectory. Rebuild the images first.
- [x] Policy-transfer metric (2026-11-04): selector `policy_transfer`, CEM fling policy trained in the candidate and
  run in the source, against a source-trained oracle; `[[transfer_tasks]]` in five manifests (point grasp only).
  Rebuild the images first.
- [x] Metric normalization against trivial baselines (2026-11-04): `sbatch scripts/baselines.slurm experiments/runs/ID`
  writes `baseline-scores.json` (empty and bounding-box candidates); `sim2sim.baselines.normalize`. Scored for all 53
  reported runs (2026-11-05); not yet added to the results report. Score wave 10 and the replicates when they finish.
- [ ] Replicates (deferred by user). Both `physics = "nonround"` hard variants and `pose = "jitter"` (2026-11-04: one
  rigid planar offset ±3 cm and yaw ±15° of the object group) now exist. Pose jitter refuses hinged/welded objects
  and Panda scenes.
- [ ] Longer term:
  - deformables
  - hidden dynamics (balls or liquid in a container)
  - fluids
  - other agent models
  - other simulators
  - real-world inputs

## Prompt policy change (2026-11-03, user)
- [x] Agents must not be told to use up their wall time; they finish when they believe their model matches the reference.
  New task text `TASK_CONVERGE` (experiments/configs/generate-waves.py) and rewritten harness prompt/finish guard
  (examples/sandbox_agent.py) need the rebuilt images (main build job `57592918`, tools image after it). All results so
  far used the "use the full budget" prompt, so new ablations need their baselines rerun under the new prompt before comparing.
  Time to finish then becomes an outcome (when does the agent believe it is done?).

## Comparisons at equal budget (2026-11-04, user)
- [ ] Report every ablation at matched wall time (e.g. 2 h of 4 h) and at matched cumulative tokens, not only at the final
  checkpoint. Use `experiments/validation/checkpoint-index.py` (run on a compute node) and add both axes to the results report.
- [ ] Copy finished runs from home (hot storage) to lab storage one run at a time, after confirming the lab quota has headroom
  (2026-11-03 incident: a bulk copy filled the shared quota and killed 16 runs). The quota is not visible from the cluster
  (`df` shows the 2 PB filesystem; no `lfs`); ask the user for headroom before copying.

## Scale identifiability (2026-11-04, from the no-interaction results)
- [x] ~~Does one stated mass fix the scale?~~ Wave 6: yes for the spring (exact); blocks only anchor the named block.
  Wave 7: also exact for the no-interaction spring; the lever's mass/inertia/damping follow it, but friction loss stays 0.
- [x] ~~Lever friction loss 0 under the new prompt without force~~ Wave 8: with force it is exact; without force the hinge
  hint recovers it (×0.257 near the ×0.30 common factor). Friction is identifiable but overlooked without a hint (n=1).
- [x] ~~Lever with stated mass AND hinge hint~~ Wave 9: near-full identification without force (friction ×0.93) and
  no-interaction (friction ×1.18).
- [ ] Scale anchor via a second object: the stated red-block mass did not propagate to blue through collisions (×1.41).
  A "non-verbal" anchor is not possible without force sensing. Any anchor must be a stated mass, so the open question is
  propagation across bodies.
- [ ] Blocks mass ratio was lost under the new prompt without force sensing (old prompt recovered it). Replicate before
  claiming a prompt effect (n=1); it may just be noise.
- [ ] Agents default to round masses (0.1 kg, 0.2 kg) when the scale is unidentifiable. Report it as a behavior. Also
  check that their uncertainty is visible (do they say the scale is unknown?).
- [ ] The no-interaction and no-force ablations under the new convergence prompt, if the prompt changes behavior.
- [ ] Time to finish is an outcome under `TASK_CONVERGE`. Spring base finished at 56 min; lever base was exact at 40 min
  but kept testing to 235 of 240 min (n=1 each). Report it, and check what the agent was testing after it was exact.

## Wave 18 sealed boxes (user, 2026-11-13)
- [ ] Review holds with the wave 16 rule: resume toward 6 h while the agent is working out the box contents; otherwise
  stop.
- [ ] After the runs end:
  - compare the final models' composite CoM trajectories per probe (plots under `version-*/evaluation/artifacts`);
  - check whether each agent found a body inside, and its mass ratio;
  - inspect the models with the new property editor.
- [x] Fixed `test_dashboard_physics_tab_with_live_backend`: headless Chromium's virtual clock stalls with live Viser viewers, so the check now polls the page in real time. Inspector and dashboard 47 passed (job `60359870`).
- [x] Wave 19 (hinted marble boxes): `-001` is invalid (camera clipping). Of `-002`, heavy is valid (it inferred the ball) and equal is invalid (it carried the box around the camera). The equal rerun is `-003`, job `60555357`. Review holds as before: review holds under the same rule. After the runs, compare the
  CoM errors with wave 18 and with the solid-box baseline.
- [ ] The wrist camera is exempt from the covered-lens rule. Design a rule before using it with sealed boxes.
