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

## Waiting on the user (do not start without an answer)
- [ ] **Spring levels review.** `w11-tetra-spring-k` and `w11-tetra-spring-block` are prepared on netscratch. The user
  wants to review the scenes first (previews in `experiments/validation/tetra-spring/`). After approval, launch them with
  `SIM2SIM_EXPERIMENTS_ROOT` set to netscratch. Option offered: a no-statement control on the same scene.
- [x] ~~Replicates~~ Approved 2026-11-05, n≤3 for the key findings. Done: results table in log 2026-11-06 21:30. 20 runs were queued (log 2026-11-05 03:15) for the
  force-sensing contrast on lever, spring and blocks, the stated mass on the spring, and the lever hint, mass + hint and
  no-interaction chain. Still without replicates: assembly contrasts (baseline COM error varies 7× over 3 old-prompt runs),
  old- vs new-prompt lever friction, the force-sensing effect on other seeds (`pose = "jitter"` or new nonround seeds), and
  wave 10. When they finish, analyze them with the originals as n=3 per condition (mean and spread), and add replicate
  pairs to `compare.py`.
- [ ] **Lab-storage copy.** Copy finished runs from home to lab only after the user states lab quota headroom (details
  under "Comparisons at equal budget").
- [ ] **Results report for waves 7–9.** The 2026-11-04 report covers wave 6 and earlier. Rebuild with
  `experiments/validation/build-results-report.py` plus a new spec file when the user wants it.
- [ ] **SAM-3D weights (user):** accept the license at https://huggingface.co/facebook/sam-3d-objects and run
  `/n/netscratch/qianqian_lab/Lab/aejxie/sam3d/env/bin/hf auth login`. Then the orchestrator does the rest:
  - `sbatch --account=qianqian_lab scripts/fetch-sam3d.slurm`
  - `sbatch --exclude=holygpu8a19102 scripts/sam3d-smoke.slurm full` (check the camera-frame z range)
  - start `scripts/sam3d-service.slurm`
  - run a `{"tools": ("sam3d",)}` condition vs its baseline on `trusted-image-sam3d.json` with `SIM2SIM_SAM3D_SPOOL` exported, one service per compared run
- [ ] **Netscratch → lab:** replicates live on netscratch, which purges after 90 days. Copy them to lab after quota headroom is confirmed.
- [ ] Retro-score earlier final submissions with `friction`/`expert`/`policy_transfer` (needs a metrics override in
  `candidate_history` or `baselines`-style tool). This would show whether scale errors (lever ×0.33) matter downstream.

## Config-only ablations (no new infrastructure)
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
- [ ] SAM-3D tool. SAM 3D Objects needs an NVIDIA GPU with ≥32 GB VRAM and gated Hugging Face checkpoints. GPU approved
  2026-11-05: a `kempner_gpu_priority` service (A100-40GB or RTX Pro 6000, never H100/H200), relayed by the controller, so
  agent containers stay CPU-only and offline. Merged 2026-11-05 (log 06:30); the weights are blocked on the user's HF login.
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

