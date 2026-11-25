# Experiment TODO (handoff)

What remains to run or build, so the next agent can continue. Dated log of what has
run: [experiment-log.md](experiment-log.md). Current plan and pre-registered
hypotheses: [experiments/plan-20261102.md](../experiments/plan-20261102.md).
Keep this list current: strike items when a log entry covers them.

## Standing constraints
- Strict OCI sandbox only for comparable results. Development-mode runs are not
  comparable: no filesystem boundary and a different harness.
- Model: `gemini-4-neon` is the default since 2026-11-23 (AGENTS.md "Gemini models"); `gpt-6-astra` only if Gemini rate
  limits would be hit. Medium reasoning.
- One run per condition. **No replicates until the user says so** (user, 2026-11-24; this supersedes the 2026-11-05
  n=3 rule). Reruns of runs killed by infrastructure are not replicates and continue.
- Slurm `test` or `sapphire` (CPU) only, never GPU partitions. Pack 2–3 runs per `scripts/experiment-strict.slurm` job.
- Cost: one pool per ET day, $1000 since 2026-11-06 (user; $700 from 2026-11-04, $500 before) (`cost_pool = "sim2sim-YYYYMMDD"`), and per-run caps (usually $44). Agents never see
  spend. The orchestrator picks and launches the next wave without asking, within the day's pool.
- Only the declared factor may differ between paired runs. Since wave 6 the task is `TASK_CONVERGE` (finish when nothing
  differs); compare new-prompt runs only with new-prompt baselines.
- Run on netscratch (`SIM2SIM_EXPERIMENTS_ROOT=/n/netscratch/qianqian_lab/Lab/aejxie/sim2sim-experiments`); the lab root
  is over its quota (2026-11-23).

## Roles (user, 2026-11-24)
- Level building belongs to the "s2s builder role" session; the runner session (this log's author) orchestrates runs:
  it prepares, queues, reviews holds and logs batches once a builder hands over merged configs and an image record.
  Touch-tabletop (wave 27, WIP branch `worktree-agent-a873b0bc2546edb50`) and gear-train (wave 28) were handed over.

## Queue priority (user, 2026-11-24)
0. **Wave 31, resets vs no resets (user, 2026-11-25): top priority.** Prepare and `scontrol top` as soon as the runtime's
   `[simulation] resets` switch is merged (builder).
1. Stochasticity experiments: fixture (wave 23) and die (wave 24).
2. Explanation-depth gear-train runs (wave 28), as soon as the builder hands them over: put them at the top
   (`scontrol top`).
   Built (builder, 2026-11-24; option (b): public grip slip, 0.5 ms step). Launch only after a trusted image that contains
   `gear_train.py` and the `grasp_slip_m` knob is built and recorded; see the gear-train section below.
3. Other new-capability runs: waves 25, 26 and 29 (and wave 27 once built).
4. Carbon evaluation last: its reruns carry `Nice=200000`.

## Inkling 2 Mini on the five hard levels (user, 2026-11-24): configs ready, not launched
- `experiments/configs/inkling-20261124/*.toml`, zero cost, self-hosted (AGENTS.md "Inkling 2 Mini, self-hosted").
- **Launch:**
  1. `sbatch scripts/inkling-service.slurm` from the repository root. For requeue, add
     `-p kempner_requeue --qos=normal`. The lab's 16-GPU priority QoS was full on 2026-11-25.
  2. Wait until `$INKLING_ROOT/service/service.json` exists and the log says `healthy`.
  3. Prepare and submit the five runs like the Carbon ones (`trusted-image-isolated.json`, sapphire). No key is needed.
- **Capacity:** one service, 5 slots and 262144 shared cache cells, serves all five runs. Expect slow calls, about 14
  tokens/s per run when all five generate at once.
  - Keep the service's `-t` longer than the 6 h walls plus queueing.
  - A preempted or dead service is held as a provider outage (36 h allowance).
- **Report:** compare at matched tokens as well as matched wall time, and report the timeout and malformed-call rates.

## 2026-11-23 batch (launched; see the log)
- [ ] Recordings (a1a65b1): jobs submitted before 2026-11-24 ~07:30 UTC keep their submit-time copy of
  `experiment-strict.slurm`, so they do not record. Run `.venv/bin/python experiments/validation/run-video.py RUN_DIR`
  on a test node for each such run before it goes into a report (all queued sapphire jobs of waves 23–26, 29 and Carbon).
- [ ] One `test` slot is reserved for the user's dashboard (user, 2026-11-23, via the results-explorer session) until
  the user releases it: at most 4 strict jobs on `test`; use sapphire otherwise.
- [ ] Review holds (4 h checkpoint: resume to 6 h while the agent works on the physics or mechanism) for waves 23–26 and
  the Carbon batch; jobs in `experiments/validation/jobs-20261123.txt`.
- [ ] Carbon vs Neon/Hydrogen/astra on the five hard levels (`model-compare-20261119.py`, add the Carbon set); record the
  best overall model in AGENTS.md "Gemini models" and make it the default.
- [ ] Wave 25: touch harness (local controller + pausable bash) vs the w21 harness, on Neon; report to the analyst session.
- [ ] Wave 26: low-rank answer vs full MJCF on cube_spring_block (n=2 each): parameter errors vs wall minutes and tokens;
  extract the MJCF runs' cube mass/COM/inertia and spring stiffness from their submissions for the same errors.
- [ ] Hydrogen configs of waves 23/24 exist but were not launched (Neon is the default).

## Cable tangles (user, 2026-11-24): designed, to build and queue
- [ ] Build and queue per [level-designs/cable-tangle-20261124.md](level-designs/cable-tangle-20261124.md): C1 one knotted
  cable vs C2 cable + hidden unlinked closed loop (both show two ends); new two-point-grasp capability; {base, 2 grasps,
  + MJCF} × {C1, C2} = 6 runs, point grasp + wrench, RGB-D movable camera, Neon, n=1, 4 h. Settings are user-approved;
  do not re-ask.

## Explanation-depth gear train (user, 2026-11-24): designed, to build and queue
- [ ] Build and queue per [level-designs/gear-train-depth-20261124.md](level-designs/gear-train-depth-20261124.md): visible
  ~20-gear train, deep and shallow models equal in prediction (deep ≥ 10× costlier); neutral task × {none, cheap-use,
  intervention-use} sentence = 3 runs, point grasp + wrench, RGB-D movable camera, Neon, n=1, 4 h. Settings are
  user-approved; do not re-ask.
- [x] Built (builder session, 2026-11-24). **Ready to launch once the image below exists.**
  - Configs `experiments/configs/wave28-20261124/` (`RUNS_W28`): `w28-gear-recon`, `w28-gear-cheap`, `w28-gear-deep`, one
    run each. gemini-4-neon, point grasp + wrench, RGB-D, movable camera, `sequence` tool, lock-step, 14400 s wall
    (`wall_only`), no checkpoint hold, budget hold 129600, provider hold 129600, pool `gemini-20261124`, evaluation
    timeout 600 s, public `simulation.grasp_slip_m = 0.002`.
  - Image: not built by the builder (the agent/evaluator image split is in progress). The evaluator image must contain
    `src/sim2sim/gear_train.py` and the `grasp_slip_m` knob (`simulation.py`, `candidate_worker.py`, `realtime.py`). Strict
    runs on an older image would ignore the slip in the agent's reconstruction world and fail evaluation (unknown metric).
    No existing baseline needs a rerun: the knob defaults off and the configs of other waves are byte-identical.
  - Validation: check job `63463232` (all premises; see the `docs/environments.md` gear-train section) and 49 tests in the
    same job. Smoke: `--prepare-only` of `w28-gear-cheap` on the login node; the dashboard lists the level (with the
    generic base prompt, since the task is neutral rather than `TASK_CONVERGE`).
  - Watch in the first runs: how often the agent's grip slips (`grasped` false after a move; pulls of more than 2 mm past
    the handle's reach or very fast turns), and whether that alone stalls the agent; the depth class of each submission
    (`scores.gear_train.structure`); and `scores.gear_train.accuracy.candidate_timestep_s`. Agents choose their own
    timestep; a G-cheap model at a large step is legitimate if it passes the gate.
  - Weakened from the design: evaluation probes stay within reach and below the slip threshold, so they are gentle
    (120° in 1.8 s or more). After an over-pull slips, deep and shallow still differ by up to 21 % of the gate, not under
    10 %. That is far below the 3.39 rad seen without the slip, but an agent comparing over-pulls closely might still
    notice.

## Touch-only tabletop levels (user, 2026-11-24): designed, to build and queue
- [ ] Build and queue per [level-designs/touch-tabletop-20261124.md](level-designs/touch-tabletop-20261124.md): Franka +
  Shadow hand, touch only; L1 abstract shapes, L2 mug/banana/hammer (categories unstated vs stated), L3 spiky ball in a
  bin; × {w21, w25} harness = 8 runs, Neon, n=1, 4 h. Settings are user-approved; do not re-ask.
- [x] Built (builder session, 2026-11-24). **All 8 runs ready (L3 settled by the user 2026-11-24: hard identification task,
  no pass bar, 40 mm walls; check job `63460653`). Held only for the isolated image (agent/evaluator split).**
  - Configs `experiments/configs/wave27-20261124/`: `w27-shapes`, `w27-objects-unstated`, `w27-objects-stated` and
    `w27-bin-spiky`, each also as `-touch` (w25 harness: + local controller copy, pausable bash). gemini-4-neon, touch only
    (tactile, no images, no wrench), 14400 s wall, no checkpoint hold, provider hold 129600, pool `gemini-20261124`.
  - Image `b56436aa9137…`: record `experiments/images/trusted-image-touch.json` (job `63327059`, built from the merged
    branch `d59a5b2`). Point strict jobs at it with `SIM2SIM_TRUSTED_IMAGE_RECORD`. No existing baseline needs a rerun:
    the harness change is additive (a new robot family).
  - Validation:
    - check job `63325077` (all premises; see the `docs/environments.md` section);
    - tests `63325338` and `63325491` (tabletop, local controller, runtime, simulation);
    - OCI `63327523` (8 passed, 1 skipped) and `63328095` (new panda_shadow end-to-end test passed);
    - smoke `--prepare-only` of `w27-objects-stated` and `w27-bin-spiky-touch` (job `63328177`).
  - Watch in the first runs: whether agents find the flexed-wrist pose. With the wrist straight the forearm hangs ~4 cm
    under the palm and hits low objects, the table and the bin rim. Also watch L3 balls knocked over the 30 mm wall.
    After each run: `python -m sim2sim.tabletop RUN_DIR` (Slurm `test`) for grab success and the contact coverage view.
  - L3 reference floors (scripted identification, 3 presses per ball): palm taxels 25% (curled) / 47.5% (straight), contact
    height 97.5%; grab possible but hard (1 of 108 open-loop tries). See `docs/environments.md`.

## Wave 24 biased die (user, 2026-11-23): launched on Neon only (4 runs)
- [ ] Configs `experiments/configs/wave24-20261123/` (8 runs): {point grasp without wrench, Franka with joint efforts} ×
  {25 frames/s baseline, 100 frames/s control} × {gemini-4-neon, gemini-4-hydrogen}.
  - Every run: real time, RGB-D, movable camera, `sequence` + `channel` tools, die metric.
  - 6 h wall with the 4 h checkpoint hold, `provider_hold_seconds` 14400, accounting-only caps, pool `gemini-20261123`.
  - Single factors: the actuator (grasp vs franka at equal frame rate), and the frame rate (25 vs 100 at equal actuator).
  - The 100 frames/s control keeps the throws, evaluation trials and baselines identical, so it tests whether the agent
    adapts its modeling approach (e.g. a mechanics model from precise release estimates) to more precise observations.
- [ ] Before launching:
  - Use the merged main image `42f70708fdcc…` (`experiments/images/trusted-image.json`, job `63142237`); the
    branch-only image in `trusted-image-die.json` predates the merge (predictor runner moved to `sim2sim.predictor`).
  - Pack **one die run per strict job**. The controller's real-time physics thread, the agent (4 CPU) and, during
    evaluation, the predictor container (4 CPU) all need CPUs at once.
  - Use lab or netscratch storage (`audit_bytes` 32 GiB: every frame batch the agent's code reads is audited).
- [ ] Review:
  - Brier score vs experiment time against the uniform, empirical and mechanics baselines (`RUN/die-view.html`).
  - Whether agents model the throw-dependent outcome distribution, use the controller channel, and report estimates.
  - Latency and late-answer rates.
  - Compare at matched wall minutes and tokens.

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

## Wave 20 (user, 2026-11-16): outside-start tunnels, point-grasp tetra/cube springs, 1D spring–mass(–damper), lighting ablation
- [ ] Launch when the three level builders report back:
  - tunnel_{spring,latch}_out: point grasp with wrench, RGBD, `sequence`, fixed camera.
  - tetra/cube spring levels:
    - point grasp without force, RGBD, movable camera, `sequence`;
    - the prompt stresses identifying the inertial properties of the cluster or cube and states that the spring is ideal;
    - give k in the spring + object levels, and the block mass in the + block levels.
  - msd_{spring,damper}_{visible,hidden}: point grasp with wrench, RGBD, movable camera, `sequence`.
  - All runs: 6 h visible budget with a 4 h wall checkpoint hold (new `checkpoint_wall_seconds`), plus the $44 hold.
- [ ] **Lighting ablation (user):**
  - The basic level of each family (tunnel_spring_out, tetra_spring, cube_spring, msd_spring_visible) gets a twin run
    whose prompt also gives the ground-truth lighting: the exact `<light>`/`<headlight>` MJCF from the scene.
  - Next report: does the given lighting reduce the time agents spend on lighting/appearance (share of wall time and
    of experiments, from progress notes and actions), and does that time go to the physics instead (physics metrics at
    matched time)? The user cares about physics, not appearance fitting.
- [ ] After launch: build the dex-hand tactile levels (Shadow Hand, no vision, resets; 4 objects; com + chamfer).
- [ ] **Cloth levels (user, 2026-11-16).** Being built in a worktree; merge when done.
  - Levels:
    - base square, uniform stiffness;
    - quadrant stiffnesses with uniform colours;
    - quadrant stiffnesses with quadrant colours;
    - quadrant stiffnesses with colours split by the diagonals;
    - an odd shape with random holes and cuts;
    - a centre circle hole with a stiff rim;
    - a centre circle hole with a soft rim.
  - Ground truth: a simple triangle mesh, modelled on "RoboSim studio" default cloths.
  - Evaluation:
    - a local stiffness map with the cloth laid flat;
    - drops onto a bar from random angles;
    - a drop with the centre over a cone;
    - videos of every evaluation.
  - Conditions: RGBD, `sequence`, movable camera, point grasp with wrench, no Panda.
- [ ] **Dex-hand levels.** Being built in a separate worktree (Shadow Hand fixture); merge when done.
- [ ] **Budget (user, 2026-11-16):** $2000 for 2026-11-16 only. From 2026-11-17, a HARD cap of $200 per ET day across all pools; spend after midnight ET from older pools counts toward the new day.
- [ ] Controller hang:
  - The controller waited 13 h on a crashed inference child (`ModuleNotFoundError`) and the wall watchdog did not end
    the run. Make `InferenceBroker.call` fail loudly when the child exits without replying, and check why `WallClock`
    did not fire.
  - Launches now freeze `src/` per job, which removes this trigger.
- [ ] **Node-packing hang (2026-11-16):** with three strict jobs (six runs) on one node, every run hung at its first
  candidate-change evaluation (pause plus evaluator step).
  - Find the cause: the shared per-UID nproc limit, or rootless Podman contention.
  - Until then, keep at most two strict jobs per node.
  - Make the controller fail loudly on a stalled pause or step launch.
  - Redo the lost replicates as the $200/day budget allows: cylinder, marble heavy and equal hint, spring-out,
    msd-spring-visible-lit, hand prism, offset-com and smooth.
- [ ] **Image-load hang (2026-11-17/18):** strict jobs on sapphire nodes holy8a24208 and holy8a26105 hung inside
  `podman-job.sh`/`load_trusted_image` for 10–12 h.
  - Add timeouts to the podman init and load (fail fast so Slurm frees the node), and log each step.
  - Find out what state on a node causes it (stale pause process or lock?).


## Wave 23 positioning fixture (user, 2026-11-23): launched on Neon only (3 runs)
- [ ] Launch `experiments/configs/wave23-20261123/` (6 runs: `fixture_rigid`, `fixture_rubber`, `fixture_bimodal` ×
  `gemini-4-neon`, `gemini-4-hydrogen`; Franka with joint torques, RGB-D, movable camera, sequence tool; 6 h visible wall,
  4 h checkpoint hold, `provider_hold_seconds = 14400`, accounting-only caps, pool `gemini-20261123`). Levels:
  `docs/environments.md` (Spring-powered positioning-fixture levels). The submission is a predictor program
  (`predictor.py`: `predict`, `settings`), scored by `fixture` (CRPS at 25 held-out settings × 16 releases, Brier, hit
  rates at the chosen settings × 40 releases); each evaluation takes a few minutes (`timeout_seconds = 1800`).
- [ ] Launch with the merged main image `42f70708fdcc…` (`experiments/images/trusted-image.json`, job `63142237`; OCI
  job `63143283` passed). The branch-only image in `trusted-image-fixture.json` predates the merge (unified predictor
  runner and predictor-task prompt). Rebuild if `src/` or the harness changes before launch.
- [ ] Gemini fairness rules (AGENTS.md) need an astra reference only if a comparison is wanted; none exists for this level.
- [ ] Review per run: the dashboard links each submission's fixture view (predicted vs measured histograms, settings map,
  trial replay). Check whether agents find the bimodal level's minor (folded-back) mode and avoid mean-based settings for
  the narrow target.
