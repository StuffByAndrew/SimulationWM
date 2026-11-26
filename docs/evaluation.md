# Private evaluation

`sim2sim.evaluation.evaluate_submission(zip_bytes, config)` runs in the resource-limited evaluation OCI worker, never in the agent container. The runtime excludes scoring from the active timer and retains reports privately. The evaluator creates a fresh source simulation, independent of the interactive state. Container memory, CPU, process and wall-clock limits are mandatory because native XML/asset parsing is not a security boundary.

Configuration supplies `source_dir`, `output_dir`, optional `source_entrypoint` and candidate `entrypoint` (default `scene.xml`), camera/render settings, and private `probes`. A source-relative `manifest` defaults to `manifest.toml`; its `probes`, `correspondence`, and `camera` populate absent configuration fields. Correspondence is optional. Missing corresponding bodies explicitly fail the privileged parameter component while generic visual evaluation continues. Other invalid submissions, compilation errors, and failed probe execution yield `status: failed`, an explicit error and no scores.

The candidate uses the trusted source camera position, orientation and intrinsics. Each private probe supplies a pixel, or a source body and local point projected to a pixel. Source and candidate independently receive the same reset, pixel grasp, four displacement samples, release and three waiting samples. A missed candidate grasp advances time without motion; no privileged body attachment is used. With `simulation.grasp = "weld"` the grasp is a weld in both worlds, and a probe may declare a world `rotation` vector (radians), split over the four moves like the displacement (docs/protocol.md "Weld grasp"). RGB, depth and segmentation-derived object silhouettes are compared at every sample. Masks include visible geometry on non-world bodies and exclude world geometry such as the floor. This does not provide per-object segmentation correspondence.

Successful reports contain `scores.reset`, `scores.probes` and `scores.parameters`. RGB error is mean absolute RGB difference divided by 255. Depth error is mean absolute depth difference in meters clipped to one. Silhouette error is mask symmetric difference divided by union. Image trajectory error is distance between mask centroids (normalized image coordinates) and mean masked depth in meters; this mixed-unit metric measures observable motion rather than privileged body pose. Optional privileged mass, friction, hinge stiffness, damping and range errors compare declared corresponding bodies. Errors divide by the larger of source magnitude and 0.01 and clip to one. Different parameter array shapes receive error one. No calibrated aggregate success threshold or rotational trajectory metric is currently supplied.

RGB, depth and mask PNGs for reset and each probe are saved privately under `output_dir/artifacts/<digest>`. Each probe also stores its exact action sequence and observed trajectories in JSON. Reports and these artifacts must never be returned to the agent.

ZIP submissions are limited to 8 MiB compressed, 32 MiB declared expanded size and 256 entries. Base64 is validated before decoding with an encoded-length bound. Absolute paths, traversal, backslashes, URI/drive syntax, duplicate entries, symlinks, special files, XML entities/DOCTYPE, native MJCF plugins/extensions and external MJCF paths are rejected. Included XML is inspected recursively regardless of filename extension. Archive checks do not replace OCI resource limits.

Accepted project versions are SHA-256 addressed under `output_dir/versions/<digest>/project`, with the original archive retained alongside. Repeated submissions reuse the existing version. Reports contain a relative `project_path`. Host-owned version directories must never be writable by the agent.

Candidate physics uses the source timestep so identical tick sequences span identical physical time. Every probe records source/candidate grasp success and a `status` (`ok` or `missed_grasp`), and the report includes `probe_grasp_failure_rate`. A source grasp miss invalidates the reference probe and fails evaluation explicitly. A candidate grasp miss is retained as a scored failed interaction; its subsequent moves advance the corresponding ticks without displacement. Parameter metrics also include a dedicated absolute sliding-friction coefficient (`sliding_mu_absolute_error`).

Researchers may configure a trusted `metrics_hook = "package.module:function"` in host evaluation configuration. The module must be installed in the evaluation image; submissions cannot supply hooks or Python modules. The callable receives `(source_simulation, candidate_simulation, context)` after built-in probes. Context contains `scores`, `artifact_dir` and trusted configuration. It returns JSON-compatible finite custom metrics, saved as `scores.custom`; exceptions explicitly fail evaluation. The hook can reset either independent simulation for bespoke material or dynamics probes, and write private artifacts inside the supplied artifact directory.

Reference simulation settings inherit trusted `source_config`; explicit evaluation camera, dimensions, timestep, grasp softness (`grasp_solref`, `grasp_solimp`) and tick settings override it. Source manifest `graspable_bodies` applies only to the reference; candidate graspable bodies are inferred from its own movable structure. The candidate receives the same timestep and grasp softness without requiring source body names.

Validation: Slurm test-partition job 56020252 passed the then-current full suite (31 tests, 40.86 seconds), including numerical/render fidelity checks. Additional final regression tests cover malformed nested asset references, missed candidate grasps, missing named correspondence and custom hook errors. Those added physics tests require a subsequent Slurm run.

## Primitive reconstruction metrics

The two seeded assembly manifests enable `[reconstruction]` with `samples = 2048` and their construction seed. The same table in trusted evaluation configuration overrides the manifest. `samples` accepts 128 through 32768, and `seed` controls deterministic sampling. Metrics reset both simulations and compare geometry in the **reset world coordinate frame**, without rigid alignment or scale fitting, so position and orientation errors count. Floor geometry and synthetic grasp targets are excluded.

`scores.reconstruction.chamfer_distance_m` is the symmetric mean **unsquared** nearest-neighbor distance in meters: half the sum of each sampled surface's mean distance to the other sampled surface. Samples represent the **exposed union surface** of supported object primitives (sphere, box, ellipsoid, cylinder, capsule, and mesh geoms taken as the convex hull of their compiled vertices, which is what MuJoCo collides); proposal points strictly inside another constituent primitive are rejected. They are area weighted across constituents, and ellipsoid samples use the sphere-transform area Jacobian. Ellipsoid area weights use deterministic numerical quadrature. Internal overlap surfaces and floor surfaces do not enter the Chamfer calculation. This is a finite sampled-surface estimate; increasing samples improves spatial resolution. Canonical geometry ordering excludes names, density and color, so identical geometry with renamed or reordered constituents yields the same samples and zero distance within numerical precision.

`center_of_mass_error_m` is the norm of predicted minus source compiled object COM, with the signed difference in `center_of_mass_error_vector_m` and both COM positions retained. Each COM uses compiled body mass and reset-world inertial positions. It therefore honors explicit body inertial overrides. COM remains available independently when primitive surface or density evaluation is unavailable.

`density.per_primitive` reports `source_name`, `candidate_name`, source and predicted density in kg/m³, `absolute_error_kg_m3`, and `relative_error` (null for zero source density). Each constituent has one uniform density; the composite may be heterogeneous. Effective densities come from parsed MuJoCo specifications, including inherited defaults and explicit geom mass overrides divided by the analytic primitive volume. Shape correspondence uses minimum-cost one-to-one assignment of geometry-only, 128-point pairwise surface Chamfer costs in the reset world frame. Neither names, color, density nor mass influence matching. This makes primitive order and naming irrelevant, while ambiguous geometrically identical constituents can have ambiguous density assignments.

`density.coverage`, `unmatched_source` and `unmatched_candidate` explicitly expose unequal primitive counts. Missing constituents receive no artificial zero density error. Source and candidate `*_compiled_mass_consistent` flags compare summed constituent density-times-volume against compiled object mass; explicit body inertial overrides or compiler mass scaling can make these disagree. Overlapping constituent masses follow MuJoCo's additive geom-mass convention; density is not reinterpreted as a non-overlapping material field. Shell-inertia densities are unavailable because they are surface rather than volumetric densities. Unsupported object geoms explicitly make the primitive surface/density components unavailable and retain generic visual metrics and independently available COM metrics. All reconstruction metrics and their plots are host-private.

The `*_compiled_center_of_mass_consistent` density flags also compare the center of mass implied by constituent densities and primitive centers to compiled body COM. They detect explicit inertial-position changes even when total mass agrees. A false consistency flag alone leaves parsed constituent density errors available with their declared meaning. A matched primitive's density is instead a gap (`status: unavailable`, `unavailable_reason`, null predicted density and errors) when its declared mass does not set its compiled body's mass distribution: the body has an explicit `<inertial>` (not overridden by `inertiafromgeom="true"`), the geom is massless (zero mass or density, or outside `inertiagrouprange`), or it is a `contype=conaffinity=0` visual-only geom with a massless colliding twin in the same body (center within its bounding radius), which geometry-only matching cannot tell apart. The same rules apply to the source side. Matching stays geometry-only. `center_of_mass_error_m` remains the mass comparison for such candidates.

## Named hinge damping and friction loss

A trusted manifest `[hinge_correspondence]` maps source hinge names to candidate hinge names. `scores.hinges` independently records each source hinge's status, both joint names, `source_damping`, `predicted_damping`, `damping_absolute_error`, `damping_relative_error`, and the equivalent four `frictionloss` and `stiffness` fields. Damping uses N·m·s/rad, friction loss N·m and stiffness N·m/rad (from evaluator image builds after 2026-11-03; earlier images omit stiffness). Relative error divides by the absolute source value and is null when that value is zero. Missing or non-hinge mapped joints are explicitly unavailable. Named mappings resolve each joint's compiled degree-of-freedom address; joint declaration order does not determine correspondence. The older body parameter summary also includes `hinge_frictionloss`, alongside damping, and uses the declared named hinge pairs when provided.

Private `metric-history.json` preserves raw hinge measurements independently of primitive reconstruction availability. `hinge-history.png` adds three checkpoint/version panels showing absolute damping, friction-loss and stiffness errors for every source hinge. Failed checkpoints and missing named hinges create gaps rather than zero errors. The existing three-panel reconstruction history remains available. The damped lever enables these hinge diagnostics alongside ordinary geometry/density/COM metrics and private dynamics probes; the unequal-mass tetrahedral cluster uses the same reconstruction metrics and its own private dynamics probes.

The damped-lever manifest uses `hinge_matching = "single"` instead of requiring the agent to reproduce a private joint name. When both models have exactly one hinge, the evaluator matches those hinges regardless of names (including an unnamed candidate hinge). Multiple or missing hinges explicitly make this component unavailable. An explicit `[hinge_correspondence]` takes precedence when a researcher needs named correspondence.


## Independent reconstruction selectors and unavailable assignments

Select `chamfer`, `center_of_mass`, and `density` independently in
`evaluation.metrics` (or `experiment.evaluation_metrics`). Disabled components
skip their sampling, density parsing and scoring. `reconstruction` retains the
legacy bundle. Metric-only reset operations skip camera rendering; appearance
and dynamics still use their trusted full-modality captures when selected.

`evaluation.reconstruction.matching_same_type = true` and
`matching_max_distance_m = 0.08` constrain geometry-only assignment in the new
sphere preset. Invalid pairs are excluded from the assignment and listed as
unmatched; density is not assigned to an unrelated box or distant object merely
because one exists. These thresholds are an explicit geometric eligibility rule,
not proof of unique physical identifiability. Unmatched source curves retain
NaN (`null` in strict JSON) at that checkpoint. COM compares the entire compiled
object and requires no primitive correspondence. Hinge matching stays unavailable
until a declared pair or unambiguous single-hinge pair exists.

## Panda object and arm evaluation

Panda and Shadow hand submissions include an unchanged copy of the public robot fixture.
The evaluator compares it byte-for-byte to the trusted public fixture and
requires a single `robot/panda.xml` (or `robot/shadow_hand.xml`) include. Shadow hand
levels use the same object selectors, fixture exclusion (the `rh_forearm` subtree) and
`[[arm_probes]]`, whose steps carry `hand_target` fields (`actuator_targets`, `ticks`);
the probe score then compares joint positions, velocities and efforts and adds
`tactile_mae_n`, the mean absolute difference over every taxel and body total. The arm is an instrument, not a
reconstruction target: private object selectors exclude it from surface,
density and compiled COM metrics. A fixed cube is retained as a jointless
nonworld body so its box enters target geometry sampling. Single-hinge matching
excludes the seven Panda joints before matching the spring or damped lever.

`dynamics` selects private arm-target probe sequences from the source manifest.
Fresh source and candidate worlds execute identical joint/gripper commands.
The researcher report records mean absolute joint position, joint velocity,
actuator effort, finger position/velocity and gripper-width differences, plus
target-contact tick counts. Vision-enabled Panda levels also record RGB error
at probe checkpoints. Blind levels use proprioception and private geometry
metrics without an agent-visible image. Probe commands, scores, trajectories,
contacts and image artifacts stay researcher-only. A probe that never contacts
its target cannot establish contact-based fidelity and must be corrected during
environment validation.

Each arm probe also samples, every 10 ticks (20 ms), the composite centre of mass
of every non-robot body not welded to the world, in source and candidate alike
(model-agnostic: no body correspondence; a box and its hidden contents count
together). `object_com` reports the mean, final and maximum source-candidate COM
distance along the probe, both total free masses and their absolute difference.
Without positive free mass on either side the COM errors are `null` with
`status = "unavailable"`, never zero. The raw COM samples go into the probe's
`trajectory.json`, with a `com.png` plot of x, y, z and the distance over time.

## Independent RGB image MSE

Select `image_mse` in `evaluation.metrics` or `experiment.evaluation_metrics`
for private `scores.reset.rgb_mse`, independently of `appearance`, dynamics and
reconstruction metrics. Fresh source and candidate simulations are reset and
rendered with the trusted source camera position, orientation, intrinsics and
image dimensions, regardless of agent observation modalities. This is the fixed
global camera, even when an optional Panda wrist camera is enabled. Wrist
images are observations and recorded evidence, not inputs to `image_mse`. MSE averages
`(source_rgb / 255 - candidate_rgb / 255)^2` over every pixel and all three
channels of the full RGB image, including background; no silhouette mask,
alignment or cropping is applied. Its range is zero through one. Selecting
`appearance` also includes `rgb_mse` alongside the existing RGB absolute error,
depth and silhouette metrics, including visual dynamics checkpoints. Scores and
checkpoint history remain researcher-only and are never returned to the agent.

Candidate loads can be privately evaluated without counting them as submissions by
setting top-level `evaluate_candidate_changes = true`. Each successful load records
an `evaluation` audit event with `checkpoint_kind = "candidate_change"` and a
separate `change_index`; scores remain private and the original hard wall deadline
still bounds evaluation. This option defaults off.

Historical successful loads can be recovered from verified audit request wires:
`sbatch scripts/candidate-history.slurm experiments/runs/RUN_ID ...`.
The bounded native development evaluator uses the hash-verified original source
snapshot and trusted recorded configuration, adds the `image_mse` selector, and
writes `candidate-change-history.json` plus private per-change artifacts. It never
modifies the original sealed audit. Original change wall times and actual new
evaluation timestamps remain separate; retrospective results are labeled explicitly.
Repeated invocations retain completed entries, including explicit failures and
unrecoverable gaps. Filesystem edits that were never loaded or saved cannot be
reconstructed. This is retrospective development evaluation, not an OCI run or a
new model experiment.

## Per-primitive contact friction

Select `friction` in `evaluation.metrics` (or `experiment.evaluation_metrics`) for
private `scores.reconstruction.friction`. It uses the same geometry-only primitive
assignment as density (`matching_same_type`, `matching_max_distance_m`, coverage
and unmatched lists), never names, color, density or mass. Each matched record keeps
the compiled `[sliding, torsional, rolling]` coefficients and `condim` of both
geoms, `sliding_mu_absolute_error` between the geoms' own sliding coefficients,
and `support_sliding_mu_absolute_error` between the coefficients MuJoCo actually
applies against the world geom straight below each primitive's center at reset
(`source_support_sliding_mu`, `predicted_support_sliding_mu`). The applied value
follows MuJoCo's mixing rule: the higher-`priority` geom's coefficients, otherwise
the elementwise maximum; `condim = 1` is frictionless. The two errors differ by
design: a block whose geom μ is below its table's μ slides exactly as if it had
the table's μ. The support value is a gap with `unavailable_reason` when an
explicit `<pair>` overrides the geom, nothing lies below, the primitive rests on
another object, or the world geom below cannot collide with it. Upper primitives
of a rigid cluster report the world geom under their center even though only the
lowest ones touch it. Object-on-object friction is not compared.

## Expert manipulation video

Select `expert` for private `scores.expert`. Each source manifest
`[[expert_trajectories]]` entry has a `name`, `frame_ticks` (1–1000, default 10)
and `actions`: point-grasp scenes use `grasp` (a `pixel`, or `body` plus
`local_point`), `move`, `release` and `wait`; Panda scenes use `arm_target` and
`wait`. Fresh source and candidate worlds replay the identical script from reset.
A body grasp resolves to the source pixel of that body point at the moment of the
grasp, and the candidate receives those resolved pixels; a missed candidate grasp
turns later moves into equal waits. A trusted frame is captured at reset and every
`frame_ticks` ticks. Each record reports the frame count, both grasp outcomes,
mean RGB absolute error, RGB MSE (also its maximum), depth and silhouette errors,
and the mask-centroid image trajectory error. A source grasp miss fails the
evaluation. Private artifacts hold `expert_NNN/side_by_side.mp4` (source left,
candidate right, real-time rate) and the resolved script with both trajectories.
The six point-grasp environment manifests carry one scripted manipulation each.
Panda manifests carry none yet; `expert` fails explicitly without one.

## Policy transfer

Select `policy_transfer` (point-grasp scenes only) for private
`scores.policy_transfer`. Each manifest `[[transfer_tasks]]` entry gives a grasp
point (`pixel`, or `body` plus `local_point`) and `goal_offset_m`; the goal is the
source object COM at reset plus that offset. An open-loop policy grasps there,
moves the handle by a displacement `d` over `move_ticks` (default 40), releases,
and waits `settle_ticks` (default 400); its cost is the distance from the settled
object COM to the goal. A cross-entropy search (`iterations`, default 6, of 16
samples and 4 elites, `d` within ±`max_displacement_m`, default 0.3 m, started
from the naive `d = goal_offset_m`) trains `d` in the candidate. That policy then
runs in the source: `transfer_error_m`. The same search in the source gives
`oracle_error_m`, and `transfer_regret_m` is their difference; the report also
keeps `naive_error_m`, `candidate_predicted_error_m` (the candidate's own belief),
both policies and whether the candidate grasp succeeds. Searches are seeded and
MuJoCo is deterministic, so evaluating the source against itself gives zero
regret. Five manifests define a fling task; the spring defines none because its
outcome cannot depend on the policy. A Panda selection fails explicitly.

## Mechanism force profile

`force_profile` (selector; needs a private manifest `[force_profile]`, see `src/sim2sim/force_profile.py`) measures a
hidden mechanism the same way in the source and the candidate, independent of how the candidate models it: the
trusted-camera ray through `grasp_pixel` (or, if that misses, the nearest pixel within `grasp_radius_px` whose ray hits a
movable body) attaches an ideal point grasp to the first movable body (the stick handle),
exactly as an agent's pixel grasp does (agents often build in a shifted world frame that still matches the image), at
time 0 and then held still for `settle_ticks` (a candidate not at rest would otherwise drift off the pixel), and
the grasp target follows the polyline `path` through `strokes` (`[arc_length_m, speed_m_s]` targets: slow push, retract,
push, retract, then one fast push and retract). Every `sample_ticks` it records the grasp reaction and `resistance_n`,
its component against the path tangent. Panda scenes run this probe with point grasps enabled and the robot at home.
Scores: `rms_resistance_error_n`, `max_resistance_error_n`, `source_rms_resistance_n`, plus both raw sample series
(`stroke`, `arc_length_m`, `resistance_n`, `force_world_n`) for plotting the ground-truth force-position profile
against the model's; loading and unloading strokes stay separate, so latch hysteresis and damping are plottable. A
candidate handle the ray misses is `missed_grasp` with null errors (a gap). The empty baseline has no handle.

## Cloth (2D flex)

`cloth` (selector; needs a private manifest `[cloth]`, see `src/sim2sim/cloth.py` and the `cloth_*` manifests) compares
cloths independently of mesh resolution or MJCF layout. A model's cloth is the union of its 2D flexes, whose vertex
bodies must move on three orthogonal slide joints (flexcomp `dof="full"` or explicit vertex bodies); a candidate without a
2D flex, or with another vertex parameterization, is `unavailable` (with the error), never zero. Fresh source and
candidate worlds gain two researcher-only static fixtures far from the scene (a horizontal bar and a cone, 1.5 m either
side of the source cloth's reset centre, their tops 0.6 m above it); placements and probe points come from the source and
apply to both worlds in the world frame, without alignment.

- `stiffness_map`: on a `lift_grid`² grid over the source cloth's reset bounding box (inset `lift_margin_m`), a vertical
  ray from above grasps the cloth as an agent's point grasp does (nearest vertex of the hit flex face, at the hit offset)
  and lifts it `lift_height_m` at `lift_speed_m_s`, recording the grasp's vertical reaction every `sample_ticks`. A
  stiffer cloth lifts a wider tent, so the final lift force and the force/height slope map local bending stiffness (plus
  the lifted weight). Reports both per-point maps, raw curves, `lift_force_mae_n`, `lift_force_rms_error_n`,
  `lift_force_relative_error` (MAE over the mean source force), `stiffness_mae_n_m`, `stiffness_rms_error_n_m`,
  `probed_points` and `candidate_missed_points`. Source-ray misses (holes, outside the outline) are gaps.
- `drops`: each `[[cloth.drops]]` (seeded in the generator: three bar, two cone) rotates the reset layout by `yaw_deg`
  about z and `tilt_deg` about a horizontal axis at `tilt_axis_deg`, releases it at rest 5 cm above the bar or the cone
  apex, and runs `drop_ticks`. Every `frame_ticks` the symmetric mean Chamfer distance between `samples` area-weighted
  barycentric surface points of each cloth (fixed on each mesh) gives `chamfer_series_m`, `mean_chamfer_m`,
  `final_chamfer_m` (the drape at rest) and `max_chamfer_m`; `drop_mean_chamfer_m` and `drop_final_chamfer_m` average the
  drops. `rest_chamfer_m` compares the two flat cloths at reset.

Private artifacts under `artifacts/<digest>/cloth/`: `stiffness_map.json`, `stiffness_map.png` (source, candidate and
absolute-error lift-force maps) and `drop_NNN/side_by_side.mp4` (source left, candidate right, real time). Point-grasp
interaction with any flex is described in `docs/environments.md` (Cloth levels). Mask-based metrics (`appearance`,
`dynamics`, `expert` silhouettes/centroids) count geoms only and ignore flexes; use `image_mse` and `cloth` for cloth levels.

## Touch tabletop objects (`objects`, `identification`)

Selectors for the touch-only `panda_shadow` levels (`src/sim2sim/tabletop.py`; private manifest `[tabletop]`), on fresh
reset states of both worlds.

- `objects`: each source object (a top-level `object_bodies` body with its descendants) is assigned at most one candidate
  object (a top-level non-robot candidate body) by geometry only: a one-to-one assignment minimising the symmetric
  Chamfer distance between exposed primitive-union surfaces, never names, color, density or mass. Per source object: the
  assigned candidate body, its Chamfer, the surface-centroid error, `found` (centroid error within the source object's
  bounding radius), `label_correct` (the candidate body name contains a private `[tabletop.labels]` synonym; only where a
  list exists, L2) and the prior-defeating dimensions (`[[tabletop.prior_dimensions]]`: exact extent of both objects
  along a world axis). Totals: `found_fraction`, `mean_assigned_chamfer_m`, `label_accuracy`. An unassigned object has no
  Chamfer, centroid, label or dimension: a gap with a reason, never zero. The existing `chamfer`, `center_of_mass` and
  `density` selectors still score the scene as a whole.
- `identification` (L3): the submitted project's root `parameters.json` `{"spiky_ball_xyz": [x, y, z]}` (world m, initial
  centre) against the initial ball positions: `correct` when the nearest ball is `[tabletop] spiky`, plus the distance to
  it. A missing or malformed file leaves identification `unavailable` with the reason; the rest is scored as usual.
- `prior` baseline (L2): `sim2sim.baselines` also evaluates a level's private `prior.xml` (canonical mug, banana and hammer
  at typical sizes at the true poses) where one exists, so the prior-defeating dimensions compare with a prior-only answer.

Post-run, researcher-only: `python -m sim2sim.tabletop RUN_DIR` (Slurm `test`) reads the verified audit and source
snapshot and writes `RUN/tabletop.json` and `RUN/touch-view.html`: L3 grab success on the reference world's last recorded
state (spiky ball COM ≥ rim + 2 cm, touching the robot, every smooth ball in the bin) and the contact coverage map (every
taxel contact in world coordinates over the true scene, with each object's covered surface fraction within 1 cm). The
dashboard links the view from the run page.

## Die prediction (`die`)

The real-time die levels (`docs/environments.md`) score a **predictor program**, not a scene. `src/sim2sim/die.py`
implements the metric; the stream itself is described in `docs/protocol.md`.

**Trials.** At a run's first submission the host simulates fresh throws from the trusted source scene and caches them in
`RUN/die/` with the baselines. This takes a few minutes, inside the paused private evaluation:

- `[die] trials_per_protocol` throws of each public launcher protocol, from launcher stream 1: new perturbation draws
  from the same distribution as the agent's session (stream 0).
- Each trial records the true release state, its frames before the deadline (the agent's modalities, from the trusted
  fixed camera at its initial pose), the trajectory at 50 Hz until rest, the outcome, and the state at the deadline.
- The die is at rest when it is slower than 10 mm/s and 0.2 rad/s for 0.2 s, or after 6 s. The outcome is the face
  whose normal is closest to +z, also for a die resting cocked against a tray wall (top cosine < 0.995, flagged `cocked`;
  about 2 % of throws).

**Scoring.** Each submission's program runs in its own container (agent image and bounds, nothing else mounted). It
receives every trial in real time and gives one allocation per trial.

- **Brier score:** the multiclass Brier score Σₖ (pₖ − yₖ)², from 0 (perfect) to 2. The uniform allocation scores
  1 − 1/F (11/12 for the d12).
- **Accuracy:** top-face accuracy; a tie shares the credit equally.
- **Latency:** the median answer time after release, over on-time answers.
- **Status counts:** `on_time`, `late`, `malformed` and `missing`.
- **Missed answers:** late, malformed and missing answers are scored as the uniform allocation. They carry no
  information, but are never better than uniform. `predictor_ready: false` means the program never started; every trial
  then counts as uniform.
- Results are reported overall and per protocol, with the face probabilities depending on how the die is thrown.

**Baselines,** on the same trials:

- `uniform`.
- `empirical`: per-protocol face frequencies from `[die] empirical_throws` separate source throws (stream 2).
- `mechanics`, the fitted deterministic mechanics model: the true source rigid-body model, i.e. the best possible fit
  of geometry, mass distribution and contact. It is simulated once from the release state *as observed* with Gaussian
  error (`[die] mechanics_noise`: 1 mm, 0.02 rad, 0.02 m/s, 0.5 rad/s per axis; stream 3), and it predicts that outcome
  with certainty. It measures how much an exact model gains from an imperfectly measured release.
- `oracle`: the same model from the exact release state, in a separately built simulation. It scores Brier 0 exactly
  when the source is deterministic, so every evaluation re-checks determinism.

**Artifacts** (researcher-only):

- `RUN/die/trials.json`: release states, trajectories, outcomes, baselines and geometry.
- `RUN/die/frames/NNNN.json`: the streamed frames and the final image.
- Per version: `die-predictions.json` (per-trial status, latency, allocation, estimates, Brier and accuracy for every
  allocation) and `evaluation.json`; in `RUN/predictor-NNN/`, `predictor-output.jsonl` (every output line with its time)
  and `predictor.stderr.log` (the stream-mode runner of `src/sim2sim/predictor.py`).

**View.** `RUN/die-view.html` is self-contained and rewritten after every evaluation; regenerate it with
`python -m sim2sim.die RUN`. It shows:

- predicted versus observed face frequencies, with the empirical baseline;
- a calibration plot of binned allocations;
- Brier score versus experiment wall time over the run's checkpoints, with the baseline levels and a table;
- a protocol and checkpoint selector;
- replay of individual trials: the die wireframe projected through the trusted camera over the release or final frame;
- evaluator-only overlays: true COM, release velocity and angular velocity, and the agent's optional `estimates`
  (dashed).

**Limitations.**

- Trials are fixed per run, so checkpoints are compared on identical throws.
- The mechanics noise is a researcher choice, not an estimate of what the frames allow.
- Wall-clock streaming makes latency depend on the node's load: CPU starvation of the predictor container shows up as
  late answers.

## Trivial baselines and normalization

`sim2sim.baselines` gives error metrics a scale that compares across
environments. `baseline_archive` builds two candidates from the trusted source:
`empty` (the world fixtures with every object body removed) and `bounding_box`
(one free box filling the objects' reset world bounding box with MuJoCo's built-in
density 1000 kg/m³ and friction `1 0.005 0.0001`). Panda baselines carry the
unchanged public robot. `sbatch scripts/baselines.slurm experiments/runs/RUN_ID ...`
evaluates both with the run's recorded trusted configuration and hash-verified
source snapshot, as `candidate_history` does, into `RUN_DIR/baseline-scores.json`.
`normalize(scores, baseline_scores)` divides every error leaf (fields containing
`error`, `mse`, `_mae_`, `difference`, or `chamfer_distance_m`) by the same leaf
of a baseline: 1 is no better than the trivial scene and 0 is exact. A baseline
that cannot produce a value (an empty scene's Chamfer or COM, hinges without a
hinge, or a zero baseline error) leaves a gap, never a ratio. Hinge relative
errors already compare with the zero-parameter default.

## Low-rank parameter answer (`lowrank`)

`lowrank` (selector; needs a private manifest `[lowrank]` table naming a `body` and a spring `joint`; `evaluation._lowrank`)
scores a JSON answer instead of a scene and combines with no other metric. The answer holds exactly `mass` (kg),
`stiffness` (N/m), `com` (world m) and `inertia` (3x3, kg m², about the COM in world axes), the last two in the scene's
initial state (`qpos0`; keyframed scenes are rejected). Truth comes from the fresh source: the body's mass, its world
COM, R·diag(I)·Rᵀ, and the joint's stiffness. Scores: relative mass and stiffness errors, COM distance (m), and the
inertia's relative Frobenius error; the answer and truth are kept with them. A malformed answer fails the evaluation.

## Positioning fixture (`fixture`)

`fixture` (selector; needs a private manifest `[fixture]` table and a predictor submission, see
[predictor submissions](protocol.md#predictor-submissions-srcsim2simpredictorpy); `src/sim2sim/fixture.py`) scores a
calibrated model of the positioning-fixture levels and combines with no scene metric. The isolated predictor first
answers `settings()` and `predict(compression, pressure, n = samples)` at every held-out setting and at each chosen
setting. The evaluator then releases the fresh source fixture repeatedly: each trial starts from a world reset, sets both
dials directly to the setting (joint positions, at rest), runs `reset_carriage` with a perturbation drawn from
`numpy.random.default_rng([seed, setting index, trial])` (the same distribution as the agent's resets, a different
stream), releases, and runs until the carriage moves slower than 1 mm/s for 0.1 s (or `max_release_ticks`). The stop is
the carriage index in ruler mm.

- `prediction`: at the `settings` grid (25 seeded points, one per cell of a 5 × 5 grid over both dials, `repeats` = 16
  releases each): `crps_mm`, the mean sample CRPS `mean_y(E|X − y|) − E|X − X'|/2` of the predicted samples against each
  release; baselines from the same releases: `oracle_crps_mm` (the CRPS the true distribution would get, half the mean
  pairwise release difference), `mean_only_crps_mm` (a point mass at each setting's true mean), `pooled_crps_mm` (all grid
  releases pooled, a settings-blind predictor) and `crps_skill_vs_pooled = 1 − crps / pooled`. `brier[target]` scores
  the predicted hit probability (fraction of samples inside the region) against each release's hit, with `mean_only`
  (1 if the true mean lies inside) and `oracle` (`p(1 − p)` at the true rate). Any failed or malformed query makes
  `status = failed` with null predicted scores (`failed_queries`, first `error`), never a partial average.
- `targets[name]`: the chosen setting released `chosen_repeats` = 40 times: `measured_hit_fraction`, `hits`, `trials`,
  `wilson95`, and `predicted_hit_fraction` from the predictor at that setting. Context from the grid releases:
  `grid_best_hit_fraction` and `mean_choice_hit_fraction` (the grid setting whose mean is nearest the region centre, the
  choice a mean-only model would make). A missing or invalid `settings()` gives null hit fractions with `error`.
- `view`: `{path, sha256}` of the researcher view.

Private artifacts under `artifacts/<digest>/fixture/`: `trials.json` (every release's stop, settle-and-release
trajectory sampled every `trajectory_every` ticks, the predicted samples, call errors and all scores) and `view.html`
(`src/sim2sim/fixture_view.py`, self-contained, no network): per target the predicted distribution (line) over the
histogram of measured stops with the target band and predicted vs measured hit-rate bars with the Wilson interval; a
2-D map of all held-out and chosen settings coloured by predicted success, measured success or their difference (toggles;
target selector); clicking a setting shows its histograms and replays all its trials together, carriage trajectories
overlaid in time and as moving markers on a rail strip. The dashboard links each submission's view under "Verified
original recordings" after checking its hash against the audited report; otherwise open
`RUN/version-NNN/evaluation/artifacts/<digest>/fixture/view.html` directly. Evaluation cost: 480 releases of a
Panda scene at 22–31 µs per step (about 0.2–3 s of simulated time each), plus the predictor's own calls; configs allow
1800 s. The strict end-to-end path (agent → fixture actions → predictor container → evaluator container) is covered by
`tests/test_oci.py::test_fixture_level_predictor_submission_end_to_end` with a reduced table.

## Gear train (`gear_train`)

`gear_train` (selector; needs a private manifest `[gear_train]` table and `[[probes]]` with `settle_ticks`;
`src/sim2sim/gear_train.py`) scores a scene submission for the explanation-depth level `environments/gear_train`
([design](level-designs/gear-train-depth-20261124.md)) against the deep source and the private shallow reference
(`shallow_reference`, `shallow.xml`: crank and hand only, one coupling, the gears' inertia and damping lumped into the crank
joint). The candidate must name its crank and dial-hand bodies `crank` and `hand` (the public task says so); without them the
evaluation fails with that error. It combines with other scene metrics but the level's configs select it alone: the generic
`dynamics` image metric scores every visible body and so can never rate a model without gears like one with them.

- `accuracy`: each probe (manifest `body`, `local_point`, `moves` = `[dx, dy, dz, ticks]` list, with a weld optionally `[dx, dy, dz, ticks, rx, ry, rz]` adding a world rotation vector in rad, `settle_ticks`) grasps the
  body point's pixel through the source camera, runs its moves, releases and waits, from reset, in the fresh source and the
  candidate, both with the run's public grasp settings including `grasp`, `grasp_slip_m` and `grasp_slip_rad`. The level's moves follow the crank handle's
  or hand tip's circle in 5° chords, within reach and below the slip threshold; a source grip that misses or slips fails the
  evaluation, a candidate's is kept as `candidate_grasp_held = false` and its remaining moves become waits. Unlike the
  other scene metrics the candidate keeps **its own submitted timestep**: each action lasts the same time (rounded to its
  steps) and its samples are interpolated to the source's. Every step records the **world** rotation of `crank` and `hand` about the panel normal (the source crank's hinge axis) relative to
  their reset orientation, unwrapped, so a flipped hinge axis, another zero angle or other joint names cost nothing. Per probe
  and body the RMS difference (rad) and the source's range are kept; `probe_error_rad` is their mean over probes and both bodies.
- `structure`: hinge bodies other than crank and hand whose axis is within 5° of the panel normal are axles. Source gears are
  matched one-to-one to candidate axles by axle position in the panel plane within `axle_tolerance_m` (nearest first;
  names, sizes and masses play no part). `depth_class` is `shallow` (no gear matched), `deep` (≥ 80 % matched) or `partial`;
  also hinge/axle body counts, unmatched candidate axles and couplings (equality constraints by type, tendons).
- `sim_cost`: candidate, shallow and deep reference, each compiled as submitted (no evaluation grasp targets) at **its own
  timestep** (the deep reference at the source's, the shallow reference at the largest step that passes the gate, the
  candidate at its submitted one: big steps are part of why a shallow model is cheap), from rest with a constant crank torque `cost_torque_nm` about the panel normal: wall seconds per simulated second
  of `mj_step(nstep)` over `cost_seconds`, `cost_runs` runs interleaved across the three models, median (`runs` kept), and
  `relative_to_shallow`; plus body, joint, dof, equality, tendon, geom, peak contact and peak constraint-row counts. Times
  come from the evaluation worker (2 CPUs) and are only comparable within one report.
- `cheap` (G-cheap): `status = fails_accuracy_gate` with `cost_relative_to_shallow = null` when `probe_error_rad` exceeds
  `cheap_gate_rad` (never a zero score); otherwise `cost_relative_to_shallow`. The gate is the **worst** probe error of the
  shallow reference with its lumped armature and damping each scaled by 0.9 or 1.1 (four combinations), computed by
  `experiments/validation/gear-scenes.py` into the manifest. `probe_error_rad` is always kept, so another gate can be applied
  to recorded reports.
- `removal` (G-deep, reported for every condition): for each source gear in `removals` the source loses that body and the
  candidate its matched body, each with its subtree and every equality, tendon, actuator, sensor and contact element naming
  anything inside it (repeated to a fixpoint; keyframes are dropped), then the probes rerun. A gear without a match is
  `cannot_answer`, a gap that never becomes an error value; a candidate whose edited model cannot run is `failed` with its
  error. `answered`, `cannot_answer`, `failed` and `mean_probe_error_rad` (answered gears only, else null) summarize them.

Artifacts under `artifacts/<digest>/gear_train/`: `probes.json` (actions and both worlds' rotations every 5 ticks) and
`scores.json`. Not yet built: the design's H4 trace classifier (did the agent build and time models of two depths) and a
dashboard per-version structure summary; read `scores.gear_train.structure` and `sim_cost` from the private reports.
