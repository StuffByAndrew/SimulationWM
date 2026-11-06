# Private evaluation

`sim2sim.evaluation.evaluate_submission(zip_bytes, config)` runs in the resource-limited evaluation OCI worker, never in the agent container. The runtime excludes scoring from the active timer and retains reports privately. The evaluator creates a fresh source simulation, independent of the interactive state. Container memory, CPU, process and wall-clock limits are mandatory because native XML/asset parsing is not a security boundary.

Configuration supplies `source_dir`, `output_dir`, optional `source_entrypoint` and candidate `entrypoint` (default `scene.xml`), camera/render settings, and private `probes`. A source-relative `manifest` defaults to `manifest.toml`; its `probes`, `correspondence`, and `camera` populate absent configuration fields. Correspondence is optional. Missing corresponding bodies explicitly fail the privileged parameter component while generic visual evaluation continues. Other invalid submissions, compilation errors, and failed probe execution yield `status: failed`, an explicit error and no scores.

The candidate uses the trusted source camera position, orientation and intrinsics. Each private probe supplies a pixel, or a source body and local point projected to a pixel. Source and candidate independently receive the same reset, pixel grasp, four displacement samples, release and three waiting samples. A missed candidate grasp advances time without motion; no privileged body attachment is used. RGB, depth and segmentation-derived object silhouettes are compared at every sample. Masks include visible geometry on non-world bodies and exclude world geometry such as the floor. This does not provide per-object segmentation correspondence.

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

`scores.reconstruction.chamfer_distance_m` is the symmetric mean **unsquared** nearest-neighbor distance in meters: half the sum of each sampled surface's mean distance to the other sampled surface. Samples represent the **exposed union surface** of supported object primitives (sphere, box, ellipsoid, cylinder and capsule); proposal points strictly inside another constituent primitive are rejected. They are area weighted across constituents, and ellipsoid samples use the sphere-transform area Jacobian. Ellipsoid area weights use deterministic numerical quadrature. Internal overlap surfaces and floor surfaces do not enter the Chamfer calculation. This is a finite sampled-surface estimate; increasing samples improves spatial resolution. Canonical geometry ordering excludes names, density and color, so identical geometry with renamed or reordered constituents yields the same samples and zero distance within numerical precision.

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

Panda submissions include an unchanged copy of the public robot fixture.
The evaluator compares it byte-for-byte to the trusted public fixture and
requires a single `robot/panda.xml` include. The arm is an instrument, not a
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
