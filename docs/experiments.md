# Experiments

Dated record of batches: [experiment-log.md](experiment-log.md).

The shared store is `experiments/`: editable TOML specifications in `configs/`,
private attempt records in `runs/<id>/`, generated dashboard in `dashboard/`, and
trusted development checks in `validation/`. Existing historical `runs/` fixtures
remain historical evidence and are not Astra pilots.

The [Astra random-assembly preset](../experiments/configs/astra-random-assembly.toml)
uses exactly `gpt-6-astra` through the host Responses broker, with a hard
1800-second wall limit, 12000 shared physics ticks, six submissions and explicit
API limits (100 calls, 16384 output tokens per call and a 200000 reported-total
token guard). RGB, calibration and optional grasp wrench are enabled; depth is off.
No source counts, masses, densities, joint values or private descriptions are
included in the agent goal. A returned provider summary is labeled as such;
timestamped agent commentary contains only what the agent explicitly emits.

## Configure and run

`[experiment]` requires a display name, environment label, text-valued private
`physical_parameters`, model-constraint strings and an `evaluation_metrics`
array. `[source]`, `[sandbox]`, `[budget]`, `[simulation]`, `[evaluation]` and
`[inference]` retain the native runtime settings. The wrapper removes experiment
metadata before calling the runtime and appends only explicit model constraints
to the public task. Model constraints are public task instructions, not a
claim that every modeling choice is machine enforced. Paths resolve from the
invocation directory; the batch script invokes from the repository root.

From the repository root, inside a Slurm `test` allocation:

```sh
.venv/bin/sim2sim experiment experiments/configs/astra-random-assembly.toml --id astra-pilot-001
# Or submit the same command through the repository batch script:
sbatch scripts/experiment.slurm experiments/configs/astra-random-assembly.toml --id astra-pilot-001
```

Use a fresh ID for each attempt. A prepared run contains `experiment.json`,
`config-private.json`, `status.json` and a private `source/` copy of its scene,
manifest and assets. The runtime records additional immutable versions, audit,
videos and metrics there. The wrapper records status/preflight and a boundary
report even when the run is blocked before agent launch.

To change actual physical values without modifying a shared environment:

```sh
.venv/bin/sim2sim experiment CONFIG.toml --id density-study-001 --prepare-only
# Edit experiments/runs/density-study-001/source/scene.xml and its private manifest.
# Describe the resolved values in experiment.json physical_parameters.
.venv/bin/sim2sim experiment --start density-study-001
```

Changing metadata alone does not change physics. Apply requested density, mass,
friction, damping or joint friction-loss attributes to the private MJCF and verify
them before starting. Explicit geom mass overrides density. The final entrypoint and complete source-tree hashes
and whether the tree changed after preparation are recorded. Native checks require
Slurm; source inspection and configuration checks are lightweight.

### Seeded color perturbations

An optional private `[perturbation]` table recolors object geoms in the prepared
`source/` copy (never `environments/`), e.g. as a vision leak canary:

```toml
[perturbation]
seed = 20261102
colors = "random"          # or "uniform_gray": every object geom rgba 0.4537 0.4537 0.4537 1
```

Object roots are the manifest `object_bodies`, else its `[correspondence]`
keys; each must be a top-level scene body, and its subtree geoms must match any
declared `object_geoms`. Robot, table, floor and fixture geoms are untouched.
Authored colors, including default classes, are resolved by MuJoCo's spec
parser; object geoms with materials are rejected because a material could
override or tint the rgba. `random` draws opaque HSV colors (saturation and value
in [0.5, 1]) with four decimals and no channel within 0.0005 of a multiple of 0.05,
at least 0.35 RGB from the geom's own authored color and from a fixed palette of
stereotypical colors (pure and CSS red, green, blue, yellow, orange, purple/violet,
cyan, magenta, white, black, MuJoCo default 0.5 gray), and 0.25 from fixtures and
each other. The record stores this rule as `colors.rule` (`PALETTE`/`COLOR_RULE`
in `perturbation.py`). Runs prepared before 2026-11-03 used the earlier saturated,
hue-spaced rule without the palette; their recorded values remain authoritative. `uniform_gray` makes object geoms indistinguishable by color (e.g. the
tetrahedral spatial-memory ablation). Parsing drops XML comments in the perturbed copy.

`pose = "jitter"` (alone or with `colors`/`physics`; applied last) gives replicates
a different layout of the same objects. With `random.Random(f"{seed}/pose")` it
moves all object bodies together as one rigid group: one planar offset
(±0.03 m in x and y) and one yaw (±15°) about the vertical through their
centroid. Heights, the objects' poses relative to each other and every
non-object body stay unchanged, so stacking and contacts between objects hold
and a yaw about the vertical keeps each lowest point. A draw with object contact
deeper than 1 mm, or where any private grasp probe's pixel no longer raycasts
onto its body, is redrawn (at most 100 attempts). It requires exactly one free
joint on each top-level object body: hinged or welded objects are anchored to
world fixtures, and Panda scenes are refused because their arm probes target
fixed object poses. The record (`perturbation.pose`) keeps the range, offset,
yaw, centroid and every original and jittered body position and quaternion;
`verify` checks them in the compiled model, plus non-penetration, settling and
probe hits.

`physics = "nonround"` (alone or with `colors`) makes a hard variant whose values
careful measurement plus rounding cannot recover exactly. With
`random.Random(f"{seed}/physics")` it rescales every object geom size component
(box, sphere, capsule, cylinder, ellipsoid; ×[0.8, 1.25]), its mass, or its
density when no mass is set (×[0.6, 1.6]), and nonzero object hinge damping,
frictionloss and stiffness (×[0.6, 1.6]; zero stays zero). Each value is rounded
to four significant digits and resampled until its mantissa m (1000–9999) has a
last digit other than 0/5 and lies more than 5 from a multiple of 50 and 50 from
a multiple of 500 (for 0.01–0.1: not within 0.0005 of a multiple of 0.005).
Geom positions, `fromto` endpoints (capsule lengths), joint anchors/axes, ranges,
friction and robot/fixture geoms stay unchanged, so probes and manifests stay
valid. Free or welded object roots are moved only in z so their lowest point keeps
its authored height; hinged roots stay at their axes. A draw with any object
contact deeper than 1 mm is redrawn (at most 100 attempts). Meshes, explicit
`<inertial>`, unnamed/slide/ball object joints, polynomial damping/stiffness and
scenes whose authored reset already has >1 mm object penetration are refused
(`panda/tetrahedral_cluster` did until its 2026-11-03 cluster move; see
`docs/environments.md`). Penetration is measured at the simulator's reset pose (Panda home), so
preparation compiles the scene: prepare such runs on a `test` node.
Evaluation always compares against the perturbed private `source/` copy.

`experiment.json` records `perturbation` (exact original and written rgba,
sizes, masses/densities, hinge values and shifted body positions, object geoms,
seed and original scene hash) and text `perturbation.colors`/`perturbation.physics`
entries in `physical_parameters`. The table is removed before the runtime sees
the configuration. `--start` runs `perturbation.verify` on its Slurm node
before launch: it compiles both scenes and checks compiled object rgba, no
material, unchanged fixture colors and a finite 500-tick settle (identical for
colors alone). With physics it also checks compiled sizes, body masses
(mass or density × primitive volume), hinge values, non-roundness, unchanged
anchors and other geoms, ≤1 mm initial penetration after reset, settle
displacement at most 0.1 m beyond the original's, point-probe rays still hitting
their bodies and Panda arm probes still contacting their targets. Its report
goes to `status.json`.

## Select evaluation

Choose supported families with `experiment.evaluation_metrics`:

| Family | Computation |
| --- | --- |
| `appearance` | Reset RGB, depth and object silhouette errors. |
| `dynamics` | Private interaction rollouts, motion/image errors and missed grasps. |
| `reconstruction` | Legacy bundle: exposed-union Chamfer, geometry-matched primitive density and COM. |
| `chamfer` | Total exposed-union symmetric Chamfer only. |
| `center_of_mass` | Compiled object COM position/error and full vectors only. |
| `density` | Geometry-matched per-primitive density measurements/errors only. |
| `parameters` | Optional declared body correspondence parameters. |
| `hinges` | Declared or single-hinge damping/friction-loss comparisons. |

Disabled families skip their work, including dynamics probes. Omitting the native
`evaluation.metrics` selector preserves previous defaults; selecting
`reconstruction` enables it even when the source manifest omitted or disabled
that opt-in; explicit family selection takes precedence.
The pilot selects appearance, dynamics and reconstruction. Generic geometry,
density and COM do not require hidden body names. Optional `parameters` can be
unavailable when an independently named reconstruction lacks declared body
correspondence; that is not a generic reconstruction failure.

## View and compare

```sh
.venv/bin/sim2sim dashboard --root experiments
.venv/bin/sim2sim replay experiments/runs/ID --output experiments/runs/ID/replay
```

The dashboard reads private recorded artifacts and active snapshots; it does not
run a model or simulation. Its separate interactive physics tab does simulate,
only on a Slurm `test` node started by `scripts/serve-dashboard.sh`; see
[the dashboard guide](dashboard.md#interactive-physics-inspector). Replay verifies stored bytes and exports exact frames,
state and wire records without regenerating absent evidence. Keep blocked runs,
partial execution and unavailable metrics visibly distinct from successful
experiments. The dashboard and all parameter descriptions remain private.

## Pilot prerequisites

For `strict_oci`, the host must have the configured credential variable, checked
by presence only, and a prebuilt researcher-trusted image. The image must exclude
source scenes, private experiment files and host credentials. Actual finite
CPU/memory/process limits must be verified before its agent starts. Selecting
the `rlimit` process backend cannot repair missing CPU/memory enforcement; on
Slurm use `slurm_step` with `scripts/experiment-strict.slurm`.
Earlier cluster diagnostics found unenforced CPU/memory bounds; a strict pilot
must be reported as blocked unless its own current preflight succeeds. Never
substitute the development backend automatically for a blocked strict run.

## Explicit development backend

The [development preset](../experiments/configs/astra-random-assembly-development.toml)
uses the same scene, observations, metrics and 1800-second wall budget, but omits
API inference settings. Its `execution_mode = "development_shared_filesystem"`
requires the deliberate `--development` flag:

```sh
sbatch scripts/experiment.slurm experiments/configs/astra-random-assembly-development.toml --id astra-development-001 --development
```

This mode is for the explicitly approved spawned-agent development workflow.
It is **not OS isolated**: subagents share the workspace and can technically
access sibling private files. Giving the model only `public/task.json` and the
curated `public/rpc.py` bridge is a usage convention, not filesystem access
control. Source/parameter confidentiality cannot be guaranteed in this mode.
The dashboard labels the execution mode instead of calling it isolated.

The controller starts on Slurm and exposes only permitted observations/actions
through a public request/response directory. The orchestrator starts a fresh
`gpt-6-astra` subagent without inherited private history after the bridge is ready,
and interrupts it at the controller deadline or STOPPED signal. The controller
cannot interrupt an independently spawned agent by itself. Native model workers
remain bounded and validated; no API credential is copied or required. This is
never an automatic fallback when strict OCI preflight fails. The strict API
preset is retained for later use on a compatible host.

The development preset exposes `[development_worker]` limits: 4 GiB address
space, 1 GiB data, 128 MiB output file size, 256 open files and EGL rendering.
Each native worker also receives a CPU-time limit derived from the controller's
remaining wall budget and is pinned to one available CPU. These limits bound
worker resource use; they do not isolate filesystem access or the external
spawned model agent.

Current production evidence: job `56063002` built the current trusted image,
passed 126 tests (one OCI test skipped), and exercised the Chromium dashboard.
Image `d991f5411f7be34293e0bf3f986c10d92db56d27d3f5c9e2c17789819b5eaf46`
inspection found no environment/configuration/experiment/source/credential
payloads and no host bind mounts. Actual resource inspection still returned
`memory.max=max` and `cpu.max=max 100000`, so Podman-cgroup strict execution is
blocked there. The `slurm_step` backend (Slurm-enforced step cgroups) passed OCI
integration in job `57174710`; see `experiments/validation/sandbox-backend/`.
No API pilot ran in that validation job.

Slurm `test` job `56066912` subsequently passed 149 tests with one OCI skip in
82.76 seconds. A real prepared experiment then exposed an NFS `.nfs` guard
failure. A narrow exception for a regular, non-symlink hexadecimal `.nfs` file
passed 15 focused preparation, start, development and metadata checks in Slurm
job `56069922` (24.26 seconds). The full suite preceded that fix. Neither
validation job launched a model pilot.

## Recorded development pilot

The explicitly authorized `gpt-6-astra` development attempt
`experiments/runs/astra-development-20261028-002/` completed on Slurm `test` job
`56069945` with `development_shared_filesystem`. The controller recorded
`agent_finish` after 1698.70 seconds of execution wall time, within its
1800-second limit, with three accepted submission checkpoints and 3000 physics
ticks (1400 reference, 1600 reconstruction). The external model returned its
final report after finishing through the public RPC. The Slurm wrapper completed
postprocessing with exit code 0. This is a real model development pilot, with
shared filesystem access and no enforced OS privacy boundary; it is not the
strict OCI/API pilot. Private per-checkpoint scores and source values remain in
the run artifacts. Public RPC records, explicit progress notes and provider
summaries do not constitute the external agent's full tool transcript.

Artifact-only verification in Slurm `test` job `56073575` passed after the
wrapper completed. The sealed audit contained 6346 verified events; fresh exact
replay exported the same event count and 666 recorded frames across both worlds.
All four indexed MP4s decoded at their declared 30 fps with frame counts
85, 85, 13 and 1. A refreshed dashboard showed a closed, verified snapshot;
Chromium selected the pilot and sought its final recorded wall time. The private
report and screenshot are under `experiments/validation/pilot002-final-*`, and
the exact replay is in the run's `replay/` directory. This check only read and
replayed recorded evidence.

## Two-hour medium Astra environment batch

The `astra-medium-*-2h.toml` presets cover all six current environments:
`blocks`, `spring`, `damped_lever`, `tetrahedral_cluster`, `random_assembly`,
and `primitive_boat`. They use exactly `gpt-6-astra`, medium reasoning, and the
explicitly authorized `development_shared_filesystem` backend. Spawn each model
with `fork_turns="none"` and `reasoning_effort="medium"`, giving it only its
public task and RPC paths. The model identity/effort in configuration records
intent; the orchestrator must actually select both when spawning.

`budget.wall_only = true` requires finite `wall_seconds = 7200`. Active-time,
cumulative tick and submission-count quotas are disabled and reported as `null`.
Actual cumulative ticks and submission counts are still recorded; resets never
clear these counters. Two hours includes initialization, external inference,
interaction and private scoring. A model may finish earlier. Byte/archive,
worker memory and per-request safety bounds remain enforced; these are not
simulation tick or checkpoint quotas. The audit allowance is an upper bound of
8 GiB, not a preallocated file. Keep actions reasonably sized to fit the bounded
audit reservation and remaining wall time.

| Environments | Computed metrics |
| --- | --- |
| Random assembly, primitive boat, blocks | Total Chamfer and compiled COM |
| Tetrahedral spheres | Total Chamfer and each ball's density |
| Spring, damped lever | Viscous hinge damping and dry friction loss |

Hinge friction comprises the viscous coefficient (`damping`, N·m·s/rad) and dry
friction torque (`frictionloss`, N·m); neither is a dimensionless contact μ.
Single-hinge matching permits independent joint names and remains unavailable
until both models contain exactly one hinge. Density matching in these presets
requires the same primitive type and surface matching distance at most 0.08 m;
rejected assignments produce NaN gaps. It never matches by color or density.

Both ten-primitive runs use a **uniform 500 kg/m³** in their prepared private
MJCF copies, with geom mass overrides removed. Shared environment originals
remain reusable. For another uniform-density attempt, prepare the selected
configuration with `--prepare-only`, set `density="500"` on all ten object geoms
in that attempt's private `source/scene.xml`, remove their `mass` attributes,
and then start the prepared ID. This physical edit is part of preparation; the
TOML private description alone does not apply it. Compile/check the copied
source on Slurm before launch. The batch record in `experiments/validation/` identifies each
prepared source and actual launch; preparation alone is not an executed eval.

Start each prepared run in its own job:

```sh
sbatch scripts/experiment-two-hour.slurm PREPARED_RUN_ID
```

The wrapper allocates 2 h 10 min on `test`: a 2 h native execution deadline and
10 min for archive finalization. Start the external model only when that run's
`public/task.json` exists. Interrupt it at `deadline_unix` or `stopped.json`.
Do not consume a run's two hours waiting for an available model slot. With
three concurrent model slots, launch two batches of three independent jobs.
Slurm dependencies start controllers, not external agents: the dispatcher must
remain available to launch each fresh model when its public task appears.
Resolve scheduler permissions before starting dependent controllers. If a
controller expires without its model, preserve that attempt and use a new run
ID; never reset or extend the expired run's budget in place.
Each job defers dashboard generation; refresh the dashboard after a batch using
a `test` allocation for any browser rendering. This is not strict OCI/API
execution and does not use host inference credentials.

Prelaunch validation, 2026-10-30: Slurm `test` job `56344135` compiled all
six prepared source scenes, verified both uniform-density compiled masses, and
passed **161 tests, 1 skipped** (OCI only) in 88.37 seconds. A preceding native
mailbox regression timed out while generating optional PNG plots; its private
evaluation itself succeeded. The mailbox regression now uses the same
JSON-only checkpoint mode as the new batch. Standalone plotting retains its
separate regression coverage. These validations are not model eval results.

The executed batch produced **13 accepted checkpoints across all six
environments**. Each native controller finished through the public `finish`
operation within its 7200-second deadline. Five runs completed on 2026-10-30;
the replacement damped-lever run completed on 2026-11-02 (UTC).

| Environment | Slurm job | Native wall time (min) | Accepted checkpoints |
| --- | --- | ---: | ---: |
| Random assembly | `56344207` | 117.7 | 3 |
| Primitive boat | `56344209` | 69.2 | 2 |
| Tetrahedral cluster | `56344212` | 84.7 | 3 |
| Damped lever | `56976448` | 46.4 | 2 |
| Spring | `56346040` | 40.9 | 2 |
| Blocks | `56350726` | 22.6 | 1 |

The original damped-lever controller, job `56346037`, started while the
dispatcher was waiting for scheduler-command approval. No damped-lever model
was spawned for that attempt; its controller expired at two hours with zero
ticks and zero submissions. That failed attempt remains archived. The fresh
attempt is `astra-medium-damped_lever-20261101-001`; its clock was not extended
or reset from the failed attempt. Runtime model-call fields remain unknown;
the separate orchestrator records identify the actual model spawns.

These are explicitly authorized shared-filesystem development runs, without
an enforced filesystem isolation boundary. External final reports are saved
where returned; the blocks agent was interrupted after its controller stopped,
so its native finish record is available without an external final report.
Full external tool transcripts and token counts are unavailable. The
[batch record](../experiments/validation/astra-medium-20261029-batch.json)
contains the actual run IDs, job IDs, selected model/effort and superseded attempt.

Final verification in Slurm `test` job `56977064` passed **34 dashboard/replay
tests** in 3.68 seconds and verified sealed, complete audits for all six runs.
Slurm accounting records all six successful jobs as `COMPLETED`, exit `0:0`.
The final view-only check, job `56984168`, passed both browser-control tests
(7.45 seconds) and four checks on recorded data: grouped metrics, hinge friction,
the grasp-overlay toggle/path, and grouped tetrahedral density curves. This
last check refreshed static view assets using the saved verified snapshots;
it did not repeat the archive scan or simulate anything. Both validation jobs
completed. The [verification report](../experiments/validation/medium-final-report.json)
contains actual timestamps, counts and accounting output.

The six current audit archives contain 138,986,579 logical bytes in 91,170
regular files; the filesystem reports 3,082,305,536 allocated file bytes.
The dashboard catalog's unique verified assets across its recorded runs total
12,101,761 logical bytes in 832 assets. These figures exclude stale cache files
and other experiment-directory contents. All six new runs have zero duplicate
MP4s and no per-checkpoint metric PNGs. An earlier artifact check timed out in an
optional recursive `du` scan; storage reporting now reuses the mandatory audit
scan and already verified asset sizes, avoiding that extra traversal.

## Panda arm six-hour batch

The eight `astra-medium-panda-*-2h.toml` presets use the pinned public Panda
fixture and the explicit `development_shared_filesystem` backend. The six
existing-scene variants expose RGB and calibration alongside arm
proprioception. `blocks_proprio` and `fixed_cube_proprio` expose arm
proprioception only. Each preset retains `gpt-6-astra`, medium reasoning and a
21600-second wall-only budget. They select the corresponding existing metric
family plus private Panda arm dynamics probes. Source scenes and manifests
remain private; only the robot fixture is staged as public development input.
The `-2h` filenames predate the approved six-hour batch and are retained so
prepared experiment records continue to refer to the exact configuration path;
the recorded `wall_seconds` value is authoritative.

The eight clean-agent development attempts use separate `test` jobs through
`scripts/experiment-six-hour.slurm`, with up to three models active at once.
The public bridge is made available before each fresh model starts so waiting
for a model slot never consumes that run's six-hour controller window. Strict
OCI/API mode remains subject to its existing CPU/memory enforcement preflight.

The 2026-11-02 batch completed all eight native controllers via `agent_finish`.
Each run had one accepted submission checkpoint, except vision blocks, which
had two; all nine private evaluation checkpoints recorded status `ok`. The
individual Slurm job IDs, finish timestamps and counters are in
[`panda-six-hour-20261102-batch.json`](../experiments/validation/panda-six-hour-20261102-batch.json).
These are shared-filesystem development runs, not strict OCI/API executions.

## Continuing a run at its spend cap (budget hold)

To decide whether a run deserves more time without a special-cased continuation, configure the full wall budget the
run may ever get (`budget.wall_seconds`, visible to the agent from the start) with a lower `max_cost_usd`, and set
`[inference] budget_hold_seconds`. At the cap the run holds frozen (containers paused, clocks stopped) until the
orchestrator writes `RUN/hold-decision.json` (`{"action":"resume","max_cost_usd":X,"cost_pool_usd":Y}` or
`{"action":"stop"}`); see [protocol.md](protocol.md#budget-holds-continuing-a-run-without-a-special-case). Submit the
Slurm job with `--time` covering the wall budget, the hold limit and finalization. A held run occupies its
allocation; the daily spend rule applies to the money spent after resuming, so raise `cost_pool_usd` accordingly.

## Continuing a finished attempt

(The section below is the older, separate-session mechanism. Prefer a budget hold for new runs.)

Accepted `submission-NNN.zip` projects and the corresponding immutable versions
can seed a new attempt. A finished controller cannot be restarted in place: use a
new run ID with its own fresh budget; the parent is never modified. The saved
project resumes the reconstruction, not a serialized external model conversation
or live physics state. Never provide private scores, source snapshots or probe
artifacts as continuation context.

Strict OCI runs implement this with a `[continuation]` table in a copy of the
parent's configuration (development mode does not support it):

```toml
[continuation]
parent = "w1-lever-base-20261102-003"  # finished strict run under experiments/runs/
submission = "last"                    # or a 1-based accepted submission number
```

Preparation refuses unless both runs are `strict_oci`; the parent status is
`completed` with result `finished` (agent finish or wall deadline); the selected
`submission-NNN.zip` hash equals its `version-NNN` accepted version; the new
source tree hash and entrypoint equal the parent's resolved ones; the
`[simulation]` table (physics and observation modalities) is identical; and the
public task (including model constraints) is identical. Budgets may differ.
`--start` refuses a continuation whose source was edited after preparation.

Preparation stages `prior/` in the new run: `project/` (the extracted selected
submission, agent-authored) and `progress-notes.txt` (only `source=agent`
progress notes with `model` or `adapter` origin, as `[elapsed origin] text`).
Nothing else is carried over: no scores, metric history, evaluations, source,
manifests, configuration, observations or provider summaries. Preparation fails
if a carried path or byte sequence contains the parent run ID or a non-plain-word
environment identifier (`damped_lever`, `random_assembly`, ...); plain words such
as `blocks` or `spring` are physical descriptions the agent may legitimately use.
`experiment.json` records `continuation` (parent, submission number and sha256,
parent status/result/reason, parent source tree hash, public file hashes). The
runtime re-hashes `prior/`, records a `public_fixture` audit event and mounts it
as a fresh read-only volume at `/opt/public/prior` (container-root owned). The
task gains one fixed sentence naming that path; the rest equals the parent task.

```sh
cp experiments/configs/wave1-20261102/w1-lever-base.toml experiments/configs/wave1-20261102/c1-lever-base-003.toml
# edit the copy: budget.wall_seconds = 7200; append the [continuation] table above
.venv/bin/python -m sim2sim.cli experiment experiments/configs/wave1-20261102/c1-lever-base-003.toml --id c1-lever-base-20261103-001 --prepare-only
sbatch --time=02:40:00 scripts/experiment-strict.slurm c1-lever-base-20261103-001
```

The current continuation inventory is
[`continuation-inventory.json`](../experiments/validation/continuation-inventory.json).
No continuation model runs were launched when this inventory was created.
Future experiment presets enable private evaluation at each successful candidate
load and include fixed-camera `image_mse`. Historical candidate-change scores are
separate retrospective artifacts, retaining original change times and actual
evaluation timestamps; original sealed audits and submission scores are preserved.

Historical prompt provenance is indexed in
[`prompt-provenance.json`](../experiments/validation/prompt-provenance.json).
All 15 completed model runs retain their original controller task in
`public/task.json`, matching the public task in the previously verified audit
snapshot. One failed attempt also retains its task. Exact external-agent launch
and steering messages were not archived, and retrospective Panda orchestration
notes are not verbatim prompts. Revised presets do not change these records.
Future launches must archive the exact authored launch and steering messages as
specified in `skills/run-experiment/SKILL.md`.

## Optional agent tools (ablations)

`[tools] segmentation = true` and/or `planning = true` (both default false,
strict OCI only) give the agent a tool; omitting it is the matched control. The
tool is a CLI/Python module mounted read-only at `/opt/public/tools/<name>` only
when enabled, and the task gains exactly the fixed sentence in `AGENT_TOOLS`
(`src/sim2sim/runtime.py`); the harness, system prompt and everything else are
unchanged. Staged file hashes are recorded as a `public_fixture` audit event.

- `segmentation`: `tools/segmentation/segment.py` with MobileSAM (Apache-2.0,
  commit `f706ad9c`), fetched once on the host by `scripts/fetch-segmentation.sh`
  into the git-ignored `tools/segmentation/vendor/`; the runtime refuses to stage
  weights whose sha256 is not
  `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`.
  It needs CPU PyTorch, so it requires the derived tools image
  ([`trusted-image-tools.json`](../experiments/images/trusted-image-tools.json):
  `Containerfile.tools` on top of the verified base image, label
  `sim2sim.tool-runtime=torch-cpu`, checked before launch). The image contains no
  tool code or weights. Its archive lives under
  `/n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-images/` to spare the home quota.
  Measured in the 4-CPU/8 GB agent container (Slurm job `57329543`): model load
  3.0 s; image embedding 0.5 s; one prompted CLI call 4.4 s wall / 7 s CPU;
  `--auto` (16x16 grid) 17 s wall / 58 s CPU. Reuse a `Segmenter` in Python to
  pay the load once.
- `planning`: `tools/planning/plan.py` (MuJoCo + NumPy, runs in either image) for
  `robot = "panda"`: damped-least-squares IK for the `panda_ee` fingertip site or
  hand, joint interpolation, and a collision-checked straight-line/RRT-connect
  planner on the agent's own scene, returning `arm_target` waypoints.
  One IK or plan CLI call takes about 0.3–0.4 s including model load.
- `local_controller`: `tools/local_controller/localsim.py` for `robot = "shadow_hand"` (base image): a local copy of the
  reconstruction world for the agent's own project and tactile cell positions for any hand pose; see
  [protocol](protocol.md). Pair it with the harness flag `--bash-mode pausable` for the touch harness (wave 25).

Both conditions of a pair must use the same image, so start both with the tools
record. `SIM2SIM_TRUSTED_IMAGE_RECORD` defaults to `trusted-image.json`, so runs
without the variable are unchanged:

```sh
cp experiments/configs/wave3b-20261104/w3-ps-assembly-fixed.toml experiments/configs/tools/t-ps-assembly-base.toml
# copy again as t-ps-assembly-seg.toml / t-ps-assembly-plan.toml and append: [tools] segmentation = true (or planning = true)
.venv/bin/python -m sim2sim.cli experiment experiments/configs/tools/t-ps-assembly-base.toml --id t-ps-assembly-base-001 --prepare-only
.venv/bin/python -m sim2sim.cli experiment experiments/configs/tools/t-ps-assembly-seg.toml --id t-ps-assembly-seg-001 --prepare-only
SIM2SIM_TRUSTED_IMAGE_RECORD=experiments/images/trusted-image-tools.json \
    sbatch --export=ALL scripts/experiment-strict.slurm t-ps-assembly-base-001 t-ps-assembly-seg-001
```

To rebuild the tools image: `SIM2SIM_CONTAINERFILE=Containerfile.tools
SIM2SIM_BASE_IMAGE_RECORD=experiments/images/trusted-image.json
SIM2SIM_TRUSTED_IMAGE_RECORD=experiments/images/trusted-image-tools.json
SIM2SIM_IMAGE_ARCHIVE_DIR=<dir> sbatch --export=ALL scripts/build-image.slurm`.
