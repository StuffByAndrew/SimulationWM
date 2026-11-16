# API access
~/.env file contains
INFERENCE_BASE_URL and INFERENCE_API_KEY 
This is an endpoint which is essentially the same as open router which is openai compatible. We can use these to run truely sandbox agents for evaluation

# Development continuity

## Persistent user compute instruction

Do not run heavy computation on a login node. Use Slurm's **`test` partition**
for all physics, rendering, full tests, container builds and benchmark runs.
Static source inspection, syntax compilation and lightweight pure validation
checks may run on the login node. This is a user constraint, not a suggestion.
Do not launch local simulations to investigate a failed compute job.

## Architecture and boundaries

- `src/sim2sim/simulation.py`: private MuJoCo model/data, fixed camera rendering,
  raycasts, connect-constraint `PointGrasp`, action stepping and observation
  encoding. Public observations must not disclose model state or ray hits.
- `src/sim2sim/runtime.py`: rootless Podman transport, bounded protocol input,
  cumulative budgets, host inference broker, evaluation worker orchestration,
  researcher logs and video. Keep credentials and source files out of agent
  containers; never silently fall back to an unisolated agent or evaluator.
- `src/sim2sim/evaluation.py`: submission validation, immutable project versions,
  fresh source/candidate rollouts, image and optional corresponding-parameter
  metrics. Untrusted native model parsing belongs in the constrained worker.
- `src/sim2sim/primitive_metrics.py`: exposed primitive-union surface sampling,
  symmetric Chamfer, geometry-only primitive assignment, per-primitive density
  and compiled rigid-body COM comparisons. Do not use density or color to match
  primitives, or count internal overlapping surfaces as exposed geometry.
- `src/sim2sim/plotting.py`: researcher-only checkpoint history JSON and PNG
  curves. Preserve raw per-primitive measurements and COM vectors; unavailable
  measurements must remain gaps rather than fabricated zero errors.
- `src/sim2sim/audit.py`: bounded private event/blob storage and integrity
  verification. Preserve exact wire bytes and actual partial execution, including
  failed actions. Do not truncate or substitute data silently to fit a budget.
- `src/sim2sim/replay.py`: verified playback of archived frames, state and wires,
  with no model parsing, simulation or external API calls. Playback must never
  regenerate missing evidence or silently reinterpret an incomplete log.
- `src/sim2sim/inspector.py`, `inspector_worker.py`: researcher-only Viser
  physics inspector for recorded versions (Slurm `test` node only). Versions come
  from the verified audit; candidate/source MJCF is parsed only in limited
  development workers. Launch via `scripts/serve-dashboard.sh`. Each viewer has a
  researcher-only property editor driven by the `inspector.PROPERTIES` registry
  (opacity in the viewer; mass, friction via a worker `set` op on the live
  MjModel + `mj_setConst`) and an object CoM marker (non-welded, non-robot
  bodies). Edits never touch recorded runs, audits or scores.
- `src/sim2sim/client.py`: agent-facing versioned JSON client.
- `src/sim2sim/sam3d.py`: optional `[tools] sam3d` relay. The strict controller forwards the agent's validated
  image+mask over a private 0700 spool to a researcher GPU service (`scripts/sam3d-service.slurm`,
  `kempner_gpu_priority`, A100 only, never H100/H200; env/weights under netscratch `SAM3D_ROOT`). A missing,
  stale or silent service stops the run; there is no CPU path. Calls are audited (`sam3d_call`).
- `environments/*`: reusable scene XML plus private manifest probes and body
  correspondence. `examples/` contains the protocol demonstration agent.
- `configs/example.toml`, `Containerfile`, `docs/`: runnable configuration,
  reviewed image recipe, protocol and environment/evaluation contracts.

Read the relevant docs before changing these interfaces. Coordinate action
payloads, callback behavior, tick accounting and modality filtering across core,
runtime and evaluator. Resets must never refund cumulative ticks. Evaluation
must use fresh state and trusted camera settings, independent of the agent's
selected observation modalities. Scores and privileged probe artifacts stay
private. A point grasp controls translation without imposing orientation.
Optional `simulation.wrench` defaults false; when disabled no wrench observation
key is exposed. When enabled, report only the active connect constraint's
object-on-handle force in world coordinates and zero torque about the actual
grasp point. Do not disclose grasp coordinates, body COM lever arms, mass, or
forces from unrelated contacts. Refreshing force must not advance time/ticks.

The default sandbox process-limit backend is `cgroup` and requires enforced
CPU, memory and pids controllers. Explicit `sandbox.process_limit_backend =
"rlimit"` is valid only on hosts enforcing CPU/memory but lacking delegated pids:
`pids` becomes hard `RLIMIT_NPROC` for the fixed nonroot subordinate real UID.
Keep capabilities dropped and `no_new_privs` enabled. The bound covers threads
and processes across containers sharing that UID and may cause slot starvation.
`"slurm_step"` (Slurm allocations only) requires integer `cpus` and whole-MiB
`memory`; optional `[sandbox.worker]` bounds the evaluator and candidate worker.
Never silently switch backend or disable CPU/memory constraints to run a test.

Slurm sandbox backend (2026-11-02): Podman `--memory/--cpus` stay unenforced on
Slurm `test` nodes (`memory.max=max`, historical job `56024690`), so strict runs
there use `sandbox.process_limit_backend = "slurm_step"`. Each container (probe,
agent, candidate worker, evaluator) runs in its own `srun --exact` step with
`podman run --cgroups=disabled`; Slurm's step cgroup enforces `memory.max`
(swap 0) and a dedicated cpuset, and trusted `prlimit` sets `RLIMIT_NPROC` as the
sandbox UID (crun's `--ulimit=nproc` fails spuriously: it counts the host user's
processes). The controller verifies every container from the host before any
input: own numbered step, finite memory/swap, cpuset size, subordinate UID, zero
capabilities, `no_new_privs`, nproc and the pause-process user namespace. Probe
jobs `57163534`/`57163814` showed OOM kill over the step limit, 2.0 CPUs of
throughput for 6 busy processes on a 2-CPU step, and fork stop at 63 children
(`experiments/validation/sandbox-backend/`). Job `57174710` passed the OCI
integration and Panda-fixture tests with this backend. `RLIMIT_NPROC` is shared by
all containers of that UID on a node. Job scripts must source
`scripts/podman-job.sh` (job-private node-local storage; Slurm strips
`XDG_RUNTIME_DIR`) and load the trusted image archive by recorded ID.

## Setup and validation

Python 3.11+ is required (`tomllib` is used). Install with
`.venv/bin/python -m pip install -e '.[test]'` after creating the virtualenv.
Do not add alternate interpreters or dependencies without updating project
metadata and setup documentation.
SciPy (`scipy>=1.11,<2`) supplies cKDTree and linear assignment; Matplotlib
(`matplotlib>=3.8,<4`) supplies scientific metric plots. Use binary wheels for
lightweight dependency installation; plotting and metric evaluation run on Slurm.

From the repository root, submit `sbatch scripts/test.slurm`. The script uses
EGL by default, accepts `MUJOCO_GL=osmesa` through the exported environment, and
runs `.venv/bin/python -m pytest`. Set `SIM2SIM_ROOT` for submissions made outside
the repository and `SIM2SIM_PYTHON` to choose a different installed interpreter.
Arguments to the script are forwarded to pytest. Inspect Slurm output and record
the actual pass/failure result; a submitted job is not a successful test.

OCI integration is skipped by default. Build and record the trusted image with
`sbatch scripts/build-image.slurm` (archive and ID in
`experiments/images/trusted-image.json`), then submit
`SIM2SIM_OCI_TEST=1 SIM2SIM_PROCESS_LIMIT_BACKEND=slurm_step sbatch -c 4 --mem=8G --export=ALL scripts/test.slurm tests/test_oci.py`.
This test must remain on `test` compute nodes and must not be treated as passed
when skipped. Strict runs: `sbatch scripts/experiment-strict.slurm ID [ID]`.

For lightweight login-node syntax validation only:

```sh
.venv/bin/python -m compileall -q src tests
```

For a full run, allocate a `test` compute node, build the image with rootless
Podman, and execute the README commands there. Do not run container builds on the
login node. Keep host-only inference keys out of repository files and logs.

Validation record, 2026-10-28: Slurm `test` job `56050549` passed **97 tests,
1 skipped**, EGL, in 156.28 seconds. Job `56053020` then passed **21** focused
video/runtime checks after final cadence and label fixes. Job `56047534` validated
both new dynamics scenes, all five probes, unequal-mass COM and hinge errors.
Earlier job `56041119` validated density variants and actual three-checkpoint
metric plots. The trusted offline fixture under `runs/dual_world_validation`
produced a verified 254-event, 55-frame two-world replay with an actual partial
action; it is not an isolated-agent integration run. Complete production OCI
integration remains blocked by the unenforced CPU/memory limits described above.
Do not describe the skipped OCI test or the controlled fixture as that run.

Later validation, 2026-10-28: Slurm `test` job `56066912` passed **149 tests,
1 skipped** (OCI only), EGL, in 82.76 seconds. A real prepared experiment then
exposed an NFS `.nfs` file guard failure; the runtime guard received a narrow
regular-file, non-symlink, hexadecimal `.nfs` exception. Focused preparation,
start, development and metadata checks passed **15 tests** in Slurm job
`56069922` (24.26 seconds) after that change. The full suite predates this
last fix. Job `56063002` built and inspected trusted image
`d991f5411f7be34293e0bf3f986c10d92db56d27d3f5c9e2c17789819b5eaf46`:
it contained no source/configuration/experiment files or credentials, but its
live probe still found `memory.max=max` and `cpu.max=max 100000`. Strict OCI
remains blocked; these checks are not an isolated-agent run.

The current design is in `docs/2026-10-28-design.md`. Preserve minimal scope:
MuJoCo XML projects, one ideal point grasp, explicit submissions, configurable
observations, and researcher-only evaluation. See `docs/evaluation.md` for metric
limitations. No commits unless the active user request authorizes them.


Interactive inspector validation, 2026-11-02: Slurm `test` job `57112482`
passed 91 focused tests (inspector live physics plus Chromium, dashboard,
replay, simulation, Panda). Job `57111546` ran `scripts/serve-dashboard.sh`
end to end on a real Panda run. Full-suite job `57110407` had two timing
flakes in `tests/test_runtime.py`; job `57111289` passed that file on rerun.


Storage (2026-11-03): `/n/home00` has a 95 GB quota and a 4 h strict run writes about 3 GB. New runs use
the lab root `/n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-experiments`, via `--root` or an exported
`SIM2SIM_EXPERIMENTS_ROOT` for `scripts/experiment-strict.slurm`. Image archives live in
`/n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-images` (paths in `experiments/images/*.json`).
Run directories must not be symlinks.

Partitions (2026-11-03, user): strict experiments and tests may also run on `sapphire` (same `holy8a` nodes, QoS
`normal`, no 5-job/112-CPU per-user cap). Strict OCI passed there (job `57583004`). Never use GPU partitions for this.
Dashboard generation and inspection may run on the login node.

API spend is orchestrator-only. Agents under investigation must never see dollars, spend, caps or
remaining budgets: not in the task, replies, tool results or container memory. Caps are host-enforced.
Exhaustion suspends the run (`budget_suspended`); the orchestrator may resume it later with
`[continuation]`.

## Repository experiment skills

Record every launched or planned batch in [docs/experiment-log.md](docs/experiment-log.md)
(dated, newest first); remaining experiments for the next agent are in
[docs/experiment-todo.md](docs/experiment-todo.md). For configuring or running experiments, read
[skills/run-experiment/SKILL.md](skills/run-experiment/SKILL.md). View experiments
directly in the dashboard; there is no separate viewing or experiment-selection
skill. Future experiment filters belong in the dashboard. The protected
`.agents` directory is not a skill installation destination here.

New experiment artifacts belong under `experiments/runs/<id>/`; development
checks belong under `experiments/validation/`. Configurations live under
`experiments/configs/`. See `docs/experiments.md`. Prepare/start copies the source
privately and records its final hash. Requested physical changes must alter the
MJCF, not just parameter-description text. Keep research text, source values and
manifests out of public tasks and image contents. Model constraints are public
instructions; explicit metric selectors skip disabled families.

The original pilot preset is exactly `gpt-6-astra` with hard wall time 1800 seconds,
including initialization, inference and evaluation. Never replace the requested
model, use Codex credentials, or bypass failed isolation. Check credential
presence only. Current image availability and effective limits require a fresh
compute-node probe; historical evidence is not proof of the current image.
Timestamp actual progress and completion, label provider summaries, and do not
call either hidden reasoning. Preserve null/unknown execution evidence after a
partial failure instead of claiming no model calls.


Explicit development mode is `development_shared_filesystem`, selected with
`--development` and its own preset. It is not OS isolated: all subagents share
workspace access. Never describe public-only prompts as an enforced filesystem
boundary or use this mode as an automatic production fallback. The user has
explicitly approved this development workflow in the current session. Spawn the
requested model without inherited private history, provide only curated public
bridge/task paths, and interrupt it at the native controller deadline/STOPPED
signal. Preserve the strict API preset for later; no API pilot is authorized by
the latest steering.

Development pilot record, 2026-10-28: `astra-development-20261028-002` used
exactly `gpt-6-astra` in Slurm `test` job `56069945` and finished by public RPC
within the controller's 1800-second wall limit (1698.70 seconds recorded).
Three submissions were accepted; 3000 total physics ticks were used (1400
reference, 1600 reconstruction). The external model returned a final report,
and the Slurm wrapper completed with exit code 0. This is a shared-filesystem
development pilot, not an isolated OCI/API run. Private scores remain in its
run directory; no full external model/tool transcript or token count is
available.


Latest requested batch (2026-10-29, prepared before launch): use the new
`astra-medium-*-2h.toml` presets for all six environments, exactly `gpt-6-astra`
with medium reasoning. `wall_only=true`, 7200 seconds including initialization,
inference and evaluation; no active-time/tick/submission-count quotas. Retain
protocol byte and worker resource bounds and actual cumulative counters. Both
ten-primitive prepared sources use uniform 500 kg/m³; original environments are
unchanged. Each run gets a separate Slurm `test` job. With three model slots,
start two batches without charging waiting-for-slot time to another run. This
supersedes the original pilot budget only for the new batch. Keep development
mode explicit; strict OCI/API is still not authorized as a replacement.

Prelaunch validation, 2026-10-30: Slurm `test` job `56344135` compiled all
six prepared source scenes, verified both uniform-density compiled masses, and
passed **161 tests, 1 skipped** (OCI only) in 88.37 seconds. A preceding native
mailbox regression timed out while generating optional PNG plots; its private
evaluation itself succeeded. The mailbox regression now uses the same
JSON-only checkpoint mode as the new batch. Standalone plotting retains its
separate regression coverage. These validations are not model eval results.

Executed medium batch: all six environments completed by public `finish`
within their 7200-second wall limits, with 13 accepted checkpoints. Slurm jobs:
random assembly `56344207` (3), boat `56344209` (2), tetrahedral cluster
`56344212` (3), spring `56346040` (2), blocks `56350726` (1), and damped lever
`56976448` (2). All used exactly `gpt-6-astra`, medium reasoning, fresh context,
and explicit shared-filesystem development mode. The first five completed on
2026-10-30; damped lever completed on 2026-11-02 UTC. The original damped-lever
job `56346037` expired without an orchestrated model while the dispatcher waited
on scheduler-command approval; its zero-tick/zero-submission attempt is preserved.
The retry has a new ID and fresh budget, not an extension of the expired run.
Keep the native unknown inference fields; model spawns and returned final reports
are recorded separately. No full external transcript or token counts are known.
See `docs/experiments.md` and `experiments/validation/astra-medium-20261029-batch.json`.

Final batch validation: Slurm `test` job `56977064` passed 34 dashboard/replay
tests (3.68 seconds), verified all six sealed audits, and confirmed the six
successful eval jobs exited `0:0`. Job `56984168` passed two browser-control
tests (7.45 seconds) and four actual-recording view checks after the final
Chamfer-only title correction. It reused saved verified snapshots and refreshed
only static view assets; no model calls, simulation or archive rescan occurred.
Both validation jobs completed. Storage statistics reuse the required integrity
scan; do not add another recursive disk-usage pass merely to produce those
same figures. Dashboard asset bytes cover the verified catalog asset set, not
stale cache files or whole run-directory usage. The original failed damped-lever
attempt and earlier optional-storage-scan timeout remain recorded.

## Excluded runs (user decision, 2026-11-05)

Results, reports and analyses use **only strict runs**: `execution_mode = "strict_oci"` (isolated OCI
container under Slurm step limits, model called through the API). The earlier non-API
`development_shared_filesystem` runs (`astra-development-*`, `astra-medium-*`, 2026-10-28 to 2026-11-02) and the
offline `trusted-native-dual-world-fixture` are excluded and must not be cited, compared or shown, including
in the sections above that describe them. The list is `experiments/validation/excluded-runs.json`;
`build-results-report.py` asserts that every reported run is strict and unlisted.
