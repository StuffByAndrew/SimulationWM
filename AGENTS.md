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

## Git workflow (user, 2026-11-23)

- Every agent may commit without asking, and should commit regularly in small logical commits (one coherent change
  each; note in the message what was validated). Never commit secrets (`.env*`, credentials) or run outputs; never
  force-push.
- Work in a git worktree (`.claude/worktrees/<name>`) whenever another agent might be editing the same checkout.
- Merge a worktree's branch into `main` and remove the worktree and its branch as soon as its work is done, unless
  the user said to keep it persistent. Before merging, check the main checkout's `git status` and never discard or
  overwrite another agent's uncommitted changes.

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
- `src/sim2sim/monitor.py` (+ `monitor_view.html`): researcher-only lightweight job monitor kept up on the login node by
  `scripts/serve-monitor.sh` (port 8090; also reachable over the user's Tailscale node `holylogin08-aejxie`). Stdlib only:
  running/queued strict jobs, failures with reasons, per-job Slurm CPU/RAM use (stored in `experiments/monitor/usage.json`,
  backfilled from sacct), API spend per key (ledger) and storage headroom. It shows spend, so never expose it to agents.
- `src/sim2sim/robots.py`: public robot fixture registry (Panda `arm_target`, Shadow hand `hand_target`: include file,
  fixture root body, actuator/joint order, home pose); no MuJoCo import so the host runtime can use it. Robot levels
  expose only proprioception and optional taxels (`simulation.TACTILE`); the fixture is verified byte-for-byte.
- `src/sim2sim/client.py`: agent-facing versioned JSON client (and `Channel`, the `[tools] channel` code channel that the
  sandbox harness serves during bash; unrelated to `local_controller`, a local reconstruction-world copy in the sandbox).
- `src/sim2sim/predictor.py`: the one isolated runner for agent-submitted predictor programs (one container per
  submission, no network/mounts/source/controller). `call` mode (evaluator image, worker bounds; per-call alarm, 1 MiB
  JSON values, prints to stderr) and real-time `stream` mode (agent image and bounds; timed answers). The runtime's
  `PREDICTOR_METRICS` lookup maps a metric to its module's `evaluate_submission(predictor.Job)`; see `docs/protocol.md`.
  Strict OCI only.
- `src/sim2sim/fixture.py`, `fixture_view.py`: positioning-fixture levels (`environments/panda_small/fixture_*`): the
  private carriage mechanism behind `reset_carriage`/`release_carriage` (seeded reset perturbation of the pad; the seed,
  pad state and mappings never reach a public observation, which carries only the dial readings), and the `fixture` metric
  (call mode; held-out releases vs predictor samples: CRPS, Brier, hit rates at the chosen settings) with its researcher
  HTML view.
- `src/sim2sim/realtime.py`: generic real-time world (`[simulation] realtime`, `frame_rate`; any level, either
  embodiment): host thread at wall-clock rate, FIFO actions on the live world, fixed frame grid rendered from recorded
  states, no pause/tick-advance/state access for the agent. Host holds pause it; a provider-failed inference attempt is
  rewound (`mark`/`rewind`) as if frozen, while successful-call latency advances it. `launcher.py`: optional private `[launcher]` (perturbed resets, `throw`
  protocols, seeded streams). `die.py`: biased-die metric `die` (stream mode; host-generated fresh throws, baselines,
  predictor scoring, `RUN/die-view.html`). Keep die-specific logic out of the generic real-time, launcher and predictor
  modules.
- `src/sim2sim/sam3d.py`: optional `[tools] sam3d` relay. The strict controller forwards the agent's validated
  image+mask over a private 0700 spool to a researcher GPU service (`scripts/sam3d-service.slurm`,
  `kempner_gpu_priority`, A100 only, never H100/H200; env/weights under netscratch `SAM3D_ROOT`). A missing,
  stale or silent service stops the run; there is no CPU path. Calls are audited (`sam3d_call`).
- `environments/*`: reusable scene XML plus private manifest probes and body
  correspondence. `examples/` contains the protocol demonstration agent.
- `configs/example.toml`, `Containerfile`, `docs/`: runnable configuration,
  reviewed image recipe, protocol and environment/evaluation contracts.
- Agent-image isolation (user, 2026-11-24): `Containerfile` builds two images (targets `agent`, `evaluator`) recorded
  together in one two-image record (`experiments/images/trusted-image-isolated.json`; `podman-job.sh` loads and verifies
  both; it still loads older single-image records, whose prepared configs name `localhost/sim2sim:latest` for both images and
  run as declared, by explicit dispatch in `scripts/trusted-images.py`). The agent image (`localhost/sim2sim-agent:latest`, `sandbox.image`: agent
  container, stream-mode predictors) ships only `sandbox_agent.py` and `sim2sim` `client`, `simulation`, `robots`,
  `candidate_worker`, `predictor`, `submission`. Evaluation and metric modules live only in the evaluator image
  (`localhost/sim2sim:latest`, `sandbox.evaluator_image`: candidate worker, evaluator, call-mode predictors). Agent-side
  modules import only each other (other modules lazily, on evaluator-side paths only); adding a module to the
  agent image needs the Containerfile list and a reason. The runtime refuses two different images that lack the matching
  `sim2sim.image-role` label; the build greps the agent image for every absent module's path, name and docstring.

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
With `point_grasps = 2` the same rule holds per grasp: each reading is only that grasp's own connect force.

Weld grasp (user, 2026-11-25): from now on, "ideal point grasp with wrench" in a new level means a **weld grasp**: the
grasped object is rigidly fixed to the handle at the grasp point; the agent translates the grasp and rotates the object
about the grasp point; the wrench reports the weld's full force and torque on the object at the grasp point, in world
coordinates. It is selected by a `[simulation]` key (to be built); the default stays the translation-only connect grasp
above, so existing configs and recorded runs are unchanged. The other wrench rules (only the active grasp's own
constraint, nothing from unrelated contacts, no grasp coordinates, lever arms or mass, no time/ticks on refresh) still
apply. Applies to the childproof bottle, gear-train and cable levels (`docs/level-designs/`).

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

OCI integration is skipped by default. Build and record the trusted images with
`SIM2SIM_TRUSTED_IMAGE_RECORD=$PWD/experiments/images/trusted-image-isolated.json sbatch scripts/build-image.slurm`
(both archives and IDs in that record), then submit with the same `SIM2SIM_TRUSTED_IMAGE_RECORD`
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
MuJoCo XML projects, one or two (configured `simulation.point_grasps`) ideal point grasps, explicit submissions, configurable
observations, and researcher-only evaluation. See `docs/evaluation.md` for metric
limitations.


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

Out of storage (2026-11-16, user): if a storage location fills up, explore the other storage locations this user has on the
cluster (lab storage, netscratch, home; see the FASRC/Kempner docs linked in CLAUDE.md) and continue there. Try not to stop
running work, and never delete anything to make space.

Frozen controller code (2026-11-16): the strict controller re-imports `sim2sim` in a spawned child on every inference
call. A merge that briefly left `runtime.py` importing a not-yet-copied module crashed one child and hung
`w20-msd-spring-visible-20261116-001` for 13 h; the 6 h wall watchdog did not end it. `scripts/experiment-strict.slurm`
now copies `src/sim2sim` to a job-private temp dir at launch and puts it first on `PYTHONPATH`. Still avoid editing
`src/` in the main checkout while runs that predate this change are active; develop in a worktree and merge when idle.

Partitions (2026-11-03, user): strict experiments and tests may also run on `sapphire` (same `holy8a` nodes, QoS
`normal`, no 5-job/112-CPU per-user cap). Strict OCI passed there (job `57583004`). Never use GPU partitions for this.
Dashboard generation and inspection may run on the login node.

API spend is orchestrator-only. Agents under investigation must never see dollars, spend, caps or
remaining budgets: not in the task, replies, tool results or container memory. Caps are host-enforced.
Exhaustion suspends the run (`budget_suspended`); the orchestrator may resume it later with
`[continuation]`.

## Gemini models (user, 2026-11-20)

`GEMINI_API_KEY` (in `~/.env`) reaches Google's API directly through `[inference] api = "gemini"` (see
docs/protocol.md). The models are `gemini-4-neon` and `gemini-4-hydrogen`.

- **Limits:**
  - Spend is unlimited: there is no dollar cap on this key.
  - The key's rate limits are 200 requests/min, 2M tokens/min and 100K requests/day.
  - Keep concurrent runs within these limits. Each run makes about 1 call/min, with up to about 100K context tokens per
    call before the harness compacts.
  - The broker retries HTTP 429/5xx within each call's deadline.
  - Gemini configs still carry a host-priced `max_cost_usd`/pool for accounting only; answer their holds with resume.
- **Second key (user, 2026-11-24):** `GEMINI_API_KEY_BYCOST` (in `~/.env`) is billed by cost, with a **lifetime cap of
  $15000**.
  - Every run on it uses the single cost pool `gemini-bycost` with `cost_pool_usd = 15000`; that pool is the lifetime
    cap, so never rename or reset it.
  - Generator: merge `BYCOST` into a run's options (`api_key_env`, pool and a $1000 per-run safety cap).
  - Use it for extra capacity, for example when the first key's rate limits would be hit.
  - Never answer a hold on this pool with `cost_pool_usd` above 15000.
  - There is no CPU-only Kempner partition: the Kempner cluster is GPU-only by policy, so use `test` or `sapphire`.
- **Fair comparison:** despite unlimited spend, compare Gemini fairly with the existing gpt-6-astra runs.
  - Use the astra original's config, changing only the model route and the cost lines.
  - Keep the same harness, image, task, tools, observations, `reasoning_effort`, `max_output_tokens` and wall budget.
    Never give Gemini a longer budget or extra prompting.
  - Report at matched wall minutes and matched cumulative tokens, including astra's final budget where astra stopped at
    its spend cap (e.g. w11 at $44).
  - Tokens are each provider's own count.
  - Report malformed-call and timeout rates.
- **Provider outages:** Google's latency can spike from 2 s to more than 290 s per call. Four consecutive timeouts kill the
  harness.
  - Set `[inference] provider_hold_seconds` so outages are held outside the run: the failed attempt is credited back and
    the backoff runs paused.
  - Successful-call latency still counts.
  - Treat a run that died of provider outages as an infrastructure failure and rerun it.

## Inkling 2 Mini, self-hosted (user, 2026-11-24)

The user forbids paying for Inkling: never route an Inkling request to a paid API (the aqinference gateway's
`thinkingmachines/inkling-2-mini` and Tinker are paid). It runs only on the researcher's own service.

- **Service:** `scripts/inkling-service.slurm` (Kempner A100s only, never H100/H200).
  - vLLM and SGLang need Hopper or Blackwell for this checkpoint, so llama.cpp b12706 serves it.
  - It serves the official `thinkingmachines/Inkling-2-Mini@c9d0283`, whose files are SHA-256-verified against the hub.
  - llama.cpp's converter turned it into BF16 GGUF. Its MXFP8 experts are exact in BF16, its NVFP4 experts are rounded to
    BF16 after their fp32 scale, and activations are BF16 rather than FP4/FP8.
  - It uses two A100-40GB GPUs.
  - Files are under `INKLING_ROOT=/n/netscratch/qianqian_lab/Lab/aejxie/inkling`: `model/`, `gguf/`, `src/`, `logs/`.
- **Broker:** `[inference] api = "local_openai"`, with `endpoint` set to the service's private `service/service.json`.
  - That record is a 0600 heartbeat holding the address and a per-job token.
  - Cost is always 0. A dead or requeued service is a provider outage, held by `provider_hold_seconds`.
- **Configs:** `experiments/configs/inkling-20261124/` follows the gemini-carbon-20261124 recipe with the local route.
  - It has no `reasoning_summary`: raw reasoning is not recorded.
  - Its per-call `timeout_seconds` is 1800.
- **Launch:**
  - Start the service first and keep its `-t` longer than the runs.
  - Wait for `service.json` to appear.
  - Then submit the strict runs as usual. The controller needs no key for Inkling.
- **Speed (job 63571190):**
  - About 26 tokens/s for one stream and 67 tokens/s for five streams.
  - About 10 tokens/s at 143K context.
  - A cold 143K-token prefill takes 13 min. A cached re-call takes 12 s.
  - Compare Inkling with other models at matched tokens as well as wall time.

## Repository experiment skills

Record every launched or planned batch in [docs/experiment-log.md](docs/experiment-log.md)
(dated, newest first); remaining experiments for the next agent are in
[docs/experiment-todo.md](docs/experiment-todo.md). For configuring or running experiments, read
[skills/run-experiment/SKILL.md](skills/run-experiment/SKILL.md); for designing and adding a new level, read
[skills/add-level/SKILL.md](skills/add-level/SKILL.md). View experiments
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
