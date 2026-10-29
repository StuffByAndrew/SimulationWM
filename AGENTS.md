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
- `src/sim2sim/client.py`: agent-facing versioned JSON client.
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

The default sandbox process-limit backend is `cgroup` and requires enforced
CPU, memory and pids controllers. Explicit `sandbox.process_limit_backend =
"rlimit"` is valid only on hosts enforcing CPU/memory but lacking delegated pids:
`pids` becomes hard `RLIMIT_NPROC` for the fixed nonroot subordinate real UID.
Keep capabilities dropped and `no_new_privs` enabled. The bound covers threads
and processes across containers sharing that UID and may cause slot starvation.
Never silently switch backend or disable CPU/memory constraints to run a test.

Current infrastructure blocker: diagnostic Slurm job `56024690` on
`holy8a24401` showed `--memory=1g --cpus=1` leaves `memory.max=max` and
`cpu.max=max 100000`, with or without private cgroup namespaces. Podman accepting
flags and reporting controllers is insufficient evidence of enforcement.
The trusted evaluator-image resource probe must refuse to start the agent when
actual limits are absent. `rlimit` cannot repair this CPU/memory gap. Full OCI
runs need administrator delegation or another compatible host; do not describe
the current Slurm environment as supporting the sandbox.

## Setup and validation

Python 3.11+ is required (`tomllib` is used). Install with
`.venv/bin/python -m pip install -e '.[test]'` after creating the virtualenv.
Do not add alternate interpreters or dependencies without updating project
metadata and setup documentation.

From the repository root, submit `sbatch scripts/test.slurm`. The script uses
EGL by default, accepts `MUJOCO_GL=osmesa` through the exported environment, and
runs `.venv/bin/python -m pytest`. Set `SIM2SIM_ROOT` for submissions made outside
the repository and `SIM2SIM_PYTHON` to choose a different installed interpreter.
Arguments to the script are forwarded to pytest. Inspect Slurm output and record
the actual pass/failure result; a submitted job is not a successful test.

OCI integration is skipped by default. To run it after building the image,
submit `SIM2SIM_OCI_TEST=1 sbatch --export=ALL scripts/test.slurm`. On a host enforcing CPU/memory but lacking pids, add
`SIM2SIM_PROCESS_LIMIT_BACKEND=rlimit` explicitly. This test must
remain on `test` compute nodes and must not be treated as passed when skipped.

For lightweight login-node syntax validation only:

```sh
.venv/bin/python -m compileall -q src tests
```

For a full run, allocate a `test` compute node, build the image with rootless
Podman, and execute the README commands there. Do not run container builds on the
login node. Keep host-only inference keys out of repository files and logs.

Validation record: 2026-10-28, latest completed compute suite **41 passed,
1 skipped**, EGL. OCI integration is the skipped opt-in test. Numerical, image and video
checks pass; complete OCI integration is blocked by the unenforced CPU/memory
limits above. The suite after resource preflight changes is pending. Earlier job `56020252` on
`holy8a26602` passed 31 tests in 40.86 seconds. Do not describe a pending or
skipped integration check as successful.

The current design is in `docs/2026-10-28-design.md`. Preserve minimal scope:
MuJoCo XML projects, one ideal point grasp, explicit submissions, configurable
observations, and researcher-only evaluation. See `docs/evaluation.md` for metric
limitations. No commits unless the active user request authorizes them.
