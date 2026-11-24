# sim2sim

A small MuJoCo benchmark for reconstructing a scene from camera or arm
observations. In the original environments, an agent observes RGB, depth or
camera calibration, grasps a pixel, moves the grasp point in world coordinates,
and submits an MJCF project.
An optional grasp wrench reports world-frame contact force and zero point torque.
The researcher retains the source scene, hidden probes, scores and recordings.

Eight opt-in [Panda arm environments](docs/environments.md#panda-arm-variants)
replace pixel grasping with seven-joint and gripper commands. Six retain RGB
vision; the two-block and fixed-cube variants use arm proprioception only.

The host runs physics and the protocol. In strict OCI mode, a rootless Podman
container runs the agent without network access or host directory mounts, and a
separate constrained container evaluates submitted XML and assets. The explicit
development mode uses native workers and a shared filesystem instead; it has no
OS isolation. The included scripted agent demonstrates the protocol; it is not
a reconstruction solver.

## Setup

Use Python 3.11 or newer. From the repository root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

SciPy supplies nearest-neighbor surface distances and primitive assignment;
Matplotlib generates standalone metric-progress plots. Both install through the
project dependencies above. The `test` extra also installs PyYAML for validating
the repository experiment skills; no alternate interpreter is needed.

MuJoCo rendering needs a working EGL or OSMesa driver. On this cluster, EGL
works on the tested compute node; the login node lacks the required software
rendering driver. Rootless Podman with working cgroup resource limits is required
for complete strict OCI benchmark runs. That mode fails closed when rootless
Podman is unavailable.

**Run all heavy computation on Slurm's `test` partition.** Do not run physics,
rendering, the full test suite, image builds or benchmark runs on a login node.
Request a compute allocation before building the example image:

```sh
srun --partition=test --time=00:15:00 --mem=4G --cpus-per-task=2 --pty bash
# Inside the allocation, from the repository root:
podman build -f Containerfile --target agent -t localhost/sim2sim-agent:latest .
podman build -f Containerfile --target evaluator -t localhost/sim2sim:latest .
```

The image must be built before the run: runtime containers use `--pull=never`.
Local cluster configuration may require a different allocation duration or
Podman setup.

The default `sandbox.process_limit_backend = "cgroup"` requires delegated CPU,
memory and pids controllers. Hosts with working CPU/memory cgroups but no pids
controller can explicitly select `process_limit_backend = "rlimit"` under
`[sandbox]`. This applies a hard `RLIMIT_NPROC` using `pids` (default 64), with
a fixed nonroot subordinate UID, dropped capabilities and `no_new_privs`.
Linux counts threads against this real UID; concurrent containers may share the
accounting and exhaust each other's slots. This is not a per-container pids
cgroup. [Linux documents the hard limit and its privilege rules](https://man7.org/linux/man-pages/man2/getrlimit.2.html).

Controller names alone do not establish enforcement. Before any agent starts,
the runtime launches a small probe in the researcher-trusted evaluator image
and checks its effective limits, UID, privilege flags and environment. On Slurm
`test` nodes Podman silently ignores `--memory/--cpus` (`memory.max=max`), so use
`process_limit_backend = "slurm_step"` there: every container runs in its own
`srun --exact` step whose cgroup Slurm enforces, and the controller verifies each
container's step memory, cpuset, UID, capabilities and process bound from the host
before sending input. Build the image once with `sbatch scripts/build-image.slurm`
(both archives and IDs in `experiments/images/trusted-image-isolated.json`); job scripts
load both into job-private storage and refuse different IDs. Launch one or two
prepared strict runs per allocation with `sbatch scripts/experiment-strict.slurm
ID [ID]`. Evidence: `experiments/validation/sandbox-backend/`.

## Configured experiments

Use [experiment configuration and viewing](docs/experiments.md) to select an
environment, actual private physical values, model instructions and evaluation
families. The [Astra preset](experiments/configs/astra-random-assembly.toml) targets
exactly `gpt-6-astra` with a 30-minute hard wall limit. Prepare a private source
copy before changing densities or joint values; metadata alone does not apply
physical changes. Researcher descriptions and values stay out of the agent task.

```sh
sbatch scripts/experiment.slurm experiments/configs/astra-random-assembly.toml --id astra-pilot-001
.venv/bin/sim2sim dashboard --root experiments
```

The separately selected [development preset](experiments/configs/astra-random-assembly-development.toml)
uses a spawned Astra agent and a public bridge with `--development`. It has no OS
filesystem isolation and is never an automatic substitute for a blocked OCI run.
The API preset is retained for later use. See the experiment guide for the
explicit workflow and confidentiality limits.

The [two-hour medium Astra batch](docs/experiments.md#two-hour-medium-astra-environment-batch)
covers all six environments with separate Slurm jobs and a wall-only budget:
7200 seconds, without tick or submission-count quotas. Its approved development
workflow selects `gpt-6-astra` with medium reasoning. The
[results dashboard](docs/dashboard.md) groups comparable curves, provides a
metric checklist, and has a checkbox for recorded grasp points and trajectories.
Its **Interactive physics** tab runs the source scene and any recorded
reconstruction version in two live MuJoCo/Viser viewers. Launch it from a login
node with `scripts/serve-dashboard.sh`, which requests a CPU-only Slurm `test` node.
The optional `inspect` extra (`viser>=1.1,<2`, also in `test`) supplies the viewer.

New attempts and their artifacts live under `experiments/runs/<id>/`; the private
multi-experiment dashboard is `experiments/dashboard/index.html`. Strict OCI
broker runs require a host credential and verified container limits. The explicit
development mode uses neither the broker credential nor OCI limits. A blocked
attempt records its status without pretending an agent ran. Configure and run experiments
with [run-experiment](skills/run-experiment/SKILL.md); view recorded experiments
directly in the [dashboard](docs/dashboard.md).

## Run

On a compatible host with enforced limits, edit
[configs/example.toml](configs/example.toml), then run inside the compute allocation:

```sh
MUJOCO_GL=egl .venv/bin/sim2sim run configs/example.toml
```

Paths resolve relative to the invocation directory. Choose a fresh `output_dir`
for every run. Configuration selects the private source, agent image/command,
active time and physics budgets, submission limits, camera modalities, video,
and evaluation probes. An optional host inference broker forwards requests to
one configured HTTPS endpoint; the API key stays on the host.

The output directory retains the resolved configuration, event log, submitted
versions, private evaluation artifacts and reset-separated videos. Evaluation
reports are researcher-only. The agent receives submission validation feedback.
For enabled reconstruction metrics, `metric-history.png` tracks Chamfer distance,
per-original-primitive density error and center-of-mass error at explicit
submission checkpoints; `metric-history.json` retains the full measurements and
COM error vectors. Missing measurements appear as gaps. These history files also
remain private.

Recorded runs can be inspected without rerunning physics, parsing candidate
models, or making inference calls:

```sh
.venv/bin/sim2sim replay experiments/runs/sandbox-example --output experiments/runs/sandbox-example/replay
```

Replay verifies the audit files and exports the archived frames, state snapshots,
wire bytes and a local `index.html` viewer. This is playback of the recorded
history, not a new simulation or a reconstructed API response.

- [Protocol and observation formats](docs/protocol.md)
- [Environment authoring](docs/environments.md)
- [Evaluation metrics, artifacts and limits](docs/evaluation.md)
- [Accepted design](docs/2026-10-28-design.md)
- [Development continuity and compute policy](AGENTS.md)

## Validation and limits

Submit tests from the repository root:

```sh
sbatch scripts/test.slurm
# Select another supported rendering backend when needed:
MUJOCO_GL=osmesa sbatch --export=ALL scripts/test.slurm
# Optional OCI integration; requires the prebuilt image and working Podman:
SIM2SIM_OCI_TEST=1 sbatch --export=ALL scripts/test.slurm
# On Slurm test nodes (Slurm step cgroups enforce CPU/memory):
SIM2SIM_OCI_TEST=1 SIM2SIM_PROCESS_LIMIT_BACKEND=slurm_step sbatch -c 4 --mem=8G --export=ALL scripts/test.slurm tests/test_oci.py
```

On 2026-10-28, Slurm `test` job `56066912` completed **149 passed, 1 skipped**
in 82.76 seconds with EGL; the skip was OCI integration. After a real prepared
experiment exposed an NFS `.nfs` file guard issue, a narrow fix passed **15**
focused preparation, start, development and metadata checks in job `56069922`
(24.26 seconds). The latter job is the validation after that fix; the full suite
predates it. Earlier jobs `56050549`, `56053020` and `56047534` covered native
candidate execution, video/runtime behavior and dynamics scenes.

A controlled trusted two-world fixture produced a verified replay with **254
events and 55 frames**, including a real three-tick partial action. It is an
offline test fixture, not a production isolated-agent run. The OCI test remains
explicitly opt-in and skipped in the full suite.

The trusted image was built and inspected in job `56063002`; its image ID is
`d991f5411f7be34293e0bf3f986c10d92db56d27d3f5c9e2c17789819b5eaf46`.
Its inspected contents excluded source scenes, configurations, experiments,
documentation, repository metadata and credentials. The live resource probe still
reported `memory.max=max` and `cpu.max=max 100000`. With the `slurm_step` backend,
job `57174710` passed the isolated agent/evaluator OCI integration test.

The explicitly approved shared-filesystem Astra development pilot completed in
Slurm job `56069945`: the controller recorded `agent_finish` within its
30-minute limit, three accepted submissions and 3000 physics ticks. Its private
record is `experiments/runs/astra-development-20261028-002/`. This pilot has no
OS filesystem isolation claim; the strict API pilot remains unrun.

Metrics report discrepancies;
they do not establish unique physical parameter identification or provide a
calibrated aggregate pass threshold. See the evaluation documentation for metric
scope and failure behavior.
