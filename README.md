# sim2sim

A small MuJoCo benchmark for reconstructing a scene from camera observations and
pixel interactions. An agent observes RGB, depth or camera calibration, grasps a
pixel, moves the grasp point in world coordinates, and submits an MJCF project.
The researcher retains the source scene, hidden probes, scores and recordings.

The host runs physics and the protocol. A rootless Podman container runs the
agent without network access or host mounts. A separate constrained container
evaluates submitted XML and assets. The included scripted agent demonstrates the
protocol; it is not a reconstruction solver.

## Setup

Use Python 3.11 or newer. From the repository root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

MuJoCo rendering needs a working EGL or OSMesa driver. On this cluster, EGL
works on the tested compute node; the login node lacks the required software
rendering driver. Rootless Podman with working cgroup resource limits is required
for complete benchmark runs. The runtime fails closed when rootless Podman is
unavailable.

**Run all heavy computation on Slurm's `test` partition.** Do not run physics,
rendering, the full test suite, image builds or benchmark runs on a login node.
Request a compute allocation before building the example image:

```sh
srun --partition=test --time=00:15:00 --mem=4G --cpus-per-task=2 --pty bash
# Inside the allocation, from the repository root:
podman build -f Containerfile -t localhost/sim2sim:latest .
```

The image must be built before the run: runtime containers use `--pull=never`.
Local cluster configuration may require a different allocation duration or
Podman setup.

The default `sandbox.process_limit_backend = "cgroup"` requires delegated CPU,
memory and pids controllers. Hosts with enforced CPU/memory cgroups but no pids
controller may explicitly select `process_limit_backend = "rlimit"` under
`[sandbox]`. This applies hard `RLIMIT_NPROC` using `pids` (default 64) to the
fixed nonroot subordinate real UID. The bound counts threads/processes across
containers sharing that UID, so they can exhaust each other's slots.

**Full sandbox runs are blocked on this cluster.** Diagnostic Slurm job
`56024690` on `holy8a24401` showed that rootless Podman accepted `--memory=1g`
and `--cpus=1` while exposing unlimited `memory.max` and `cpu.max`, including
with a private cgroup namespace. The `rlimit` process alternative cannot supply
missing CPU/memory enforcement. A trusted evaluator-image resource preflight
refuses to start an agent unless actual cgroup limits are enforced. Running the
complete benchmark requires administrator-provided delegation or a compatible
host; choosing `rlimit` alone does not make this cluster suitable.

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
# On a host enforcing CPU/memory but lacking pids delegation:
SIM2SIM_OCI_TEST=1 SIM2SIM_PROCESS_LIMIT_BACKEND=rlimit sbatch --export=ALL scripts/test.slurm
```

On 2026-10-28, the latest completed compute-node suite reported **41 passed,
1 skipped**, using EGL. The skipped OCI integration test is opt-in through
`SIM2SIM_OCI_TEST=1`. Earlier Slurm job `56020252` completed 31 tests in 40.86
seconds; changes since those runs require validation.

Numerical, image and video tests pass in the compute suite. Full rootless OCI
integration remains blocked by the cluster's unenforced CPU/memory controls;
agent/evaluator integration is not validated here. The final suite after adding
resource preflight is pending.
Metrics report discrepancies;
they do not establish unique physical parameter identification or provide a
calibrated aggregate pass threshold. See the evaluation documentation for metric
scope and failure behavior.
