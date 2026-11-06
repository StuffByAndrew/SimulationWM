#!/usr/bin/env bash
#SBATCH --partition=test
#SBATCH --job-name=sim2sim-strict
#SBATCH --time=04:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --output=experiments/validation/experiment-strict-%j.log
# Strict OCI: one to three prepared run IDs whose controllers share this allocation (defaults sized for two;
# for three submit with --cpus-per-task=30 --mem=72G).
# Sizing per run: controller (batch step) + agent step (4 CPU/8G) + candidate and
# evaluator worker steps (2 CPU/4G each); the remainder is controller headroom.
# Each run keeps its own native wall deadline; the extra scheduler time is reserved
# for private finalization. Override with sbatch --time for longer run limits.
set -euo pipefail
# SIM2SIM_EXPERIMENTS_ROOT (exported, optional) selects another experiments root, e.g. lab storage for large audits.
[[ "${SLURM_JOB_PARTITION:-}" =~ ^(test|sapphire)$ ]] || { echo "Strict experiments require a Slurm test or sapphire allocation." >&2; exit 2; }
(( $# >= 1 && $# <= 3 )) || { echo 'usage: sbatch scripts/experiment-strict.slurm RUN_ID [RUN_ID [RUN_ID]]' >&2; exit 2; }
(( $# < 3 || SLURM_CPUS_PER_TASK >= 30 )) || { echo 'three runs need --cpus-per-task=30 --mem=72G' >&2; exit 2; }
cd "${SIM2SIM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
source scripts/podman-job.sh
load_trusted_image  # fails closed before any controller starts
export MUJOCO_GL="${MUJOCO_GL:-egl}" MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# Host credentials enter only the controller environment, after the pause process exists;
# every container step is launched with an explicit minimal environment.
set +x
set -a
source "$HOME/.env"
set +a
pids=()
for id in "$@"; do
    "${SIM2SIM_PYTHON:-.venv/bin/python}" -m sim2sim.cli experiment --start "$id" --root "${SIM2SIM_EXPERIMENTS_ROOT:-experiments}" --no-dashboard \
        > "experiments/validation/experiment-strict-$SLURM_JOB_ID-$id.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
