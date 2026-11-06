#!/usr/bin/env bash
#SBATCH --partition=test
#SBATCH --job-name=sim2sim-experiment
#SBATCH --time=00:35:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --output=experiments/validation/experiment-launch-%j.log
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == test ]] || { echo 'Experiments require a Slurm test allocation.' >&2; exit 2; }
repo_root="${SIM2SIM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$repo_root"
export MUJOCO_GL="${MUJOCO_GL:-egl}" MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
config="${1:-experiments/configs/astra-random-assembly.toml}"
shift "$(( $# > 0 ? 1 : 0 ))"
exec "${SIM2SIM_PYTHON:-$repo_root/.venv/bin/python}" -m sim2sim.cli experiment "$config" "$@"
