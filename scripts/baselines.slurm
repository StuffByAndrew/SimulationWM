#!/bin/bash
# Trivial-baseline scores (empty and bounding-box candidates) per run: sbatch scripts/baselines.slurm experiments/runs/RUN_ID ...
#SBATCH --partition=test
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=experiments/validation/baselines-%j.log
set -euo pipefail
cd "${SIM2SIM_ROOT:-/n/home00/aejxie/sim2sim}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
.venv/bin/python -m sim2sim.baselines "$@"
