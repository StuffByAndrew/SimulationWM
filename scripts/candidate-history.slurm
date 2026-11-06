#!/bin/bash
#SBATCH --partition=test
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=experiments/validation/candidate-history-%j.log
set -euo pipefail
cd "${SIM2SIM_ROOT:-/n/home00/aejxie/sim2sim}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
.venv/bin/python -m sim2sim.candidate_history "$@"
