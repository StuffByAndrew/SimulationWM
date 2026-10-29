#!/usr/bin/env bash
#SBATCH --job-name=sim2sim-tests
#SBATCH --partition=test
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=sim2sim-tests-%j.out
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit this script through sbatch on the test partition; do not run it on a login node.' >&2
    exit 2
fi
if [[ "${SLURM_JOB_PARTITION:-}" != test ]]; then
    echo 'This project requires the Slurm test partition for heavy computation.' >&2
    exit 2
fi
if [[ -n "${SIM2SIM_ROOT:-}" ]]; then
    repo_root="$SIM2SIM_ROOT"
else
    repo_root="$(git -C "${SLURM_SUBMIT_DIR:?}" rev-parse --show-toplevel)"
fi
cd "$repo_root"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${SIM2SIM_PYTHON:-$repo_root/.venv/bin/python}"
printf 'node=%s job=%s backend=%s repository=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$MUJOCO_GL" "$repo_root"
exec "$python_bin" -m pytest -q "$@"
