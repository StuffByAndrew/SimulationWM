#!/bin/bash
#SBATCH --partition=sapphire
#SBATCH --job-name=sim2sim-compact
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=experiments/validation/compact-%j.log
# Compact run audits (src/sim2sim/compact.py): one file per stream instead of one per blob. Usage:
#   sbatch scripts/compact-runs.slurm RUN_DIR [RUN_DIR ...]   specific runs (e.g. a strict job's runs once they end)
#   sbatch scripts/compact-runs.slurm                         every run with loose blobs in the three experiment roots
# compact.py itself refuses runs that are live (named by a Slurm job, preparing, or recently written and unsealed), so
# this is safe to submit while experiments run. One JSON line per run; PROGRESS lines count runs done of the total.
set -euo pipefail
cd "${SIM2SIM_ROOT:-$SLURM_SUBMIT_DIR}"
python_bin="${SIM2SIM_PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if (( $# )); then
    runs=("$@")
else
    mapfile -t runs < <(for root in /n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-experiments \
                                    /n/netscratch/qianqian_lab/Lab/aejxie/sim2sim-experiments "$PWD/experiments"; do
        for run in "$root"/runs/*/; do
            [[ -n "$(find "${run}audit/blobs" -maxdepth 1 -type f -print -quit 2>/dev/null)" ]] && echo "${run%/}"
        done
    done)
fi
total=${#runs[@]}
echo "PROGRESS 0/$total $(date -Is)"
printf '%s\n' "${runs[@]}" | xargs -P "${SIM2SIM_COMPACT_PARALLEL:-4}" -n 1 "$python_bin" -m sim2sim.compact |
    awk -v total="$total" '{print; fflush(); done++; printf "PROGRESS %d/%d\n", done, total; fflush()}' ||
    echo "NOTE some runs were refused or failed and keep their loose blobs; see their JSON lines"
echo "DONE $(date -Is)"
