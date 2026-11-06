#!/usr/bin/env bash
#SBATCH --job-name=sam3d-service
#SBATCH --partition=kempner_gpu_priority
#SBATCH --account=kempner_qianqian_lab
#SBATCH --qos=kempner_qianqian_priority
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=sam3d-service-%j.log
# Researcher-run SAM 3D Objects service for strict runs with [tools] sam3d = true (never H100/H200). It verifies every
# pinned weight file (SAM3D_ROOT/sam3d-lock.json), loads the model once, works offline, and answers requests from the
# private spool. Strict controllers on test/sapphire need the same spool exported: SIM2SIM_SAM3D_SPOOL=$SAM3D_ROOT/spool.
# Start it before the experiment jobs and keep -t longer than they run; when it ends, waiting runs stop loudly.
set -euo pipefail
repo_root="${SIM2SIM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
root="${SAM3D_ROOT:-/n/netscratch/qianqian_lab/Lab/aejxie/sam3d}"
spool="${SIM2SIM_SAM3D_SPOOL:-$root/spool}"
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
[[ "$gpu" != *H100* && "$gpu" != *H200* ]] || { echo "refusing H100/H200 GPU: $gpu" >&2; exit 2; }
mkdir -p -m 700 "$spool"
[[ ! -e "$spool/service.json" ]] || { echo "another service may own $spool (service.json exists)" >&2; exit 2; }
export PYTHONPATH="$repo_root/src" CONDA_PREFIX="$root/env" CUDA_HOME="$root/env" PATH="$root/env/bin:$PATH" PYTHONUNBUFFERED=1
export HF_HUB_CACHE="$root/cache/hf-hub" TORCH_HOME="$root/cache/torch" XDG_CACHE_HOME="$root/cache/xdg" TORCH_EXTENSIONS_DIR="$root/cache/torch-extensions"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
printf 'node=%s job=%s gpu=%s spool=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$gpu" "$spool"
exec "$root/env/bin/python" -m sim2sim.sam3d --spool "$spool" --root "$root"
