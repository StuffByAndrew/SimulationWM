#!/usr/bin/env bash
#SBATCH --job-name=sam3d-env
#SBATCH --partition=test
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --output=sam3d-env-%j.log
# Researcher-side build of the SAM 3D Objects service environment (CPU node; CUDA extensions are compiled for A100,
# sm_80). Everything lives under SAM3D_ROOT (netscratch), never on /n/home00. Weights are fetched separately by
# scripts/fetch-sam3d.sh after the user accepts the gated Hugging Face license.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit through sbatch; do not build on a login node.' >&2; exit 2; }
root="${SAM3D_ROOT:-/n/netscratch/qianqian_lab/Lab/aejxie/sam3d}"
commit=f91db411c50efee93d8db7aeb323885650f6f722  # facebookresearch/sam-3d-objects main, 2026-06-02
repo="$root/sam-3d-objects" env="$root/env"
export CONDA_PKGS_DIRS="$root/conda-pkgs" PIP_CACHE_DIR="$root/pip-cache" XDG_CACHE_HOME="$root/cache" TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT
[[ -d "$repo" ]] || git clone https://github.com/facebookresearch/sam-3d-objects.git "$repo"
git -C "$repo" fetch --depth 1 origin "$commit" && git -C "$repo" checkout -q "$commit"
[[ "$(git -C "$repo" rev-parse HEAD)" == "$commit" ]]
[[ ! -e "$env" ]] || { echo "$env already exists" >&2; exit 2; }
/n/sw/Miniforge3-25.3.1-0/bin/mamba env create -y -p "$env" -f "$repo/environments/default.yml"
pip="$env/bin/python -m pip"
# conda-forge's cuda-nvcc-tools 12.1.105 `fatbinary` segfaults on these Rocky 8 hosts (even `fatbinary --version`, job
# 57925987), so CUDA extensions compile with the cluster CUDA 12.2 toolkit (minor-compatible with torch cu121) and env gcc 12.
cuda=/n/sw/helmod-rocky8/apps/Core/cuda/12.2.0-fasrc01/cuda
export PATH="$cuda/bin:$env/bin:$PATH" CUDA_HOME="$cuda" FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=16
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"  # doc/setup.md also lists pypi.ngc.nvidia.com, which no longer resolves
cd "$repo"
$pip install -e '.[dev]'
# pytorch3d, flash_attn and gsplat import torch in setup.py, so they build against the installed torch. The flash_attn
# 2.8.3 wheel needs GLIBC 2.32 (Rocky 8: 2.28) and SAM 3D forces flash_attn on A100, so it is compiled here for sm_80.
$pip install --no-build-isolation "$(grep '^pytorch3d' requirements.p3d.txt)"
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=12 NVCC_THREADS=2 \
    $pip install --no-build-isolation --no-deps --no-cache-dir "$(grep '^flash_attn' requirements.p3d.txt)"
# The kaolin 0.17.0 wheel needs GLIBC 2.29 (Rocky 8 has 2.28, job 57936579): build the same tag from source.
grep -v '^kaolin' requirements.inference.txt > "$TMPDIR/inference.txt"
$pip install --no-build-isolation -r "$TMPDIR/inference.txt" 'kaolin @ git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.17.0'
$pip install 'huggingface-hub[cli]<1.0'  # the hf CLI for scripts/fetch-sam3d.slurm, as doc/setup.md pins it
"$env/bin/python" patching/hydra
"$env/bin/python" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_arch_list())'
echo "sam3d env built: $env (GPU import and visibility check: scripts/sam3d-smoke.slurm)"
