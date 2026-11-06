#!/usr/bin/env bash
#SBATCH --job-name=sam3d-fetch
#SBATCH --partition=sapphire
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=sam3d-fetch-%j.log
# Researcher-side, one-time download of the gated SAM 3D Objects checkpoints (SAM License) at a pinned revision into
# SAM3D_ROOT/checkpoints. Prerequisites (user): accept the license at https://huggingface.co/facebook/sam-3d-objects,
# then run `$SAM3D_ROOT/env/bin/hf auth login` once in a terminal. The token stays where hf keeps it; this script never
# reads it. Submit from the repository root: sbatch --account=qianqian_lab scripts/fetch-sam3d.slurm
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit through sbatch; do not download on a login node.' >&2; exit 2; }
repo_root="${SIM2SIM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
root="${SAM3D_ROOT:-/n/netscratch/qianqian_lab/Lab/aejxie/sam3d}"
pins="$repo_root/experiments/sam3d/checkpoint-pins.json"
revision="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$pins")"
[[ ! -e "$root/checkpoints" ]] || { echo "$root/checkpoints already exists" >&2; exit 2; }
# Caches go to netscratch; HF_HOME stays default so hf finds the login token.
export HF_HUB_CACHE="$root/cache/hf-hub" HF_XET_CACHE="$root/cache/hf-xet" HF_HUB_DISABLE_TELEMETRY=1
download="$root/hf-download-$SLURM_JOB_ID"
"$root/env/bin/hf" download facebook/sam-3d-objects --repo-type model --revision "$revision" --local-dir "$download" \
    --include 'checkpoints/*' LICENSE --max-workers 1
python3 - "$pins" "$download" <<'EOF'
import hashlib, json, sys
from pathlib import Path
pins, root = json.load(open(sys.argv[1])), Path(sys.argv[2])
for name, pin in pins["small_files"].items():
    data = (root / name).read_bytes()
    assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == pin["git_blob_sha1"], name
for name, size in pins["lfs_bytes"].items():
    assert (root / name).stat().st_size == size, name
print("pinned checkpoint files verified:", len(pins["small_files"]) + len(pins["lfs_bytes"]))
EOF
mv "$download/checkpoints" "$root/checkpoints"
mv "$download/LICENSE" "$root/checkpoints/SAM-LICENSE"
rm -rf "$download"
echo "next: sbatch scripts/sam3d-smoke.slurm full (writes $root/sam3d-lock.json)"
