#!/usr/bin/env bash
# Researcher-side, one-time fetch of the optional segmentation tool's third-party code and weights: pinned MobileSAM
# (Apache-2.0) into tools/segmentation/vendor/. Never runs in an agent container; the runtime re-verifies the weights'
# sha256 (AGENT_TOOLS in src/sim2sim/runtime.py) every time it stages the tool volume.
set -euo pipefail
cd "$(dirname "$0")/.."
commit=f706ad9c4eb7f219c00d9050e46328518ffb65d2
archive_sha256=afe5c71ff9d562ce8811bdfb18712dc2d637c33f0c73c1e7300d9717ebcf91b3
vendor=tools/segmentation/vendor
[[ ! -e "$vendor" ]] || { echo "$vendor already exists" >&2; exit 2; }
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
curl -sSfL -o "$work/source.tar.gz" "https://codeload.github.com/ChaoningZhang/MobileSAM/tar.gz/$commit"
echo "$archive_sha256  $work/source.tar.gz" | sha256sum -c --quiet
tar xzf "$work/source.tar.gz" -C "$work" "MobileSAM-$commit/mobile_sam" "MobileSAM-$commit/weights/mobile_sam.pt" "MobileSAM-$commit/LICENSE"
mkdir "$vendor"
mv "$work/MobileSAM-$commit/mobile_sam" "$work/MobileSAM-$commit/weights/mobile_sam.pt" "$work/MobileSAM-$commit/LICENSE" "$vendor/"
sha256sum "$vendor/mobile_sam.pt"
