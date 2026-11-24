# Sourced by Slurm scripts: job-private rootless Podman state on node-local /tmp.
# Private storage means no other job can retag or remove the trusted image. The
# first podman call below starts the user-namespace pause process in the batch
# step, so it outlives every container step of this job. Rootless storage reads
# rootless_storage_path (not graphroot); Slurm strips XDG_RUNTIME_DIR from steps,
# which the runtime restores explicitly for each container step.
podman_root="/tmp/sim2sim-podman-${SLURM_JOB_ID:?Slurm allocation required}"
mkdir -p -m 700 "$podman_root/run"
printf '[storage]\ndriver = "overlay"\ngraphroot = "%s/root"\nrunroot = "%s/runroot"\nrootless_storage_path = "%s/root"\n[storage.options.overlay]\nmountopt = "nodev,metacopy=on"\n' \
    "$podman_root" "$podman_root" "$podman_root" > "$podman_root/storage.conf"
export XDG_RUNTIME_DIR="$podman_root/run" CONTAINERS_STORAGE_CONF="$podman_root/storage.conf"
trap 'podman system reset --force >/dev/null 2>&1; rm -rf "$podman_root"' EXIT
# Serialize the first podman call per node: concurrent jobs of one user race initializing the shared rootless
# SHM lock segment ("failed to open 2048 locks ... bad file descriptor", job 57583489).
# Fail fast (2026-11-18): on some nodes this first call hung for 10-12 h, holding the job until its time limit.
[[ "$(timeout 300 flock -w 240 /tmp/sim2sim-podman-init.lock timeout 120 podman info --format '{{.Store.GraphRoot}}')" == "$podman_root/root" ]] || { echo "Job-private Podman storage was not selected (or podman init timed out on $(hostname))." >&2; exit 2; }
# A record holds two images built from one Containerfile: "images" -> "agent" and "evaluator", each with tag, image_id,
# archive and archive_sha256 (scripts/build-image.slurm). Old single-image records are refused. SIM2SIM_TRUSTED_IMAGE_RECORD
# opts into another record (e.g. experiments/images/trusted-image-tools.json).
trusted_record="${SIM2SIM_TRUSTED_IMAGE_RECORD:-$PWD/experiments/images/trusted-image-isolated.json}"

load_trusted_image() {  # Fail closed unless both shared archives load as the record's agent and evaluator image IDs.
    local entries role tag id archive sha
    entries="$(python3 -c 'import json, sys
images = json.load(open(sys.argv[1]))["images"]
assert sorted(images) == ["agent", "evaluator"] and len({images[r]["tag"] for r in images}) == len({images[r]["image_id"] for r in images}) == 2, images
for role in ("agent", "evaluator"):
    print(role, images[role]["tag"], images[role]["image_id"], images[role]["archive"], images[role]["archive_sha256"])' "$trusted_record")" \
        || { echo "$trusted_record is not a two-image (agent, evaluator) trusted record." >&2; exit 2; }
    while read -r role tag id archive sha; do
        [[ "$(sha256sum "$archive" | cut -d' ' -f1)" == "$sha" ]] || { echo "Trusted $role image archive digest mismatch." >&2; exit 2; }
        timeout 900 podman load -q -i "$archive" >/dev/null || { echo "podman load ($role) failed or timed out on $(hostname)." >&2; exit 2; }
        [[ "$(podman image inspect "$tag" --format '{{.Id}}')" == "$id" ]] || { echo "Loaded $tag is not trusted $role image $id." >&2; exit 2; }
        echo "trusted_${role}_image_id=$id"
    done <<< "$entries"
}
