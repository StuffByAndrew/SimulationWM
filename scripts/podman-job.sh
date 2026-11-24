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
# A record is a two-image record (agent + evaluator, scripts/build-image.slurm; e.g. experiments/images/trusted-image-isolated.json,
# whose agent image new configs name) or an older single-image record (localhost/sim2sim:latest, which older prepared configs
# name for both images); scripts/trusted-images.py dispatches on the shape. SIM2SIM_TRUSTED_IMAGE_RECORD selects the record.
trusted_record="${SIM2SIM_TRUSTED_IMAGE_RECORD:-$PWD/experiments/images/trusted-image.json}"

load_trusted_image() {  # Fail closed unless every archive of the record loads as its recorded image ID.
    local entries name tag id archive sha
    entries="$(python3 "$(dirname "${BASH_SOURCE[0]}")/trusted-images.py" "$trusted_record")" \
        || { echo "$trusted_record is not a trusted image record." >&2; exit 2; }
    while read -r name tag id archive sha; do
        [[ "$(sha256sum "$archive" | cut -d' ' -f1)" == "$sha" ]] || { echo "Trusted $name archive digest mismatch." >&2; exit 2; }
        timeout 900 podman load -q -i "$archive" >/dev/null || { echo "podman load ($name) failed or timed out on $(hostname)." >&2; exit 2; }
        [[ "$(podman image inspect "$tag" --format '{{.Id}}')" == "$id" ]] || { echo "Loaded $tag is not trusted $name $id." >&2; exit 2; }
        echo "trusted_${name}_id=$id"
    done <<< "$entries"
}
