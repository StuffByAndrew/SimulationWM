#!/usr/bin/env bash
#SBATCH --partition=test
#SBATCH --job-name=sim2sim-image
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=experiments/validation/build-image-%j.log
# Build, inspect, archive and record the trusted sandbox image. No agent or model runs.
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == test ]] || { echo 'Image builds require a Slurm test allocation.' >&2; exit 2; }
cd "${SIM2SIM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
# Optional: SIM2SIM_CONTAINERFILE (default Containerfile); SIM2SIM_BASE_IMAGE_RECORD loads a verified base image as
# localhost/sim2sim:latest for a derived Containerfile; SIM2SIM_TRUSTED_IMAGE_RECORD (podman-job.sh) names the record
# written; SIM2SIM_IMAGE_ARCHIVE_DIR (default experiments/images) holds the archive.
containerfile="${SIM2SIM_CONTAINERFILE:-Containerfile}"
source scripts/podman-job.sh
[[ -z "${SIM2SIM_BASE_IMAGE_RECORD:-}" ]] || trusted_record="$SIM2SIM_BASE_IMAGE_RECORD" load_trusted_image
podman build -f "$containerfile" -t localhost/sim2sim:latest .
id="$(podman image inspect localhost/sim2sim:latest --format '{{.Id}}')"
inspection="$(podman run --rm --pull=never --network=none --read-only --cap-drop=ALL --user=65532:65532 --entrypoint= \
    localhost/sim2sim:latest python -I -c '
import json, os
from pathlib import Path
forbidden = ["environments", "configs", "experiments", "runs", "docs", "README.md", ".git", ".env", "manifest.toml", "scene.xml",
             "config-private.json", "experiment.json", "credentials", ".secrets", ".codex", ".aws",
             "mobile_sam", "mobile_sam.pt"]  # optional agent tools are mounted per run, never baked in
hits = sorted(str(p) for root in ("/opt", "/workspace", "/home", "/root", "/") for p in (Path(root).glob("*") if root == "/" else Path(root).rglob("*"))
              if p.name in forbidden or p.suffix == ".toml" and p.name != "pyproject.toml" and "site-packages" not in str(p))
print(json.dumps({"forbidden_hits": hits, "env_names": sorted(os.environ), "agent_files": sorted(p.name for p in Path("/opt/agent").iterdir()),
                  "root_read_only": bool(os.statvfs("/").f_flag & os.ST_RDONLY), "uid": os.getuid()}))')"
.venv/bin/python - "$id" "$inspection" "$trusted_record" "$containerfile" "${SIM2SIM_BASE_IMAGE_RECORD:-}" "${SIM2SIM_IMAGE_ARCHIVE_DIR:-experiments/images}" <<'PY'
import datetime, hashlib, json, os, re, socket, subprocess, sys
from pathlib import Path
image_id, inspection, record_path, containerfile, base_record, archive_dir = sys.argv[1], json.loads(sys.argv[2]), *sys.argv[3:]
assert re.fullmatch(r"[0-9a-f]{64}", image_id), image_id
assert not inspection["forbidden_hits"], inspection["forbidden_hits"]
assert not [n for n in inspection["env_names"] if re.search(r"(?<!GPG_)KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|INFERENCE|OPENAI|AWS", n)], inspection["env_names"]
assert inspection["root_read_only"] and inspection["uid"] == 65532, inspection
archive = Path(archive_dir) / ("sim2sim-" + image_id[:12] + ".tar")
archive.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["podman", "save", "--format=docker-archive", "-o", str(archive), "localhost/sim2sim:latest"], check=True)
digest = hashlib.sha256()
with archive.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
record = {"image_id": image_id, "archive": str(archive), "archive_sha256": digest.hexdigest(), "archive_bytes": archive.stat().st_size,
          "built_job": os.environ["SLURM_JOB_ID"], "node": socket.gethostname(), "built_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "containerfile": containerfile, "containerfile_sha256": hashlib.sha256(Path(containerfile).read_bytes()).hexdigest(),
          **({"base_image_id": json.loads(Path(base_record).read_text())["image_id"], "base_record": base_record} if base_record else {}),
          "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(), "inspection": inspection,
          "note": "Working tree may be uncommitted; the image ID is the trust anchor."}
Path(record_path).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY
podman rmi -f localhost/sim2sim:latest >/dev/null
load_trusted_image  # Prove the shared archive round-trips to the recorded ID.
