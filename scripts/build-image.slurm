#!/usr/bin/env bash
#SBATCH --partition=test
#SBATCH --job-name=sim2sim-image
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=experiments/validation/build-image-%j.log
# Build, inspect, archive and record the trusted agent and evaluator images. No agent or model runs.
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == test ]] || { echo 'Image builds require a Slurm test allocation.' >&2; exit 2; }
cd "${SIM2SIM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
# Optional: SIM2SIM_CONTAINERFILE (default Containerfile, targets agent and evaluator); SIM2SIM_BASE_IMAGE_RECORD loads a
# verified two-image record for a derived Containerfile (FROM the agent image), which replaces only the agent image and keeps
# the base record's evaluator entry; SIM2SIM_TRUSTED_IMAGE_RECORD (podman-job.sh) names the record written;
# SIM2SIM_IMAGE_ARCHIVE_DIR (default experiments/images) holds the archives.
containerfile="${SIM2SIM_CONTAINERFILE:-Containerfile}"
source scripts/podman-job.sh
declare -A tags=([agent]=localhost/sim2sim-agent:latest [evaluator]=localhost/sim2sim:latest)
if [[ -n "${SIM2SIM_BASE_IMAGE_RECORD:-}" ]]; then
    trusted_record="$SIM2SIM_BASE_IMAGE_RECORD" load_trusted_image
    roles=(agent)
    podman build -f "$containerfile" -t "${tags[agent]}" .
else
    roles=(agent evaluator)
    for role in "${roles[@]}"; do podman build -f "$containerfile" --target "$role" -t "${tags[$role]}" .; done
fi
# First docstring line of every package module: the agent inspection greps the agent image for each absent module's
# file path, dotted name and docstring phrase.
docstrings="$(.venv/bin/python -c 'import ast, json; from pathlib import Path
print(json.dumps({p.stem: (ast.get_docstring(ast.parse(p.read_text())) or "").split("\n")[0] for p in sorted(Path("src/sim2sim").glob("*.py"))}))')"
inspections=()
for role in "${roles[@]}"; do
    inspections+=("$role" "${tags[$role]}" "$(podman run --rm --pull=never --network=none --read-only --cap-drop=ALL --user=65532:65532 --entrypoint= \
        "${tags[$role]}" python -I -c '
import json, os, re, sys
from pathlib import Path
role, docstrings = sys.argv[1], json.loads(sys.argv[2])
forbidden = ["environments", "configs", "experiments", "runs", "docs", "README.md", ".git", ".env", "manifest.toml", "scene.xml",
             "config-private.json", "experiment.json", "credentials", ".secrets", ".codex", ".aws",
             "mobile_sam", "mobile_sam.pt"]  # optional agent tools are mounted per run, never baked in
hits = sorted(str(p) for root in ("/opt", "/workspace", "/home", "/root", "/") for p in (Path(root).glob("*") if root == "/" else Path(root).rglob("*"))
              if p.name in forbidden or p.suffix == ".toml" and p.name != "pyproject.toml" and "site-packages" not in str(p))
result = {"forbidden_hits": hits, "env_names": sorted(os.environ), "root_read_only": bool(os.statvfs("/").f_flag & os.ST_RDONLY), "uid": os.getuid()}
if role == "agent":  # everything this uid can read, outside /proc, /sys and /dev
    import sim2sim
    present = sorted(p.stem for p in Path(sim2sim.__file__).parent.iterdir())
    excluded = sorted(set(docstrings) - set(present))
    names = "|".join(excluded)
    pattern = re.compile(("sim2sim/(?:%s)\\.py|sim2sim\\.(?:%s)\\b" % (names, names)).encode() + b"".join(b"|" + re.escape(docstrings[m].encode()) for m in excluded if docstrings[m]))
    found, scanned, unreadable = [], 0, 0
    for top in sorted(p for p in Path("/").iterdir() if p.name not in ("proc", "sys", "dev") and not p.is_symlink()):
        for directory, _, files in os.walk(top):
            for name in files:
                path = os.path.join(directory, name)
                if os.path.islink(path) or not os.path.isfile(path):
                    continue
                try:
                    match = pattern.search(Path(path).read_bytes())
                except PermissionError:
                    unreadable += 1
                    continue
                scanned += 1
                if match:
                    found.append([path, match.group().decode(errors="replace")])
    result.update(agent_files=sorted(p.name for p in Path("/opt/agent").iterdir()), agent_modules=present, excluded_modules=excluded,
                  excluded_hits=found, scanned_files=scanned, unreadable_files=unreadable)
print(json.dumps(result))' "$role" "$docstrings")")
done
.venv/bin/python - "$trusted_record" "$containerfile" "${SIM2SIM_BASE_IMAGE_RECORD:-}" "${SIM2SIM_IMAGE_ARCHIVE_DIR:-experiments/images}" "${inspections[@]}" <<'PY'
import datetime, hashlib, json, os, re, socket, subprocess, sys
from pathlib import Path
record_path, containerfile, base_record, archive_dir, *rest = sys.argv[1:]
METRIC_MODULES = {"evaluation", "runtime", "worker", "fixture", "fixture_view", "die", "launcher", "tabletop", "cloth", "force_profile",
                  "primitive_metrics", "baselines", "gear_train"}  # a guard: never in the agent image (the scan covers every absent module)
images = {"evaluator": json.loads(Path(base_record).read_text())["images"]["evaluator"]} if base_record else {}
for role, tag, raw in zip(rest[0::3], rest[1::3], rest[2::3]):
    inspection = json.loads(raw)
    image_id = subprocess.run(["podman", "image", "inspect", tag, "--format", "{{.Id}}"], capture_output=True, text=True, check=True).stdout.strip()
    labels = json.loads(subprocess.run(["podman", "image", "inspect", tag, "--format", "{{json .Config.Labels}}"], capture_output=True, text=True, check=True).stdout)
    assert re.fullmatch(r"[0-9a-f]{64}", image_id), image_id
    assert labels.get("sim2sim.image-role") == role, labels
    assert not inspection["forbidden_hits"], inspection["forbidden_hits"]
    assert not [n for n in inspection["env_names"] if re.search(r"(?<!GPG_)KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|INFERENCE|OPENAI|AWS", n)], inspection["env_names"]
    assert inspection["root_read_only"] and inspection["uid"] == 65532, inspection
    if role == "agent":
        assert not inspection["excluded_hits"] and METRIC_MODULES <= set(inspection["excluded_modules"]), inspection
        assert inspection["agent_files"] == ["sandbox_agent.py"] and inspection["scanned_files"] > 1000, inspection
    archive = Path(archive_dir) / f"sim2sim-{role}-{image_id[:12]}.tar"
    archive.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["podman", "save", "--format=docker-archive", "-o", str(archive), tag], check=True)
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    images[role] = {"tag": tag, "image_id": image_id, "archive": str(archive), "archive_sha256": digest.hexdigest(),
                    "archive_bytes": archive.stat().st_size, "labels": labels, "inspection": inspection}
assert set(images) == {"agent", "evaluator"} and images["agent"]["image_id"] != images["evaluator"]["image_id"], images
record = {"images": images, "built_job": os.environ["SLURM_JOB_ID"], "node": socket.gethostname(),
          "built_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "containerfile": containerfile, "containerfile_sha256": hashlib.sha256(Path(containerfile).read_bytes()).hexdigest(),
          **({"base_record": base_record, "base_agent_image_id": json.loads(Path(base_record).read_text())["images"]["agent"]["image_id"]} if base_record else {}),
          "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
          "note": "Working tree may be uncommitted; the image IDs are the trust anchors."}
Path(record_path).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY
podman rmi -f "${tags[@]}" >/dev/null
load_trusted_image  # Prove both shared archives round-trip to the recorded IDs.
