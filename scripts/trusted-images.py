"""Print `NAME TAG IMAGE_ID ARCHIVE SHA256` per image of a trusted image record (scripts/podman-job.sh load_trusted_image).

Two shapes, dispatched explicitly: a two-image record (scripts/build-image.slurm since 2026-11-24) has only "images" ->
"agent"/"evaluator" with tag, image_id, archive and archive_sha256, and prints agent_image and evaluator_image; an older
single-image record has top-level image_id, archive and archive_sha256 and no "images", and prints one line, image, tagged
localhost/sim2sim:latest (what its prepared configs name for both sandbox images). Anything else fails. Standard library
only; runs under the host python3.
"""
import json
import sys

record = json.load(open(sys.argv[1]))
if "images" in record:
    images = record["images"]
    assert "image_id" not in record and sorted(images) == ["agent", "evaluator"], sorted(record)
    assert len({images[r]["tag"] for r in images}) == len({images[r]["image_id"] for r in images}) == 2, images
    rows = [(role + "_image", images[role]) for role in ("agent", "evaluator")]
else:
    assert {"image_id", "archive", "archive_sha256"} <= set(record), sorted(record)
    rows = [("image", dict(record, tag="localhost/sim2sim:latest"))]
for name, image in rows:
    print(name, image["tag"], image["image_id"], image["archive"], image["archive_sha256"])
