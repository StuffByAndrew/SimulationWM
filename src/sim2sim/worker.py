"""Internal evaluator entrypoint. Runs only inside the constrained OCI container."""
import base64
import json
from pathlib import Path
import sys

from .runtime import decode_submission, extract_bundle, zip_directory


def main():
    try:
        line = sys.stdin.buffer.readline(180 * 1024 * 1024 + 1)
        if len(line) > 180 * 1024 * 1024:
            raise ValueError("Worker input exceeds byte limit")
        payload = json.loads(line)
        source = Path("/workspace/source")
        output = Path("/workspace/output")
        extract_bundle(decode_submission(payload["source_b64"], 64 * 1024 * 1024), source)
        config = dict(payload["config"])
        config.update(source_dir=str(source), output_dir=str(output))
        from .evaluation import evaluate_submission
        report = evaluate_submission(decode_submission(payload["submission_b64"], 8 * 1024 * 1024), config)
        artifacts = zip_directory(output, 96 * 1024 * 1024)
        response = {"report": report, "artifacts_b64": base64.b64encode(artifacts).decode()}
    except Exception as exc:
        response = {"error": type(exc).__name__ + ": " + str(exc)}
    print(json.dumps(response, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
