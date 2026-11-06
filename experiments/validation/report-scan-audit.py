# Run on a compute node: srun -p sapphire -c 24 --mem=48G .venv/bin/python experiments/validation/report-scan-audit.py WORK/traces RUN_DIR...
"""Researcher-side fast scan of audit envelopes (no hash-chain verification; sim2sim.audit.read_audit is the verified path):
keep model inference responses (text + tool calls) and controller action_begin ops. Usage: python scan_audit.py OUT_DIR RUN_DIR..."""
import json, sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path


def envelope(blobs, pointer):
    data = (blobs / pointer["event_sha256"]).read_bytes()
    assert len(data) == pointer["event_size"]
    env = json.loads(data)
    event = env["payload"]
    for item in env["attachments"]:
        if item["encoding"] != "json":
            continue  # wires and images are not needed here
        target = event
        for part in item["path"][:-1]:
            target = target[part]
        target[item["path"][-1]] = json.loads((blobs / item["sha256"]).read_bytes())
    return event


def keep(event):
    kind = event.get("kind")
    if kind == "action_begin":
        return dict(kind="action", t=event["wall_elapsed"], op=event["resolved_action"].get("op"), world=event.get("world"))
    payload = event.get("payload")
    if kind == "boundary" and event["direction"] == "response" and isinstance(payload, dict) and "inference" in payload:
        inf = payload["inference"]
        return dict(kind="response", t=event["wall_elapsed"], content=inf.get("content"),
                    tool_calls=inf.get("tool_calls"), budget_stop=inf.get("budget_stop", False))
    return None


def one(args):
    run_dir, out_dir = map(Path, args)
    pointers = [json.loads(line) for line in (run_dir / "audit" / "events.jsonl").read_text().splitlines()]
    blobs = run_dir / "audit" / "blobs"
    with ThreadPoolExecutor(24) as pool:
        events = [e for e in pool.map(lambda p: keep(envelope(blobs, p)), pointers) if e]
    (out_dir / f"{run_dir.name}.json").write_text(json.dumps(events))
    return run_dir.name, len(pointers), sum(e["kind"] == "response" for e in events)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(exist_ok=True)
    with ProcessPoolExecutor(len(sys.argv) - 2) as pool:
        for row in pool.map(one, [(r, out) for r in sys.argv[2:]]):
            print(*row, flush=True)
