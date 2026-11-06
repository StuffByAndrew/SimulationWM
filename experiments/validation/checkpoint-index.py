"""Index every scored checkpoint of a run by wall time AND by cumulative model tokens, so conditions can be compared at
equal elapsed time (e.g. the 2 h mark of a 4 h run) or at equal tokens consumed (compute proxy).
Usage (compute node; NFS-latency bound): .venv/bin/python experiments/validation/checkpoint-index.py OUT.json RUN_DIR...
Fast researcher-side scan of audit envelopes (no hash-chain re-verification; read_audit is the verified path).
Token sources: old runs' inference replies carried cumulative_tokens; cost-blind runs record it in the host-only
`inference_accounting` event. A submission's tokens = cumulative tokens of the last inference completed before it."""
import json, sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path


def envelope(blobs, pointer):
    env = json.loads((blobs / pointer["event_sha256"]).read_bytes())
    event = env["payload"]
    for item in env["attachments"]:
        if item["encoding"] == "json":
            target = event
            for part in item["path"][:-1]:
                target = target[part]
            target[item["path"][-1]] = json.loads((blobs / item["sha256"]).read_bytes())
    return event


def keep(event):
    kind, t = event.get("kind"), event.get("wall_elapsed")
    if kind == "inference_accounting":
        return ("tokens", t, event["cumulative_tokens"])
    payload = event.get("payload")
    if kind == "boundary" and event.get("direction") == "response" and isinstance(payload, dict) \
            and isinstance(payload.get("inference"), dict) and payload["inference"].get("cumulative_tokens") is not None:
        return ("tokens", t, payload["inference"]["cumulative_tokens"])
    if kind == "action_begin" and (event.get("resolved_action") or {}).get("op") == "submit":
        return ("submit", t, None)
    return None


def index(run_dir):
    run_dir = Path(run_dir)
    pointers = [json.loads(line) for line in (run_dir / "audit" / "events.jsonl").read_text().splitlines()]
    with ThreadPoolExecutor(24) as pool:
        kept = sorted((e for e in pool.map(lambda p: keep(envelope(run_dir / "audit" / "blobs", p)), pointers) if e),
                      key=lambda e: e[1])
    tokens, submits = 0, []
    for kind, t, value in kept:
        if kind == "tokens":
            assert value >= tokens, (run_dir.name, t, value, tokens)
            tokens = value
        else:
            submits.append(dict(wall_min=t / 60, tokens=tokens))
    history = json.loads((run_dir / "metric-history.json").read_text())["checkpoints"]
    assert len(history) == len(submits), (run_dir.name, len(history), len(submits))
    for row, checkpoint in zip(submits, history):
        row.update(submission=checkpoint["submission"], status=checkpoint["status"])
    return run_dir.name, dict(checkpoints=submits, final_tokens=tokens, final_wall_min=kept[-1][1] / 60 if kept else None)


def at(checkpoints, wall_min=None, tokens=None):
    """Last scored checkpoint available at a wall-time or token budget (None if nothing submitted yet)."""
    key, limit = ("wall_min", wall_min) if wall_min is not None else ("tokens", tokens)
    return ([c for c in checkpoints if c[key] <= limit] or [None])[-1]


if __name__ == "__main__":
    with ProcessPoolExecutor(min(16, len(sys.argv) - 2)) as pool:
        result = dict(pool.map(index, sys.argv[2:]))
    Path(sys.argv[1]).write_text(json.dumps(result, indent=1))
    for name, r in result.items():
        print(name, "final_tokens=%d" % r["final_tokens"], [(round(c["wall_min"]), c["tokens"]) for c in r["checkpoints"]])
