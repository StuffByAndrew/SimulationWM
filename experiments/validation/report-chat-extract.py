"""Researcher-side: the full model conversation of strict runs, for the results report's per-run pages. Compute node:
  .venv/bin/python experiments/validation/report-chat-extract.py OUT_DIR RUN_DIR...
writes OUT_DIR/<run>/chat.js (window.CHAT = {...}) and OUT_DIR/<run>/img/<sha256>.<ext>. Reads audit envelopes without re-verifying
the hash chain (sim2sim.audit.read_audit is the verified path); never parses MJCF or runs simulation.
Each model call lists the messages that are new since the previous call (everything after the request's last assistant message; all
messages for the first call), then the model's reply. Every message therefore appears once, at the call where it first entered the
context, with its images intact (the harness later replaces old images with a placeholder and old turns with a model-written summary).
Images are the exact bytes the model received, written once per content hash. Private: never mount the output in an agent sandbox."""
import json, shutil, sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

COMPACT = "Your context is nearly full, so older turns will now be removed."  # examples/sandbox_agent.py COMPACT prompt


def load(blobs, pointer):
    data = (blobs / pointer["event_sha256"]).read_bytes()
    assert len(data) == pointer["event_size"], pointer["seq"]
    return json.loads(data)


def place(event, path, value):
    target = event
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def one(args):
    run_dir, out_dir = map(Path, args)
    blobs, out = run_dir / "audit" / "blobs", out_dir / run_dir.name
    pointers = [json.loads(line) for line in (run_dir / "audit" / "events.jsonl").read_text().splitlines()]
    with ThreadPoolExecutor(24) as pool:
        envelopes = list(pool.map(lambda p: load(blobs, p), pointers))
    (out / "img").mkdir(parents=True, exist_ok=True)
    calls, replies, marks, images = [], {}, [], set()
    for env in envelopes:
        event, payload = env["payload"], env["payload"].get("payload")
        kind = event.get("kind")
        if kind == "boundary" and event["direction"] == "request" and isinstance(payload, dict) and payload.get("op") == "inference":
            for item in env["attachments"]:
                if item["encoding"] != "data_url":
                    continue
                assert item["path"][:2] == ["payload", "messages"] and item["prefix"].startswith("data:image/"), item["path"]
                name = f"img/{item['sha256']}.{item['prefix'][len('data:image/'):].split(';')[0].replace('jpeg', 'jpg')}"
                if item["sha256"] not in images:
                    shutil.copyfile(blobs / item["sha256"], out / name)
                    images.add(item["sha256"])
                place(event, item["path"], name)
            calls.append(dict(action_id=event["action_id"], t=event["wall_elapsed"], messages=payload["messages"]))
        elif kind == "boundary" and event["direction"] == "response" and isinstance(payload, dict) and "inference" in payload:
            replies[event["action_id"]] = dict(t=event["wall_elapsed"], **payload["inference"])
        elif kind == "boundary" and event["direction"] == "response" and isinstance(payload, dict) and "submission" in payload:
            marks.append(dict(kind="submission", t=event["wall_elapsed"], submission=payload["submission"], status=payload["status"]))
        elif kind == "termination":
            marks.append(dict(kind="termination", t=event["wall_elapsed"], reason=event["reason"], submissions=event["submissions"]))
    turns, acted, shown = [], None, set()
    for call in calls:
        messages = call["messages"]
        last = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
        # The harness's context-summary request appends one user message; it keeps only the reply's text, never runs its tool calls,
        # and the next request resumes from the turn before the summary request.
        compaction = messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], str) and messages[-1]["content"].startswith(COMPACT)
        if acted is not None and last >= 0 and acted.get("tool_calls") and messages[last].get("tool_calls"):
            assert [c["id"] for c in messages[last]["tool_calls"]] == [c["id"] for c in acted["tool_calls"]], (run_dir.name, call["action_id"])
        new = [m for m in messages[last + 1:] if json.dumps(m, sort_keys=True) not in shown]  # each message once, where it first appeared
        shown.update(json.dumps(m, sort_keys=True) for m in new)
        reply = replies.get(call["action_id"])  # None: the run ended (or the call failed) before a reply was recorded
        turns.append(dict(t=call["t"], new=new, context=len(messages), reply=reply, compaction=compaction))
        if not compaction:
            acted = reply
    (out / "chat.js").write_text("window.CHAT=" + json.dumps(dict(run=run_dir.name, turns=turns, marks=marks), separators=(",", ":")) + ";")
    return run_dir.name, len(pointers), len(turns), len(images), sum(f.stat().st_size for f in (out / "img").iterdir())


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(min(16, len(sys.argv) - 2)) as pool:
        for row in pool.map(one, [(r, out) for r in sys.argv[2:]]):
            print(*row, flush=True)
