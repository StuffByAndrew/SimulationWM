"""Researcher study of text-only model turns (a reply with no tool call: the 'intention-only' loop) from report-chat-extract.py output.
Per run: model turns (context-compaction summaries excluded), text-only turns and their share of turns and of run time (each turn's time is
until the next turn's request; the last turn's is its own latency), the longest consecutive text-only streak (length, start minute, share of
run time) and the most frequent text-only replies. Usage: .venv/bin/python experiments/validation/tool-loop-study.py CHAT_DIR [RUN ...]  (Inkling 2026-11-27: report-work-20261127-inkling/tool-loop-study.jsonl)"""
import json
import sys
from collections import Counter
from pathlib import Path

chat_dir = Path(sys.argv[1])
runs = sys.argv[2:] or sorted(p.name for p in chat_dir.iterdir() if (p / "chat.js").is_file())
for run in runs:
    s = (chat_dir / run / "chat.js").read_text()
    turns = [t for t in json.loads(s[len("window.CHAT="):-1])["turns"] if not t.get("compaction") and t.get("reply")]
    starts = [float(t["t"]) for t in turns]
    spans = [(starts[i + 1] if i + 1 < len(turns) else float(t["reply"]["t"])) - starts[i] for i, t in enumerate(turns)]
    text_only = [not t["reply"].get("tool_calls") for t in turns]
    total = sum(spans)
    best, cur, best_start, cur_start = 0, 0, None, None
    for i, flag in enumerate(text_only):
        if flag:
            cur_start = i if cur == 0 else cur_start
            cur += 1
            if cur > best:
                best, best_start = cur, cur_start
        else:
            cur = 0
    streak_time = sum(spans[best_start:best_start + best]) if best else 0.0
    phrases = Counter((t["reply"].get("content") or "").strip()[:70] for t, f in zip(turns, text_only) if f)
    n = sum(text_only)
    print(json.dumps(dict(run=run, turns=len(turns), text_only=n, share_turns=round(n / len(turns), 3),
                          share_time=round(sum(sp for sp, f in zip(spans, text_only) if f) / total, 3),
                          longest_streak=best, streak_start_min=None if best_start is None else round(starts[best_start] / 60, 1),
                          streak_share_time=round(streak_time / total, 3), first_text_only_min=next((round(st / 60, 1) for st, f in zip(starts, text_only) if f), None),
                          run_minutes=round(total / 60, 1), top_text_only=phrases.most_common(3))))
