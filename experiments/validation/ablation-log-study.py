"""Researcher log study for an ablation (ablation-pairs-plot.py config): per run, the listed controller ops (events.jsonl) with the model's own
progress note just before each (what it was for), optional tool-call counts from the chat extract (report-chat-extract.py), and the model's
notes matching a pattern. Reads events.jsonl, progress.jsonl and chat.js only (no audit, nothing simulated).
Usage: .venv/bin/python experiments/validation/ablation-log-study.py CONFIG.json [--ops reset] [--pattern REGEX] [--chat DIR --calls NAME=REGEX ...]
  Reset ablation (defaults): --ops reset and the coping pattern below.
  Touch harness: --ops none --chat <work>/chat --calls local_controller='LocalController|localsim|taxels\\(' --calls bash_resume='^bash_resume$'
                 --calls bash_kill='^bash_kill$' --pattern 'local ?controller|localsim|LocalController|taxel|offline|locally|resume'
A --calls regex is matched against "name" alone when it starts with ^, otherwise against the call's name and arguments."""
import argparse
import json
import re
from pathlib import Path

ROOTS = (Path("/n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-experiments/runs"), Path("/n/netscratch/qianqian_lab/Lab/aejxie/sim2sim-experiments/runs"),
         Path("/n/home00/aejxie/sim2sim/experiments/runs"))
COPING = (r"\breset|restor|stuck|jam|fell|fall|drop|knock|tipped|flipp|disturb|irrevers|cannot undo|can't undo|careful|gently|"
          r"put (it )?back|return(ed)? (it|the)|original (position|pose|state)|starting (position|pose|state)")


def run_dir(run):
    hits = [d / run for d in ROOTS if (d / run / "events.jsonl").is_file()]
    assert len(hits) == 1, (run, hits)
    return hits[0]


def calls(chat_dir, run, rules):
    """Per rule: the minutes of the matching tool calls (reply times) in the run's chat extract."""
    s = (chat_dir / run / "chat.js").read_text()
    turns = [t for t in json.loads(s[len("window.CHAT="):-1])["turns"] if t.get("reply") and not t.get("compaction")]
    out = {name: [] for name in rules}
    for t in turns:
        for tc in t["reply"].get("tool_calls") or []:
            for name, rule in rules.items():
                text = tc["name"] if rule.startswith("^") else f"{tc['name']} {tc.get('arguments', '')}"
                if re.search(rule, text):
                    out[name].append(float(t["reply"]["t"]) / 60)
    return out, sum(len(t["reply"].get("tool_calls") or []) for t in turns)


def study(run, ops, pattern, chat_dir, rules):
    d = run_dir(run)
    events = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines() if l.strip()]
    notes = [json.loads(l) for l in (d / "progress.jsonl").read_text().splitlines() if l.strip()]
    notes = [n for n in notes if n.get("origin") == "model" and isinstance(n.get("text"), str)]
    listed = [e for e in events if e.get("op") in ops]
    counts = {}
    for e in events:
        if e.get("op"):
            counts[e["op"]] = counts.get(e["op"], 0) + 1
    print(f"\n=== {run}: {len(listed)} {'/'.join(sorted(ops)) or 'listed ops'}, {len(notes)} notes, ops {dict(sorted(counts.items(), key=lambda kv: -kv[1])[:8])}")
    if chat_dir:
        found, total = calls(chat_dir, run, rules)
        for name, mins in found.items():
            span = f", minutes {mins[0]:.0f}–{mins[-1]:.0f}" if mins else ""
            print(f"  calls {name}: {len(mins)} of {total} tool calls{span}")
    for e in listed:
        before = [n for n in notes if n["wall_elapsed"] <= e["wall_elapsed"]]
        why = before[-1]["text"].replace("\n", " ")[:260] if before else "(no note before)"
        print(f"  {e['op']} {e.get('world'):14s} min {e['wall_elapsed'] / 60:5.1f} ok={e.get('ok')} | last note: {why}")
    for n in notes:
        if pattern.search(n["text"]):
            print(f"  note min {n['wall_elapsed'] / 60:5.1f}: {n['text'].replace(chr(10), ' ')[:300]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--ops", default="reset", help="comma-separated events.jsonl ops to list ('none' for no ops)")
    ap.add_argument("--pattern", default=COPING, help="regex for the model notes to quote")
    ap.add_argument("--chat", type=Path, help="report-chat-extract.py output folder, for --calls")
    ap.add_argument("--calls", action="append", default=[], metavar="NAME=REGEX", help="tool calls to count (needs --chat)")
    a = ap.parse_args()
    rules = dict(c.split("=", 1) for c in a.calls)
    assert not rules or a.chat, "--calls needs --chat"
    ops = set() if a.ops == "none" else set(a.ops.split(","))
    C = json.loads(Path(a.config).read_text())
    for title, _, sides in C["rows"]:
        print("\n##########", title.replace("\n", " "))
        for cond, runs in sides.items():
            for r in runs:
                print(f"[{cond}]", end="")
                study(r, ops, re.compile(a.pattern, re.I), a.chat, rules)


if __name__ == "__main__":
    main()
