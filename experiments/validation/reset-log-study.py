"""Researcher log study for the reset ablation (wave 31): per run, how often and when the agent reset each world, the model's own progress
note just before each reset (what it was for), and, in no-reset runs, the notes that mention resetting, a stuck, dropped or disturbed object,
or working around the missing reset. Reads events.jsonl and progress.jsonl only (no audit, nothing simulated).
Usage: .venv/bin/python experiments/validation/reset-log-study.py CONFIG.json  (the ablation-pairs-plot.py config)"""
import json
import re
import sys
from pathlib import Path

ROOTS = (Path("/n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-experiments/runs"), Path("/n/netscratch/qianqian_lab/Lab/aejxie/sim2sim-experiments/runs"),
         Path("/n/home00/aejxie/sim2sim/experiments/runs"))
COPING = re.compile(r"\breset|restor|stuck|jam|fell|fall|drop|knock|tipped|flipp|disturb|irrevers|cannot undo|can't undo|careful|gently|"
                    r"put (it )?back|return(ed)? (it|the)|original (position|pose|state)|starting (position|pose|state)", re.I)


def run_dir(run):
    hits = [d / run for d in ROOTS if (d / run / "events.jsonl").is_file()]
    assert len(hits) == 1, (run, hits)
    return hits[0]


def study(run):
    d = run_dir(run)
    events = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines() if l.strip()]
    notes = [json.loads(l) for l in (d / "progress.jsonl").read_text().splitlines() if l.strip()]
    notes = [n for n in notes if n.get("origin") == "model" and isinstance(n.get("text"), str)]
    resets = [e for e in events if e.get("op") == "reset"]
    ops = {}
    for e in events:
        if e.get("op"):
            ops[e["op"]] = ops.get(e["op"], 0) + 1
    print(f"\n=== {run}: {len(resets)} resets ({sum(e.get('world') == 'reference' for e in resets)} hidden world, "
          f"{sum(e.get('world') == 'reconstruction' for e in resets)} own model), {len(notes)} notes, ops {dict(sorted(ops.items(), key=lambda kv: -kv[1])[:8])}")
    for e in resets:
        before = [n for n in notes if n["wall_elapsed"] <= e["wall_elapsed"]]
        why = before[-1]["text"].replace("\n", " ")[:260] if before else "(no note before)"
        print(f"  reset {e.get('world'):14s} min {e['wall_elapsed'] / 60:5.1f} ok={e.get('ok')} | last note: {why}")
    for n in notes:
        if COPING.search(n["text"]):
            print(f"  note min {n['wall_elapsed'] / 60:5.1f}: {n['text'].replace(chr(10), ' ')[:300]}")


C = json.loads(Path(sys.argv[1]).read_text())
for title, _, sides in C["rows"]:
    print("\n##########", title.replace("\n", " "))
    for cond, runs in sides.items():
        for r in runs:
            print(f"[{cond}]", end="")
            study(r)
