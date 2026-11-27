"""Write a report work folder's checkpoints-baselines.json: the checkpoint-index.py entries of a spec's comparison runs that the folder's own
checkpoints-cost.json lacks, one source per run (the cost-annotated Gemini-report index first, else the cumulative report's indexes, which must
agree with each other). Avoids the builder's merge conflict between the same run indexed with and without cumulative cost.
Usage: .venv/bin/python experiments/validation/baseline-checkpoints.py SPEC.py WORK_DIR"""
import importlib.util
import json
import sys
from pathlib import Path

spec_path, work = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("spec", spec_path)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
own = json.loads((work / "checkpoints-cost.json").read_text())
cost = {k: v for p in S.G.CHECKPOINTS for k, v in json.loads(Path(p).read_text()).items()}
plain = [json.loads(Path(p).read_text()) for p in S.B.CHECKPOINTS]
out = {}
for run in S.NAMES:
    if run in own:
        continue
    if run in cost:
        out[run] = cost[run]
        continue
    hits = [d[run] for d in plain if run in d]
    assert hits, (run, "no checkpoint index in any source")
    assert all(h == hits[0] for h in hits), (run, "sources disagree")
    out[run] = hits[0]
(work / "checkpoints-baselines.json").write_text(json.dumps(out))
print(len(out), "baseline runs;", len([r for r in S.NAMES if r in own]), "from the folder's own index")
