"""Researcher-only: did an agent on a spring level use the spring as a scale in the hidden scene?

usage (compute node; reads every recorded reference tick state): .venv/bin/python experiments/validation/spring-scale-usage.py OUT.json RUN_DIR...

For every 10th recorded state of the reference world the hidden scene is posed (qpos, mj_forward) and checked for:
- spring_in_hole: the bottom plate sits more than 8 cm below where it rests at reset (it starts lying on the table top, and only
  the 9 cm hole lets it drop that far);
- <body>_on_spring: a contact between the body (cluster or cube, block, plug) and the top plate;
- <body>_weighed: that contact while the spring is in the hole (the weighing the scene was built for: compression = m g / k).
Per run: first wall time (s) and number of sampled states for each, plus the agent's progress notes that mention the hole or
weighing. Agents may also have weighed things in their own local simulator; that is not counted here.
"""
import base64
import json
import re
import struct
import sys
from pathlib import Path

import mujoco
import numpy as np

EVERY = 10
NOTE = re.compile(r"\bhole\b|\bweigh|calibrat|scale for|spring as a (scale|balance)", re.I)


def scan(run):
    private = json.loads((run / "config-private.json").read_text())
    scene = run / "source" / private["evaluation"]["source_entrypoint"]
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    body = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    base, top = body("spring_base"), body("spring_top")
    assert base >= 0 and top >= 0, run.name
    targets = {n: body(n) for n in ("cluster", "cube", "block", "plug") if body(n) >= 0}
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    rest_z = float(data.xpos[base][2])
    first, count, n, tick = {}, {}, 0, 0
    audit = run / "audit"
    blob = lambda sha: json.loads((audit / "blobs" / sha).read_bytes())
    for line in open(audit / "events.jsonl"):
        event = blob(json.loads(line)["event_sha256"])
        payload = event["payload"]
        if payload.get("kind") != "tick" or payload.get("world") != "reference":
            continue
        tick += 1
        if tick % EVERY:
            continue
        raw = base64.b64decode(blob(event["attachments"][0]["sha256"])["arrays"]["qpos"]["data_b64"])
        qpos = struct.unpack("<%dd" % (len(raw) // 8), raw)
        assert len(qpos) == model.nq, (run.name, len(qpos), model.nq)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        n += 1
        flags = {"spring_in_hole": data.xpos[base][2] < rest_z - 0.08}
        touching = {model.geom_bodyid[c.geom1] for c in data.contact[:data.ncon] if model.geom_bodyid[c.geom2] == top} | \
                   {model.geom_bodyid[c.geom2] for c in data.contact[:data.ncon] if model.geom_bodyid[c.geom1] == top}
        for name, b in targets.items():
            flags[f"{name}_on_spring"] = b in touching
            flags[f"{name}_weighed"] = b in touching and flags["spring_in_hole"]
        for k, v in flags.items():
            if v:
                count[k] = count.get(k, 0) + 1
                first.setdefault(k, round(payload["wall_elapsed"]))
    notes = [dict(min=round(x["wall_elapsed"] / 60), text=x["text"]) for x in map(json.loads, (run / "progress.jsonl").read_text().splitlines())
             if x.get("origin") == "model" and NOTE.search(x["text"])]
    return dict(run=run.name, sampled_states=n, every=EVERY, targets=sorted(targets), first_wall_s=first, sampled_counts=count, notes=notes)


if __name__ == "__main__":
    rows = [scan(Path(r)) for r in sys.argv[2:]]
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r["run"], r["sampled_states"], r["first_wall_s"], len(r["notes"]), flush=True)
