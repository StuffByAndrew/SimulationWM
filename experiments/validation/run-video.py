"""Researcher-only: compact scrubbable recording of one strict run, encoded from the frames its verified audit already holds.

usage (compute node): .venv/bin/python experiments/validation/run-video.py RUN_DIR [RUN_DIR ...]
writes <run root>/recordings/<run id>/{reference.mp4, reconstruction.mp4, recording.json} next to the run's root/runs folder.

Every RGB observation the controller recorded (the reference world at its recording rate while time advances, every
observation returned to the agent, and the agent's reconstruction world) is read from the hash-verified audit
(sim2sim.audit.read_audit, the same frames the dashboard's recordings view shows) and encoded as H.264, one MP4 per world,
at a fixed 10 frames per second with a keyframe every 10 frames so a player can seek any frame quickly. recording.json maps
each video frame to its run-clock time (s), simulation time, episode and action, and lists the run's actions, submissions,
evaluations and model progress notes on the same clock. Nothing is simulated, rendered or inferred; frames never recorded
stay absent. The audit is only read.
"""
import io
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from sim2sim.audit import read_audit
from sim2sim.replay import _observations

FPS = 10  # video frames per second of encoded video; the player maps frames to run-clock time through recording.json
WORLDS = ("reference", "reconstruction")


def collect(run):
    """Frames (PNG bytes) per world and the run's timeline, in audit order."""
    frames = {w: [] for w in WORLDS}
    actions, notes, submissions, evaluations, active, seen = [], [], [], [], {}, set()
    for pointer, event in read_audit(run):
        kind, world, wall = event.get("kind"), event.get("world"), event.get("wall_elapsed")
        if kind == "action_begin" and world in WORLDS and (event.get("resolved_action") or {}).get("op") not in ("progress", "inference"):
            op = (event.get("resolved_action") or {}).get("op")
            active[world] = dict(world=world, op=op, start=wall, end=None, action_id=event.get("action_id"), index=len(actions))
            actions.append(active[world])
            if op == "submit":
                submissions.append(dict(t=wall, world=world))
        elif kind == "action_end" and world in active:
            active[world].update(end=wall, outcome=event.get("outcome"))
        elif kind == "progress" and event.get("origin") == "model" and isinstance(event.get("text"), str):
            notes.append(dict(t=wall, text=event["text"]))
        elif kind == "evaluation":
            report = event.get("report") or {}
            evaluations.append(dict(t=wall, submission=event.get("submission"), status=report.get("status"),
                                    checkpoint_kind=event.get("checkpoint_kind", "submission")))
        if world not in WORLDS or not isinstance(wall, (int, float)):
            continue
        sources = [] if kind == "boundary" and event.get("direction") == "request" else [event]
        for value in sources:
            for _, obs, images in _observations(value):
                png = images.get("rgb")
                if png is None:
                    continue
                key = (world, wall, obs.get("time"), hash(png))  # the same frame stored twice in one event counts once
                if key in seen:
                    continue
                seen.add(key)
                frames[world].append(dict(png=png, t=wall, sim_time=obs.get("time"), ticks=obs.get("ticks"),
                                          action=active[world]["index"] if world in active else None))
    return frames, dict(actions=actions, notes=notes, submissions=submissions, evaluations=evaluations)


def encode(path, frames):
    """H.264, constant 10 fps, keyframe every 10 frames, yuv420p (plays in every browser). Frame sizes must agree."""
    first = np.asarray(Image.open(io.BytesIO(frames[0]["png"])).convert("RGB"))
    with imageio.get_writer(path, fps=FPS, codec="libx264", macro_block_size=8, quality=None, pixelformat="yuv420p",
                            ffmpeg_params=["-crf", "28", "-preset", "veryfast", "-g", str(FPS)]) as writer:
        for f in frames:
            image = np.asarray(Image.open(io.BytesIO(f["png"])).convert("RGB"))
            assert image.shape == first.shape, (path, image.shape, first.shape)
            writer.append_data(image)
    return list(first.shape[:2])


def record(run):
    run = Path(run).resolve()
    assert run.parent.name == "runs", run
    out = run.parent.parent / "recordings" / run.name
    out.mkdir(parents=True, exist_ok=True)
    frames, timeline = collect(run)
    index = dict(schema_version=1, run=run.name, fps=FPS, source="verified audit frames (sim2sim.audit.read_audit); nothing simulated or rendered",
                 worlds={}, **timeline)
    for world, items in frames.items():
        if not items:
            continue
        size = encode(out / f"{world}.mp4", items)
        index["worlds"][world] = dict(file=f"{world}.mp4", frames=len(items), height=size[0], width=size[1],
                                      t=[round(f["t"], 3) for f in items], sim_time=[f["sim_time"] for f in items],
                                      action=[f["action"] for f in items])
    (out / "recording.json").write_text(json.dumps(index, separators=(",", ":")))
    sizes = {w: (out / v["file"]).stat().st_size for w, v in index["worlds"].items()}
    print(run.name, {w: v["frames"] for w, v in index["worlds"].items()}, {w: f"{s / 1e6:.1f} MB" for w, s in sizes.items()}, flush=True)


if __name__ == "__main__":
    for run in sys.argv[1:]:
        record(run)
