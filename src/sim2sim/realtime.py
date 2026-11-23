"""Real-time reference world (with the optional seeded launcher of launcher.py).

`[simulation] realtime = true` with an integer `frame_rate` (default false: every lock-step level is unchanged) is a generic
environment setting for any level and either embodiment (point grasp or robot): a host thread advances the reference world
at one simulated second per wall-clock second, also while the agent reasons or its code runs. Agent actions apply to the
live world in arrival order; there is no pause, slow motion, tick advance, reconstruction world or state access. Camera
frames (the configured width x height and modalities) exist only on a fixed grid of `frame_rate` per simulated second since
the latest reset and are rendered on request from the recorded state. A level whose private manifest has a `[launcher]`
table (launcher.py) additionally gets perturbed resets and the `throw` action; without one, reset restores the scene's
initial state. See docs/protocol.md.
"""
from __future__ import annotations

import base64
from collections import deque
from contextlib import contextmanager
import math
import threading
import time

import mujoco
import numpy as np

from .launcher import Launcher


class WorldFailure(RuntimeError):
    """The real-time physics thread stopped; the run cannot continue."""


def restore(sim, state, camera):
    """Load a recorded snapshot_simulation state and observation-camera pose into sim (a separate rendering copy)."""
    for key, value in state["arrays"].items():
        array = np.frombuffer(base64.b64decode(value["data_b64"]), dtype=value["dtype"]).reshape(value["shape"])
        (sim._model.eq_data if key == "model.eq_data" else getattr(sim._data, key))[...] = array
    sim._data.time, sim.ticks, sim.total_ticks = state["time"], state["ticks"], state["total_ticks"]
    sim._grasp = tuple(state["grasp"]) if state["grasp"] is not None else None
    sim._model.cam_pos[sim._camera_id], sim._model.cam_quat[sim._camera_id] = camera
    mujoco.mj_forward(sim._model, sim._data)


class RealtimeWorld:
    """The live reference world, stepped by a host thread at the wall-clock rate.

    Commands run in arrival order at tick boundaries. A timed command (move, arm_target, hand_target) occupies its ticks and
    later commands wait for it, so chained commands (move, then release) take effect back to back whatever the transport
    latency; every command returns when it takes effect. Frames are captured every frame_ticks ticks since the latest
    reset or throw (a launch captures its frame 0 at once) and kept for history_seconds of simulated time. Images are
    rendered in the caller's thread from the recorded state with a second Simulation of the same scene, because MuJoCo
    rendering is bound to the thread of its context; the physics thread never renders."""
    TIMED = ("move", "arm_target", "hand_target")
    INSTANT = ("grasp", "release", "camera", "reset", "throw")

    def __init__(self, sim, render_sim, frame_rate, launcher=None, history_seconds=20.):
        from .candidate_worker import snapshot_simulation
        self.snapshot = snapshot_simulation
        self.sim, self.render_sim, self.launcher = sim, render_sim, launcher
        self.dt = float(sim._model.opt.timestep)
        self.frame_ticks = round(1 / (frame_rate * self.dt))
        assert self.frame_ticks >= 1 and math.isclose(self.frame_ticks * self.dt * frame_rate, 1), "frame period must be whole ticks"
        self.kept = deque(maxlen=max(1, round(history_seconds * frame_rate)))
        self.lock = threading.Condition()
        self.pending, self.motion = deque(), None
        self.launches = self.episode = self.next_frame = 0
        self.stopped, self.error, self.paused_at, self.max_lag = False, None, None, 0
        with self.lock:
            self._launch(None)
        self.origin = time.monotonic() - sim.total_ticks * self.dt
        self.thread = threading.Thread(target=self._loop, name="realtime-world", daemon=True)
        self.thread.start()

    # Physics thread -----------------------------------------------------------------------------------------------------
    def _due(self):
        return int((time.monotonic() - self.origin) / self.dt)

    def _loop(self):
        try:
            while True:
                with self.lock:
                    while not self.stopped and (self.paused_at is not None or
                                                (self.sim.total_ticks >= self._due() and not (self.pending and self.motion is None))):
                        self.lock.wait(None if self.paused_at is not None else max(1e-4, self.origin + (self.sim.total_ticks + 1) * self.dt - time.monotonic()))
                    if self.stopped:
                        return
                    self._start_pending()
                    due = self._due()
                    self.max_lag = max(self.max_lag, due - self.sim.total_ticks)
                if self.sim.total_ticks < due:
                    self._tick()
        except BaseException as exc:
            with self.lock:
                self.error = exc
                for _, slot in self.pending:
                    slot["error"] = exc
                self.pending.clear()
                self.lock.notify_all()

    def _tick(self):
        sim = self.sim
        if self.motion is not None:
            ticks, apply, i = self.motion
            apply(i)
            self.motion = None if i + 1 == ticks else (ticks, apply, i + 1)
        mujoco.mj_step(sim._model, sim._data)
        sim.ticks += 1
        sim.total_ticks += 1
        if sim.ticks % self.frame_ticks == 0:
            with self.lock:
                self._capture()

    def _capture(self):
        model = self.sim._model
        camera = self.sim._camera_id
        self.kept.append({"frame": self.next_frame, "episode": self.episode, "state": self.snapshot(self.sim),
                          "camera": (model.cam_pos[camera].copy(), model.cam_quat[camera].copy())})
        self.next_frame += 1
        self.lock.notify_all()

    def _launch(self, protocol):
        """Reset (protocol None) or throw: with a launcher, the next launch of its session stream; otherwise a plain reset."""
        if self.launcher is None:
            self.sim.action({"op": "reset"}, render=False)
        else:
            self.launcher.place(self.sim, *self.launcher.state(protocol, Launcher.SESSION, self.launches))
        self.launches += 1
        self.episode += 1
        self._capture()

    def _start_pending(self):
        while self.pending and self.motion is None:
            action, slot = self.pending.popleft()
            try:
                op = action["op"]
                frame = self.next_frame
                if op in self.TIMED:
                    ticks, apply = self.sim.motion(action)
                    self.motion = (ticks, apply, 0)
                elif op in ("reset", "throw"):
                    if op == "throw" and (self.launcher is None or action.get("protocol") not in self.launcher.protocols):
                        raise ValueError("throw needs a launcher protocol: " + ", ".join(self.launcher.protocols if self.launcher else ()))
                    self._launch(action.get("protocol"))
                else:
                    assert op in self.INSTANT and not action.get("ticks"), action
                    self.sim.action(action, render=False)
                slot.update(state=self.snapshot(self.sim), frame=frame)
            except Exception as exc:
                slot["error"] = exc
            self.lock.notify_all()

    # Caller thread ------------------------------------------------------------------------------------------------------
    def _check(self):
        if self.error is not None:
            raise WorldFailure("real-time world stopped: %r" % self.error)

    def _wait(self, ready, deadline):
        while not ready():
            self._check()
            if time.monotonic() >= deadline:
                raise TimeoutError("real-time world wait passed its deadline")
            self.lock.wait(deadline - time.monotonic())

    def command(self, action, deadline):
        """Queue action; return (state when it took effect, the first frame captured at or after that moment)."""
        assert action["op"] in self.TIMED + self.INSTANT, action
        slot = {}
        with self.lock:
            self._check()
            self.pending.append((action, slot))
            self.lock.notify_all()
            self._wait(lambda: "state" in slot or "error" in slot, deadline)
            if "error" in slot:
                raise slot["error"]
            self._wait(lambda: self.next_frame > slot["frame"], deadline)
            return slot["state"], self._entry(slot["frame"])

    def _entry(self, index):
        entry = next((entry for entry in self.kept if entry["frame"] == index), None)
        assert entry is not None, "frame %d left the history" % index
        return entry

    def latest(self):
        with self.lock:
            self._check()
            return self.kept[-1]

    def frames(self, since, limit, deadline):
        """Kept frames with index > since, oldest first, at most limit; waits for the next frame if none is newer yet."""
        with self.lock:
            self._wait(lambda: self.next_frame - 1 > since or time.monotonic() >= deadline, deadline + 1)
            return [entry for entry in self.kept if entry["frame"] > since][:limit]

    def render(self, entry):
        """Complete observation of a kept frame (private: filter with public_observation before it leaves the host)."""
        restore(self.render_sim, entry["state"], entry["camera"])
        return {**self.render_sim.observe(full=True), "frame": entry["frame"], "episode": entry["episode"]}

    @contextmanager
    def paused(self):
        """Host-side holds only (budget/provider holds, private evaluation): the world clock stops with the run clocks."""
        with self.lock:
            self.paused_at = time.monotonic()
        try:
            yield
        finally:
            with self.lock:
                self.origin += time.monotonic() - self.paused_at
                self.paused_at = None
                self.lock.notify_all()

    def close(self):
        with self.lock:
            self.stopped = True
            self.lock.notify_all()
        self.thread.join()
        return {"total_ticks": self.sim.total_ticks, "frames": self.next_frame, "launches": self.launches, "max_lag_ticks": self.max_lag,
                "error": None if self.error is None else repr(self.error)}
