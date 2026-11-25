"""Gateway broker parsing and in-container harness loop; no live API, container or native simulation."""
import base64
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import time
import sys
import zipfile
from unittest.mock import patch

import pytest

from sim2sim.client import Client
from sim2sim.cli import main as cli
from sim2sim.runtime import TOOL_TRANSPORT_NOTE, _inference_process, BudgetStop, InferenceBroker, RuntimeFailure, WallClock, read_cost_pool, run

spec = importlib.util.spec_from_file_location("sandbox_agent", Path(__file__).resolve().parents[1] / "examples/sandbox_agent.py")
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)

CONFIG = {"endpoint": "https://api.aqinference.com/v1/responses", "provider": "openai", "model": "gpt-6-astra",
          "reasoning_effort": "medium", "reasoning_summary": "auto", "api_key_env": "INFERENCE_API_KEY",
          "metadata": {"project": "amish-research", "feature": "sim2sim-agent-eval"}, "max_calls": 5, "max_cost_usd": 0.01}
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * 16).decode()
# Recorded gateway response shapes (ids shortened; values as returned on 2026-11-02).
TOOL_CALL = {"id": "resp_08552d", "content": "", "toolCalls": [{"id": "call_43TM", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"echo hi\"}"}}],
             "finishReason": "tool_calls", "usage": {"promptTokens": 89, "completionTokens": 19, "totalTokens": 108, "cacheReadTokens": 0, "cacheWriteTokens": 0,
             "reasoningTokens": 0, "cacheStatus": "uncached"}, "cost": {"input": 0.00089, "output": 0.00095, "request": None, "image": None, "total": 0.00184,
             "currency": "USD", "source": "openrouter_catalog", "confidence": "estimated"}, "model": "gpt-6-astra", "provider": "openai", "latencyMs": 1500,
             "route": {"provider": "openai", "model": "gpt-6-astra", "routedVia": "direct", "fallbackUsed": False, "attempts": [{"attempt": 0, "provider": "openai",
             "model": "gpt-6-astra", "status": "success", "statusCode": 200, "latencyMs": 1500, "retryCount": 0, "fallbackReason": None, "error": None}]}}
CACHED = {**TOOL_CALL, "content": "91", "toolCalls": None, "finishReason": None, "cost": {"total": 0.010058, "currency": "USD"},
          "usage": {"promptTokens": 3, "completionTokens": 5, "totalTokens": 8, "cacheReadTokens": 9503, "cacheWriteTokens": 22, "reasoningTokens": 0, "cacheStatus": "cache_read_write"}}
TOOLS = [{"name": "bash", "description": "Run a command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}]


class Response:
    def __init__(self, value):
        self.data = io.BytesIO(json.dumps(value).encode())
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read1(self, count): return self.data.read(count)


def test_gateway_payload_parsing_and_cost_budget():
    seen = []
    class Opener:
        def open(self, request, timeout):
            seen.append((json.loads(request.data), request.headers))
            return Response(TOOL_CALL)
    broker = InferenceBroker(CONFIG)
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + PNG}}]},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "name": "bash", "arguments": "{\"command\":\"ls\"}"}]},
                {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "scene.xml"}]}]
    with patch.dict(os.environ, {"INFERENCE_API_KEY": "HOST-ONLY-TEST-KEY"}), patch("sim2sim.runtime.urllib.request.build_opener", return_value=Opener()):
        answer = broker._call({"op": "inference", "messages": messages, "tools": TOOLS}, 1)
    payload = seen[0][0]
    assert payload["model"] == "gpt-6-astra" and payload["provider"] == "openai" and payload["store"] is False
    assert payload["metadata"] == CONFIG["metadata"] and payload["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert payload["tools"] == [{"type": "function", "function": {**TOOLS[0]}}]
    assert payload["input"] == [{"role": "system", "content": [{"type": "input_text", "text": TOOL_TRANSPORT_NOTE}]},
        {"role": "system", "content": [{"type": "input_text", "text": "rules"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "look"}, {"type": "input_image", "imageUrl": "data:image/png;base64," + PNG}]},
        {"role": "user", "content": [{"type": "input_text", "text": "[host] Result of your bash call (id call_1, arguments {\"command\":\"ls\"}):"},
                                     {"type": "input_text", "text": "scene.xml"}]}]
    assert answer["tool_calls"] == [{"id": "call_43TM", "name": "bash", "arguments": "{\"command\":\"echo hi\"}"}]
    assert answer["cost_usd"] == 0.00184 and answer["usage"]["accountedTokens"] == 108 and answer["summaries"] == []
    assert "route" not in answer and "HOST-ONLY-TEST-KEY" not in json.dumps(answer)
    assert broker.parse(CACHED, {"bash"})["usage"]["accountedTokens"] == 8 + 9503 + 22
    broker.calls = 1
    assert broker.charge(answer)["remaining_cost_usd"] == pytest.approx(0.01 - 0.00184)
    broker.charge(broker.parse(CACHED, {"bash"}))
    with pytest.raises(RuntimeFailure, match="cost budget exhausted"):
        broker.call({"messages": messages}, 1)
    with pytest.raises(RuntimeFailure, match="omitted cost"):
        InferenceBroker(CONFIG).charge(broker.parse({**CACHED, "cost": {"total": None}}, {"bash"}))


def test_gateway_rejections():
    broker = InferenceBroker(CONFIG)
    for result in ({**TOOL_CALL, "model": "gpt-6-sol"}, {**TOOL_CALL, "route": {**TOOL_CALL["route"], "fallbackUsed": True}},
                   {**TOOL_CALL, "route": {**TOOL_CALL["route"], "provider": "openrouter"}}, {**TOOL_CALL, "route": None}):
        with pytest.raises(RuntimeFailure, match="substitution"):
            broker.parse(result, {"bash"})
    with pytest.raises(RuntimeFailure, match="invalid tool call"):
        broker.parse(TOOL_CALL, {"observe"})
    user = {"role": "user", "content": "x"}
    for request in ({"messages": [user, {"role": "tool", "tool_call_id": "call_9", "content": "x"}]},
                    {"messages": [{"role": "assistant", "content": "", "tool_calls": [{"id": "c", "name": "rm", "arguments": "{}"}]}], "tools": TOOLS},
                    {"messages": [{"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + PNG}}]}]},
                    {"messages": [user], "model": "gpt-6-sol"}, {"messages": [user], "tools": [{"name": "bad name", "parameters": {"type": "object"}}]},
                    {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]}]}):
        with pytest.raises(RuntimeFailure):
            broker.payload(request)
    for config in ({**CONFIG, "api": "chat"}, {k: v for k, v in CONFIG.items() if k != "metadata"}, {**CONFIG, "endpoint": "http://example.com"}):
        with pytest.raises(RuntimeFailure):
            InferenceBroker(config)


def test_wall_clock_kills_registered_work_and_rejects_nonfinite_budgets():
    for value in (float("nan"), float("inf"), -float("inf"), 0, -1, True):
        with pytest.raises(RuntimeFailure):
            WallClock(value)
    process = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(5)"])
    clock = WallClock(.05)
    try:
        clock.register(process)
        assert clock.expired.wait(1)
        assert process.wait(timeout=1) != 0
        with pytest.raises(RuntimeFailure, match="Hard wall"):
            clock.check()
    finally:
        clock.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def test_configured_provider_missing_key_cannot_launch_agent(tmp_path):
    config = {"sandbox": {"image": "test", "command": ["agent"]}, "inference": CONFIG, "output_dir": str(tmp_path / "run")}
    with patch.dict(os.environ, {}, clear=True), patch("sim2sim.runtime.check_rootless"), patch("sim2sim.runtime.check_limits"), \
         patch("sim2sim.runtime.subprocess.Popen") as launch:
        with pytest.raises(RuntimeFailure, match="credential is missing"):
            run(config)
    launch.assert_not_called()


class Controller(Client):
    """The real client over an in-memory controller that records wire requests."""
    def __init__(self, replies, now, wall=3600.):
        self.task = {"type": "task", "task": "Reconstruct the scene.", "deadline_unix": now() + wall,
                     "actions": ["observe", "grasp", "move", "release", "reset", "load_reconstruction", "submit", "inference", "progress", "finish"],
                     "action_schema": {"observe": {"ticks": "optional"}, "grasp": {"pixel": "[u,v]"}, "move": {"displacement": "[dx,dy,dz]", "ticks": 20},
                                       "release": {"ticks": "optional"}, "reset": "restore", "load_reconstruction": {"project_b64": "zip"},
                                       "submit": {"project_b64": "zip"}, "finish": "end run"},
                     "budgets": {"wall_only": True, "active_seconds": None, "wall_seconds": wall},
                     "inference": {"model": "gpt-6-astra", "enabled": True}, "submission": {"max_bytes": 1 << 20},
                     "observation": {"rgb": PNG, "width": 4, "height": 4, "time": 0., "calibration": {"intrinsics": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}}
        self.replies, self.requests, self.notes = list(replies), [], []

    def request(self, op, **kwargs):
        self.requests.append({"op": op, **kwargs})
        if op == "inference":
            InferenceBroker(CONFIG).payload(self.requests[-1])  # The harness wire shape passes strict broker validation.
            if not self.replies:
                return {"ok": True, "inference": {"budget_stop": True, "reason": "host stop", "deadline_unix": self.task["deadline_unix"]}}
            calls = [{"id": "call_%d_%d" % (len(self.requests), i), "name": name, "arguments": json.dumps(args)} for i, (name, args) in enumerate(self.replies.pop(0))]
            return {"ok": True, "inference": {"content": "Testing the next hypothesis.", "tool_calls": calls, "usage": None,
                                              "deadline_unix": self.task["deadline_unix"]}}
        if op == "progress":
            self.notes.append((kwargs["origin"], kwargs["text"]))
            return {"ok": True}
        if op == "submit":
            return {"ok": True, "submission": 1, "status": "ok"}
        if op == "finish":
            return {"ok": True, "status": "finished"}
        return {"ok": True, "world": kwargs["world"], "observation": {"rgb": PNG, "time": .04, "grasped": False}, "remaining_ticks": None}


def test_harness_loop_bash_actions_guarded_finish(tmp_path):
    replies = [[("bash", {"command": "mkdir -p project && echo '<mujoco/>' > project/scene.xml && cat project/scene.xml"})],
               [("load_reconstruction", {}), ("observe", {"world": "reference", "ticks": 2}), ("observe", {"world": "reconstruction", "ticks": 2})],
               [("finish", {})], [("submit", {})], [("finish", {})], [("finish", {})]]
    controller = Controller(replies, now=lambda: 1000.)
    loop = agent.Agent(controller, workspace=tmp_path, now=lambda: 1000.)
    loop.run()
    ops = [r["op"] for r in controller.requests if r["op"] not in ("inference", "progress")]
    assert ops == ["load_reconstruction", "observe", "observe", "submit", "finish"]
    inferences = [r for r in controller.requests if r["op"] == "inference"]
    assert len(inferences) == 6 and all(r["messages"][0] == {"role": "system", "content": agent.SYSTEM} for r in inferences)
    assert {tool["name"] for tool in inferences[0]["tools"]} == {"bash", "view_image", "observe", "grasp", "move", "release", "reset", "load_reconstruction", "submit", "finish"}
    results = [m for m in inferences[-1]["messages"] if m["role"] == "tool"]
    assert "exit code 0\n<mujoco/>" in results[0]["content"][0]["text"]
    assert all("remaining wall time 1:00:00" in m["content"][-1]["text"] for m in results)
    assert any(part["type"] == "image_url" for part in results[2]["content"]) and (tmp_path / "observations/00003-reference-rgb.png").exists()
    assert "no reconstruction has been submitted" in results[4]["content"][0]["text"]
    assert "you have NOT validated" in results[6]["content"][0]["text"]
    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode([r for r in controller.requests if r["op"] == "submit"][0]["project_b64"])))
    assert archive.read("scene.xml") == b"<mujoco/>\n"
    assert [text for origin, text in controller.notes if "Intercepted" in text] == ["Intercepted an early finish request with 1:00:00 remaining."]
    assert ("model", "Testing the next hypothesis.") in controller.notes


def test_harness_deadline_compaction_and_image_pruning(tmp_path):
    clock = [0.]
    controller = Controller([[("observe", {"world": "reference"})]] * 30, now=lambda: clock[0], wall=1000.)
    loop = agent.Agent(controller, workspace=tmp_path, max_images=4, keep_turns=2, reserve_seconds=100, keep_image_files=3, now=lambda: clock[0])
    original = loop.infer
    def infer(messages):
        clock[0] += 60
        reply = original(messages)
        reply["usage"] = {"promptTokens": 10, "cacheReadTokens": 200000 if len(loop.turns) == 6 else 0, "cacheWriteTokens": 0}
        return reply
    loop.infer = infer
    loop.run()
    assert controller.requests[-1]["op"] == "finish" and "automatically" in controller.notes[-1][1]
    assert loop.summary == "Testing the next hypothesis." and len(loop.turns) <= 2 + 15
    assert len(list((tmp_path / "observations").iterdir())) == 3
    images = [p for m in loop.messages()[2:] if isinstance(m["content"], list) for p in m["content"] if p["type"] == "image_url"]
    assert len(images) <= 4


def test_shared_cost_pool_ledger_threshold_and_monitor(tmp_path, capsys):
    config = {**CONFIG, "max_cost_usd": 58., "cost_pool": "wave1", "cost_pool_usd": 10., "cost_ledger_dir": str(tmp_path), "call_reserve_usd": 1.}
    (tmp_path / "wave1").mkdir()
    (tmp_path / "wave1/other.json").write_text(json.dumps({"cost_usd": 5., "inflight_reserve_usd": 1., "calls": 3, "tokens": 900}))
    broker = InferenceBroker(config, run_id="mine")
    assert broker.stop_reason() is None
    broker.calls = 1
    broker.charge(broker.parse({**CACHED, "cost": {"total": 2.5, "currency": "USD"}}, {"bash"}))
    broker.write_ledger(0.)
    assert broker.stop_reason() == "Shared inference cost pool exhausted"  # 10 - (5 + 1) - 2.5 < 2 reserves
    with pytest.raises(BudgetStop, match="pool exhausted"):
        broker.call({"messages": [{"role": "user", "content": "x"}]}, 1)
    record = read_cost_pool(tmp_path / "wave1")["mine"]
    assert set(record) == {"run_id", "pool", "cost_usd", "inflight_reserve_usd", "calls", "tokens", "updated_at"} and record["cost_usd"] == 2.5
    assert cli(["cost", "--pool", "wave1", "--ledger", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["total_cost_usd"] == 7.5
    with pytest.raises(RuntimeFailure, match="already exists"):
        InferenceBroker(config, run_id="mine")
    solo = InferenceBroker({**config, "cost_pool_usd": 1000.}, run_id="solo")
    with pytest.raises(RuntimeFailure, match="omitted cost"):
        solo.charge(solo.parse({**CACHED, "cost": None}, {"bash"}))
    assert solo.stop_reason() == "Provider omitted cost; further inference is disabled"
    # A transiently torn/stale read (NFS replace race, wave-1 job 57186237) is retried, not treated as exhaustion.
    real = Path.read_text
    flaky = iter([OSError(116, "Stale file handle"), "{"])
    def read_text(self, *args, **kwargs):
        if self.name == "other.json":
            value = next(flaky, None)
            if isinstance(value, Exception):
                raise value
            if value is not None:
                return value
        return real(self, *args, **kwargs)
    with patch.object(Path, "read_text", read_text), patch("sim2sim.runtime.time.sleep"):
        assert read_cost_pool(tmp_path / "wave1")["other"]["cost_usd"] == 5.
    (tmp_path / "wave1/other.json").write_text("{")
    assert "unreadable" in InferenceBroker({**config, "cost_pool_usd": 1000.}, run_id="third").stop_reason()
    assert InferenceBroker({**CONFIG, "max_cost_usd": 15., "call_reserve_usd": 3.}).stop_reason() is None
    # The spawned network worker must not rebuild the pool (regression: smoke run 57180717 failed here).
    import multiprocessing
    parent, child = multiprocessing.Pipe(duplex=False)
    with patch.object(InferenceBroker, "_call", lambda self, request, timeout: {"ledger": self.ledger}):
        _inference_process(config, {}, 1, child)
    assert parent.recv() == (True, {"ledger": None}, False)
    assert InferenceBroker({**CONFIG, "max_cost_usd": 5., "call_reserve_usd": 3.}).stop_reason() == "Run inference cost budget exhausted"


@pytest.mark.parametrize("submit_first", [False, True])
def test_harness_budget_stop_submits_unsubmitted_work_then_finishes(tmp_path, submit_first):
    replies = [[("bash", {"command": "mkdir -p project && echo '<mujoco/>' > project/scene.xml"})]] + ([[("submit", {})]] if submit_first else [])
    controller = Controller(replies, now=lambda: 1000.)
    agent.Agent(controller, workspace=tmp_path, now=lambda: 1000.).run()
    assert [r["op"] for r in controller.requests if r["op"] not in ("inference", "progress")] == ["submit", "finish"]
    assert any(text.startswith("Inference budget exhausted (host stop)") for _, text in controller.notes)


def test_wall_clock_hold_extends_the_deadline_and_excludes_the_hold():
    clock = WallClock(.4)
    try:
        with clock.hold():
            time.sleep(.6)
            assert not clock.expired.is_set() and clock.remaining() > .3
        assert clock.held > .55 and clock.remaining() > .3 and not clock.expired.is_set()
        assert clock.expired.wait(2)
    finally:
        clock.close()


def test_harness_reanchors_its_deadline_on_every_reply(tmp_path):
    controller = Controller([[("observe", {"world": "reference"})]], now=lambda: 1000.)
    loop = agent.Agent(controller, workspace=tmp_path, now=lambda: 1000.)
    controller.task["deadline_unix"] += 500  # e.g. a host-side hold of 500 s
    loop.infer(loop.messages())
    assert loop.left() == 4100


def test_harness_sequence_runs_agent_written_steps_and_saves_outputs(tmp_path):
    controller = Controller([], now=lambda: 1000.)
    controller.task["actions"].insert(4, "sequence")
    controller.task["action_schema"]["sequence"] = {"steps_file": "path"}
    loop = agent.Agent(controller, workspace=tmp_path, now=lambda: 1000.)
    assert "sequence" in loop.schemas
    steps = [{"op": "observe", "ticks": 5}, {"op": "move", "displacement": [0, 0, .01], "ticks": 3}, {"op": "release"}]
    (tmp_path / "steps.json").write_text(json.dumps(steps))
    parts, done = loop.execute({"id": "c", "name": "sequence", "arguments": json.dumps({"world": "reconstruction", "steps_file": str(tmp_path / "steps.json")})})
    sent = [r for r in controller.requests if r["op"] != "progress"]
    assert not done and [r["op"] for r in sent] == ["observe", "move", "release"] and all(r["world"] == "reconstruction" for r in sent)
    records = json.loads((tmp_path / "sequences/0001.json").read_text())
    assert [r["op"] for r in records] == ["observe", "move", "release"] and "rgb" not in records[0]["observation"]
    assert "Ran 3 of 3 steps" in parts[0]["text"] and any(p["type"] == "image_url" for p in parts)
    (tmp_path / "bad.json").write_text(json.dumps([{"op": "finish"}]))
    parts, _ = loop.execute({"id": "d", "name": "sequence", "arguments": json.dumps({"world": "reference", "steps_file": str(tmp_path / "bad.json")})})
    assert parts[0]["text"].startswith("ERROR: steps_file must hold")


def test_pausable_bash_runs_without_limit_only_while_the_agent_waits(tmp_path):
    controller = Controller([[("bash", {"command": "sleep 100", "check_seconds": 1})]], now=time.time)
    loop = agent.Agent(controller, workspace=tmp_path, bash_mode="pausable", now=time.time)
    assert {"bash", "bash_resume", "bash_kill", "view_image"} <= set(loop.schemas) and set(loop.schemas["bash"]["properties"]) == {"command", "check_seconds"}
    call = lambda name, **args: loop.execute({"id": "c", "name": name, "arguments": json.dumps(args)})[0][0]["text"]
    def ours():  # live processes started from this test's workspace
        found = {}
        for pid, state in agent.descendants().items():
            try:
                if os.readlink("/proc/%d/cwd" % pid) == str(tmp_path):
                    found[pid] = state
            except OSError:
                pass
        return found
    assert call("bash", command="echo done") == "exit code 0\ndone\n"
    # A detached (double-forked, new session) loop and a foreground loop: both may only run between check-ins.
    loops = "(setsid bash -c 'while true; do echo x >> detached.txt; sleep .05; done' &); while true; do echo tick; echo x >> fg.txt; sleep .05; done"
    out = call("bash", command=loops, check_seconds=1)
    assert out.startswith("paused after 1 s of running time") and "tick" in out
    assert len(ours()) >= 2 and all(state in "tT" for state in ours().values())  # both loops (a short sleep may be between runs)
    sizes = lambda: [(tmp_path / name).stat().st_size for name in ("fg.txt", "detached.txt")]
    before = sizes()
    time.sleep(.5)
    assert sizes() == before
    assert call("observe", world="reference").startswith("ERROR: a bash command is paused")
    assert call("bash", command="echo no").startswith("ERROR: a bash command is paused")
    out = call("bash_resume", check_seconds=1)
    assert out.startswith("paused after 2 s") and all(a > b for a, b in zip(sizes(), before))
    assert call("bash_kill").startswith("killed by bash_kill after 2 s") and ours() == {} and loop.job is None
    assert call("bash_kill").startswith("ERROR: no bash command is paused")
    # A finished command leaves nothing behind, even a detached process.
    assert call("bash", command="(setsid sleep 100 &); echo started") == "exit code 0\nstarted\n" and ours() == {}
    # Only the session deadline ends a command without the agent's choice.
    loop.deadline = time.time() + loop.reserve + 2
    assert call("bash", command="sleep 100", check_seconds=50).startswith("killed because the session is ending") and ours() == {}
    # A command still paused when the host stops the session is killed before finishing.
    loop.deadline = time.time() + 3600
    loop.run()
    assert loop.job is None and ours() == {} and controller.requests[-1]["op"] == "finish"


def test_harness_realtime_predictor_task_frames_and_controller_channel(tmp_path):
    controller = Controller([], now=lambda: 1000.)
    task = controller.task
    task["actions"] = ["observe", "grasp", "move", "release", "frames", "throw", "reset", "submit", "inference", "progress", "finish"]
    task["action_schema"].update(frames={"since": "index"}, throw={"protocol": "name"})
    task.update(realtime={"frame_rate_hz": 25}, submission={"max_bytes": 1 << 20, "kind": "predictor"},
                code_channel={"path": str(tmp_path / "controller.sock"), "ops": ["observe", "frames", "throw"]})
    loop = agent.Agent(controller, workspace=tmp_path, now=lambda: 1000.)
    assert loop.system == agent.SYSTEM_PREDICTOR % agent.PREDICTOR_WORLDS["realtime"] and loop.guard == agent.GUARD_PREDICTOR
    lockstep = agent.Agent(Controller([], now=lambda: 1000.), workspace=tmp_path, now=lambda: 1000.)
    lockstep.task["submission"]["kind"] = "predictor"
    assert agent.prompts(lockstep.task)[0] == agent.SYSTEM_PREDICTOR % agent.PREDICTOR_WORLDS["lockstep"] and agent.prompts(controller.task)[2] == agent.GUARD_PREDICTOR
    lockstep.task["submission"]["kind"] = "parameters"
    assert all("predictor" not in text and "estimates" in text for text in agent.prompts(lockstep.task))
    assert "world" not in loop.schemas["throw"]["properties"] and loop.schemas["observe"]["required"] == []
    rendered = loop.render({"ok": True, "frames": [{"rgb": PNG, "frame": 3}, {"rgb": PNG, "frame": 4}]}, "reference")
    reply = json.loads(rendered[0]["text"])
    assert len(rendered) == 1 and [f["frame"] for f in reply["frames"]] == [3, 4] and all(Path(f["saved_images"][0]).exists() for f in reply["frames"])
    script = tmp_path / "channel.py"
    script.write_text("import json\nfrom sim2sim.client import Channel\nc = Channel(%r)\nprint(json.dumps(c.request('throw', protocol='drop', world='reference')))\n"
                      "try:\n    c.request('submit', project_b64='')\nexcept RuntimeError as e:\n    print('rejected', e)\n" % str(tmp_path / "controller.sock"))
    output = loop.bash("PYTHONPATH=%s python3 %s" % (Path(agent.__file__).resolve().parents[1] / "src", script), 60)
    assert output.startswith("exit code 0") and '"ok": true' in output and "rejected op must be one of" in output, output
    assert [r["op"] for r in controller.requests if r["op"] != "progress"] == ["throw"] and loop.channel_requests == 2
    assert not (tmp_path / "controller.sock").exists()
    # Pausable bash: the channel serves only while the command runs and closes with it.
    paused = agent.Agent(controller, workspace=tmp_path, bash_mode="pausable", now=time.time)
    paused.deadline = time.time() + 3600
    script.write_text("import json, time\nfrom sim2sim.client import Channel\nc = Channel(%r)\nprint(c.request('observe', world='reference')['ok'], flush=True)\n"
                      "time.sleep(100)\n" % str(tmp_path / "controller.sock"))
    before = len(controller.requests)
    out = paused.bash_start("PYTHONPATH=%s python3 %s" % (Path(agent.__file__).resolve().parents[1] / "src", script), 2)
    assert out.startswith("paused after") and "True" in out and paused.channel_requests == 1, out
    assert [r["op"] for r in controller.requests[before:]] == ["observe"] and (tmp_path / "controller.sock").exists()
    assert paused.bash_end("killed").startswith("killed") and not (tmp_path / "controller.sock").exists()


def test_harness_two_point_grasps_tools_require_grasp_index():
    task = Controller([], now=lambda: 1000.).task
    single = {tool["name"]: tool["parameters"] for tool in agent.tool_schemas(task)}
    assert all("grasp_index" not in single[op]["properties"] for op in ("grasp", "move", "release"))
    for op in ("grasp", "move", "release"):
        task["action_schema"][op]["grasp_index"] = "0 or 1"
    two = {tool["name"]: tool["parameters"] for tool in agent.tool_schemas(task)}
    for op in ("grasp", "move", "release"):
        assert two[op]["properties"]["grasp_index"] == {"type": "integer", "minimum": 0, "maximum": 1} and "grasp_index" in two[op]["required"]
    assert two["observe"] == single["observe"]


def test_harness_weld_move_tool_offers_optional_rotation():
    task = Controller([], now=lambda: 1000.).task
    connect = {tool["name"]: tool["parameters"] for tool in agent.tool_schemas(task)}
    assert "rotation" not in connect["move"]["properties"]
    task["action_schema"]["move"]["rotation"] = "optional [rx,ry,rz] world rotation vector"
    weld = {tool["name"]: tool["parameters"] for tool in agent.tool_schemas(task)}
    assert weld["move"]["properties"]["rotation"] == agent.NUMBERS(3) and "rotation" not in weld["move"]["required"]
    assert {op: weld[op] for op in weld if op != "move"} == {op: connect[op] for op in connect if op != "move"}
