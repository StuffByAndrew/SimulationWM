import base64
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from contextlib import nullcontext
import zipfile

from sim2sim.runtime import (RuntimeFailure, Transport, check_rootless, container_command,
                             decode_submission, extract_bundle, InferenceBroker, NoRedirect, validate_effective_limits)
from sim2sim.client import Client


class RuntimeTests(unittest.TestCase):
    def test_reject_rootful_and_missing_runtime(self):
        with patch("sim2sim.runtime.subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout="false\n")):
            with self.assertRaises(RuntimeFailure):
                check_rootless()
        with patch("sim2sim.runtime.subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaises(RuntimeFailure):
                check_rootless()

    def test_require_delegated_cgroup_v2_controllers(self):
        host = {"security": {"rootless": True}, "cgroupVersion": "v2", "cgroupControllers": ["cpu", "memory", "pids"]}
        def invoke(backend="cgroup"):
            with patch("sim2sim.runtime.subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps({"host": host}))):
                return check_rootless(backend)
        invoke()
        host["cgroupVersion"] = "v1"
        with self.assertRaises(RuntimeFailure):
            invoke()
        host["cgroupVersion"] = "v2"
        host["cgroupControllers"] = ["pids"]
        with self.assertRaises(RuntimeFailure):
            invoke()
        host["cgroupControllers"] = ["cpu", "memory"]
        with self.assertRaises(RuntimeFailure):
            invoke()
        invoke("rlimit")
        cmd = container_command("local/image", ["python"], {"process_limit_backend": "rlimit"}, "test")
        self.assertIn("--pids-limit=-1", cmd)
        self.assertEqual(cmd[-4:], ["/usr/bin/prlimit", "--nproc=64:64", "--", "python"])

    def test_slurm_step_command_and_in_container_checks(self):
        limits = {"process_limit_backend": "slurm_step", "memory": "8g", "cpus": 4, "pids": 128}
        with patch.dict("os.environ", {"SLURM_JOB_ID": "7", "XDG_RUNTIME_DIR": "/tmp/x", "OPENAI_API_KEY": "secret"}):
            cmd = container_command("local/image", ["python"], limits, "n")
        self.assertEqual(cmd[:8], ["srun", "--jobid=7", "--exact", "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--mem=8192M", "--quiet"])
        self.assertIn("XDG_RUNTIME_DIR=/tmp/x", cmd)
        self.assertFalse(any("secret" in x or x.startswith(("--memory", "--cpus=", "--pids")) for x in cmd))
        self.assertIn("--cgroups=disabled", cmd)
        self.assertEqual(cmd[-4:], ["/usr/bin/prlimit", "--nproc=128:128", "--", "python"])
        with self.assertRaises(RuntimeFailure):
            container_command("local/image", ["python"], {**limits, "cpus": 1.5}, "n")
        observed = {"memory.max": "max", "cpu.max": "max 100000", "nproc": [128, 128], "uid": 65532, "cap_eff": "0000000000000000", "no_new_privs": "1"}
        validate_effective_limits(observed, limits)
        with self.assertRaises(RuntimeFailure):
            validate_effective_limits({**observed, "nproc": [256, 256]}, limits)

    def test_sandbox_has_no_host_mounts_or_network(self):
        cmd = container_command("local/image", ["python", "agent.py"], {}, "test")
        for flag in ("--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--http-proxy=false", "--image-volume=ignore", "--cgroups=enabled", "--log-driver=none"):
            self.assertIn(flag, cmd)
        self.assertFalse(any(x in cmd for x in ("-v", "--volume", "--mount", "--privileged", "--env-host")))

    def test_effective_limits_fail_closed_on_ignored_flags(self):
        observed = {"memory.max": "1073741824", "cpu.max": "100000 100000", "pids.max": "64", "nproc": [64,64], "uid": 65532, "cap_eff": "0000000000000000", "no_new_privs": "1"}
        validate_effective_limits(observed, {})
        observed["pids.max"] = None
        validate_effective_limits(observed, {"process_limit_backend": "rlimit"})
        with self.assertRaises(RuntimeFailure):
            validate_effective_limits(observed, {})
        observed["memory.max"] = "max"
        with self.assertRaises(RuntimeFailure):
            validate_effective_limits(observed, {"process_limit_backend": "rlimit"})
        observed["memory.max"] = "1073741824"
        observed["cpu.max"] = "max 100000"
        with self.assertRaises(RuntimeFailure):
            validate_effective_limits(observed, {"process_limit_backend": "rlimit"})

    def test_submission_limit_and_invalid_base64(self):
        with self.assertRaises(RuntimeFailure):
            decode_submission("!", 20)
        with self.assertRaises(RuntimeFailure):
            decode_submission(base64.b64encode(b"12345").decode(), 4)

    def test_bundle_traversal_is_rejected(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("../escape", "bad")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeFailure):
                extract_bundle(data.getvalue(), directory)

    def test_transport_reads_json_and_enforces_deadline(self):
        proc = subprocess.Popen([sys.executable, "-c", "import sys,time; print('{\"op\":\"observe\"}',flush=True); time.sleep(2)"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            transport = Transport(proc, 1024)
            self.assertEqual(transport.receive(time.monotonic() + 1), {"op": "observe"})
            with self.assertRaises(RuntimeFailure):
                transport.receive(time.monotonic() + .02)
        finally:
            proc.kill()
            proc.wait()
            proc.stdin.close()
            proc.stdout.close()

    def test_transport_rejects_oversize(self):
        proc = subprocess.Popen([sys.executable, "-c", "print('x'*100,flush=True)"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            with self.assertRaises(RuntimeFailure):
                Transport(proc, 10).receive(time.monotonic() + 1)
        finally:
            proc.wait()
            proc.stdin.close()
            proc.stdout.close()

    def test_fixed_inference_endpoint_and_no_redirects(self):
        for endpoint in ("http://example.com", "https://user:pass@example.com", "file:///secret"):
            with self.assertRaises(RuntimeFailure):
                InferenceBroker({"endpoint": endpoint})
        with self.assertRaises(RuntimeFailure):
            NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://elsewhere")
        with self.assertRaises(RuntimeFailure):
            InferenceBroker({}).call({}, 1)

    def test_gemini_transcript_and_host_priced_accounting(self):
        model = "gemini-4-neon"
        cfg = {"api": "gemini", "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % model, "model": model,
               "api_key_env": "K", "reasoning_effort": "medium", "reasoning_summary": "auto",
               "price_usd_per_mtok": {"input": 3.0, "cached_input": 0.15, "output": 15.0}}
        for bad in ({"endpoint": cfg["endpoint"].replace("neon", "hydrogen")}, {"metadata": {"project": "p", "feature": "f"}},
                    {"price_usd_per_mtok": {"input": 3.0, "output": 15.0}}):
            with self.assertRaises(RuntimeFailure):
                InferenceBroker({**cfg, **bad})
        broker = InferenceBroker(cfg)
        tools = [{"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "additionalProperties": False}}]
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "bash", "arguments": "{}"}]},
                    {"role": "tool", "tool_call_id": "c1", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]}]
        body = broker.gemini_payload(broker.payload({"messages": messages, "tools": tools})[0])
        # Same labelled-text transport as the gateway route: no native functionCall/functionResponse parts in history.
        self.assertEqual([c["role"] for c in body["contents"]], ["user", "user"])
        self.assertEqual(body["contents"][1]["parts"][1], {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}})
        self.assertEqual(len(body["systemInstruction"]["parts"]), 2)
        self.assertEqual(body["generationConfig"]["thinkingConfig"], {"includeThoughts": True, "thinkingLevel": "medium"})
        self.assertIn("additionalProperties", body["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"])
        reply = {"modelVersion": model, "candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"text": "plan", "thought": True}, {"text": "ok"}, {"functionCall": {"name": "bash", "args": {"command": "ls"}}, "thoughtSignature": "x"}]}}],
            "usageMetadata": {"promptTokenCount": 1000, "cachedContentTokenCount": 400, "candidatesTokenCount": 50, "thoughtsTokenCount": 150}}
        out = broker.parse_gemini(reply, {"bash"})
        self.assertEqual((out["content"], out["summaries"], out["tool_calls"][0]["arguments"]), ("ok", ["plan"], '{"command": "ls"}'))
        self.assertEqual(out["usage"], {"promptTokens": 600, "completionTokens": 200, "totalTokens": 800, "cacheReadTokens": 400,
                                        "cacheWriteTokens": 0, "reasoningTokens": 150, "accountedTokens": 1200})
        self.assertAlmostEqual(out["cost_usd"], (600 * 3.0 + 400 * 0.15 + 200 * 15.0) / 1e6)
        with self.assertRaises(RuntimeFailure):
            broker.parse_gemini({**reply, "modelVersion": "gemini-4-hydrogen"}, {"bash"})

    def test_local_openai_service_record_transcript_and_zero_cost(self):
        import http.server
        import os
        import threading
        from sim2sim.runtime import ProviderUnavailable
        seen = []
        reply = {"model": "inkling-2-mini", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "reasoning_content": "private thinking",
            "tool_calls": [{"id": "chatcmpl-tool-1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200, "prompt_tokens_details": {"cached_tokens": 400}}}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append((self.path, self.headers["Authorization"], json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                data = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as directory:
            os.chmod(directory, 0o700)
            record = Path(directory) / "service.json"
            cfg = {"api": "local_openai", "endpoint": str(record), "model": "inkling-2-mini", "reasoning_effort": "medium", "max_output_tokens": 64}
            for bad in ({"api_key_env": "K"}, {"reasoning_summary": "auto"}, {"endpoint": "http://127.0.0.1/v1/chat/completions"}):
                with self.assertRaises(RuntimeFailure):
                    InferenceBroker({**cfg, **bad})
            broker = InferenceBroker(cfg)
            tools = [{"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}]
            messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                        {"role": "assistant", "content": "thinking aloud", "tool_calls": [{"id": "c1", "name": "bash", "arguments": "{}"}]},
                        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "out"},
                                                                           {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]}]
            with self.assertRaises(ProviderUnavailable):  # no service yet: an outage, covered by provider_hold_seconds
                broker._call({"messages": messages, "tools": tools}, 10)

            def advertise(age=0., **change):
                record.write_text(json.dumps({"endpoint": "http://127.0.0.1:%d/v1/chat/completions" % server.server_port, "model": "inkling-2-mini",
                                              "token": "t0k", "heartbeat_unix": time.time() - age, **change}))
                os.chmod(record, 0o600)
            advertise(age=600.)
            with self.assertRaises(ProviderUnavailable):
                broker._call({"messages": messages, "tools": tools}, 10)
            advertise(model="other")
            with self.assertRaises(RuntimeFailure):
                broker._call({"messages": messages, "tools": tools}, 10)
            advertise()
            os.chmod(record, 0o644)
            with self.assertRaises(RuntimeFailure):
                broker._call({"messages": messages, "tools": tools}, 10)
            os.chmod(record, 0o600)
            out = broker.call({"messages": messages, "tools": tools}, 10)  # through the spawned host-only inference process
        path, auth, body = seen[-1]
        self.assertEqual((path, auth, body["model"], body["max_tokens"], body["chat_template_kwargs"]),
                         ("/v1/chat/completions", "Bearer t0k", "inkling-2-mini", 64, {"reasoning_effort": "medium"}))
        # Same labelled transcript as the other APIs: transport note + system, user, assistant text, labelled tool result.
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "system", "user", "assistant", "user"])
        self.assertEqual(body["messages"][3]["content"], "thinking aloud")
        self.assertEqual(body["messages"][4]["content"][2], {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}})
        self.assertEqual(body["tools"][0]["function"]["name"], "bash")
        self.assertEqual((out["content"], out["tool_calls"], out["finish_reason"], out["summaries"], out["cost_usd"]),
                         ("", [{"id": "chatcmpl-tool-1", "name": "bash", "arguments": '{"command": "ls"}'}], "tool_calls", [], 0.))
        self.assertEqual(out["usage"], {"promptTokens": 600, "completionTokens": 200, "totalTokens": 800, "cacheReadTokens": 400,
                                        "cacheWriteTokens": 0, "reasoningTokens": None, "accountedTokens": 1200})
        with self.assertRaises(RuntimeFailure):
            broker.parse_chat({**reply, "model": "gpt-6-astra"}, {"bash"})

    def test_client_protocol(self):
        writer = io.StringIO()
        client = Client(io.StringIO('{"type":"task"}\n{"ok":true,"observation":{"ticks":2}}\n'), writer)
        self.assertEqual(client.observe(ticks=2), {"ticks": 2})
        self.assertEqual(json.loads(writer.getvalue()), {"op": "observe", "ticks": 2, "world": "reference"})


if __name__ == "__main__":
    unittest.main()

class HostBudgetTests(unittest.TestCase):
    def run_agent(self, directory, script, budget=None, evaluate=None, candidate_factory=None, tick_delay=0, simulation=None, extra=None, manifest=None):
        import types
        from sim2sim.runtime import run

        class FakeSimulation:
            def __init__(self, *args):
                self.total_ticks = self.ticks = 0
                self._callback = None
            def observe(self):
                return {"total_ticks": self.total_ticks, "original_mass": "PRIVATE-CANARY", "qpos": [1,2,3], "body_hits": [9]}
            def action(self, request):
                if request["op"] == "reset":
                    self.ticks = 0
                else:
                    for _ in range(request.get("ticks", 0)):
                        if tick_delay:
                            time.sleep(tick_delay)
                        self.total_ticks += 1
                        self.ticks += 1
                        if self._callback:
                            self._callback(self)
                return self.observe()
            def close(self):
                pass

        def stop(proc, name):
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                stream.close()

        source = Path(directory) / "source"
        source.mkdir()
        (source / "scene.xml").write_text("<mujoco/>")
        if manifest is not None:
            (source / "manifest.toml").write_text(manifest)
        if (simulation or {}).get("robot"):
            from sim2sim.robots import ROBOTS
            (source / "robot/assets").mkdir(parents=True)
            (source / "robot" / ROBOTS[simulation["robot"]][0]).write_text("<mujoco/>")
            (source / "robot/IMPORT.md").write_text("Robot notes.\n\nThe sibling scenes hold PRIVATE blocks.\n")
        config = {"source": {"scene": str(source / "scene.xml")}, "sandbox": {"image": "test", "command": ["agent"]}, "record_video": False,
                  "output_dir": str(Path(directory) / "run"), "budget": budget or {"physics_ticks": 3, "active_seconds": 2},
                  "evaluation": {}, "simulation": simulation or {}, **(extra or {})}
        self.host_operations = []
        def host_command(command, **kwargs):
            self.host_operations.append(command[1])
            return subprocess.CompletedProcess(command, 0, stdout="test-image-id")
        def plot_history(*args, **kwargs):
            self.host_operations.append("plot")
        cmd = [sys.executable, "-u", "-c", "import json,sys,time\ntask=json.loads(input())\n" + script]
        candidate_patch = patch("sim2sim.runtime.CandidateSession", candidate_factory) if candidate_factory else nullcontext()
        with candidate_patch, patch.dict(sys.modules, {"sim2sim.simulation": types.SimpleNamespace(Simulation=FakeSimulation)}), \
             patch("sim2sim.runtime.check_rootless"), patch("sim2sim.runtime.check_limits", return_value={}), patch("sim2sim.runtime.check_image_roles"), patch("sim2sim.runtime.container_command", return_value=cmd), \
             patch("sim2sim.runtime.stop_container", side_effect=stop), \
             patch("sim2sim.runtime.evaluate_in_container", side_effect=evaluate), \
             patch("sim2sim.plotting.update_metric_history", side_effect=plot_history), \
             patch("sim2sim.runtime.subprocess.run", side_effect=host_command):
            return run(config)

    def test_continuation_prior_is_verified_hashed_and_mounted(self):
        import hashlib
        from sim2sim import runtime
        with tempfile.TemporaryDirectory() as directory:
            prior = Path(directory) / "prior"
            (prior / "project").mkdir(parents=True)
            (prior / "project/scene.xml").write_text("<mujoco/>")
            (prior / "progress-notes.txt").write_text("notes\n")
            hashes = {"project/scene.xml": hashlib.sha256(b"<mujoco/>").hexdigest(), "progress-notes.txt": hashlib.sha256(b"notes\n").hexdigest()}
            with patch("sim2sim.runtime.launch_container", wraps=runtime.launch_container) as launch:
                self.run_agent(directory, 'print(\'{"op":"finish"}\',flush=True)\ninput()\n',
                               extra={"continuation": {"public_dir": str(prior), "files_sha256": hashes}})
            ((volume, target),) = launch.call_args.kwargs["mounts"]
            self.assertTrue(volume.startswith("sim2sim-public-"))
            self.assertEqual(target, "/opt/public/prior")
            from sim2sim.audit import read_audit
            fixtures = [event for _, event in read_audit(Path(directory) / "run") if event.get("kind") == "public_fixture"]
            self.assertEqual([(event["path"], event["files_sha256"]) for event in fixtures], [("/opt/public/prior", hashes)])
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "prior").mkdir()
            (Path(directory) / "prior/progress-notes.txt").write_text("changed\n")
            with self.assertRaisesRegex(RuntimeFailure, "changed after preparation"):
                self.run_agent(directory, "", extra={"continuation": {"public_dir": str(Path(directory) / "prior"), "files_sha256": {}}})

    def test_panda_action_schema_payload_and_shared_ticks(self):
        script = '''
assert "arm_target" in task["actions"]
assert not set(["grasp", "move", "release"]) & set(task["actions"])
assert task["public_tool_config"]["robot"] == "panda"
assert task["public_robot_path"] == "/opt/public/robot"
for action in [{"op":"arm_target","joint_targets":[0]*7,"gripper_width_m":0.04,"ticks":2}, {"op":"reset"}, {"op":"arm_target","joint_targets":[0]*7,"gripper_width_m":0.08,"ticks":2}, {"op":"finish"}]:
 print(json.dumps(action),flush=True)
 json.loads(input())
'''
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_agent(directory, script, simulation={"robot": "panda", "rgb": False, "depth": False, "calibration": False})
            self.assertEqual(result["total_ticks"], 2)
            self.assertIn("volume", self.host_operations)
            from sim2sim.audit import read_audit
            events = [event for _, event in read_audit(Path(directory)/"run")]
            actions = [event["resolved_action"] for event in events if event.get("kind") == "action_begin" and event["resolved_action"]["op"] == "arm_target"]
            self.assertEqual(actions, [{"op": "arm_target", "world": "reference", "ticks": 2, "joint_targets": [0]*7, "gripper_width_m": .04}])

    def test_two_point_grasps_contract_forwarding_and_default_unchanged(self):
        script = '''
assert task["public_tool_config"]["point_grasps"] == 2 and "point_grasps" in task["observation_contract"]
assert all("grasp_index" in task["action_schema"][op] for op in ("grasp", "move", "release"))
for action in [{"op":"grasp","pixel":[1,1],"grasp_index":1}, {"op":"move","displacement":[0,0,.1],"ticks":2,"grasp_index":1}, {"op":"finish"}]:
 print(json.dumps(action),flush=True)
 json.loads(input())
'''
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.run_agent(directory, script, simulation={"point_grasps": 2})["total_ticks"], 2)
            from sim2sim.audit import read_audit
            actions = [event["resolved_action"] for _, event in read_audit(Path(directory)/"run") if event.get("kind") == "action_begin"]
            self.assertEqual([action.get("grasp_index") for action in actions if action["op"] in ("grasp", "move")], [1, 1])
        script = '''
assert "point_grasps" not in task["public_tool_config"] and "point_grasps" not in task["observation_contract"]
assert "grasp_index" not in json.dumps(task["action_schema"])
for action in [{"op":"grasp","pixel":[1,1],"grasp_index":1}, {"op":"finish"}]:
 print(json.dumps(action),flush=True)
 json.loads(input())
'''
        with tempfile.TemporaryDirectory() as directory:
            self.run_agent(directory, script)
            from sim2sim.audit import read_audit
            grasps = [event["resolved_action"] for _, event in read_audit(Path(directory)/"run") if event.get("kind") == "action_begin"]
            self.assertEqual([action for action in grasps if action["op"] == "grasp"], [{"op": "grasp", "world": "reference", "ticks": 0, "pixel": [1, 1]}])
        for simulation in ({"point_grasps": 3}, {"point_grasps": 2, "robot": "panda"}):
            with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeFailure, "point_grasps"):
                self.run_agent(directory, "", simulation=simulation)

    def test_weld_grasp_contract_rotation_forwarding_and_default_unchanged(self):
        script = '''
assert task["public_tool_config"]["grasp"] == "weld" and task["public_tool_config"]["grasp_slip_rad"] == .5
assert "rotation" in task["action_schema"]["move"] and "grasp" in task["observation_contract"]
assert "couple" in task["observation_contract"]["wrench"]
for action in [{"op":"grasp","pixel":[1,1]}, {"op":"move","displacement":[0,0,.1],"rotation":[0,0,1],"ticks":2}, {"op":"finish"}]:
 print(json.dumps(action),flush=True)
 json.loads(input())
'''
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.run_agent(directory, script, simulation={"grasp": "weld", "grasp_slip_rad": .5})["total_ticks"], 2)
            from sim2sim.audit import read_audit
            moves = [event["resolved_action"] for _, event in read_audit(Path(directory)/"run") if event.get("kind") == "action_begin"
                     and event["resolved_action"]["op"] == "move"]
            self.assertEqual(moves, [{"op": "move", "world": "reference", "ticks": 2, "displacement": [0, 0, .1], "rotation": [0, 0, 1]}])
        script = '''
assert not {"grasp", "grasp_slip_rad"} & set(task["public_tool_config"]) and "grasp" not in task["observation_contract"]
assert "rotation" not in task["action_schema"]["move"] and "ideal point torque zero" in task["observation_contract"]["wrench"]
for action in [{"op":"move","displacement":[0,0,.1],"rotation":[0,0,1],"ticks":2}, {"op":"finish"}]:
 print(json.dumps(action),flush=True)
 json.loads(input())
'''
        with tempfile.TemporaryDirectory() as directory:
            self.run_agent(directory, script)
            from sim2sim.audit import read_audit
            moves = [event["resolved_action"] for _, event in read_audit(Path(directory)/"run") if event.get("kind") == "action_begin"
                     and event["resolved_action"]["op"] == "move"]
            self.assertEqual(moves, [{"op": "move", "world": "reference", "ticks": 2, "displacement": [0, 0, .1]}])

    def test_shadow_hand_action_schema_contract_and_payload(self):
        script = '''
assert task["actions"][1] == "hand_target" and not set(["grasp", "move", "release", "arm_target"]) & set(task["actions"])
assert "rh_A_LFJ0" in task["action_schema"]["hand_target"]["actuator_targets"]
assert "robot/shadow_hand.xml" in task["action_schema"]["load_reconstruction"]["robot"]
contract = task["observation_contract"]
assert "rh_THJ1" in contract["proprioception"] and "actuator_effort_nm[20]" in contract["proprioception"]
assert "tactile" not in contract and "wrench" not in contract
for action in [{"op":"hand_target","actuator_targets":[0.1]*20,"ticks":2}, {"op":"arm_target","joint_targets":[0]*7,"gripper_width_m":0.04,"ticks":1}, {"op":"grasp","pixel":[1,1]}, {"op":"reset"}, {"op":"finish"}]:
 print(json.dumps(action),flush=True)
 reply=json.loads(input())
 assert reply["ok"] is (action["op"] in ("hand_target", "reset", "finish")), reply
'''
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_agent(directory, script, simulation={"robot": "shadow_hand", "rgb": False, "depth": False, "calibration": False})
            self.assertEqual(result["total_ticks"], 2)
            from sim2sim.audit import read_audit
            actions = [event["resolved_action"] for _, event in read_audit(Path(directory)/"run") if event.get("kind") == "action_begin" and event["resolved_action"]["op"] == "hand_target"]
            self.assertEqual(actions, [{"op": "hand_target", "world": "reference", "ticks": 2, "actuator_targets": [0.1]*20}])

    def test_arm_with_hand_offers_both_actions_without_gripper(self):
        script = '''
assert task["actions"][1:3] == ["arm_target", "hand_target"] and not set(["grasp", "move", "release"]) & set(task["actions"])
assert set(task["action_schema"]["arm_target"]) == {"joint_targets", "ticks"}
assert "robot/panda_shadow.xml" in task["action_schema"]["hand_target"]["actuator_targets"]
contract = task["observation_contract"]["proprioception"]
assert "joint_positions_rad[31]" in contract and "joint1, joint2" in contract and "actuator_effort_nm[27]" in contract
for action in [{"op":"arm_target","joint_targets":[0]*7,"ticks":1}, {"op":"hand_target","actuator_targets":[0.1]*20,"ticks":1}, {"op":"finish"}]:
 print(json.dumps(action),flush=True)
 assert json.loads(input())["ok"]
'''
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.run_agent(directory, script, simulation={"robot": "panda_shadow", "rgb": False, "depth": False,
                                                                           "calibration": False})["total_ticks"], 2)

    def test_fixture_actions_contract_private_seed_and_reference_only(self):
        script = '''
assert task["actions"].index("reset_carriage") < task["actions"].index("reset") and "release_carriage" in task["actions"]
assert task["submission"]["entrypoint"] == "predictor.py" and set(task["submission"]["predictor"]["functions"]) == {"predict", "settings"}
assert "fixture" in task["observation_contract"] and "fixture_seed" not in json.dumps(task)
for action, ok in [({"op":"release_carriage","world":"reconstruction"}, False), ({"op":"reset_carriage","ticks":3}, False),
                   ({"op":"release_carriage","ticks":2}, True), ({"op":"finish"}, True)]:
 print(json.dumps(action),flush=True)
 reply=json.loads(input())
 assert reply["ok"] is ok, reply
'''
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_agent(directory, script, manifest="[fixture]\nseed = 1\n",
                                    extra={"evaluation": {"entrypoint": "predictor.py", "metrics": ["fixture"]}})
            self.assertEqual(result["total_ticks"], 2)
            self.assertIn("fixture_seed", json.loads((Path(directory) / "run/manifest.json").read_text())["resolved_source_config"])
        script = '''
assert "release_carriage" not in task["actions"] and task["submission"]["entrypoint"] == "scene.xml" and "predictor" not in task["submission"]
for action in [{"op":"release_carriage"}, {"op":"finish"}]:
 print(json.dumps(action),flush=True)
 reply=json.loads(input())
assert True
'''
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.run_agent(directory, script)["total_ticks"], 0)

    def test_observe_only_reference_rejects_interaction(self):
        script = '''
assert task["worlds"]["reference"].startswith("Observe-only") and task["action_schema"]["grasp"]["world"] == "reconstruction only"
for action, ok in [({"op":"grasp","pixel":[1,1]}, False), ({"op":"move","displacement":[0,0,.1],"ticks":1}, False), ({"op":"observe","ticks":1}, True)]:
 print(json.dumps(action),flush=True)
 reply=json.loads(input())
 assert reply["ok"] is ok, reply
 assert ok or "observe-only" in reply["error"], reply
print('{"op":"finish"}',flush=True)
input()
'''
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_agent(directory, script, simulation={"reference_interaction": False})
            self.assertEqual(result["total_ticks"], 1)

    def test_no_reset_run_has_no_reset_action_in_either_world(self):
        script = '''
assert "reset" not in task["actions"] and "reset" not in task["action_schema"], task["actions"]
for action, ok in [({"op":"observe","ticks":2}, True), ({"op":"reset"}, False), ({"op":"reset","world":"reconstruction"}, False),
                   ({"op":"observe","ticks":1}, True)]:
 print(json.dumps(action),flush=True)
 reply=json.loads(input())
 assert reply["ok"] is ok, reply
 assert ok or "reset is not available" in reply["error"], reply
print('{"op":"finish"}',flush=True)
input()
'''
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.run_agent(directory, script, simulation={"resets": False})["total_ticks"], 3)
        default = '''
assert "reset" in task["actions"] and "reset" in task["action_schema"]
print('{"op":"finish"}',flush=True)
input()
'''
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.run_agent(directory, default)["total_ticks"], 0)

    def test_reset_never_refunds_ticks(self):
        script = '''
for action in [{"op":"observe","ticks":2},{"op":"reset"},{"op":"observe","ticks":2}]:
 print(json.dumps(action),flush=True)
 reply=json.loads(input())
 if action.get("ticks")==2 and action is not None and reply.get("remaining_ticks")==1: pass
print('{"op":"finish"}',flush=True)
input()
'''
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_agent(directory, script)
            self.assertEqual(result["total_ticks"], 2)
            events = [json.loads(x) for x in (Path(directory)/"run/events.jsonl").read_text().splitlines()]
            self.assertFalse(events[2]["ok"])
            from sim2sim.audit import read_audit
            responses = [event["payload"] for _,event in read_audit(Path(directory)/"run")
                         if event.get("kind") == "boundary" and event.get("direction") == "response"]
            self.assertTrue(all("original_mass" not in response.get("observation", {}) and "qpos" not in response.get("observation", {})
                                and "body_hits" not in response.get("observation", {}) for response in responses))
            termination = json.loads((Path(directory)/"run/termination.json").read_text())
            self.assertEqual(termination["remaining_ticks"], 1)

    def test_active_deadline_stops_idle_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeFailure):
                self.run_agent(directory, "time.sleep(2)", {"physics_ticks": 3, "active_seconds": .05})
            termination = json.loads((Path(directory)/"run/termination.json").read_text())
            self.assertEqual(termination["status"], "failed")
            self.assertEqual(termination["remaining_active_seconds"], 0)

    def test_scoring_pauses_budget_and_only_validity_returns(self):
        script = '''
print('{"op":"submit","project_b64":"eA=="}',flush=True)
r=json.loads(input())
assert r == {"ok":True,"submission":1,"status":"failed"}, r
time.sleep(.1)
print('{"op":"finish"}',flush=True)
input()
'''
        def evaluate(*args):
            time.sleep(.2)
            return {"status": "failed", "scores": {"secret": 123}}
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_agent(directory, script, {"physics_ticks": 3, "active_seconds": .3}, evaluate)
            self.assertEqual(result["submissions"], 1)
            self.assertLess(self.host_operations.index("pause"), self.host_operations.index("plot"))
            self.assertLess(self.host_operations.index("plot"), self.host_operations.index("unpause"))

    def test_freeze_and_thaw_time_is_not_charged_to_the_active_budget(self):
        """Slow freeze/thaw RPCs (up to 76 s, probe job 63755982) are infrastructure: the evaluation pause starts before the
        STOP and ends after the CONT, so 0.5 s of freezing does not exhaust a 0.3 s active budget."""
        script = '''
print('{"op":"submit","project_b64":"eA=="}',flush=True)
r=json.loads(input())
assert r["ok"], r
print('{"op":"finish"}',flush=True)
input()
'''
        with tempfile.TemporaryDirectory() as directory, patch("sim2sim.runtime._set_paused", side_effect=lambda *a: time.sleep(.25)):
            result = self.run_agent(directory, script, {"physics_ticks": 3, "active_seconds": .3},
                                    lambda *args: {"status": "failed", "scores": None})
            self.assertEqual(result["submissions"], 1)
            termination = json.loads((Path(directory) / "run/termination.json").read_text())
            self.assertEqual(termination["status"], "finished")
            self.assertGreater(termination["remaining_active_seconds"], 0)

    def test_wrench_default_is_independent_and_absent_from_observation(self):
        script = """assert task['observation_contract']['modalities']['wrench'] is False
assert task['observation_contract']['modalities']['rgb'] is True
assert 'wrench' not in task['observation']
assert 'original_mass' not in task['observation'] and 'qpos' not in task['observation'] and 'body_hits' not in task['observation']
print('{"op":"finish"}',flush=True)
input()
"""
        with tempfile.TemporaryDirectory() as directory:
            self.run_agent(directory, script)

    def test_lossless_boundaries_include_rejected_and_inference_requests(self):
        from sim2sim.audit import read_audit
        script = '''
for action in [{"op":"observe","world":"invalid","pixel":"long-"*500},{"op":"reset","ticks":-1},{"op":"inference","messages":[{"role":"user","content":"complete private inference prompt"}]}]:
 print(json.dumps(action),flush=True)
 assert not json.loads(input())["ok"]
print('{"op":"finish"}',flush=True)
input()
'''
        with tempfile.TemporaryDirectory() as directory:
            self.run_agent(directory, script)
            records = [event for _, event in read_audit(Path(directory)/"run")]
            requests = [event for event in records if event.get("kind") == "boundary" and event.get("direction") == "request"]
            self.assertEqual(len(requests), 4)
            self.assertEqual(requests[0]["payload"]["pixel"], "long-"*500)
            self.assertEqual(requests[2]["payload"]["messages"][0]["content"], "complete private inference prompt")
            for event in requests:
                self.assertEqual(json.loads(base64.b64decode(event["wire_b64"])), event["payload"])
            self.assertTrue(any(event.get("kind") == "rejection" for event in records))

    def test_agent_never_sees_spend_and_budget_stop_suspends(self):
        """Cost is orchestrator-only: no USD, cap or remaining-budget field reaches the agent side."""
        from sim2sim.audit import read_audit
        from sim2sim.runtime import BudgetStop
        script = r'''
import re
assert not re.search(r"cost|usd|dollar|max_calls|max_total_tokens|remaining_(calls|tokens|cost)", json.dumps(task), re.I), task["inference"]
print('{"op":"inference","messages":[{"role":"user","content":"hi"}]}',flush=True)
reply=json.loads(input())["inference"]
assert set(reply) == {"content","tool_calls","usage","deadline_unix"} and set(reply["usage"]) == {"promptTokens","cacheReadTokens","cacheWriteTokens"}, reply
print('{"op":"inference","messages":[{"role":"user","content":"hi"}]}',flush=True)
reply=json.loads(input())["inference"]
assert reply.pop("deadline_unix") is None and reply == {"budget_stop": True, "reason": "host stop"}  # no wall budget in this test
print('{"op":"finish"}',flush=True)
input()
'''
        value = {"content": "ok", "tool_calls": [], "finish_reason": "stop", "model": "m", "summaries": [], "cost_usd": 1.25,
                 "usage": {"promptTokens": 3, "completionTokens": 1, "totalTokens": 4, "cacheReadTokens": 0, "cacheWriteTokens": 0,
                           "reasoningTokens": 0, "accountedTokens": 4},
                 "calls": 1, "remaining_calls": 9, "cumulative_tokens": 4, "remaining_tokens": None, "cumulative_cost_usd": 1.25,
                 "remaining_cost_usd": 0.75}
        with tempfile.TemporaryDirectory() as directory, \
                patch("sim2sim.runtime.InferenceBroker.call", side_effect=[value, BudgetStop("Run inference cost budget exhausted")]):
            self.run_agent(directory, script)
            status = json.loads((Path(directory)/"run/termination.json").read_text())
            self.assertEqual((status["status"], status["reason"]), ("finished", "budget_suspended"))
            records = [event for _, event in read_audit(Path(directory)/"run")]
            self.assertEqual([e["cost_usd"] for e in records if e.get("kind") == "inference_accounting"], [1.25])
            self.assertEqual([e["reason"] for e in records if e.get("kind") == "inference_suspended"], ["Run inference cost budget exhausted"])

    def test_budget_hold_freezes_the_run_and_resumes_with_raised_caps(self):
        """A spend-cap hold pauses the containers and both clocks; resuming retries the call as if the cap had been higher."""
        import threading
        from sim2sim.audit import read_audit
        from sim2sim.runtime import BudgetStop
        script = r'''
print('{"op":"inference","messages":[{"role":"user","content":"hi"}]}',flush=True)
start=time.time()
reply=json.loads(input())["inference"]
assert reply["content"] == "ok" and reply["deadline_unix"] - start > 8.6, (reply, start)
print('{"op":"finish"}',flush=True)
input()
'''
        value = {"content": "ok", "tool_calls": [], "finish_reason": "stop", "model": "m", "summaries": [], "cost_usd": 1.0, "usage": None,
                 "calls": 1, "remaining_calls": 9, "cumulative_tokens": 4, "remaining_tokens": None, "cumulative_cost_usd": 1.0, "remaining_cost_usd": 1.0}
        with tempfile.TemporaryDirectory() as directory, \
                patch("sim2sim.runtime.InferenceBroker.call", side_effect=[BudgetStop("Run inference cost budget exhausted"), value]):
            run = Path(directory) / "run"
            def decide():
                while not (run / "hold.json").exists():
                    time.sleep(.05)
                time.sleep(1.5)
                (run / "hold-decision.json").write_text(json.dumps({"action": "resume", "max_cost_usd": 10, "cost_pool_usd": 100}))
            threading.Thread(target=decide, daemon=True).start()
            self.run_agent(directory, script, {"wall_only": True, "wall_seconds": 10}, extra={"inference": {"budget_hold_seconds": 60}})
            status = json.loads((run / "termination.json").read_text())
            self.assertEqual(status["status"], "finished")
            self.assertGreater(status["held_seconds"], 1.4)
            self.assertLess(status["wall_elapsed"], 1.4)
            self.assertEqual(status["holds"][0]["decision"]["action"], "resume")
            self.assertIn("pause", self.host_operations)
            self.assertTrue((run / "hold-001.json").exists() and (run / "hold-decision-001.json").exists())
            kinds = [e.get("kind") for _, e in read_audit(run)]
            self.assertIn("inference_hold_decision", kinds)
            self.assertNotIn("inference_suspended", kinds)

    def test_provider_outage_is_held_outside_the_run(self):
        """A provider timeout's attempt time is given back and the retry backoff runs paused; the agent only sees the answer."""
        from sim2sim.audit import read_audit
        from sim2sim.runtime import ProviderUnavailable
        script = r'''
print('{"op":"inference","messages":[{"role":"user","content":"hi"}]}',flush=True)
reply=json.loads(input())["inference"]
assert reply["content"] == "ok", reply
print('{"op":"finish"}',flush=True)
input()
'''
        value = {"content": "ok", "tool_calls": [], "finish_reason": "stop", "model": "m", "summaries": [], "cost_usd": 1.0, "usage": None,
                 "calls": 2, "remaining_calls": 8, "cumulative_tokens": 4, "remaining_tokens": None, "cumulative_cost_usd": 1.0, "remaining_cost_usd": None}
        attempts = []
        def call(*args):  # first attempt times out after 1 s; the host-side retry succeeds
            attempts.append(1)
            if len(attempts) == 1:
                time.sleep(1.0)
                raise ProviderUnavailable("Inference wall-time budget exhausted")
            return value
        with tempfile.TemporaryDirectory() as directory, patch("sim2sim.runtime.PROVIDER_BACKOFF", 0.3), \
                patch("sim2sim.runtime.InferenceBroker.call", side_effect=call):
            run = Path(directory) / "run"
            self.run_agent(directory, script, {"wall_only": True, "wall_seconds": 10}, extra={"inference": {"provider_hold_seconds": 60}})
            status = json.loads((run / "termination.json").read_text())
            self.assertEqual(status["status"], "finished")
            self.assertGreater(status["held_seconds"], 1.25)
            self.assertLess(status["wall_elapsed"], 1.0)
            self.assertIn("pause", self.host_operations)
            holds = [e for _, e in read_audit(run) if e.get("kind") == "provider_hold"]
            self.assertEqual([h["consecutive_failures"] for h in holds], [1])

    def test_wall_checkpoint_holds_once_and_stops_on_decision(self):
        """At checkpoint_wall_seconds the next inference call holds like a spend-cap hold; a stop decision suspends the run."""
        import threading
        from sim2sim.audit import read_audit
        script = r'''
time.sleep(1.2)
print('{"op":"inference","messages":[{"role":"user","content":"hi"}]}',flush=True)
reply=json.loads(input())["inference"]
assert reply.get("content") is None, reply
print('{"op":"finish"}',flush=True)
input()
'''
        with tempfile.TemporaryDirectory() as directory, patch("sim2sim.runtime.InferenceBroker.call") as call:
            run = Path(directory) / "run"
            def decide():
                while not (run / "hold.json").exists():
                    time.sleep(.05)
                (run / "hold-decision.json").write_text(json.dumps({"action": "stop"}))
            threading.Thread(target=decide, daemon=True).start()
            self.run_agent(directory, script, {"wall_only": True, "wall_seconds": 10},
                           extra={"inference": {"budget_hold_seconds": 60, "checkpoint_wall_seconds": 1}})
            status = json.loads((run / "termination.json").read_text())
            self.assertEqual(status["reason"], "budget_suspended")
            self.assertEqual([h["reason"] for h in status["holds"]], ["Wall-time checkpoint reached"])
            call.assert_not_called()  # the checkpoint holds before any provider call
            reasons = [e["reason"] for _, e in read_audit(run) if e.get("kind") == "inference_suspended"]
            self.assertEqual(reasons, ["Wall-time checkpoint stop"])

    def test_partial_timeout_records_exact_confirmed_reference_ticks(self):
        from sim2sim.audit import read_audit
        script = '''
print('{"op":"observe","ticks":10}',flush=True)
input()
'''
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeFailure):
                self.run_agent(directory, script, {"physics_ticks": 10, "active_seconds": .12}, tick_delay=.04)
            records = [event for _, event in read_audit(Path(directory)/"run")]
            ticks = [event for event in records if event.get("kind") == "tick"]
            self.assertGreater(len(ticks), 0)
            self.assertLess(len(ticks), 10)
            status = json.loads((Path(directory)/"run/termination.json").read_text())
            self.assertEqual(status["total_ticks"], len(ticks))
            end = next(event for event in records if event.get("kind") == "action_end")
            self.assertEqual(end["executed_ticks"], len(ticks))

    def test_panda_partial_timeout_keeps_targets_and_executed_ticks(self):
        from sim2sim.audit import read_audit
        action = {"op": "arm_target", "joint_targets": [0]*7, "gripper_width_m": .04, "ticks": 10}
        script = "print(" + repr(json.dumps(action)) + ",flush=True)\ninput()"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeFailure):
                self.run_agent(directory, script, {"physics_ticks": 10, "active_seconds": .12},
                               tick_delay=.04, simulation={"robot": "panda"})
            records = [event for _, event in read_audit(Path(directory)/"run")]
            ticks = [event for event in records if event.get("kind") == "tick"]
            self.assertGreater(len(ticks), 0)
            self.assertLess(len(ticks), 10)
            begin = next(event for event in records if event.get("kind") == "action_begin")
            self.assertEqual(begin["resolved_action"], dict(action, world="reference"))
            end = next(event for event in records if event.get("kind") == "action_end")
            self.assertEqual(end["executed_ticks"], len(ticks))

    def test_audit_reservation_refuses_load_before_worker_launch(self):
        from sim2sim.audit import read_audit
        def forbidden_worker(*args):
            raise AssertionError("Worker launch must follow a successful reservation")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeFailure, "audit byte budget"):
                self.run_agent(directory, "print('{\"op\":\"load_reconstruction\",\"project_b64\":\"eA==\"}',flush=True)\ninput()",
                               {"physics_ticks": 10, "active_seconds": 2, "audit_bytes": 40*1024*1024}, candidate_factory=forbidden_worker)
            records = [event for _, event in read_audit(Path(directory)/"run")]
            self.assertTrue(any(event.get("payload", {}).get("op") == "load_reconstruction" for event in records))
            self.assertFalse(any(event.get("kind") == "action_begin" for event in records))

    def test_worlds_are_independent_and_share_nonrefundable_budget(self):
        from sim2sim.candidate_worker import state_digest
        from sim2sim.audit import read_audit
        class FakeCandidate:
            def __init__(self, sandbox, output, audit):
                self.closed = False
                self.name = "candidate-test"
                self.ticks = self.total = 0
                self.last = None
            def close(self):
                self.closed = True
            def exchange(self, request, deadline, token, action_id, on_tick=None):
                if request["op"] == "load":
                    assert "graspable_bodies" not in request["config"]
                    assert "trusted_camera" not in request["config"]
                    assert request["config"]["camera"] == "own"
                else:
                    action = request["action"]
                    if action["op"] == "reset":
                        self.ticks = 0
                    for _ in range(action["ticks"]):
                        self.ticks += 1
                        self.total += 1
                        on_tick(self.packet("tick"))
                self.last = self.packet("result")
                return self.last
            def packet(self, kind):
                state = {"ticks": self.ticks, "total_ticks": self.total, "time": self.ticks*.002}
                observation = {"ticks": self.ticks, "total_ticks": self.total}
                return {"type": kind, "ok": True, "state": state, "state_sha256": state_digest(state),
                        "observation": observation, "full_observation": observation}
        script = '''
def call(action):
 print(json.dumps(action),flush=True)
 return json.loads(input())
assert call({"op":"observe","ticks":2})["observation"]["total_ticks"] == 2
assert call({"op":"load_reconstruction","project_b64":"eA==","camera":"own"})["ok"]
assert call({"op":"observe","world":"reconstruction","ticks":2})["observation"]["total_ticks"] == 2
assert call({"op":"observe","ticks":1})["observation"]["total_ticks"] == 3
assert call({"op":"reset","world":"reconstruction"})["observation"]["ticks"] == 0
assert not call({"op":"observe","world":"reconstruction","ticks":1})["ok"]
assert call({"op":"load_reconstruction","project_b64":"eA==","camera":"own"})["remaining_ticks"] == 0
assert call({"op":"observe","ticks":0})["observation"]["total_ticks"] == 3
call({"op":"finish"})
'''
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_agent(directory, script, {"physics_ticks": 5, "active_seconds": 3}, candidate_factory=FakeCandidate)
            self.assertEqual(result["world_ticks"], {"reference": 3, "reconstruction": 2})
            records = [event for _, event in read_audit(Path(directory)/"run")]
            begins = [event for event in records if event.get("kind") == "action_begin"]
            self.assertTrue(all("ticks" in event["resolved_action"] for event in begins))
            self.assertEqual(sum(event.get("kind") == "tick" for event in records), 5)

    def test_malformed_json_is_durable_before_rejection(self):
        from sim2sim.audit import read_audit
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeFailure, "invalid JSON"):
                self.run_agent(directory, "print('{not-json',flush=True)\ninput()")
            records = [event for _, event in read_audit(Path(directory)/"run")]
            received = next(event for event in records if event.get("kind") == "boundary" and event.get("direction") == "request")
            self.assertEqual(base64.b64decode(received["wire_b64"]), b'{not-json\n')
            self.assertTrue(received["complete"])
            self.assertFalse(any(event.get("kind") == "action_begin" for event in records))

    def test_progress_is_timestamped_in_sidefile_and_audit(self):
        from sim2sim.audit import read_audit
        script = '''
print('{"op":"progress","text":"Testing a public geometric hypothesis.","origin":"model"}',flush=True)
assert json.loads(input())["ok"]
print('{"op":"finish"}',flush=True)
input()
'''
        with tempfile.TemporaryDirectory() as directory:
            self.run_agent(directory, script)
            note = json.loads((Path(directory)/"run/progress.jsonl").read_text())
            self.assertEqual(note["origin"], "model")
            self.assertIn("utc_time", note)
            self.assertGreaterEqual(note["wall_elapsed"], note["active_elapsed"])
            events = [event for _,event in read_audit(Path(directory)/"run")]
            self.assertIn(note, events)

    def test_hard_wall_includes_private_evaluation_time(self):
        called = []
        def evaluate(*args):
            called.append(True)
            time.sleep(2)
            return {"status":"ok", "scores":{}}
        script = '''
print('{"op":"submit","project_b64":"eA=="}',flush=True)
input()
'''
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_agent(directory, script, {"physics_ticks":3,"active_seconds":2,"wall_seconds":1.5}, evaluate)
            self.assertEqual((result["status"], result["reason"], result["submissions"]), ("finished", "wall_deadline", 1))
            self.assertTrue(called)
            status = json.loads((Path(directory)/"run/termination.json").read_text())
            self.assertTrue(status["hard_wall_expired"])
            self.assertEqual(status["reason"], "wall_deadline")
            from sim2sim.audit import read_audit
            reports = [event["report"]["status"] for _, event in read_audit(Path(directory)/"run") if event.get("kind") == "evaluation"]
            self.assertEqual(reports, ["ok"])  # finished before the deadline-interrupted delivery

    def test_deadline_records_unfinished_evaluation_as_unevaluated(self):
        def evaluate(*args):
            time.sleep(2)
            raise RuntimeFailure("Hard wall-time budget exhausted")
        script = '''
print('{"op":"submit","project_b64":"eA=="}',flush=True)
input()
'''
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_agent(directory, script, {"physics_ticks":3,"active_seconds":2,"wall_seconds":1.5}, evaluate)
            self.assertEqual(result["reason"], "wall_deadline")
            from sim2sim.audit import read_audit
            reports = [event["report"] for _, event in read_audit(Path(directory)/"run") if event.get("kind") == "evaluation"]
            self.assertEqual([report["status"] for report in reports], ["unevaluated"])


def test_slurm_step_freeze_waits_out_slow_controller_rpcs_and_never_overlaps_them(tmp_path):
    """scancel RPCs took up to 76 s (probe job 63755982); a 30 s client timeout killed every attempt (runs 63706221,
    63706223). Each request runs to completion; one is resent only after it returned without the freezer state."""
    import pytest
    import subprocess
    from unittest.mock import patch
    from sim2sim import runtime
    events = tmp_path / "cgroup.events"
    events.write_text("populated 1\nfrozen 1\n")
    clock, calls = [0.], []

    def scancel(args, **options):
        calls.append((args, options["timeout"]))
        clock[0] += 76  # a slow controller RPC that completes
        if len(calls) == 2:  # the first CONT was lost; the second takes effect
            events.write_text("populated 1\nfrozen 0\n")

    with patch.dict(runtime.SLURM_STEPS, {"agent": ("1.2", tmp_path)}), patch.object(runtime.subprocess, "run", scancel), \
            patch.object(runtime.time, "monotonic", lambda: clock[0]), \
            patch.object(runtime.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s)):
        runtime._set_paused("agent", False, "slurm_step")
    assert [args for args, _ in calls] == [["scancel", "--signal=CONT", "1.2"]] * 2
    assert calls[0][1] == runtime.FREEZE_DEADLINE_SECONDS and calls[1][1] < calls[0][1]  # the client waits out the RPC

    def stalled(args, **options):  # an RPC still running at the deadline: fail loudly, never a second overlapping request
        raise subprocess.TimeoutExpired(args, options["timeout"])
    clock[0], calls[:] = 0., []
    with patch.dict(runtime.SLURM_STEPS, {"agent": ("1.2", tmp_path)}), patch.object(runtime.subprocess, "run", stalled), \
            patch.object(runtime.time, "monotonic", lambda: clock[0]), pytest.raises(runtime.SandboxFailure, match="within 900 s"):
        runtime._set_paused("agent", True, "slurm_step")


def test_slurm_step_freeze_failure_reports_cgroup_and_task_states(tmp_path):
    """A step that never freezes (job 63706221) fails loudly with the freezer, memory and per-task evidence."""
    import os
    import pytest
    from unittest.mock import patch
    from sim2sim import runtime
    leaf = tmp_path / "step" / "leaf"
    leaf.mkdir(parents=True)
    (leaf / "cgroup.events").write_text("populated 1\nfrozen 0\n")
    (leaf / "cgroup.freeze").write_text("1\n")
    (leaf / "cgroup.procs").write_text("%d\n" % os.getpid())
    (leaf.parent / "memory.events").write_text("low 0\nhigh 0\nmax 7\noom 0\noom_kill 0\n")
    clock = [0.]
    with patch.dict(runtime.SLURM_STEPS, {"agent": ("1.1", leaf)}), patch.object(runtime.subprocess, "run", lambda *a, **k: None), \
            patch.object(runtime.time, "monotonic", lambda: clock[0]), \
            patch.object(runtime.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s)), \
            pytest.raises(runtime.SandboxFailure) as failure:
        runtime._set_paused("agent", True, "slurm_step")
    evidence = json.loads(str(failure.value).split(": ", 1)[1])
    assert evidence["cgroups"][str(leaf)]["cgroup.freeze"] == "1" and "max 7" in evidence["cgroups"][str(leaf.parent)]["memory.events"]
    assert evidence["tasks"][str(os.getpid())]["state"] in "RS" and evidence["tasks"][str(os.getpid())]["comm"]


def test_prepared_output_accepts_continuation_prior_directory(tmp_path):
    """Regression (c1-assembly-*-20261103-001): a prepared continuation's prior/ carry-over is a directory."""
    from sim2sim.runtime import validate_prepared_output
    (tmp_path / "source").mkdir()
    (tmp_path / "prior/project").mkdir(parents=True)
    (tmp_path / "prior/project/scene.xml").write_text("<mujoco/>")
    for name in ("experiment.json", "status.json", "config-private.json"):
        (tmp_path / name).write_text("{}")
    validate_prepared_output(tmp_path)
    (tmp_path / "prior/link").symlink_to(tmp_path / "status.json")
    try:
        validate_prepared_output(tmp_path)
    except RuntimeFailure:
        return
    raise AssertionError("symlink inside prior/ must be refused")


def test_image_roles_allow_the_split_pair_or_one_declared_image_only():
    from types import SimpleNamespace
    from sim2sim.runtime import SandboxFailure, check_image_roles
    labels = {"agent": {"sim2sim.image-role": "agent"}, "eval": {"sim2sim.image-role": "evaluator"}, "old": None}
    inspect = lambda args, **options: subprocess.CompletedProcess(args, 0, stdout=json.dumps(labels[args[-1]]))
    clock = SimpleNamespace(remaining=lambda: 60)
    with patch("sim2sim.runtime.subprocess.run", side_effect=inspect):
        check_image_roles({"image": "agent", "evaluator_image": "eval"}, {}, clock)
        check_image_roles({"image": "old", "evaluator_image": "old"}, {}, clock)  # pre-split configs name one image for both
        for image, evaluator in (("eval", "agent"), ("old", "eval"), ("agent", "old")):
            try:
                check_image_roles({"image": image, "evaluator_image": evaluator}, {}, clock)
                raise AssertionError((image, evaluator))
            except SandboxFailure:
                pass


def test_trusted_image_records_dispatch_on_their_shape(tmp_path):
    root = Path(__file__).resolve().parents[1]
    lines = lambda record: subprocess.run([sys.executable, str(root / "scripts/trusted-images.py"), str(record)], capture_output=True, text=True)
    single = json.loads((root / "experiments/images/trusted-image.json").read_text())
    assert lines(root / "experiments/images/trusted-image.json").stdout.split() == \
        ["image", "localhost/sim2sim:latest", single["image_id"], single["archive"], single["archive_sha256"]]
    two = json.loads((root / "experiments/images/trusted-image-isolated.json").read_text())["images"]
    assert [line.split()[:3] for line in lines(root / "experiments/images/trusted-image-isolated.json").stdout.splitlines()] == \
        [["agent_image", "localhost/sim2sim-agent:latest", two["agent"]["image_id"]], ["evaluator_image", "localhost/sim2sim:latest", two["evaluator"]["image_id"]]]
    for bad in ({"images": two, "image_id": single["image_id"]}, {"images": {"agent": two["agent"]}}, {"image_id": single["image_id"]}):
        (tmp_path / "bad.json").write_text(json.dumps(bad))
        assert lines(tmp_path / "bad.json").returncode != 0, bad
