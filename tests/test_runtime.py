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
        self.assertIn("--ulimit=nproc=64:64", cmd)

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

    def test_client_protocol(self):
        writer = io.StringIO()
        client = Client(io.StringIO('{"type":"task"}\n{"ok":true,"observation":{"ticks":2}}\n'), writer)
        self.assertEqual(client.observe(ticks=2), {"ticks": 2})
        self.assertEqual(json.loads(writer.getvalue()), {"op": "observe", "ticks": 2})


if __name__ == "__main__":
    unittest.main()

class HostBudgetTests(unittest.TestCase):
    def run_agent(self, directory, script, budget=None, evaluate=None):
        import types
        from sim2sim.runtime import run

        class FakeSimulation:
            def __init__(self, *args):
                self.total_ticks = self.ticks = 0
                self._callback = None
            def observe(self):
                return {"total_ticks": self.total_ticks}
            def action(self, request):
                if request["op"] == "reset":
                    self.ticks = 0
                else:
                    for _ in range(request.get("ticks", 0)):
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
        config = {"source": {"scene": str(source / "scene.xml")}, "sandbox": {"image": "test"}, "record_video": False,
                  "output_dir": str(Path(directory) / "run"), "budget": budget or {"physics_ticks": 3, "active_seconds": 2},
                  "evaluation": {}}
        cmd = [sys.executable, "-u", "-c", "import json,sys,time\njson.loads(input())\n" + script]
        with patch.dict(sys.modules, {"sim2sim.simulation": types.SimpleNamespace(Simulation=FakeSimulation)}), \
             patch("sim2sim.runtime.check_rootless"), patch("sim2sim.runtime.check_limits"), patch("sim2sim.runtime.container_command", return_value=cmd), \
             patch("sim2sim.runtime.stop_container", side_effect=stop), \
             patch("sim2sim.runtime.evaluate_in_container", side_effect=evaluate), \
             patch("sim2sim.runtime.subprocess.run"):
            return run(config)

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
