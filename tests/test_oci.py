"""Opt-in real sandbox check; run on an approved compute node after building the image.

SIM2SIM_OCI_TEST=1 MUJOCO_GL=egl python -m unittest discover -s tests -p test_oci.py
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("SIM2SIM_OCI_TEST") == "1", "requires explicit rootless OCI integration opt-in")
class OCIIntegrationTests(unittest.TestCase):
    def test_full_agent_and_private_evaluator(self):
        import imageio.v2 as imageio
        from sim2sim.runtime import run
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="sim2sim-oci-") as directory:
            output = Path(directory) / "run"
            config = {
                "source": {"scene": str(root / "environments/blocks/scene.xml")},
                "sandbox": {"image": os.environ.get("SIM2SIM_AGENT_IMAGE", "localhost/sim2sim:latest"),
                            "evaluator_image": os.environ.get("SIM2SIM_EVALUATOR_IMAGE", "localhost/sim2sim:latest"),
                            "command": ["python", "/opt/agent/scripted_agent.py", "--integration"]},
                "output_dir": str(output), "record_video": True, "video_fps": 30,
                "budget": {"active_seconds": 120, "physics_ticks": 300, "submissions": 2},
                "simulation": {"camera": "fixed", "width": 320, "height": 240,
                               "rgb": True, "depth": False, "calibration": True},
                "evaluation": {"source_dir": str(root / "environments/blocks"), "camera": "fixed",
                               "width": 160, "height": 120, "timeout_seconds": 120,
                               "correspondence": {"block_red": "object", "block_blue": "other"}},
            }
            with patch.dict(os.environ, {"SIM2SIM_TEST_SECRET": "host-only"}):
                result = run(config)
            self.assertEqual(result["total_ticks"], 30)
            self.assertEqual(result["submissions"], 2)
            termination = json.loads((output / "termination.json").read_text())
            self.assertEqual(termination["status"], "finished")
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertTrue(all(event.get("ok", True) for event in events))
            self.assertTrue({"grasp", "move", "release", "reset", "submit"}.issubset({event["op"] for event in events}))
            for version in ("version-001", "version-002"):
                report = json.loads((output / version / "evaluation.json").read_text())
                self.assertEqual(report["status"], "ok", report)
                self.assertTrue(report["scores"])
            index = json.loads((output / "video-index.json").read_text())
            self.assertEqual(len(index), 2)
            self.assertEqual(index[0]["end_total_ticks"], 25)
            self.assertEqual(index[1]["start_total_ticks"], 25)
            for segment in index:
                reader = imageio.get_reader(output / segment["file"])
                try:
                    self.assertEqual(reader.get_meta_data()["fps"], 30)
                    self.assertGreater(reader.get_data(0).size, 0)
                finally:
                    reader.close()
