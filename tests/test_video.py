"""Real rendering/encoding checks; run only on the approved Slurm test partition."""
import json
from pathlib import Path
import tempfile

import imageio.v2 as imageio

from sim2sim.runtime import VideoRecorder
from sim2sim.simulation import Simulation


def test_playable_simulation_time_videos_split_on_reset():
    scene = Path(__file__).resolve().parents[1] / "environments/blocks/scene.xml"
    sim = Simulation(scene, {"camera": "fixed", "width": 96, "height": 64,
                             "rgb": True, "depth": False, "calibration": False})
    try:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            recorder = VideoRecorder(output, sim, 30)
            try:
                sim._callback = lambda simulation: recorder.tick()
                sim.action({"op": "observe", "ticks": 20})
                recorder.finish_segment()
                sim.action({"op": "reset"})
                recorder.start_segment()
                sim.action({"op": "observe", "ticks": 20})
            finally:
                recorder.close()
            index = json.loads((output / "video-index.json").read_text())
            assert len(index) == 2
            assert index[0]["end_total_ticks"] == 20
            assert index[1]["start_total_ticks"] == 20
            assert index[1]["end_total_ticks"] == 40
            for segment in index:
                assert abs(segment["end_time"] - .04) < 1e-10
                assert segment["frames"] == 3  # initial, t~=1/30, explicit final
                reader = imageio.get_reader(output / segment["file"])
                try:
                    assert reader.get_meta_data()["fps"] == 30
                    assert reader.count_frames() == segment["frames"]
                    assert reader.get_data(0).shape == (64, 96, 3)
                    assert reader.get_data(segment["frames"] - 1).size > 0
                finally:
                    reader.close()
    finally:
        sim.close()
