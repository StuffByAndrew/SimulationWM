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


def test_recorded_frame_video_cadence_is_independent_of_action_count(tmp_path):
    import base64
    import io
    from PIL import Image
    from sim2sim.runtime import FrameVideoRecorder
    png = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(png, format="PNG")
    rgb = base64.b64encode(png.getvalue()).decode()
    recorder = FrameVideoRecorder(tmp_path, 30)
    for tick in range(21):
        # Every tiny action has a full audit image, but video stays at30fps.
        recorder.push({"rgb": rgb, "time": tick*.002}, tick, 0, final=True)
    recorder.close()
    index = json.loads((tmp_path / "video-index.json").read_text())
    assert index[0]["frames"] == 3
    reader = imageio.get_reader(tmp_path / index[0]["file"])
    try:
        assert reader.count_frames() == 3
        assert reader.get_meta_data()["fps"] == 30
    finally:
        reader.close()
