"""SAM 3D relay and spool service with a fake model (no GPU, container or weights)."""
import base64
import importlib.util
import io
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image
import pytest

from sim2sim import sam3d
from sim2sim.runtime import AGENT_TOOLS, RuntimeFailure, enabled_tools
from sim2sim.sam3d import Relay, ServiceFailure, serve, validate_request

spec = importlib.util.spec_from_file_location("sandbox_agent_sam3d", Path(__file__).resolve().parents[1] / "examples/sandbox_agent.py")
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


def png(array):
    data = io.BytesIO()
    Image.fromarray(np.asarray(array, np.uint8)).save(data, "PNG")
    return data.getvalue()


RGB, MASK = png(np.full((24, 32, 3), 90)), png(np.pad(np.full((8, 8), 255), ((8, 8), (12, 12))))
B64 = lambda data: base64.b64encode(data).decode()
REQUEST = {"op": "sam3d", "image_png_b64": B64(RGB), "mask_png_b64": B64(MASK), "seed": 3}


def tetrahedron(rgb, mask, seed):
    assert rgb.shape == (24, 32, 3) and mask.dtype == bool and mask.sum() == 64
    return [[0, 0, -1], [1, 0, -1], [0, 1, -1], [0, 0, -2]], [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], [[200, 10, 10]] * 4, \
        {"rotation_wxyz": [1, 0, 0, 0], "translation": [0, 0, 1], "scale": [seed] * 3}


@pytest.fixture
def service(tmp_path):
    spool = tmp_path / "spool"
    spool.mkdir(mode=0o700)
    stop, models = threading.Event(), {"model": tetrahedron}
    thread = threading.Thread(target=serve, args=(spool, lambda *args: models["model"](*args), {"name": "fake@pinned"}, stop, .01))
    thread.start()
    while not (spool / "service.json").exists():
        time.sleep(.01)
    yield spool, models
    stop.set()
    thread.join()
    assert not (spool / "service.json").exists()


def test_round_trip_returns_bounded_mesh_and_private_record(service):
    spool, models = service
    relay = Relay(spool)
    result, record = relay.call(REQUEST, time.monotonic() + 30, lambda: None)
    mesh = base64.b64decode(result["mesh_ply_b64"])
    header, body = mesh.split(b"end_header\n")
    assert b"element vertex 4\n" in header and b"element face 4\n" in header and len(body) == 4 * 15 + 4 * 13
    assert np.frombuffer(body[:60], [("p", "<f4", 3), ("c", "u1", 3)])["p"][3].tolist() == [0, 0, -2]
    assert result["layout"]["scale"] == [3] * 3 and result["model"] == "fake@pinned" and "x right, y up, -z forward" in result["frame"]
    assert set(result) == {"mesh_ply_b64", "vertices", "faces", "layout", "seed", "frame", "model"}
    assert record["ok"] and record["service"]["model"] == {"name": "fake@pinned"} and len(record["mesh_sha256"]) == 64
    models["model"] = lambda *args: (_ for _ in ()).throw(AssertionError("degenerate mask"))
    result, record = relay.call(REQUEST, time.monotonic() + 30, lambda: None)
    assert result is None and record["error"] == "SAM 3D failed: degenerate mask"
    assert not [path for path in spool.rglob("*") if path.is_file() and path.name != "service.json"]


def test_invalid_agent_inputs_never_reach_the_service():
    validate_request(REQUEST)
    rgba = png(np.zeros((24, 32, 4)))
    validate_request({**REQUEST, "image_png_b64": B64(rgba), "seed": 0})
    for bad in ({"world": "reference"}, {"image_png_b64": B64(b"GIF89a" + b"\0" * 40)}, {"mask_png_b64": B64(RGB)},
                {"mask_png_b64": B64(png(np.zeros((24, 31))))}, {"image_png_b64": B64(MASK)}, {"image_png_b64": "@@"},
                {"seed": True}, {"seed": -1}, {"image_png_b64": B64(png(np.zeros((2049, 1, 3))))}):
        with pytest.raises(RuntimeFailure):
            validate_request({**REQUEST, **bad})


def test_missing_stale_shared_or_silent_service_fails_loudly(tmp_path, monkeypatch):
    spool = tmp_path / "spool"
    spool.mkdir(mode=0o700)
    with pytest.raises(ServiceFailure, match="not running"):
        Relay(spool)
    (spool / "service.json").write_text(json.dumps({"heartbeat_unix": time.time() - 300}))
    with pytest.raises(ServiceFailure, match="stale"):
        Relay(spool)
    spool.chmod(0o750)
    with pytest.raises(ServiceFailure, match="private"):
        Relay(spool)
    spool.chmod(0o700)
    (spool / "requests").mkdir(), (spool / "responses").mkdir()
    (spool / "service.json").write_text(json.dumps({"heartbeat_unix": time.time(), "host": "h", "job_id": "1", "model": {"name": "m"}}))
    relay = Relay(spool, timeout=.5)
    with pytest.raises(ServiceFailure, match="did not answer"):
        relay.call(REQUEST, time.monotonic() + 30, lambda: None)
    assert not list((spool / "requests").iterdir())  # the unclaimed request was withdrawn
    monkeypatch.setattr(sam3d, "HEARTBEAT", .1)
    (spool / "service.json").write_text(json.dumps({"heartbeat_unix": time.time() - 121}))
    relay.timeout = 30
    with pytest.raises(ServiceFailure, match="stale"):
        relay.call(REQUEST, time.monotonic() + 30, lambda: None)


def test_tools_config_and_task_text():
    assert enabled_tools({"tools": {"sam3d": True}}) == ("sam3d",) and AGENT_TOOLS["sam3d"][1] is None
    assert "controller action sam3d" in AGENT_TOOLS["sam3d"][0] and "/opt/public" not in AGENT_TOOLS["sam3d"][0]
    for words in ("dollar", "cost", "budget", "GPU", "spool"):
        assert words not in AGENT_TOOLS["sam3d"][0]


def test_harness_exposes_sam3d_only_when_offered_and_saves_mesh(tmp_path):
    sent = []

    class Client:
        task = {"actions": ["observe", "sam3d", "finish"], "action_schema": {"observe": {}, "sam3d": {"image_png_b64": "PNG"}, "finish": "end"}}

        def sam3d(self, image_png, mask_png, seed):
            sent.append((image_png, mask_png, seed))
            return {"mesh_ply_b64": B64(b"ply\nmesh"), "vertices": 3, "faces": 1, "layout": {}, "seed": seed, "frame": "f", "model": "m"}
    assert "sam3d" in {tool["name"] for tool in agent.tool_schemas(Client.task)}
    assert "sam3d" not in {tool["name"] for tool in agent.tool_schemas({**Client.task, "actions": ["observe", "finish"]})}
    (tmp_path / "image.png").write_bytes(RGB)
    (tmp_path / "mask.png").write_bytes(MASK)
    harness = agent.Agent.__new__(agent.Agent)
    harness.client, harness.workspace = Client(), tmp_path
    reply = json.loads(harness.sam3d({"image": str(tmp_path / "image.png"), "mask": str(tmp_path / "mask.png")}))
    assert sent == [(RGB, MASK, 0)] and Path(reply["mesh_ply"]).read_bytes() == b"ply\nmesh" and "mesh_ply_b64" not in reply
    assert json.loads(harness.sam3d({"image": str(tmp_path / "image.png"), "mask": str(tmp_path / "mask.png"), "seed": 5}))["mesh_ply"].endswith("002-mesh.ply")
