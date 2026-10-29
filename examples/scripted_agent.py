"""Protocol smoke agent. Its guessed scene is deliberately simple, not a solver."""
from pathlib import Path
import sys
from sim2sim.client import Client

client = Client()
observation = client.observe()
if "--integration" in sys.argv:
    import base64
    import io
    import os
    import socket
    import numpy as np
    from PIL import Image
    assert os.getuid() == 65532
    assert "SIM2SIM_TEST_SECRET" not in os.environ
    assert not Path("/opt/sim2sim/environments").exists()
    cgroup = Path("/sys/fs/cgroup")
    assert 0 < int((cgroup / "memory.max").read_text()) <= 1024 ** 3
    import errno
    import resource
    assert resource.getrlimit(resource.RLIMIT_NPROC) == (64, 64)
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (65, 65))
        raise AssertionError("hard process limit can be raised")
    except (ValueError, OSError):
        pass
    try:
        os.setuid(0)
        raise AssertionError("sandbox UID can become root")
    except PermissionError:
        pass
    children = []
    try:
        for _ in range(70):
            try:
                pid = os.fork()
            except OSError as exc:
                assert exc.errno == errno.EAGAIN
                break
            if pid == 0:
                import time
                time.sleep(10)
                os._exit(0)
            children.append(pid)
        else:
            raise AssertionError("process limit did not stop fork attempts")
        assert len(children) < 64
    finally:
        import signal
        for pid in children:
            os.kill(pid, signal.SIGKILL)
        for pid in children:
            os.waitpid(pid, 0)
    import threading
    stop_threads = threading.Event()
    threads = []
    try:
        for _ in range(70):
            thread = threading.Thread(target=stop_threads.wait)
            try:
                thread.start()
            except RuntimeError:
                break
            threads.append(thread)
        else:
            raise AssertionError("thread limit did not stop clone attempts")
        assert len(threads) < 64
    finally:
        stop_threads.set()
        for thread in threads:
            thread.join()
    quota, period = (cgroup / "cpu.max").read_text().split()
    assert 0 < int(quota) / int(period) <= 1
    status = Path("/proc/self/status").read_text()
    assert "CapEff:\t0000000000000000" in status
    try:
        Path("/opt/forbidden").write_text("forbidden")
        raise AssertionError("image root is writable")
    except OSError:
        pass
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=.5).close()
        raise AssertionError("network egress is reachable")
    except OSError:
        pass
    image = np.asarray(Image.open(io.BytesIO(base64.b64decode(observation["rgb"]))))
    red = (image[:, :, 0].astype(float) > 1.7 * image[:, :, 1]) & (image[:, :, 0].astype(float) > 1.7 * image[:, :, 2]) & (image[:, :, 0] > 90)
    y, x = np.nonzero(red)
    assert len(x), "red object was not visible"
    grasp = client.grasp([float(np.median(x)), float(np.median(y))])
    assert grasp["grasped"]
    client.move([0, 0, .1], ticks=10)
    client.release(ticks=10)
    client.observe(ticks=5)
    client.reset()
    client.observe(ticks=5)
# Scene guesses must come from observations; no source paths or model bytes are supplied.
project = Path("/workspace/project")
project.mkdir()
(project / "scene.xml").write_text('''<mujoco model="guess">
<option timestep="0.002"/>
<worldbody>
  <light pos="0 0 3"/>
  <camera name="fixed" pos="1.2 -1.5 1.2" xyaxes="0.78 0.62 0 -0.35 0.44 0.83"/>
  <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
  <body name="other" pos="0.15 0.1 0.1"><freejoint/><geom type="box" size="0.07 0.05 0.05" mass="0.3" rgba="0.1 0.3 0.8 1"/></body>
  <body name="object" pos="0 0 0.12"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="0.5" rgba="0.8 0.2 0.2 1"/></body>
</worldbody>
</mujoco>''')
client.submit(project)
if "--integration" in sys.argv:
    scene = project / "scene.xml"
    scene.write_text(scene.read_text().replace('mass="0.5"', 'mass="0.6"'))
    client.submit(project)
client.finish()
