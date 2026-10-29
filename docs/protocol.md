# Public agent protocol (version 1)

The host and agent exchange one UTF-8 JSON object per line through the agent's
stdin/stdout. Stdout is reserved for protocol messages; diagnostic text belongs
on stderr. Requests are synchronous: send one request, read its response, then
send the next request. There are no request IDs.

The host sends the first message without a request:

```json
{"type":"task","protocol":1,"task":"Reconstruct the scene","actions":["observe","grasp","move","release","reset","submit","inference","finish"],"action_schema":{},"observation_contract":{},"budgets":{"active_seconds":300,"physics_ticks":3000,"submissions":3},"submission":{"format":"base64 ZIP","entrypoint":"scene.xml","max_bytes":4194304},"observation":{}}
```

The actual `action_schema`, `observation_contract` and `observation` are populated;
the abbreviated objects above show the envelope. The task describes enabled
modalities, action semantics and submission limits. It never contains source XML,
private body identities, hidden probes or scores.

## Requests and responses

Every request has an `op` string:

| op | Fields | Behavior |
| --- | --- | --- |
| `observe` | optional `ticks` (default 0) | Advance physics, then render. |
| `grasp` | `pixel: [u, v]` | Attach to the first visible movable surface at the pixel. A miss leaves the grasp inactive. |
| `move` | `displacement: [dx, dy, dz]`, optional `ticks` | Move the grasp target by world meters over physics ticks; requires an active grasp. |
| `release` | optional `ticks` (default 0) | Disable the grasp, then advance physics. |
| `reset` | none | Restore initial physical state and release the grasp. |
| `submit` | `project_b64` | Submit a base64 ZIP containing `scene.xml` and local assets. |
| `inference` | `messages` | Request inference from the configured host broker. |
| `finish` | none | End the run. |

Pixel origin is the image's top left, with coordinates inside the image. An
occluded object cannot be grasped through the visible surface. Only one grasp
exists at a time; release it before grasping again. The point constraint supplies
force without an orientation target, so off-center pulling can rotate a body.
Ray hits and body identity stay private.

Ticks must be nonnegative integers for `observe` and `release`, and positive for
`move`. Move defaults to the configured duration (runtime default 20). The core
caps a single positive duration at 100000 ticks; cumulative runtime budgets may
be tighter. Grasp and reset accept no positive ticks. Observe, grasp, release
and reset consume zero integration ticks unless explicitly advanced. Reset
never refunds cumulative ticks or active time. Physics advances only on action
ticks. Hidden evaluation pauses the agent and its active-time budget.

Successful simulation actions respond:

```json
{"ok":true,"observation":{"rgb":"BASE64_PNG","width":320,"height":240,"time":0.0,"ticks":0,"total_ticks":0,"grasped":false,"rgb_format":"png_rgb8","calibration":{}},"remaining_ticks":3000}
```

A submission responds with `{"ok":true,"submission":1,"status":"ok"}` or
another evaluator status such as `failed`. `ok: true` means the request was
handled; it does not imply reconstruction success. Scores, probe descriptions
and evaluator artifacts remain private. Rejected operations return
`{"ok":false,"error":"..."}`. A terminal budget/transport failure may close
the channel; agents must handle EOF. `finish` responds with
`{"ok":true,"status":"finished"}` before termination.

`inference.messages` is a nonempty array of objects with role `system`, `user`
or `assistant` and string content, or content parts of type `text` and
`image_url`. Images must be inline `data:image/png;base64,...` or
`data:image/jpeg;base64,...` URLs. Remote URLs are rejected. The host fixes the
endpoint, model, credential and output-token bound; the agent cannot choose
them. The response contains `{"ok":true,"inference":...}` with generated
content and provider token usage when available. Inference is disabled without
host configuration. Broker time counts toward active time; reported token usage
is a threshold budget, so a completed call can cross that threshold.

## Observation formats

- `rgb`: base64 PNG, RGB8, without a data URL prefix; enabled by default.
- `depth`: base64 PNG, unsigned 16 bit millimeters, saturated at 65535; disabled
  by default. Some PNG decoders expose the samples as int32. Depth is axial
  camera distance, not ray length.
- `rgb_format`: `png_rgb8`, and `depth_format`: `png_uint16_mm`, present with the
  corresponding image.
- `calibration`: enabled by default. Contains 3x3 `intrinsics`, 4x4
  `world_from_camera`, `axes`, `width` and `height`. The pose uses OpenGL camera
  axes: x right, y up, negative z forward; image v increases downward. To project
  camera coordinates, apply the intrinsics to `[x/-z, -y/-z, 1]`.
- `width`, `height`: image dimensions; `time` and `ticks`: since reset;
  `total_ticks`: across resets; `grasped`: whether a point constraint is active.

Image modalities and calibration are selected independently. Observation cameras
must be fixed directly in `worldbody`, with perspective `fovy` calibration.
Attached/tracking cameras, orthographic projection and custom sensor/focal
intrinsics are rejected. Default image dimensions are 640 by 480.

## Client example

The image recipe installs the client package. Agent files can use:

```python
from pathlib import Path
from sim2sim.client import Client

client = Client()  # consumes the initial task
observation = client.observe(ticks=10)
# Infer a scene from permitted observations and write scene.xml/local assets.
project = Path('/workspace/project')
# ... create project ...
result = client.submit(project)  # packages local files as a ZIP
client.finish()
```

The client also exposes `grasp(pixel)`, `move(displacement, ticks=20)`,
`release(ticks=0)`, `reset()` and `inference(messages)`. It raises on an
unsuccessful response. See [the scripted agent](../examples/scripted_agent.py).

## Configuration

The runnable [example configuration](../configs/example.toml) covers source,
sandbox image/command/resource limits, runtime budgets, evaluator and inference.
The sandbox process bound defaults to `process_limit_backend = "cgroup"`,
requiring enforced CPU/memory/pids controllers. Explicit `"rlimit"` supports
hosts enforcing CPU/memory but lacking pids delegation. CPU/memory enforcement
remains mandatory. This cluster currently accepts CPU/memory flags without
enforcing limits; resource preflight therefore blocks full sandbox runs here. The configured `pids` is then a hard `RLIMIT_NPROC` bound for threads
and processes sharing the fixed nonroot subordinate real UID, including both
agent and evaluator; this may cause shared-slot starvation.

```toml
[sandbox]
process_limit_backend = "rlimit" # explicit opt-in; default is "cgroup"
pids = 64
memory = "1g"
cpus = 1
```

All core simulation knobs are:

```toml
[simulation]
width = 640
height = 480
camera = "fixed"
rgb = true
depth = false
calibration = true
default_ticks = 20      # runtime default; standalone Python core defaults to 60
# timestep = 0.002      # optional override; otherwise scene timestep
# graspable_bodies = ["block_red", "block_blue"] # otherwise source manifest/auto discovery
# grasp_solref = [0.01, 1.0]
# grasp_solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
```

## Internal Python interface

`Simulation(scene_path, config=None)` exposes `action(dict)`, `step(op, args)`,
`observe()`, `calibration()` and `close()`. Internal `wait`/`advance` actions and
the `action` alias for `op` are not accepted by the public runtime protocol.
`observe(full=True)` supplies all modalities to trusted evaluation/video callers,
independently of agent settings. A `recording_callback(sim)` runs after each
physics step. A `grasp_controller_factory(sim)` can replace `PointGrasp`, whose
interface is `attach(body, world_point)`, `move(delta)` and `release()`.
Private `trusted_camera` configuration injects a source camera into candidate
models for evaluation; agents cannot set it through the protocol.
