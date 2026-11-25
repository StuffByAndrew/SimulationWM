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
| `move` | `displacement: [dx, dy, dz]`, optional `ticks`, optional `rotation: [rx, ry, rz]` (only with `simulation.grasp = "weld"`) | Move the grasp target by world meters over physics ticks; requires an active grasp. A weld's `rotation` also turns the held object about the grasp point (below). |
| `release` | optional `ticks` (default 0) | Disable the grasp, then advance physics. |
| `reset` | none | Restore initial physical state and release the grasp. |
| `camera` | `pos: [x, y, z]`, `lookat: [x, y, z]` (only with `simulation.movable_camera = true`) | Re-aim the observation camera (up = +z) without advancing time. Both points within ±4 m, at least 0.05 m apart, and `pos` at least the near-plane frustum reach (`1.5 · znear · extent · corner factor`) from every surface, so the near clip plane can never cut a wall; a camera inside a geom or a small closed cavity is rejected too. |
| `frames` | optional `since`, `limit` (only with `simulation.realtime`) | Kept camera frames newer than `since`; see [real-time world](#real-time-world-simulationrealtime). |
| `throw` | `protocol` (only with a real-time `[launcher]`) | Reset, then launch the body with a public protocol's perturbed release state. |
| `submit` | `project_b64` | Submit a base64 ZIP containing `scene.xml` and local assets. |
| `inference` | `messages`, optional `tools` | Request inference from the configured host broker. |
| `finish` | none | End the run. |

Pixel origin is the image's top left, with coordinates inside the image. An
occluded object cannot be grasped through the visible surface. Only one grasp
exists at a time (unless `simulation.point_grasps = 2`, below); release it before grasping again. The point constraint supplies
force without an orientation target, so off-center pulling can rotate a body.
A cloth (any MuJoCo flex, from `flexcomp` or `<deformable><flex>`) is grasped at
the vertex of the hit flex nearest the ray hit, held at the hit point's offset;
the wrench then reports that vertex's pull. Ray hits and body identity stay private.

**Two point grasps** (`[simulation] point_grasps = 2`; default 1, the single grasp above, whose configs, tasks, observations
and snapshots are unchanged): two simultaneous, independent point grasps, for example one holding a strand still while the
other pulls. `grasp`, `move` and `release` take `grasp_index` 0 or 1 (default 0; the harness tools require it); each grasp
has its own connect constraint and mocap handle for every graspable body (grasp 1's are suffixed `_1`), so both may hold the
same body. A `move` moves only its grasp; the other handle stays where it is. `reset` releases both. The observation's
`grasped` becomes `[grasp 0, grasp 1]` and `wrench` a list of one reading per grasp, each only that grasp's own connect
force under the wrench rules below. The task adds `grasp_index` to those actions' schema and an
`observation_contract.point_grasps` note, and `public_tool_config.point_grasps = 2` loads the reconstruction world with two
grasps too. Ticks are charged as for one grasp (a move of either grasp costs its ticks); nothing is refunded. Point-grasp
embodiment in a lock-step world only (a robot or `realtime` config with 2 is rejected). Researcher snapshots add `grasps`
and `grasp_overlays` (both grasps; the dashboard marks both).

**Weld grasp** (`[simulation] grasp = "weld"`; default `"connect"`, the translation-only point grasp above, whose configs,
tasks, observations and snapshots are unchanged; `Simulation` rejects any other value). The grasp rigidly fixes the hit
body to its handle at the grasp point: a MuJoCo weld (body1 = object, body2 = a mocap handle; solref/solimp from
`grasp_solref`/`grasp_solimp`) holding the pinned body point at the handle and the body's orientation at grasp time.
`move` takes an optional `rotation: [rx, ry, rz]`, a world-frame rotation vector (unit axis times angle, radians): the
handle turns about the grasp point by `rotation / ticks` each tick (left-multiplied in world axes, so the total is exactly
the commanded rotation) while it translates by `displacement / ticks`, so one move can be a pure turn (`displacement`
`[0, 0, 0]`), a translation, or a screw motion. Ticks are charged exactly as for any `move` (nothing new is charged or
refunded). With the connect grasp `rotation` is not offered: the host forwards it only when the weld is configured, and
`Simulation` rejects it otherwise. A weld cannot hold a cloth (flex) vertex and needs the point-grasp embodiment, and
composes with `point_grasps = 2` (each grasp its own weld; two welds on one body form a closed loop that the soft
constraints share). The task changes only when the weld is configured: `move` gains `rotation`, `grasp`'s and `move`'s
texts say weld, `observation_contract.grasp` describes it and `observation_contract.wrench` names the couple;
`public_tool_config.grasp = "weld"` loads the reconstruction world (and `local_controller`) with welds too, and the
evaluator's source and candidate worlds use it. Harness tools offer `rotation` on `move` exactly when the contract lists
it, and a `sequence` may include it. `Client.move(displacement, ticks, rotation=[...])` sends it. A probe in a weld level's
private manifest may declare `rotation` (radians, world) beside `displacement`; it is split over the probe's four moves.
`grasp_slip_rad` (below) adds an angular slip.

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

`inference.messages` is a nonempty array of `system`, `user`, `assistant` or
`tool` messages with string content or content parts of type `text` and
`image_url` (images only in `user`/`tool` messages, as inline
`data:image/png;base64,...` or `data:image/jpeg;base64,...` URLs; remote URLs
are rejected). An `assistant` message may carry `tool_calls:
[{id,name,arguments}]`; a `tool` message carries the `tool_call_id` of an
earlier call. Optional `tools` is at most 64 `{name, description, parameters}`
function schemas. No other request fields are accepted. The host fixes the
endpoint, provider route, model, metadata, credential and output-token bound.
The response is `{"ok":true,"inference":{"content","tool_calls","finish_reason",
"model","usage","cost_usd","summaries",...}}` plus cumulative and remaining
call/token/cost counters. Inference is disabled without host configuration.

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

Image modalities and calibration are selected independently. The global observation
camera must be fixed directly in `worldbody`, with perspective `fovy` calibration.
The optional Panda wrist camera is fixed directly to its hand as described below.
Other attached cameras, tracking cameras, orthographic projection and custom
sensor/focal intrinsics are rejected. Default image dimensions are 640 by 480.

## Client example

The agent image installs the client package. Agent files can use:

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
`release(ticks=0)`, `reset()` and `inference(messages, tools=None)`. It raises on an
unsuccessful response. See [the scripted agent](../examples/scripted_agent.py).

## Configuration

The runnable [example configuration](../configs/example.toml) covers source,
sandbox image/command/resource limits, runtime budgets, evaluator and inference.
The sandbox process bound defaults to `process_limit_backend = "cgroup"`,
requiring enforced CPU/memory/pids controllers. Explicit `"rlimit"` supports
hosts enforcing CPU/memory but lacking pids delegation. CPU/memory enforcement
remains mandatory. Podman ignores CPU/memory flags on this cluster; inside a
Slurm allocation use `"slurm_step"`: each container is its own `srun --exact`
step (integer `cpus`, whole-MiB `memory`), verified from the host before input.
For `rlimit` and `slurm_step`, `pids` is a hard `RLIMIT_NPROC` set by `prlimit`
for threads and processes sharing the fixed nonroot subordinate real UID across
all containers on the node; this may cause shared-slot starvation. Optional
`[sandbox.worker]` keys bound the evaluator and candidate worker separately.
Robot runs (Panda or Shadow hand) see the public robot fixture read-only at `/opt/public/robot`.

Optional agent tools (`[tools] segmentation = true`, `planning = true`; both
default false; strict OCI only) are real ablations. An enabled tool is staged
from `tools/<name>/` as a fresh, hashed (`public_fixture` audit event),
read-only volume at `/opt/public/tools/<name>`, and one fixed sentence naming
its path and `--help` is appended to the task. A disabled tool is neither mounted
nor named, and no image contains its code or weights. `segmentation` (MobileSAM,
pinned weights sha256 `6dbb9052…6c2f`) needs the tools image (the agent image plus CPU PyTorch), whose label the
runtime checks before launch; `planning` (IK, interpolation, collision-checked
straight-line/RRT-connect on the agent's own scene) needs `robot = "panda"`. `local_controller` (a hand robot: `robot = "shadow_hand"` or `"panda_shadow"`)
mounts `tools/local_controller/localsim.py` plus a runtime-written `public_tool_config.json`, the exact public settings the
reconstruction world receives: `LocalController` runs the agent's own project with `Simulation`, the worker's
`allowed_actions` and `public_observation`, so a fresh load and the same actions give identical observations (tested
against the candidate worker and, in OCI, the strict reconstruction world); `taxels(joint_positions)` gives each tactile
cell's body-frame bounds, the pad body's world pose, and the cell centre's palmar surface point and normal from the
public fixture alone. It extracts no contact points and fits no shapes. `sequence` (no files) adds a `sequence` action that the
sandbox harness executes itself: the agent's code writes a JSON array of at most 2000 physical action objects
(`observe`, `grasp`, `move`, `release`, `arm_target`, `hand_target`, `camera`, `reset`, and on fixture levels `reset_carriage` and `release_carriage`, as offered, without `world`), the harness
sends them in order to one world as ordinary controller actions (each charged and audited as usual), stops at the
first rejected step, saves every step's non-image observation fields to `/workspace/sequences/NNNN.json` and returns
the final step's images (or none). The controller never receives a `sequence` request.

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
wrench = false         # optional 6-axis grasp-point reading
# point_grasps = 1     # 2: two simultaneous independent point grasps (grasp_index on grasp/move/release)
# grasp = "connect"    # "weld": the grasp also fixes and (move rotation) turns the object's orientation
tactile = false        # optional robot taxel normal forces (robot = "panda", "shadow_hand" or "panda_shadow" only)
default_ticks = 20      # runtime default; standalone Python core defaults to 60
# timestep = 0.002      # optional override; otherwise scene timestep
# graspable_bodies = ["block_red", "block_blue"] # otherwise source manifest/auto discovery
# grasp_solref = [0.01, 1.0]
# grasp_solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
# grasp_slip_m = 0.004  # optional: the grip slips (releases) when stretched farther; absent = it never slips
# grasp_slip_rad = 0.2  # optional, weld only: the grip also slips when twisted farther; absent = never
# resets = false        # optional no-reset condition; default true
```

`resets` (boolean, default true) offers the `reset` action. With `resets = false` the task's `actions` and
`action_schema` omit `reset`, so the harness builds no reset tool and a `sequence` may not contain one, and the host
rejects any reset request in either world (also through the code channel) with the public error "reset is not available
in this run" before anything runs: no world, tick or episode changes. Nothing else changes: `load_reconstruction`, the
fixture's `reset_carriage`/`release_carriage`, and the launcher's `throw` (which restores the die's start state internally
in real-time levels) keep working. The task text says nothing about it; a condition's task sentences are the config's.

`grasp_solref` (2 numbers) and `grasp_solimp` (5 numbers) set the point grasp's connect-constraint softness (MuJoCo
`solref`/`solimp`); the defaults above apply when they are absent, so existing configs and prepared runs are unchanged.
`grasp_slip_m` (meters, in (0, 1]; absent by default, when the grip never slips) makes the grip slip: after every physics
step, if the pinned body point lies farther than `grasp_slip_m` from the grasp target (the pinned point the agent's moves
translate), the grasp is released exactly as by `release`. The rest of a `move` that slipped still runs its ticks without
a grip. The agent sees what a release shows: `grasped` is false in the next observation, a wrench reading is zero, and a
later `move` is rejected until a new `grasp` (a `sequence` stops at that rejected step). Stretch is measured on the state the step started from, so it
lags by one step. `grasp_slip_rad` (radians, in (0, π]; absent by default; only with `grasp = "weld"`, otherwise rejected)
is the weld's angular threshold, evaluated the same way after every step: if the held body's orientation differs from the
handle's target orientation (the orientation held at grasp time, turned by the moves' rotations) by a rotation angle
larger than `grasp_slip_rad`, the grasp is released exactly as above. Either threshold releases; each grasp of
`point_grasps = 2` is checked independently. All these knobs are public: `public_tool_config` reports the values in force, and the reference world
(lock-step and real-time), the agent's reconstruction world (candidate worker) and the evaluator's source and candidate
worlds all use them, so the grasp behaves the same everywhere. `Simulation` rejects malformed values (lists of finite
numbers of those lengths; a slip that is not a number in range), so they fail the run on the host at start. MuJoCo clamps
a `solref` time constant below two timesteps. The gear-train level (`environments/gear_train`) sets `grasp_slip_m`: a
hard over-pull of the crank would otherwise stretch the soft grasp far enough that MuJoCo's inertia-scaled constraint
softness tells its equality-coupled gear train from the lumped equivalent; within the handle's reach the grip holds.

## Internal Python interface

`Simulation(scene_path, config=None)` exposes `action(dict)`, `step(op, args)`,
`observe()`, `calibration()` and `close()`. Internal `wait`/`advance` actions and
the `action` alias for `op` are not accepted by the public runtime protocol.
`observe(full=True)` supplies all modalities to trusted evaluation/video callers,
independently of agent settings. A `recording_callback(sim)` runs after each
physics step. A `grasp_controller_factory(sim)` can replace `PointGrasp`, whose
interface is `attach(body, world_point, index=0)`, `move(delta, index=0)`, `rotate(rotation, index=0)` (weld only) and
`release(index=0)` (index: the grasp).
Private `trusted_camera` configuration injects a source camera into candidate
models for evaluation; agents cannot set it through the protocol.


## Private checkpoint curves

After each explicit submission, the researcher output directory receives
`metric-history.json` and `metric-history.png`. The PNG has three panels:
Chamfer distance in meters, an absolute density-error curve in kg/m³ for each
original source primitive, and center-of-mass position error in meters. The
horizontal axis identifies submission order and the project version. Density
curves retain source primitive names when candidate primitive names change.
There is no combined scalar score.

Failed submissions, unavailable components and unmatched source primitives
produce gaps, never zeros. Each panel annotates missing checkpoint counts; the
density panel also counts missing primitive/checkpoint pairs. Disabled built-in
scoring produces unavailable panels. The JSON retains complete reconstruction
measurements, including COM error/reference/predicted vectors when available,
matched source and predicted densities, coverage, and unmatched primitive names.
Repeated submissions remain separate checkpoints even for identical versions.

History refresh and plotting run while the agent container remains paused;
their time is excluded from its active budget. Curves, JSON measurements and
scores stay on the researcher side and are never sent in agent RPC responses.


## Optional grasp wrench

Set `[simulation] wrench = true` to include `observation.wrench` in every
observation and action response. With the default `false`, the key is absent;
this setting is independent of RGB, depth and calibration.

```json
{"force":[0.0,0.0,0.0],"torque":[0.0,0.0,0.0],"frame":"world","reference":"grasp_point","sign":"object_on_handle","active":false,"force_unit":"N","torque_unit":"N m"}
```

`force` is the force the object applies to the handle, expressed in world axes
at the grasp point: the constraint force the solver applied during the latest physics step. The ideal point grasp transmits no pure rotational couple,
so its measured `torque` is exactly zero. An off-center applied force can still
produce a moment about the object's center of mass; that COM-relative moment
is not exposed. `active` indicates whether a grasp is attached. An inactive
reading contains zero force and torque. No body identity, anchor coordinates,
mass or center-of-mass position is included in this optional observation.
With `point_grasps = 2`, `wrench` is a list of two such readings (grasp 0, grasp 1), each from its own grasp's connect
rows only; a body held by both grasps shares its load between the two readings as the solver splits it.

With `grasp = "weld"` the reading is the weld's full wrench, from only that grasp's own six weld rows (same keys, frame,
reference and sign): `force` as above, and `torque` the couple the object applies to the handle about the grasp point, in
world axes. MuJoCo builds the three rotation rows as `M (ω_object − ω_handle)` with `M = ½ · torquescale ·
imag(q̄_handle ⊗ (0, ω) ⊗ q_object ⊗ relpose)` (`engine_core_constraint.c`, `mjEQ_WELD`), so a row multiplier `f` is the world
couple `Mᵀ f` on the object; the reading is `−Mᵀ f`, computed from the kinematics the rows were built from. Together, force
at the pinned point and couple reproduce exactly the weld rows' generalized force `Jᵀ efc_force` (tested). A body hanging
still from a weld at point `p` reads force `m g` and torque `(c − p) × m g` for its centre of mass `c`. The other rules hold:
no body identity, grasp or COM coordinates, lever arm or mass, and reading never advances time or ticks.


## Optional Panda tactile sensing

Set `[simulation] tactile = true` (Panda only; a boolean, default `false`) to add
`observation.tactile` to every observation in both worlds. When disabled the key
is absent. A tactile-only condition sets `rgb = depth = calibration = false` with
`tactile = true`: the agent then has proprioception and touch, and no image.

```json
{"left":{"pad_normal_force_n":[[0,0,0],[0,0,0],[0,0,0]],"finger_normal_force_n":0.0},
 "right":{"pad_normal_force_n":[[0,0,0],[0,0,0],[0,0,0]],"finger_normal_force_n":0.0},
 "unit":"N","layout":"pad rows distal to proximal, columns across the finger"}
```

Each finger reports the normal contact force summed into a 3×3 taxel grid over
its fingertip pad region (the fixture's 17 × 17 mm `fingertip_pad_collision_1`
face, binned by contact position in the finger frame) and the total over all of
that finger's collision geoms. Like a real touch sensor it counts any contact on
the finger, including the table or the other finger. Contact positions, the
touching geom or body, friction forces and object state are not exposed. The
reading is solved on the same copied state as `actuator_effort_nm`, so sensing
never changes physics, simulation time or ticks.


## Independent reference and reconstruction worlds

The same `observe`, `grasp`, `move`, `release` and `reset` requests accept
`"world":"reference"` (the default) or `"world":"reconstruction"`. The alias
`"real"` selects the reference. These worlds retain independent states. Reset
affects only the selected world; reset and reload never refund budget.

Load a candidate independently of submitting it for scoring:

```python
client.load_reconstruction("my_project", camera="fixed", entrypoint="scene.xml")
client.observe(ticks=10, world="reference")
client.observe(ticks=10, world="reconstruction")
client.grasp([320, 240], world="reconstruction")
client.move([0, 0, .1], ticks=20, world="reconstruction")
client.release(world="reconstruction")
client.reset(world="reconstruction")
```

On the wire, `load_reconstruction` contains `project_b64`, optional `entrypoint`
(default `scene.xml`) and `camera` (default `fixed`). It returns the ZIP digest
and selected-modality observation. Each load replaces the previous persistent
candidate worker. Native candidate XML parsing happens only in the constrained
worker, with no original scene files, private body names or original physical
parameters supplied. The candidate uses its own authored fixed camera.

Both worlds share one cumulative physics-tick budget. Public startup fields
`seconds_per_tick` and `public_tool_config` define the same timestep and point-
grasp softness in both worlds. Candidate mass, joints, geometry and friction
remain authored by the agent. Resolution and sensor toggles are shared public
settings. Candidate native failures stop that worker; uncertain unconfirmed
steps are explicitly marked and conservatively charged up to the requested
bound. This prevents a timeout, crash, reset or reload from refunding ticks.

## Durable boundary audit and exact recorded replay

`audit/` is the authoritative record. It saves every bounded received request
before dispatch and every attempted response before delivery, including rejected
requests, inference messages/results, load bundles, submissions and `finish`.
Raw wire bytes preserve whitespace and original encoding; parsed fields and
binary/image payloads are stored in content-addressed blobs. A response's
`delivery_complete` marker distinguishes completed writes from attempts.
Malformed JSON is retained before rejection. Oversized or timed-out incomplete
messages retain all bytes read up to the configured input cap, explicitly marked
incomplete; the runtime does not read an unbounded rejected stream.

Each accepted action durably records resolved defaults, world, sequence,
pre/post counters, episode and immutable model digest. Private typed state
checkpoints include generalized state, warm-start state, controls/applied forces,
mocap targets, equality activity/data and grasp anchors. Progress is acknowledged
and recorded for every physics tick. Lossless PNG images are captured at the
configured simulation-time video cadence and action boundaries, rather than at
every physics tick. Both worlds' encoded videos are also retained and hashed.

`[budget] audit_bytes` bounds this archive (default 512 MiB). Space for bounded
incoming messages and complete action outcomes is reserved before reading or
mutating. Exhaustion terminates the run before another action is dispatched;
a pending intent after a crash is never presented as a completed action.
The archive is private researcher data: it deliberately contains original state
and complete prompts, and must not be handed to the agent. Host API credentials
and authorization headers are never included.

```sh
sim2sim replay runs/my-run --output runs/my-replay
```

Replay verifies the content hashes, sequence chain, declared state/image hashes
and recorded video hashes, then exports exact archived frames, states, wire
bytes, videos and a standalone `index.html` inspector for both worlds. It performs
no native model parsing, physics, scoring or external inference calls. This is
exact playback of recorded observations, with integrity/divergence checks on
those saved artifacts; it does not claim a newly simulated trajectory is
bit-for-bit deterministic. The frame inspector is timing-independent; encoded
videos retain their simulation-time cadence. Arbitrary computation or model
simulations an agent performs internally in its container are outside this API
boundary and cannot be claimed to have been recorded.


## Sandbox agent, host inference broker, public progress and hard wall time

`examples/sandbox_agent.py` (image path `/opt/agent/sandbox_agent.py`) is the
strict-mode agent. Its function tools are `bash` (a command in `/workspace`
with a timeout and middle-truncated output; `/workspace` and `/tmp` are
`noexec`, so scripts run as `python3 file.py`), `view_image` for agent-made
PNG/JPEG files, and exactly the controller actions the task lists, with
descriptions taken from the task's `action_schema`. Observation images are
saved under `/workspace/observations/` (rolling retention); RGB images are
also sent to the model, other observation fields (calibration, wrench,
proprioception) as JSON text. Every tool result states the remaining wall
time. The system prompt is identical across conditions: only the controller
task text, constraints, action contract and observations differ.

`--bash-mode pausable` (in `[sandbox] command`; default `timeout`) replaces the timeout with fair, unlimited commands:
`bash` takes `check_seconds` (default 300) instead of `timeout_seconds`, and after each `check_seconds` of running time
a running command's whole process tree is frozen (SIGSTOP) and its new output returned; the agent then calls
`bash_resume` or `bash_kill`, and every other tool is refused meanwhile. Code therefore runs only while the agent waits on
it, never during model calls or other tools. The harness is a child subreaper, so detached (setsid/double-forked)
processes stay in its tree, and everything a command started is killed when it ends. Only the session deadline (less the
reserve) ends a command without the agent's choice.

For multi-hour runs the harness keeps the system prompt and task message,
drops older images beyond `--max-images`, and when the reported prompt tokens
exceed `--compact-tokens` replaces older turns with a model-written summary,
keeping the latest `--keep-turns` turns; the model keeps its own notes file. It
refuses `finish` before any submission, intercepts the first `finish` made
with more than max(15 min, 10% of the wall budget) remaining (a later-turn
`finish` is executed; each interception is an `adapter` progress note), and
finishes by itself when `--reserve-seconds` (default 360) remain, because a run
that reaches the hard deadline without `finish` is recorded as failed.

The broker targets the normalized gateway Responses route
(`https://api.aqinference.com/v1/responses`). That route accepts only
system/user/assistant messages with `input_text`/`input_image` parts, so the
broker forwards earlier tool calls as labelled assistant text and tool results
as labelled user turns; function schemas are sent as
`{"type":"function","function":{...}}`. Requests carry host `metadata`
(`project`, `feature`), `store: false`, the configured `provider` and reasoning
effort. The broker requires the exact returned model, `route.fallbackUsed ==
false`, matching route model/provider and well-formed tool calls naming
offered tools. It returns content, tool calls, finish reason, camelCase usage
(`accountedTokens` = total plus cache reads/writes) and `cost.total`, never
headers or route details. Budgets `max_calls`, `max_total_tokens` and
`max_cost_usd` are thresholds checked before each call; when a token or cost
budget is set and a response omits that accounting, the budget is exhausted
and the response withheld; inference then stops. Unknown `[inference]` settings are rejected.

`cost_pool` shares one USD cap across concurrent runs on any node. Each run's
broker alone writes `<cost_ledger_dir>/<pool>/<run-id>.json` (atomic rename;
cumulative `cost_usd`, `calls`, `tokens`, `inflight_reserve_usd`,
`updated_at`; never prompts or content). Before each call the broker sums the
other runs' cost plus in-flight reserves and its own total. When the pool or
`max_cost_usd` has less than `2 * call_reserve_usd` left, or the call/token
budget is spent, or a response omitted its cost, the controller answers
`{"ok":true,"inference":{"budget_stop":true,"reason":...}}` without calling the
provider. The sandbox agent then submits the current project if it differs from
the last submission, emits an `adapter` note and finishes; private evaluation of
that submission is host-side and costs no inference. A timed-out call is
charged `call_reserve_usd`. Ledgers stay on the host. Monitor spend with
`sim2sim cost --pool sim2sim-500`.

```toml
[inference]
endpoint = "https://api.aqinference.com/v1/responses"
provider = "openai"
model = "gpt-6-astra"
reasoning_effort = "medium"
reasoning_summary = "auto"
metadata = { project = "amish-research", feature = "sim2sim-agent-eval" }
api_key_env = "INFERENCE_API_KEY"
max_calls = 2000
max_cost_usd = 58.0
cost_pool = "sim2sim-500"
cost_pool_usd = 500.0
cost_ledger_dir = "experiments/cost-ledger"
call_reserve_usd = 3.0
max_output_tokens = 16384
max_request_bytes = 12582912
timeout_seconds = 300
```

`api = "gemini"` (default `"responses"`, the gateway above) sends the same
validated transcript, including labelled tool turns, to Google's native
`generateContent`, so both APIs see identical conversations. The endpoint must
be `.../v1beta/models/<model>:generateContent`, and the key travels in
`x-goog-api-key`. No `metadata` or `provider` is allowed. The reasoning effort
becomes `thinkingConfig.thinkingLevel`, and a summary setting becomes
`includeThoughts` (thought parts are provider summaries). Schemas go as
`parametersJsonSchema`. The broker requires `modelVersion == model`.

Usage maps to the gateway's fields:

- `promptTokens` = prompt minus cached;
- `cacheReadTokens` = cached;
- `completionTokens` = candidates plus thoughts;
- `cacheWriteTokens` = 0.

Google returns no cost, so the host prices these counts at the required
`price_usd_per_mtok = { input, cached_input, output }` list prices. Any API
retries HTTP 429/500/503/504 host-side within the call's deadline.

Provider outages are a separate failure class, `ProviderUnavailable`: a call that hits the provider's own timeout, still gets
HTTP 429/5xx after those retries, or fails in transport. With `provider_hold_seconds = N` the controller holds the outage
outside the run:

- The failed attempt's time is credited back to both clocks. A real-time world is rewound to its state at the start of the
  attempt (physics, a running timed action, kept frames and frame grid), as if it had been frozen for the attempt.
- The backoff (15 s, doubling to at most 300 s) runs with the containers paused and the clocks stopped.
- The identical request is then retried. Each hold is audited as `provider_hold` (with a real-time world, also the
  `discarded_ticks` and `discarded_frames` of the rewound attempt, which no reply or audit event ever carried).
- After N held seconds, outages reach the agent as ordinary failures again.

A successful call's latency always counts as run time.

```toml
[inference]
api = "gemini"
endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-4-neon:generateContent"
model = "gemini-4-neon"
reasoning_effort = "medium"
reasoning_summary = "auto"
api_key_env = "GEMINI_API_KEY"
price_usd_per_mtok = { input = 3.0, cached_input = 0.15, output = 15.0 }
```

`api = "local_openai"` sends the same validated transcript (labelled tool turns) to a keyless researcher-hosted OpenAI
chat-completions service, such as `scripts/inkling-service.slurm` (open-weights Inkling 2 Mini under llama.cpp on two
A100s; its header explains the engine choice and precision).

- `endpoint` is the absolute path of the service's private record `service.json`, not a URL. The record must be an
  owner-only file in an owner-only directory. While the server is healthy, the service rewrites it every 5 s with its
  address (`http://<node>:<port>/v1/chat/completions`), the served `model`, a per-job bearer `token`, `heartbeat_unix`,
  and its engine, checkpoint and precision.
- The broker reads the record on every call. Plain HTTP is allowed only to that cluster-internal address, never through a
  proxy; keyed APIs still require HTTPS.
- A missing record, a heartbeat older than 120 s, or a refused connection is a provider outage (`ProviderUnavailable`),
  so `provider_hold_seconds` holds a dead or requeued service until it is back. A record for another model, a non-chat
  address or loose permissions is a hard failure.
- The run refuses to launch without a live record and audits it once as `local_inference_service` (without the token).
- Requests carry `max_tokens` and `chat_template_kwargs.reasoning_effort`, which the model's chat template reads. The
  broker requires the returned `model` to equal `model`.
- Usage maps as for Gemini. `promptTokens` is the prompt minus prefix-cache hits (`prompt_tokens_details.cached_tokens`),
  `cacheReadTokens` is the hits, and `completionTokens` includes reasoning. `reasoningTokens` is the service's
  `completion_tokens_details.reasoning_tokens`, or null (unknown, not zero) when it reports none, as llama-server does.
- Cost is always 0. A timed-out call charges no reserve. Pool and cap lines are accounting-only.
- The model's raw reasoning is not a provider summary and is not recorded. `reasoning_summary`, `api_key_env`,
  `metadata`, `provider` and `price_usd_per_mtok` are refused.

```toml
[inference]
api = "local_openai"
endpoint = "/n/netscratch/qianqian_lab/Lab/aejxie/inkling/service/service.json"
model = "inkling-2-mini"
reasoning_effort = "medium"
```

`Client.progress(text, origin="agent")` emits a concise public note, maximum
4096 characters. `origin` is `model`, `adapter` or `agent`; the host always records
the sender as `source="agent"`. Actual provider summaries are host-recorded as
`source="provider_summary", origin="provider"`. `progress.jsonl` and the private
audit retain text, UTC time, wall/active elapsed time, model, world and action ID.
Model commentary and adapter lifecycle notes remain distinguishable. These are
public status explanations, not access to private chain-of-thought.

`[budget] wall_seconds = 1800` sets a separate hard 30-minute child-work deadline
starting at runtime entry. It includes preflight, initialization, model calls and
private evaluation/plotting; pausing containers never extends it. The existing
active-time budget still excludes private scoring. A watchdog kills registered
agent/candidate/evaluator processes and cancels the host-only inference process
at the wall deadline. Source stepping and every boundary operation also check
the deadline. Private log/video finalization may finish after child work stops;
termination metadata distinguishes execution elapsed time and finalization.
Neither model calls nor agent launch occur after a failed resource/credential
preflight. A configured provider requires a host credential before agent launch;
only its presence is checked, and values/authorization headers are never logged.

One reviewed recipe (`Containerfile`, targets `agent` and `evaluator`) builds two images, recorded together in one
trusted record (`experiments/images/trusted-image-isolated.json`). The **agent image** (`localhost/sim2sim-agent:latest`,
`sandbox.image`: the agent container and stream-mode predictors) holds the sandbox agent and only the agent-side modules
`sim2sim.client` (client and the `channel` tool), `simulation`, `robots` and `candidate_worker` (the `local_controller`
tool's local world), `predictor` and `submission` (the stream-mode predictor worker). The **evaluator image**
(`localhost/sim2sim:latest`, `sandbox.evaluator_image`: candidate worker, evaluator worker, call-mode predictors, the
limit probe) holds the whole package, including every evaluation and metric module. Images carry the label
`sim2sim.image-role` (`agent` or `evaluator`); a strict run refuses any other pairing of two images before launch (configs prepared before the split name
`localhost/sim2sim:latest` for both and run as declared), and the build
inspection greps the agent image for every absent module's path, dotted name and docstring. The scripted protocol
demonstration is not in either image; the OCI integration test passes it as an inline `python -c` command. Original environments, manifests, run stores,
repository history, credentials and researcher README are excluded. The agent
checks representative forbidden directories and credential environment names
before use; these checks supplement the rootless/no-network/no-host-directory-
mount boundary and cannot replace its mandatory effective resource probe.

## Explicit prompt-scoped development bridge

The separately selected `sim2sim.devrunner.run_development(...)` backend is for
approved development experiments on Slurm's `test` partition. It is never a
fallback from production OCI failure. Production `run(...)` still requires the
same effective rootless resource probe and fails closed.

Development publishes only `public/task.json`, the copied standard-library
`public/rpc.py` helper, public request/response records, observation PNGs and an
agent-authored `public/project/` folder. The external spawned agent is instructed
to use only this curated directory. Original scenes, experiment parameters,
private evaluation results and full audit/state artifacts remain outside it.
The public task includes its hard `deadline_unix`, remaining wall budget,
requested model identity and an explicit indication that external model token
usage is unavailable.

```sh
python PUBLIC/rpc.py --task
python PUBLIC/rpc.py '{"op":"progress","origin":"model","text":"I will compare my current shape hypothesis with the observation."}'
python PUBLIC/rpc.py '{"op":"observe","world":"reference","ticks":20}'
python PUBLIC/rpc.py '{"op":"load_reconstruction","camera":"fixed"}' --project project
python PUBLIC/rpc.py '{"op":"observe","world":"reconstruction","ticks":20}'
python PUBLIC/rpc.py '{"op":"submit"}' --project project
python PUBLIC/rpc.py '{"op":"finish"}'
```

The copied client prints compact public observations with PNG paths suitable for
image viewing. A timed-out client can recover the same outstanding request using
`--wait SEQUENCE`; it must not issue a duplicate action. Raw wires are audited
before dispatch/delivery by the same controller loop as production.

Candidate parsing and private evaluation run in separate native worker process
groups, with address-space/data, CPU-time, file-size and open-file limits, no
inherited provider/cloud credentials, and the same ZIP/include/plugin validation.
The candidate receives only its own project and public tool settings. The hard
wall watchdog stops worker groups and closes the mailbox; `stopped.json` marks
that no more actions will be accepted. The parent orchestrator must separately
interrupt the externally spawned model at that deadline.

**This mode does not enforce filesystem or network isolation.** The spawned
agent and native workers share the account's filesystem permissions; keeping
private information out of their prompts and curated tool usage is a development
agreement, not an OS security boundary. The external model cannot be OS-paused
for scoring. The hard wall cap includes that time and remains authoritative.
No host inference API is used. Only explicit public RPC actions, model-emitted
progress notes and their observations are captured here; arbitrary external
agent tool transcripts, hidden reasoning and token accounting are not claimed
to be recorded by the simulation audit.


## Wall-only runs and optional researcher plots

`budget.wall_only = true` disables active seconds, physics-tick quotas and
submission-count quotas. It requires a finite hard `wall_seconds` deadline.
Public `budgets.active_seconds`, `budgets.physics_ticks`, `budgets.submissions`
and response/termination `remaining_ticks` are `null` in this mode. Cumulative
execution counters, reset semantics and partial-action evidence are unchanged.
Message, archive and native-worker resource bounds still apply. Development
runs support a hard cap up to 21600 seconds; strict OCI retains its existing
resource checks. `plot_checkpoints = false` saves the complete checkpoint JSON
without generating PNG curves after every submission. The dashboard renders
selected grouped curves from archived evaluations. `record_video = false`
disables the extra MP4 encoding, while exact PNG recording and private state
capture continue at `video_fps` and action boundaries.

## Opt-in Panda arm protocol

With `[simulation] robot = "panda"`, the task offers `arm_target` in place of
`grasp`, `move` and `release`. Send seven joint position targets in radians,
one gripper opening width from 0 to 0.08 meters, and a positive tick duration:

```json
{"op":"arm_target","joint_targets":[0,0,0,-1.57079,0,1.57079,-0.7853],"gripper_width_m":0.08,"ticks":100}
```

The host interpolates actuator targets over the requested ticks, enforces
joint/control bounds and charges every executed tick across both worlds.
`reset` restores the Panda home pose with its gripper open and does not refund
cumulative ticks. Candidate scenes use the same arm action. The client exposes
`Client.arm_target(joint_targets, gripper_width_m, ticks, world=...)`.

`observation.proprioception` contains `joint_positions_rad[7]`,
`joint_velocities_rad_s[7]`, `actuator_effort_nm[7]`,
`finger_positions_m[2]`, `finger_velocities_m_s[2]`, `gripper_width_m` and
`gripper_opening` (fraction of the 0.08 m span). These values describe only the
named Panda joints and actuators; object poses, contacts, model arrays and ray
hits are not exposed. The proprioception-only presets omit RGB, depth,
calibration, image dimensions, point-grasp state and wrench from public
observations. Time and tick counters remain protocol metadata.

In the explicit development workflow, the task supplies `public_robot_path`.
Copy that public `robot/` directory unchanged into the candidate project,
include `robot/panda.xml` once in root `scene.xml`, and set its compiler mesh
directory to `robot/assets`. The evaluator verifies the fixture bytes before
scoring. Source object XML and private manifests are not copied to that path.

## Opt-in Shadow hand protocol

With `[simulation] robot = "shadow_hand"` the scene includes the public fixture `robot/shadow_hand.xml` (MuJoCo
Menagerie Shadow Hand E3M5, right hand, unchanged except for its compiler line; see `robot/IMPORT.md`). The forearm is
welded at the world origin with the palm facing +z and the fingers along +x. The task offers `hand_target` in place of
`grasp`, `move` and `release` (and never `arm_target`). Send the 20 position-actuator targets in fixture order
`rh_A_WRJ2, rh_A_WRJ1, rh_A_THJ5 … rh_A_THJ1, rh_A_FFJ4, rh_A_FFJ3, rh_A_FFJ0, rh_A_MFJ4 …, rh_A_RFJ4 …, rh_A_LFJ5,
rh_A_LFJ4, rh_A_LFJ3, rh_A_LFJ0` (each within its `ctrlrange`; a coupled `*J0` target is the sum of that finger's J1
and J2 angles) and a positive tick duration:

```json
{"op":"hand_target","actuator_targets":[0,0,0.3,0.4,0,0,0,0,0.25,0.3,0,0.25,0.3,0,0.25,0.3,0,0,0.25,0.3],"ticks":100}
```

The host interpolates the targets linearly from the current ones over the ticks, rejects out-of-range targets and
charges every executed tick across both worlds. `reset` restores the home pose (all joints 0 except J3 = 0.25 and
J2 = J1 = 0.15 rad on the four fingers, THJ5 = 0.3 and THJ4 = 0.4 rad, with matching targets) and the scene's initial
object state, and never refunds ticks. `Client.hand_target(actuator_targets, ticks, world=...)` wraps the action.

`observation.proprioception` holds `joint_positions_rad[24]` and `joint_velocities_rad_s[24]` in qpos order
(`rh_WRJ2, rh_WRJ1, rh_FFJ4 … rh_FFJ1, rh_MFJ4 …, rh_RFJ4 …, rh_LFJ5 … rh_LFJ1, rh_THJ5 … rh_THJ1`) and
`actuator_effort_nm[20]` in action order (absent with `effort = false`). With `tactile = true`, `observation.tactile`
has `thumb`, `first`, `middle`, `ring`, `little` and `palm`, each with `pad_normal_force_n` (a rows × columns grid) and
`total_normal_force_n` (all contacts on that body), plus `unit` and `layout`. A fingertip grid is 3 × 3 over the
palmar side (body y ≤ 1 mm) of the distal phalanx: body x −9.5…9.5 mm in columns of increasing x, body z 34…4 mm in rows,
distal first. The palm grid is 4 × 3 over the palmar face of `rh_palm` (body y ≤ −4 mm): z 96…−12 mm in rows, distal
first, x −42…44 mm in columns. Normal contact force is binned by contact position in the body frame; like the Panda
taxels it counts any contact on the body, and contact positions and identities are never exposed. Hand scenes expose no
images, calibration, wrench or point-grasp state. Candidate projects copy `robot/` unchanged, include
`robot/shadow_hand.xml` once and set `<compiler angle="radian" autolimits="true" meshdir="robot/assets"/>`; object meshes
are written inline (`<mesh vertex="..."/>`, collided as the convex hull of the vertices).

## Opt-in Franka arm with Shadow hand protocol

With `[simulation] robot = "panda_shadow"` the scene includes the public fixture `robot/panda_shadow.xml`: the compact
Franka arm (mounted as in `robot/panda.xml`, arm gains 4 times upstream, gripper removed) with the unchanged Menagerie
Shadow hand forearm attached to its flange where the Panda hand was (see `robot/IMPORT.md`). The task offers both
`arm_target` and `hand_target` (in that order after `observe`) in place of `grasp`, `move` and `release`. `arm_target`
takes the seven Panda joint targets and `ticks` only (no `gripper_width_m`; sending one is rejected); `hand_target` is the
Shadow hand action above. Each interpolates from the current targets of its own actuators, so arm and hand commands
compose. `reset` restores the arm home pose (palm down, `robots.ARM_HOME["panda_shadow"]`) with the hand home pose.
`observation.proprioception` holds `joint_positions_rad[31]` and `joint_velocities_rad_s[31]` (`joint1` … `joint7`, then
the hand qpos order) and `actuator_effort_nm[27]` (the arm actuators, then the hand action order); `tactile` is the
Shadow hand's. `Client.arm_target(joint_targets, ticks=..., world=...)` omits the gripper. The `local_controller` tool
reads the robot from the mounted fixture: `taxels(joint_positions)` then takes all 31 proprioception joint positions.
Candidate projects copy `robot/` unchanged and include `robot/panda_shadow.xml` once.

## Optional Panda wrist camera

`simulation.wrist_camera = "wrist"` enables a second camera fixed directly to
the Panda `hand` body. The global `camera` remains fixed in worldbody. Wrist
images use flat `wrist_rgb`, `wrist_rgb_format`, `wrist_depth` and
`wrist_depth_format` fields with the same dimensions and encodings as the
global images; `wrist_calibration` uses the global calibration schema. Its
`world_from_camera` pose follows the current hand state. Observing either camera
does not advance physics. The `rgb`, `depth` and `calibration` switches gate both
cameras, including responses filtered from private full observations; a blind
configuration exposes no camera data. Disabled wrist cameras expose no wrist
fields. Candidate projects preserve the supplied public robot fixture, including
the hand camera. Tracking cameras and cameras attached to scene objects are
rejected as observation cameras.

Inference replies to the agent contain only `content`, `tool_calls` and context-size `usage` (`promptTokens`,
`cacheReadTokens`, `cacheWriteTokens`). Spend accounting is host-only (`inference_accounting` audit events).
When the orchestrator's spend control stops a run, the agent receives
`{"budget_stop": true, "reason": "host stop"}`, and a subsequent `finish` is recorded as `budget_suspended`.

### Budget holds (continuing a run without a special case)

`[inference] budget_hold_seconds = N` turns a spend-cap stop (run `max_cost_usd` or the shared pool) into a hold. The
controller writes `hold.json` (reason, spend, elapsed) in the run directory, pauses the agent and candidate containers
and stops both the wall and active clocks (`WallClock.hold`). The orchestrator then writes `hold-decision.json`:
`{"action": "resume", "max_cost_usd": X, "cost_pool_usd": Y}` raises the caps and retries the held call, so the agent
receives an ordinary reply; `{"action": "stop"}` (or no decision within N seconds) suspends as above. Both files are
archived as `hold-NNN.json` / `hold-decision-NNN.json`; the audit records `inference_hold` and
`inference_hold_decision`. Hold time extends the hard deadline and is excluded from `wall_elapsed` (progress, events,
termination), and `termination.json` lists `holds` and `held_seconds`. Every inference reply carries the current
`deadline_unix`, which the sandbox harness adopts (never shown to the model), so the agent's remaining wall time is
unaffected by a hold. A resumed run is therefore the same run it would have been with the higher cap from the start;
the agent sees the full `wall_seconds` budget from its first message. Call/token caps and missing accounting never hold.


## Positioning-fixture actions

Levels whose private manifest has a `[fixture]` table (`environments/panda_small/fixture_*`, see
[environments](environments.md#spring-powered-positioning-fixture-levels)) add two reference-world actions, also allowed in
`sequence` step lists. Both are host-enforced, charged and audited like other physical actions; in the reconstruction world
they are rejected, and an observe-only reference world (`reference_interaction = false`) blocks them.

| op | Fields | Behavior |
| --- | --- | --- |
| `reset_carriage` | none | The fixture's cocking cycle: retract the spring plunger to the compression set on the compression dial, latch it with the carriage against it, set the brake pad's spring preload from the pressure dial (dial changes take effect here), and re-seat the pad, which then settles for a fixed number of physics ticks (executed, recorded and charged). Dials and robot are untouched. |
| `release_carriage` | optional `ticks` (default 0) | Open the latch of a cocked carriage, then advance physics. Rejected unless `reset_carriage` cocked it since the last release. |

Every observation on these levels has `fixture`: `{"compression_dial": d, "pressure_dial": d, "carriage_cocked": bool}`,
the two dial positions in scale divisions (0 to 10, as printed on the dial scales). The agent moves the dials only with
the robot (`arm_target`). The carriage position is not reported: it is read from images against the ruler. The pad's
contact state, the seed of the reset perturbations, the compression and pressure mappings and every other fixture value
stay private. A world `reset` restores the initial state (carriage uncocked, dials at 5).

## Predictor submissions (`src/sim2sim/predictor.py`)

A predictor metric scores a program instead of a scene: `fixture` (call mode) and `die` (stream mode, real time). The
runtime's lookup `PREDICTOR_METRICS` maps each to its module, whose `evaluate_submission(job)` (a `predictor.Job`)
returns the evaluation report; such a metric combines with no other metric. Its configuration sets `[evaluation]
entrypoint` to the program (`predictor.py`, `predict.py`); the task's `submission.entrypoint` names it and
`submission.kind = "predictor"` marks the task, so the harness uses its predictor texts (`SYSTEM_PREDICTOR` with one
line for lock-step or real-time worlds, `COMPACT_PREDICTOR`, `GUARD_PREDICTOR`); scene conditions keep theirs. The metric
states its own contract (`submission.predictor` for `fixture`, `die.predictor` for `die`).

A parameter answer replaces the scene when `[evaluation] metrics = ["lowrank"]`: the submission is a ZIP holding the JSON
file named by `[evaluation] entrypoint` (`parameters.json`), and the task's `submission.kind = "parameters"` makes the
harness use the lock-step predictor texts with the `PARAMETERS` substitutions (estimates instead of a predictor). The
task statement gives the answer's keys and units; the evaluator scores it in its ordinary container (docs/evaluation.md).

On `submit`, while the agent is paused, the metric starts one container per submission (`predictor.container`,
`python -m sim2sim.predictor MODE`): `--network=none`, a read-only root, no mounts, no source scene, manifest, controller
or credentials. The host first sends `{"op":"load","project_b64","entrypoint","call_seconds"}`; the worker extracts the
ZIP with the submission checks (XML files are still MJCF-validated) and replies `{"ok":true}` or `{"ok":false,"error"}`.
Logs go to `RUN/predictor-NNN/` (`predictor.stderr.log` and the mode's log).

**call mode** (`predictor.calls`; evaluator image, `[sandbox.worker]` limits). The worker imports the entrypoint; the
metric's `collect(call)` (for `fixture`, `fixture.collect_predictions(manifest[fixture], call)`) decides the calls,
`{"op":"call","function","kwargs"}` → `{"ok":true,"value"}` or `{"ok":false,"error"}`. Each load or call has
`call_seconds` (default 60) of wall time, enforced by an alarm that agent code cannot catch with `except Exception`; a
value must be JSON with finite numbers, at most 1 MiB. Anything the program prints goes to the stderr log, never the
protocol. One failed call is that call's error. A timeout of the whole session (`[evaluation]
predictor_timeout_seconds`, default 900), a crash or an oversized reply marks the worker broken and fails every later
call. The answers go to the private evaluator container as `evaluation["predictions"]`; the call log (arguments,
values, errors, seconds) is `predictor-calls.json`.

**stream mode** (`predictor.stream`; the agent image and the agent's sandbox bounds, because answer latency is scored and
the program gets exactly the resources it was developed with). After the load reply the worker execs
`python3 ENTRYPOINT` in the unpacked directory; its stdin/stdout are JSON lines in real time:

1. The host sends the metric's public `start` message; the program replies `{"type":"ready"}` within 120 s.
2. Per trial, the host sends the trial header at release, then frame k at k / frame rate seconds after release in real
   time (only frames before the deadline), then `trial_end` at the deadline, and `end` after the last trial.
3. The program's first `{"type":"prediction","trial":i,...}` line for the trial counts:
   - `on_time` if it is received by the deadline and the metric's `validate` accepts it (for `die`: valid probabilities);
   - `malformed` if it is not accepted, or if any other line is written during the trial;
   - `late` if it arrives after the deadline but within the 0.25 s gap before the next trial;
   - `missing` otherwise.

Every output line is logged with its receipt time (`predictor-output.jsonl`). The metric supplies the trials
(`(header, frames)` pairs) and the scoring.

Nothing a predictor returns reaches the agent; the submit reply is only the evaluator status. Predictor metrics need
`strict_oci`; the development backend fails such an evaluation explicitly. A new predictor metric adds its module with
its public contract and `evaluate_submission(job)`, and one `PREDICTOR_METRICS` entry.

## Real-time world (`simulation.realtime`)

`[simulation] realtime = true` with an integer `frame_rate` (frames per simulated second) is a generic environment setting
for any level and either embodiment (point grasp or robot). The default `false` leaves every lock-step level unchanged.
It requires `budget.wall_only` and strict OCI. Resolution and modalities are the usual `width`, `height`, `rgb`, `depth`
and `calibration` settings. `src/sim2sim/realtime.py` implements it.

- **Clock.** A host thread advances the reference world at one simulated second per wall-clock second (`RealtimeWorld`).
  It keeps running while the agent reasons, while its code runs and between requests. The agent cannot pause it, slow it,
  step it, or save and restore it. There is no reconstruction world and no `load_reconstruction`. Host-side holds outside
  the run (budget and provider holds, private evaluation) pause the world with the run clocks. A provider-failed inference
  attempt is not known to fail until it ends, so the world runs through it and is then rewound to the attempt's start
  (`RealtimeWorld.mark`/`rewind`): it continues exactly as if frozen. A successful call's latency advances the world.
- **Actions.** Agent actions take effect in arrival order at the next physics tick boundary. A timed action
  (`move`, `arm_target`, `hand_target`; `ticks` is its duration) occupies its ticks, and later actions wait for it.
  Chained actions such as `move` then `release` therefore take effect back to back, whatever the transport latency.
  Each reply arrives when its action takes effect. `observe` and `release` accept no ticks.
- **Frames.** Camera frames exist only on a fixed grid: every 1/`frame_rate` simulated seconds since the latest reset
  (frame 0 is the reset itself). Every image the agent receives is one of these frames. A reply carries the first frame
  captured at or after its action took effect.
  - Observations add `frame` (global index), `episode` (resets and launches so far) and `time` (since the episode began).
  - `frames` (`since`, `limit` 1–100, default 25) returns kept frames newer than `since`, oldest first. It waits up to
    1 s for the next frame. The latest 20 simulated seconds are kept.
  - Frames are rendered on request from the state recorded at the frame's tick, in the controller thread, with a second
    copy of the scene. MuJoCo rendering is bound to its context's thread, so the physics thread never renders.
- **Privacy.** Replies contain only the usual public observation fields; states stay private.
- **Audit.** Each action's state when it took effect (`applied` checkpoint) and each frame it returns are audited, and
  every wire is kept. Per-tick checkpoints are not recorded. `termination.json` adds `realtime_world`: total ticks,
  frames, launches, the largest lag behind the wall clock in ticks, and any physics-thread error. A physics-thread error
  ends the run.
- **Determinism.** A trajectory is fixed by its complete initial state and the recorded application tick of every action.

**Launcher (optional, `src/sim2sim/launcher.py`).** A private manifest `[launcher]` table names one free `body`, a `seed`,
a `rest` pose and `perturbation` sizes, plus `[[launcher.protocols]]`: each has a `name` and a nominal release
`position`, `quat` (wxyz), `velocity` (of the body-frame origin) and `angular_velocity` (world axes).

- With a launcher, `reset` restores a perturbed rest pose, and the `throw` action (`protocol`) resets the world and
  releases the body with the protocol's perturbed state.
- Launch k of stream s always draws `default_rng([seed, s, k])`. Stream 0 is the agent's session (k counts its resets and
  throws), 1 is evaluation trials, 2 the empirical baseline and 3 measurement noise.
- Every launch first resets the whole simulation (zero velocities and solver warm start, robot home, grasp released), so
  each trajectory is fixed by protocol, stream, index and later actions.
- The task lists the protocol names and nominal states (`launcher_protocols`). The seed and perturbation sizes stay
  private.
- Without a launcher, `reset` restores the scene's initial state and there is no `throw`. A `[launcher]` requires
  `realtime`.

## Controller channel (`[tools] channel`)

The sandbox harness relays the controller to the agent's own programs, so code can act on a live world and read frames
with low latency.

- **When.** While a `bash` command runs, the harness serves the Unix socket `task.code_channel.path`
  (`/tmp/sim2sim-controller.sock`), one connection at a time. With `--bash-mode pausable` the socket lives as long as the
  command and relays requests only while the command runs; pausing waits for a request in flight, and later requests
  wait until `bash_resume`.
- **Protocol.** Each request line is a controller action object; the reply line is the controller's reply, including
  base64 images.
- **Allowed operations.** Only the physical actions the task offers: `observe`, `frames`, `grasp`, `move`, `release`,
  `arm_target`, `hand_target`, `camera`, `reset`, `reset_carriage`, `release_carriage` and `throw`. Submit, inference,
  progress, finish and sequence are refused.
- **Client.** `from sim2sim.client import Channel; Channel().request("frames", since=-1)`.
- **Accounting.** The requests pass through the normal controller, so they are audited and charged as usual.

Without `[tools] channel` the socket never exists. It is not the `local_controller` tool, which runs a local copy of
the reconstruction world inside the sandbox and never reaches the controller. The `sequence` tool remains available
alongside and may contain `throw`.
