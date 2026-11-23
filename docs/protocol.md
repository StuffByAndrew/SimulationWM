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
| `camera` | `pos: [x, y, z]`, `lookat: [x, y, z]` (only with `simulation.movable_camera = true`) | Re-aim the observation camera (up = +z) without advancing time. Both points within ±4 m, at least 0.05 m apart, and `pos` at least the near-plane frustum reach (`1.5 · znear · extent · corner factor`) from every surface, so the near clip plane can never cut a wall; a camera inside a geom or a small closed cavity is rejected too. |
| `submit` | `project_b64` | Submit a base64 ZIP containing `scene.xml` and local assets. |
| `inference` | `messages`, optional `tools` | Request inference from the configured host broker. |
| `finish` | none | End the run. |

Pixel origin is the image's top left, with coordinates inside the image. An
occluded object cannot be grasped through the visible surface. Only one grasp
exists at a time; release it before grasping again. The point constraint supplies
force without an orientation target, so off-center pulling can rotate a body.
A cloth (any MuJoCo flex, from `flexcomp` or `<deformable><flex>`) is grasped at
the vertex of the hit flex nearest the ray hit, held at the hit point's offset;
the wrench then reports that vertex's pull. Ray hits and body identity stay private.

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
pinned weights sha256 `6dbb9052…6c2f`) needs the tools image, whose label the
runtime checks before launch; `planning` (IK, interpolation, collision-checked
straight-line/RRT-connect on the agent's own scene) needs `robot = "panda"`. `local_controller` (`robot = "shadow_hand"`)
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
tactile = false        # optional robot taxel normal forces (robot = "panda" or "shadow_hand" only)
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

- The failed attempt's time is credited back to both clocks.
- The backoff (15 s, doubling to at most 300 s) runs with the containers paused and the clocks stopped.
- The identical request is then retried. Each hold is audited as `provider_hold`.
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

The reviewed image recipe and `.containerignore` copy only generic package code,
metadata and the sandbox agent. The scripted protocol demonstration is not in
the image; the OCI integration test passes it as an inline `python -c` command. Original environments, manifests, run stores,
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

A metric may score a program instead of a scene. Its configuration sets `[evaluation] entrypoint` to the program (e.g.
`predictor.py`), the task's `submission.entrypoint` names it, and `submission.predictor` states the contract: the runner
description and the metric's functions (for `fixture`: `predict(compression, pressure, n)` returning `n` stopping
positions in ruler mm, and `settings()` returning `{target name: {"compression", "pressure"}}`). On `submit` the host
runs, before the private evaluator and while the agent is paused:

1. `run_predictor(submission, entrypoint, sandbox, output, supervisor, collect, timeout, call_seconds)` starts one
   container per submission from the evaluator image (`python -m sim2sim.predictor`) with the `[sandbox.worker]` limits,
   `--network=none`, a read-only root, no mounts and no source scene, manifest or credentials.
2. Wire, one JSON line each way: `{"op":"load","project_b64","entrypoint","call_seconds"}` extracts the ZIP with the
   submission checks (XML files are still MJCF-validated) and imports the entrypoint; `{"op":"call","function","kwargs"}`
   returns `{"ok":true,"value"}` or `{"ok":false,"error"}`. Each load or call has `call_seconds` (default 60) of wall
   time, enforced by an alarm that agent code cannot catch with `except Exception`; a value must be JSON with finite
   numbers, at most 1 MiB. Anything the program prints goes to the worker's stderr log, never the protocol.
3. The metric's `collect(call)` (for `fixture`, `fixture.collect_predictions(manifest[fixture], call)`) decides the
   calls; one failed call is that call's error. A timeout of the whole session (`[evaluation]
   predictor_timeout_seconds`, default 900), a crash or an oversized reply marks the worker broken and fails every later
   call. The result goes to the evaluator as `evaluation["predictions"]`; the call log (arguments, values, errors,
   seconds) is written to `RUN/predictor-NNN/predictor-calls.json` and the program's stderr to `predictor.stderr.log`.

Nothing a predictor returns reaches the agent; the submit reply is only the evaluator status. Predictor metrics need
`strict_oci`; the development backend fails such an evaluation explicitly. A new predictor metric adds its function
contract and a `collect_predictions(table, call)` beside its scorer and selects the runner in the runtime's submit path.
