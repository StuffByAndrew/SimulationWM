# Offline experiment dashboard

Experiment viewing uses the dashboard directly. The separate viewing skill has
been removed. Generation includes all runs under the experiment root; choosing
which experiments to display will be handled by future dashboard filters.

Run `sim2sim dashboard --root experiments` after recording experiments. `--root` may be repeated (e.g. lab storage plus home): runs from every root share one catalog, run directory names must be unique across roots, and the default output is the first root's `dashboard/`. `scripts/serve-dashboard.sh` takes several roots as a colon-separated `SIM2SIM_EXPERIMENTS`. Current strict waves: `.venv/bin/python -m sim2sim.cli dashboard --root /n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-experiments --root experiments` (login node is fine), then `SIM2SIM_EXPERIMENTS=/n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-experiments:$PWD/experiments scripts/serve-dashboard.sh`. The default output is `experiments/dashboard/index.html`; open it directly in Chromium or another modern browser. `--output` selects another artifact directory. Compatible runs live in `experiments/runs/<id>` and may include `experiment.json` and `status.json`. The dashboard never executes source scenes, candidate XML, models, inference or evaluation.

To view the existing snapshot from another machine, run this on the server:

```sh
./experiments/dashboard/serve.sh
# Optional SSH destination or config alias for the same server host:
./experiments/dashboard/serve.sh USER@HOST
```

The script serves only the dashboard directory on `127.0.0.1`, trying ports
8000, 8001, then 8002, and exits if all are in use. It prints an SSH tunnel
command to run on the client and the browser URL. It works from any working
directory. Keep the server and client
tunnel terminals open while viewing; Ctrl-C stops each. The SSH destination
must reach the exact machine running the server, including when a cluster has
multiple login nodes. If needed, add `-J USER@GATEWAY` to the printed SSH command.
The script uses `.venv/bin/python` or `SIM2SIM_PYTHON` and serves the existing
snapshot; run the dashboard generation command above to refresh its contents.

A compact experiment selector shows each run directory ID beside its display name
and status, so attempts with the same name remain distinct. It initially selects
the newest running attempt, or the newest completed/finished nonfixture attempt when none
is running; other attempts remain selectable. Reference and reconstructed camera recordings occupy two fixed side-by-side panes. The shared timeline and each recording pane's range use **elapsed wall time**, including pauses and inference/evaluation intervals. Playback holds the latest recorded frame independently for each world while it is idle. Simulation seconds and reset episodes are displayed as metadata and are never paired across worlds. Original verified MP4s are download links with their encoded timing; synchronized recording playback uses exact timestamped PNGs, avoiding a false mapping from encoded video seconds to wall time.

A checklist selects groups of recorded metrics. The default shows total Chamfer and COM error together, all per-primitive density errors together, and hinge friction errors with separate axes for their different units. Source/predicted measurements and other historical metrics remain selectable but are initially hidden. Per-primitive curves retain stable source names. Implementation seeds, sampling counts and boolean consistency flags are not numeric metric series. Failed or unavailable checkpoints create gaps. A dashed cursor follows wall time on every plot. Clicking or dragging a plot, scrubbing any recording/shared timeline, or explicitly clicking a progress line updates all panes, cursors and the selected note. The fixed-height progress panel scrolls independently: scrolling never seeks. When playback or seeking selects a new note, the panel reveals that note.

Each plot legend also has a checkbox for every recorded line. Unchecking one
line hides only that line and rescales the plot to the remaining visible values.
The plot and its legend stay available when every line is unchecked, so lines
can be restored individually. Line selections survive redraws caused by
filtering or seeking within the selected run and reset when another run loads.

Only explicit `progress` events from `source = "agent"` or official `"provider_summary"` events appear in the progress panel. Missing notes are reported as missing. Provider summaries are identified separately and are not represented as private reasoning traces. No progress is inferred from hidden reasoning, encrypted provider items or arbitrary prompt text.

Generation reads a **fixed initial audit-manifest prefix**. Every referenced blob, path, declared state/image checksum and completed video is verified. Appends after that prefix are excluded until the next generation. The resulting dashboard is a read-only snapshot and never modifies a running experiment. A snapshot reports `complete` only when its exact captured manifest is sealed and it contains a termination event. Old unsealed audits remain viewable as partial snapshots. Blocked, empty and corrupted runs remain selectable with explicit no-frame/no-metric/no-progress states. Items lacking wall timestamps are counted and excluded from synchronized playback rather than assigned fabricated times.

Images are copied byte-for-byte into content-addressed assets. Existing cached images/videos are checked and repaired from verified source bytes; symlinked assets are refused. No source XML is loaded. Raw logs remain authoritative, and regenerated dashboards are views of those logs, not replacements. Dashboard artifacts contain private camera observations, metrics and progress and must not be mounted into the agent sandbox.

Snapshot storage totals reuse the mandatory audit verification scan: logical
bytes, allocated regular-file bytes and regular-file counts need no second tree
walk. The catalog also records the unique verified asset count and logical bytes
from that refresh. These figures describe audit files and verified dashboard
assets, not total experiment-directory disk usage; stale cached assets and other
run files are outside those totals.

Pure artifact/snapshot tests may run on a login node. Browser rendering and interaction tests run only on a Slurm `test` allocation: `sbatch scripts/test.slurm tests/test_dashboard.py tests/test_replay.py`. When Chromium is installed, the browser test exercises real DOM input/click/scroll events and checks shared/per-world scrubbing, plot cursors, explicit note seeking, idle-world held frames and scroll isolation. Its clearly labeled synthetic unit-test notes and timestamps are test data, not agent output or an experiment result. Native experiment fixtures must retain their own provenance and must never be presented as successful Astra or OCI runs.

After the selector update, Slurm `test` job `56071718` passed all 15 dashboard
checks, including Chromium checks for both running and finished default
selection (6.10 seconds). Companion replay job `56071432` passed 16 checks
(1.57 seconds).

Progress records retain their `origin`. Explicit model commentary is labeled **Model progress**, adapter-generated updates **Adapter status**, official provider summaries **Provider summary**, and other explicit notes **Agent note**. Adapter status is never labeled as model reasoning.

An execution-mode badge accompanies run status. `development_shared_filesystem` is labeled **Development · shared FS** and explicitly describes a prompt-scoped workflow without a hard filesystem/tool security boundary. `strict_oci` is labeled **Strict OCI mode** as a mode declaration; actual enforcement and agent launch require the run's preflight/audit evidence. `trusted_fixture` (or explicit fixture metadata) is labeled **Trusted fixture** and makes no model or isolation claim. Missing mode provenance remains **Mode unrecorded** rather than inferred from neighboring artifacts.

## Grasp overlay and visualization cost

The recording panes have a **Show grasp point and recorded handle trajectory**
checkbox. Cyan marks the actual point on the object; gold marks the handle and
its recorded path. Paths restart at release, reset or candidate reload. The
projection is captured privately from the current state with the fixed camera;
it does not step or render the simulation. No coordinates enter public agent
observations. Old recordings without this evidence have no overlay; viewing
never reconstructs it by parsing a model or running physics. Lines join recorded
samples and do not claim unrecorded intermediate motion. Original MP4 downloads
remain unchanged; the optional overlay is drawn during dashboard playback.

Comparable curves share plots and legends. Selecting a metric affects only the
view; experiment metric selectors control whether its evaluation is computed.
Unmatched parts use JSON `null` (NaN in numeric histories) and create gaps,
including checkpoints before the first usable match. No unavailable score is
converted to zero.

The two-hour presets capture at 10 fps of simulation time plus action boundaries,
disable duplicate MP4 encoding, and save checkpoint JSON without repeatedly
rendering standalone PNG plots. The dashboard still plays the exact recorded
PNG frames. Asset hashes are verified once per generation and reused across
identical frames/runs. The view materializes RGB when available, and depth only
as a fallback; unused depth images stay in the original audit. Existing cache
files are retained, so the catalog's verified-asset total excludes stale copies.
Duplicate checkpoint references are suppressed in the
view; the complete wire/state audit is retained. Overlay trails are stored as
individual projected samples, not a growing copy of the whole trail per frame.
Consecutive stationary handle samples share one displayed path vertex, and
unavailable projections split the path. Original overlay samples remain in the
audit. Playback redraws an overlay only when its recorded frame or toggle changes.
Parallel jobs use `--no-dashboard`; the orchestrator refreshes once per batch to
avoid concurrent writers and repeated rescanning of sibling experiments.

Archive traversal uses one metadata read per entry, pins the blob directory,
and opens each hexadecimal blob name relative to that directory with no-follow
semantics. A 16 MiB cache reuses exact hash-verified bytes within one traversal;
it is discarded between refreshes, so later corruption is still detected. The
complete referenced evidence is verified. Slurm `test` job `56345558` passed
all 34 dashboard/replay checks after this optimization (2.73 seconds).

## Recorded model context and candidate changes

The expandable model-context section displays the exact recorded public goal,
public action schema, enabled observation modalities, public fixture path and
model constraints. A verified audit task takes precedence over the bounded
`public/task.json` artifact fallback, whose provenance is labeled separately.
Only public fields are copied; initial encoded observations and private
configuration/source are excluded. All displayed text uses `textContent`.
The view explains that source XML, parameters, probe artifacts and scores are
withheld by the public protocol. Shared filesystem development remains a prompt
convention without enforced filesystem/tool isolation. The known public task
is distinguished from the unavailable original orchestrator prompt and full
external model/tool transcript.

Optional `candidate-change-history.json` schema version 1 supplies retrospective
checkpoint reports. Its `checkpoints` carry `change_index`, `version`, original
`wall_elapsed`, `evaluated_at`, `origin`, `status` and `report`; `unavailable`
retains missing/failed-load evidence. Retrospective metrics use separate labeled,
dashed lines with hollow points at the original change time. Evaluation time is
shown in tooltips. Original submission metrics remain separately selectable.
Unavailable evaluated changes create gaps, never zero values. These sidecars
are private researcher evaluations, outside the sealed original audit; dashboard
viewing never evaluates a model or alters the audit. Full-image normalized RGB
MSE (`reset.rgb_mse`) is selectable and initially visible when recorded.

Recorded `orchestration_notes` appear as a **Retrospective researcher annotation**,
separate from any exact orchestrator prompt. Saved submission-only versions are
identified separately in retrospective point tooltips. Unavailable candidate
changes and their recorded timestamps appear below the plots with no scores.

## World action timeline and log

Blue reference and gold reconstruction marker rows below the wall scrubber
identify verified `action_begin` records for `observe`, `arm_target`, `grasp`,
`move` and `release`. Zero-tick observations remain actions. A separate purple
lane marks meta actions: `reset`, `load_reconstruction`, `submit` and `finish`.
Inference, progress and pre-execution protocol rejections create no action
marker. Markers use actual action
start times; the log also records completion time, outcome, requested ticks,
executed ticks, charged ticks and any unconfirmed ticks. Failed partial actions
remain visible. Missing completion evidence and negative historic reset execution
counters remain unknown, never fabricated as zero.

The text log combines explicit progress and world actions. Its type selector
filters model progress, provider summaries, meta actions or either world's actions; **All**
also retains adapter/other explicit notes. Clicking a marker seeks its start,
reveals the exact action row (changing an incompatible filter), and selects both.
Clicking an action row seeks and selects its marker. Ordinary scrubbing selects
the latest visible log entry; scrolling alone never seeks. Playback speed
choices 1×, 2×, 4×, 8× and 16× scale elapsed wall time in the shared recordings,
plots and log without changing recorded timing.

Physical world actions comprise `observe`, `arm_target`, `grasp`, `move` and
`release`. Meta actions comprise `reset`, `load_reconstruction`, `submit` and
`finish`; they have their own **Meta actions** log filter and purple timeline
lane. Both categories remain in **All** and retain their selected world and
recorded outcome. Safe public resolved arguments (joint targets, gripper width,
pixel, displacement, entrypoint and camera) appear as compact JSON alongside
requested ticks. Project archives, inference messages, private state and scores
are excluded from that argument projection. Submit and finish display as run-level
meta actions; resets and candidate loads retain their world-specific labels.

## Interactive physics inspector

The **Interactive physics** tab (next to **Recordings**) shows two live 3D
viewers: the private source scene on the left and one recorded reconstruction
version on the right. A scrubber at the bottom spans the run's elapsed wall
time; it selects the latest version loaded by that time. Markers show each
version: gold for a successful `load_reconstruction`, purple for a distinct
submission, grey when its archive could not be recovered. Clicking a marker or
the ◀/▶ buttons selects one directly. Before the first load, the right viewer
is empty rather than showing a guess. Recorded retrospective Chamfer, COM and
RGB MSE values appear beside the selected version when available.

Each viewer is a [Viser](https://viser.studio) server running real MuJoCo
physics. Click a movable body to attach the benchmark's ideal point grasp at the
clicked surface point; drag the translation gizmo to move the handle. The handle
follows at most 1.5 m/s, and the grasp imposes no orientation. Grasping starts
playback. Each viewer's side panel has Play, Pause, Step 0.1 s, Reset, Release
and speed controls. Below the viewers, **Reset both**, **Play both**,
**Pause both** and **Step both** act on both worlds together. Cameras start at
the source scene's recorded `fixed` camera and are synchronized by default.
Each world simulates independently; there is no body correspondence between
them. Panda scenes hold the arm at the benchmark home pose. Their object bodies
(not robot links) accept the point grasp, an inspector-only opt-in that agent
actions never receive. Object bodies are found in every `<worldbody>` section,
including when a model includes the robot before its own worldbody. Clicks are
raycast only against displayed geom groups 0–2, so hidden collision geometry
(for example Panda group-3 meshes) cannot block a visible body.

Starting the server:

```sh
./experiments/dashboard/serve.sh [USER@LOGIN_HOST]   # or scripts/serve-dashboard.sh
```

Run this on a login node. It picks three consecutive free ports (HTTP and one
per viewer, from 8000), then requests a CPU-only `test` allocation with `srun`
(defaults: 4 CPUs, 8G, 02:00:00; override with `SIM2SIM_INSPECT_CPUS`,
`SIM2SIM_INSPECT_MEM` and `SIM2SIM_INSPECT_TIME`). It waits for the server's
ready file, forwards the same ports from the login node to the compute node's
loopback over SSH, and prints the client tunnel command and URL. Ctrl-C stops
the tunnel and the job. If login-to-node forwarding fails, it prints a two-hop
`-J` command instead. `--static` keeps the previous recordings-only server on
the login node; its physics tab then reports that no inspector is running. The
server needs an existing snapshot (`sim2sim dashboard`). It serves the current
`index.html`/`dashboard.js` from the installed package, so adding this tab
needs no snapshot regeneration. All servers bind only to `127.0.0.1`. The
dashboard header shows the time left before the server's Slurm job ends, read from
the job's end time (falling back to the session length) and counted down locally.

Versions come from a verified audit read (`recover_changes`), the same source
used for retrospective candidate evaluation. The source bundle must match the
recorded `source_bundle_sha256`. A run without a verified audit, or with a
mismatched source, shows an explicit error instead of a fallback. The host
process never parses MJCF. Each world runs in a `sim2sim.dev_worker inspect`
process with the development worker's sanitized environment and rlimits; CPU
time is capped to the session length, and these workers are not CPU-pinned.
Candidate archives pass `extract_submission` (no plugins, entities or external
paths) inside that worker. Compilation failures appear in the viewer. Edits that
were never loaded or saved remain unavailable. API POSTs require
`application/json`, so cross-site pages cannot drive the controls without a
CORS preflight, which the server never grants. Inspector state is
researcher-only, like the rest of the dashboard.

Tests: `tests/test_inspector.py` covers primitive meshes, version selection,
audit recovery, refusal of mismatched sources and the JSON API. On a Slurm
`test` node it also runs live workers: load and plugin rejection, a pre-load
empty state, grasp-and-lift, reset, and a refused floor click. It then drives
Chromium against a live backend and against a static snapshot.

Validation, 2026-11-02: Slurm `test` job `57112482` passed **91 tests**, none
skipped: inspector, dashboard, replay, simulation, candidate history,
development and Panda suites, including both Chromium physics-tab checks and
live grasp-and-lift (42.79 seconds). The full suite in job `57110407` had two
sub-second budget timing failures in `tests/test_runtime.py`; job `57111289`
passed that file and the inspector tests (37 tests). Job `57111546` exercised
the real deploy script: it forwarded ports from the login node to
`holy8a24101`, then recovered and verified all 11 versions of
`astra-medium-panda-blocks-vision-20261102-001`, which took about three minutes
for its 1.7 GB audit. It compiled both worlds and played 5.0 s of physics in
real time without lag or instability warnings.
