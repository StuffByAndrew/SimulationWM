# Experiment log

Dated record of experiments run or in progress, newest first. One entry per batch:
date (UTC), status, question, mode, model, budget, runs, outcome. Keep it short. Link
details (batch JSON, reports) rather than copying them; never put private scores,
source values or parameter descriptions where an agent could read them. "Budget
used" = native wall time ÷ wall limit. How to run experiments: [experiments.md](experiments.md).

## 2026-11-25 05:30 UTC — Inkling 2 Mini self-hosted: service validated, five hard-level configs planned (not launched)
- **Service:** `scripts/inkling-service.slurm` (llama.cpp b12706 on 2× A100-40GB) serves the official checkpoint in BF16 at
  zero cost (see AGENTS.md).
  - Requeue job `63571190` became healthy.
  - Broker smoke `63573091` passed: tool call, tool-result round trip, image, six efforts, 143K-token needle.
- **Pipeline smoke:** `inkling-smoke-20261125-001` (`experiments/validation/inkling-smoke.toml`) ran in strict test job
  `63575871`.
  - Run root: `/n/netscratch/qianqian_lab/Lab/aejxie/inkling/smoke-root`, so no dashboard or report picks it up.
  - It is the w20-latch-out Inkling config with a 20-minute wall.
  - 22 calls, all tool calls except the last stop, with no provider errors, about 13.5K completion tokens and prefix-cache
    reuse.
  - It ended by the harness reserve with 0 submissions. It is validation, not a result.
- **Planned:** `experiments/configs/inkling-20261124/`, the five hard levels on the gemini-carbon-20261124 recipe, pool
  `inkling-20261124`, per-call timeout 1800 s.

## 2026-11-24 20:10 UTC — Netscratch quota killed all 8 running jobs (19:12 UTC); everything moved to the lab root
- Jobs `63264369`, `63264376`, `63264378`, `63264379`, `63281858`, `63281862`, `63320565` and `63320572` died between
  19:12 and 19:18 UTC on "Disk quota exceeded" writing `status.json.tmp`, so those files still say `running`.
  - Their 12 runs were at 10–32 run-minutes, with no submissions.
  - Runs: the w25 offset-com pair -002, w24-die-franka-neon-002, the four Carbon `-20261124-001` runs, the w26 lowrank and
    mjcf `-20261124-001`, and the w29 tetra-spring and cube-spring pairs `-001`.
  - All are infrastructure failures.
- The 6 still-queued netscratch-prepared jobs were cancelled before starting: `63454970`, `63454996`, `63455010`,
  `63455013`, `63455018`, `63455042`.
- **All remaining n=1 conditions were re-prepared on the lab root** with the isolated configs (`prepare-lab-20261124.sh`;
  every source byte-identical to the original):
  - wave 25, all 4 pairs, as `-20261124-001`;
  - wave 26 lowrank and mjcf as `-20261124-003`;
  - wave 29, all 8 runs, as `-20261124-003`;
  - Carbon marble, hand, latch and w11 as `-20261124-002`.
- **Jobs** (`trusted-image-isolated-gear.json`, sapphire, 3 days):
  - w25: `63479839`, `63479845`, `63479847`, `63479848`.
  - w26: `63479883`.
  - w29: `63479971`, `63479977`, `63479980`, `63479983`.
  - Carbon: `63479984`, `63479995`, with `--nice=200000`.
- **Queue order:** gear-train, then fixture and die, then wave 27, then waves 25, 26 and 29, then Carbon.
- Fixture bimodal `-003` started on kempner_cpu (job `63471018`) once the source-read audit moved to sapphire (now job
  `63480090`).
- The maintenance move to lab (job `63470430`) is running; netscratch was writable again at 19:59 UTC.

## 2026-11-25 21:20 UTC — Wave 31 on kempner_cpu (QoS now 32 CPUs / 128 GB per user = 4 strict runs)
- Latch-out and cloth-holes started on kempner_cpu at about 20:20 UTC (`63706221`, `63706223`). At 51 and 50 run-minutes
  their calls had all succeeded; cloth-holes had 1 submission.
- The kempner_cpu QoS is now 32 CPUs / 128 GB per user. The two remaining sapphire pairs (`63694746`, `63694747`,
  pending 15 h) were cancelled before starting and resubmitted as single kempner_cpu runs:
  - hand `63717613`, running;
  - cube-spring-block-mjcf `63717614`, running;
  - marble `63717620` and tetra-spring-k `63717640`, which start as slots free.
- All runs are on home; home has 50 GB free.

## 2026-11-25 20:20 UTC — Fixture-bimodal completed; wave 31 still unstarted on sapphire, two runs moved to kempner_cpu
- `w23-fixture-bimodal-neon-20261125-001` (home) **completed**: the agent called finish at 320 run-minutes, after the
  checkpoint was resumed. 179 calls, all succeeded; 3 submissions, all scored `ok`. It is the first complete stochasticity
  run on the isolated image.
- Wave 31's three home jobs had been pending on sapphire for about 14 h.
  - `63694745` (latch-out + cloth-holes) was cancelled before starting. Its runs went to kempner_cpu as single-run jobs
    `63706221` and `63706223`, which run one after the other (8 CPUs per user).
  - `63694746` and `63694747` stay at the top of sapphire.
- The other 22 jobs remain held (storage).
- Builder's analysis of the per-run file count (no code changed): reference-tick checkpoints are about 50% of audit records
  and candidate-worker boundary messages about 42%. No existing setting reduces this materially; packing blobs, or a lower
  tick cadence (which weakens per-tick replay), would need the user's go-ahead.

## 2026-11-25 05:40 UTC — Netscratch file quota nearly exhausted: queue held; wave 31 on home; Inkling merged, not launched
- **Storage:**
  - Netscratch qianqian_lab is at about 99.35M of its 100M **file** quota (builder's measurement); lab is full by bytes;
    home has about 50 GB.
  - A strict run writes **134K–240K files**, almost all `audit/blobs` (one file per content-addressed blob): gear-deep 134K,
    fixture-bimodal 240K. So netscratch fits only 3–4 more runs, and its file quota is shared with the whole lab.
- **All 25 queued strict jobs were held** (05:30 UTC).
- **User decision:** run a few on home now; the rest stay held until storage is fixed.
  - Wave 31 (no resets) re-prepared on home as `-20261125-002`, sources identical; the netscratch `-001` jobs were cancelled.
  - New jobs at the top: `63694745`, `63694746`, `63694747`.
- **Inkling 2 Mini (builder; user request):** merged (2d7d7ad).
  - New `api = "local_openai"` route to a self-hosted llama.cpp service on Kempner A100s (`scripts/inkling-service.slurm`,
    2× A100-40GB, never H100/H200).
  - Configs `experiments/configs/inkling-20261124/`: the hard five, each equal to its Carbon twin except the route, the pool,
    no reasoning_summary and `timeout_seconds = 1800`.
  - **Not launched:** no storage. Launch the service only when the runs can start.
  - For analysis (builder):
    - Precision: BF16 GGUF from the official weights (MXFP8 experts exact; NVFP4 experts rounded at most 2^-8 relative;
      BF16 activations), close to but not bit-identical with the official numerics.
    - Throughput: about 26 tokens/s for a single stream, about 14 per run with 5 runs; a cold 140K prefill takes about
      13 min. Report at matched tokens as well as matched minutes.
    - The 1800 s call timeout grants no extra budget.
    - reasoningTokens is null.

## 2026-11-25 04:55 UTC — First new-image runs: gear-deep completed; fixture-bimodal extended to 6 h
- `w28-gear-deep-20261125-001` (home, job `63530252`) **completed**: 235 run-minutes of 240, 264 real minutes, 133
  calls, all succeeded, 3 submissions, all scored `ok` by the gear-train metric. It is the first complete gear-train run.
  The agent called finish itself; every submission is depth class **deep** (all gears matched, fraction 1.0).
- `w23-fixture-bimodal-neon-20261125-001` (home, kempner_cpu `63530440`) reached its 4 h checkpoint at 04:47 UTC.
  - It had 1 submission and was measuring low-mode drift and hit rates at candidate settings for the target and the
    narrow target, i.e. still working on the mechanism.
  - **Resumed to 6 h.**

## 2026-11-25 03:10 UTC — Wave 31 planned (top priority): resets vs no resets (6 runs, waiting on a runtime switch)
- **User:** compare resets with no resets allowed, on at least 5 levels where agents reset a few times; make it the top
  priority.
- **Reset census of strict runs** (`events.jsonl` reset ops). The completed Neon runs that reset often:
  - gemini-20261120: latch-out 27, hand smooth-mesh 21, cloth-holes 17, tetra-spring-k 11, marble-heavy-hint 9;
  - w26 cube-spring-block mjcf: 17–32.
- **Configs** `experiments/configs/wave31-20261125/` (6 runs): each is its Neon baseline's config with only these changes:
  - `[simulation] resets = false`;
  - for the hand, the task's "If it falls out of the hand, use reset…" clause removed (the starting-pose sentence kept);
  - the cost pool, and the 36 h provider-hold allowance (held time is off the run clock).
- **Baselines:** the same Neon runs, which kept resets.
- **Switch merged:** builder commit 335a3c3 (tests 63554593; OCI 63554758: no reset tool, reset rejected). The generator
  uses the builder's `noreset` option (cf7b8cf). No image rebuild was needed.
- **Prepared on netscratch** as `w31-<level>-noreset-20261125-001` (`prepare-w31.sh`). Scene sources are identical to the
  baselines'; the Panda copies also hold the new sibling `panda_small` levels.
- **Jobs at the very top:** `63556096` (latch-out + cloth-holes), `63556113` (hand + cube-spring-block-mjcf), `63556116`
  (marble + tetra-spring-k). Image `trusted-image-isolated-gear.json`.

## 2026-11-25 00:50 UTC — Lab quota killed the fixture-bimodal run; everything re-prepared on home and netscratch
- **Lab hit its shared group quota at about 00:00 UTC.**
  - It killed `w23-fixture-bimodal-neon-20261123-003` (kempner_cpu job `63471018`) at 231 run-minutes, after 1
    submission that scored `ok` (the first fixture evaluation on the isolated image). Infrastructure failure, truncated.
  - The maintenance move (`63470430`) had moved 13 finished runs (about 5 GB) from netscratch to lab before lab filled;
    it stopped itself on the quota error with each source intact. Moves into lab are paused until it has headroom.
- **All 23 queued jobs (37 runs, prepared on lab) were held before starting, then cancelled.**
- **Every run re-prepared as `<stem>-20261125-001`** (`prepare-20261125.sh`; mapping in
  `experiments/validation/held-20261125-map.txt`; all 40 sources byte-identical to the lab preparations):
  - top priority on **home** (55 GB free): wave 28 gear ×3, wave 23 fixture ×3, wave 24 die ×4;
  - the rest on **netscratch** (writable): waves 25, 26, 27 and 29, and Carbon.
  - Image: `trusted-image-isolated-gear.json`.
- **Jobs:**
  - Gear: `63530252`, running at 00:50 UTC; `63530256` and `63530318` at the top.
  - Fixture: rigid + rubber `63530413`; bimodal on kempner_cpu `63530440`, running.
  - Die: `63530345`, `63530376`, `63530378`, `63530407`.
  - Wave 27: `63530328`, `63530336`, `63530339`, `63530342`.
  - Wave 25: `63530236`, `63530242`, `63530245`, `63530249`.
  - Wave 26: `63530226`.
  - Wave 29: `63530205`, `63530208`, `63530210`, `63530217`.
  - Carbon (nice): `63530189`, `63530199`.
- The hold watcher covers all three roots.
- **01:45 UTC, home protected:** the fixture run writes about 1.4 GB per hour (not the assumed 1 GB per 6 h), and home
  dropped from 55 to 49 GB free in an hour. The die runs' audits may reach 32 GiB each, and home is the user's own quota.
  - The two running home runs stay put: gear-deep `63530252` and fixture-bimodal `63530440`.
  - The 7 queued home jobs were held and cancelled before starting. Their 8 runs were re-prepared on netscratch as
    `-20261125-002`, sources byte-identical (`prepare-20261125b.sh`, map `home-held-20261125-map.txt`).
  - New jobs at the top: gear cheap `63539120`, recon `63539134`; die `63539140`, `63539168`, `63539177`, `63539182`;
    fixture rigid + rubber `63539189`.

## 2026-11-24 20:30 UTC — Wave 28 gear-train (explanation depth) queued at top priority
- Merged builder commit 83ba220 (01304fa). Premise check `63463232`: deep vs shallow probe error 3% of the gate; the deep
  model costs 51× per simulated second.
- Record `trusted-image-isolated-gear.json`: agent `53de04c98fef`, evaluator `8e8a8af0c286`. OCI `63471353` 9 passed,
  1 skipped; non-OCI `63473067`.
- Runs `w28-gear-{recon,cheap,deep}-20261124-001`: Neon, 4 h, no checkpoint, point grasp with wrench, RGB-D, movable
  camera, sequence tool, the neutral task, and `grasp_slip_m = 0.002`. Lab root; sources identical across conditions.
- Jobs `63475962`, `63475964`, `63476006` (sapphire, 3 days), moved to the top of my queue ahead of fixture and die.
- Watch for: agents stalled by the grip slipping on over-pulls. The H4 trace classifier and the dashboard structure
  summary are deferred by the builder.

## 2026-11-24 19:40 UTC — Isolated image merged; waves 23, 24 and 27 queued on it, on the lab root (netscratch full again)
- **Agent-image isolation merged** (00001b4 = builder commit 8d4ba23).
  - Record `experiments/images/trusted-image-isolated.json`: agent `a82e2587b4ab` (`localhost/sim2sim-agent:latest`)
    and evaluator `0d41008fda1c`.
  - Builder checks: grep job 63461151 found no evaluation or metric modules in the agent image; OCI 63461413 (9 passed,
    1 skipped) and 63464604; the old single-image path still works (OCI 63468586, 49 passed).
  - Configs now carry the agent tag in `[sandbox] image`. Queued old-config jobs run unchanged.
- **Netscratch over quota again at 19:25 UTC** (writes fail; lab and home writable). The 9 running jobs on netscratch are
  at risk.
  - The maintenance move (`~/sim2sim-maintenance/move-to-lab.slurm 60`, job `63470430`, log in
    `experiments/validation/maintenance/`) moves finished, idle runs to lab, each verified by checksum before the source
    is removed, to free netscratch.
  - New preparations go to the lab root.
- **Re-prepared on the lab root** with the isolated configs; sources byte-identical to the -001 runs:
  - wave 23 fixture Neon as `-003`, wave 24 die Neon as `-003` (`prepare-isolated-20261124.sh`);
  - wave 27, all 8 runs, as `-20261124-001`.
- **Jobs** (`SIM2SIM_TRUSTED_IMAGE_RECORD=trusted-image-isolated.json`, `SIM2SIM_EXPERIMENTS_ROOT` = lab):
  - Fixture rigid + rubber `63470987`, die grasp, grasp-fps100, franka and franka-fps100 `63470991`, `63470999`,
    `63471006`, `63471013`; all moved to the top of my queue.
  - Fixture bimodal on kempner_cpu, `63471018`.
  - Wave 27 pairs: `63471034`, `63471039`, `63471055`, `63471059`.
- The hold watcher now checks both roots.

## 2026-11-24 19:50 UTC — Replicates stopped (user)
- User: replicates are the lowest priority, and none run until the user says so. The running second wave-26 replicate
  pair `w26-cube-spring-block-{lowrank,mjcf}-20261124-002` (job `63281861`, about 40 min in) was cancelled as a
  user-ordered stop. Wave 26 continues with n=1 per condition (`-20261124-001`, job `63281858`).
- Reruns of infrastructure-killed runs are not replicates and stay queued.

## 2026-11-24 19:00 UTC — Infrastructure deaths, a real-time bug, an agent-image leak; reruns; kempner_cpu; wave 27 merged
- **Netscratch quota, about 10:30 UTC:** "Disk quota exceeded" killed six runs started at 10:04 UTC. Their `status.json`
  may still say `running`. All are infrastructure failures.
  - w25-hand-prism (+control) -002, job `63264367`.
  - w23-fixture-rigid, -rubber -002, `63264372`.
  - w23-fixture-bimodal -002, `63264373`.
  - w24-die-grasp-neon -002, `63264374`.
  - Lab and netscratch were both writable again by 18:30 UTC.
- **Podman inspect/volume timeouts at start (18:07 UTC):** four jobs died within 4 min; their nodes are now excluded.
  - w25 smooth-mesh and spiky pairs, `63264370` and `63264371`.
  - w29 tetra-spring-block pair and w29 cube-spring-block-lowrank-prompt + w11, `63320577` and `63320578`.
  - Excluded nodes: holy8a26208, 26210, 28204, 28203.
- **Real-time bug:** w24 die fps100 grasp and franka -002 (`63264375`, `63264377`) failed with "event exceeds pre-action
  audit reservation" after about 3 run-minutes.
  - The real-time world kept stepping and recording through provider-held failed attempts: 166,907 frames against
    175 s of run clock and 1,966 s held.
  - Root cause (builder, corrected): a real-time `frames` reply of up to 100 frames was reserved as a single
    observation (about 3.5 MB), yet one reply was 4.8 to 5.6 MB. It fails at any frame rate once a call asks for
    about 70 frames or more.
  - The fix is merged into main (a3b8623, regression test jobs 63455271/63456054/63457376) and applies to jobs launched
    after it.
  - A separate semantic issue: the live world kept running during failed provider attempts, which are only credited
    back afterwards. Runner's decision: **B**, snapshot at each call and restore plus clock shift on a failure, so failed
    attempts are frozen as docs/protocol.md says. Merged into main (builder commit 4c73571; job 63459448, 62 passed).
    The `provider_hold` audit event records `discarded_ticks`/`discarded_frames`. Runs launched earlier keep the
    old behaviour.
- **Agent-image leak (designer and builder; user approved the fix):**
  - The trusted agent image installs all of `src/sim2sim`, so evaluator modules are readable in the agent container:
    `fixture.py` describes the hidden pad perturbation; `tabletop.py` names labels and prior-defeating dimensions.
  - No values or scores leak.
  - Fix: a split image, with the agent image holding only the harness, the client and what the public tools import
    (record `trusted-image-isolated.json`, pending).
  - **Waves 23, 24, 27 and 28 are held until it is verified.**
  - A source-read audit of all past runs is running (job `63454413`, `experiments/validation/source-read-audit.sh`).
- **Reruns on the current image** (levels without code-described hidden mechanisms):
  - w25 hand-prism, smooth-mesh and spiky pairs as `-003`: jobs `63454970`, `63454996`, `63455010`.
  - w29 tetra-spring-block pair as `-002`: `63455013`.
  - w29 cube-spring-block-lowrank-prompt `-002`: `63455018`.
  - w29 w11 tetra-spring-k-lowrank-prompt `-002` on **kempner_cpu**: `63455042`.
  - All sources byte-identical to their -001 runs (`prepare-20261124c.sh`).
- **kempner_cpu** (new): CPU-only jobs on Kempner H100 nodes; account `kempner_qianqian_lab`; QoS 8 CPUs / 32 GB per
  user, so one strict run at a time; 7-day limit. Allowed in `experiment-strict.slurm` and `test.slurm` (08b0703). OCI
  passed there (job `63452421`, 8 passed, 1 skipped).
- **Wave 27** (touch-tabletop, builder session) merged into main (1beccb4); it waits for the isolated image.
- **Running at 19:00 UTC** (started 18:07 UTC on image 42f70708fdcc):
  - w25-hand-prism-offset-com pair -002 and w24-die-franka-neon -002. The die run is exposed to the real-time bug, and
    the leak makes it suspect.
  - wave 26 reruns ×4, w29 tetra-spring and cube-spring pairs, and Carbon marble, hand, latch and w11.

## 2026-11-24 — Wave 29 queued: low-rank prompt on the spring-as-a-scale levels (8 runs)
- **Question (user):** does asking the agent to find the scene's important low-rank structure make it converge faster on the
  properties that matter (target mass, spring stiffness)? To be compared with the earlier runs for the report.
- **Factor:** one sentence appended to each original task: "First identify the important low-rank structure of this
  scene: the few physical properties that govern most of its behavior. Estimate those first, and make sure your model gets
  them right." It names no property. Everything else equals the originals.
- **Runs** (`experiments/configs/wave29-20261124/`, gemini-4-neon, 36 h provider-hold allowance, pool `gemini-20261124`):
  - Low-rank prompt on w20 tetra-spring, cube-spring, tetra-spring-block and cube-spring-block (6 h, 4 h checkpoint), and
    on w11 tetra-spring-k (the gemini-20261120 recipe: original wall, no checkpoint).
  - Neon baselines with the original prompt where none existed: w20 tetra-spring, cube-spring and tetra-spring-block.
  - Other baselines: cube-spring-block is `w26-cube-spring-block-mjcf` (reruns `-20261124-00{1,2}`); w11 is
    `w11-tetra-spring-k-neon-20261120-001`; the astra originals are the w20/w11 runs.
- **Sources:** byte-identical to the originals' (`prepare-w29.sh`). The only differences are private-side: the
  cube-spring-block manifest's unused `[lowrank]` table, and the new sibling `panda_small` levels in the w11 copy.
- **Jobs** (sapphire, 3 days, each prompted run paired with its baseline; image `42f70708fdcc`): see
  `experiments/validation/jobs-20261123.txt` (wave 29 lines).
- **Analysis plan:** per submission, the target mass (and COM) and the spring stiffness from the candidate MJCF, plus the
  `expert` weighing metric, against wall minutes and cumulative tokens. Measure the time to within 10% of the truth, prompted vs
  baselines (Neon and astra).

## 2026-11-24 03:10 UTC — Provider-hold allowance exhausted: two Carbon runs died; unstarted runs re-prepared with 36 h
- **Died (infrastructure, rerun):** `w19-marble-heavy-hint-carbon-20261123-001` (206 run-minutes) and
  `w21-hand-smooth-mesh-carbon-20261123-001` (193), job `63158114`, at about 02:30 UTC, with no submissions.
  - Google's 503 "high demand" outages had used up each run's 4 h provider-hold allowance (`held_seconds` ≈ 14,420 and
    14,716 s). After that, failed calls reached the harness, which exited: marble on an inference deadline, hand on a 503.
  - The surviving 2026-11-23 runs (Carbon cloth and w11, wave 26 ×4) have been held about as long and may die the same way;
    they continue as launched.
- **Allowance raised to 36 h** (`PROVIDER_HOLD = 129600` in `generate-waves.py`) for unstarted runs and reruns. Held time
  is outside the run clock, so the agent's conditions are unchanged.
  - The 10 held sapphire jobs of waves 23–25 were cancelled before starting. Their `-001` runs stay prepared and never
    started, because moving prepared run directories was not permitted.
  - Waves 23–25 were re-prepared as `-20261123-002` from the regenerated configs. Each source is byte-identical to its
    `-001` (job `63263855`).
- **Carbon reruns** (`experiments/configs/gemini-carbon-20261124/`, sources identical to the -001 runs):
  `w19-marble-heavy-hint-carbon-20261124-001`, `w21-hand-smooth-mesh-carbon-20261124-001`, and
  `w20-latch-out-carbon-20261124-001` (replaces the never-started -003, whose job `63210168` was cancelled).
- **03:50 UTC, four more died the same way** (`held_seconds` 14,425–14,792):
  - `w11-tetra-spring-k-carbon-20261123-001`: 237 run-minutes, 1 submission.
  - `w26-cube-spring-block-lowrank-20261123-001` (221 min) and `-002` (231 min): neither submitted.
  - `w26-cube-spring-block-mjcf-20261123-001`: 241 min, 1 submission.
  - All count as infrastructure failures (truncated). `mjcf-002` was resumed at its 4 h checkpoint, then died the same
    way at 04:30 UTC (312 run-minutes, 1 submission, held 14,855 s).
  - `w22-cloth-holes-carbon-20261123-001` **completed** at about 05:30 UTC (356 run-minutes, 644 real, 1 submission):
    the only full-length run of the 2026-11-23 launch. It is the first complete Carbon data point.
- **Sapphire backlog:** about 4,400 pending jobs and no start estimate for the 15 reruns. `test` (12 h cap) is too short
  with these outages; other partitions are not approved for strict runs.
- **Wave 26 rerun in full:** both conditions, n=2, from `experiments/configs/wave26-20261124/` (36 h allowance, image
  `d9cc983d1fc8`), so the low-rank vs MJCF pair stays matched. Runs `w26-cube-spring-block-{lowrank,mjcf}-20261124-00{1,2}`,
  jobs `63281858` and `63281861`.
- **Carbon w11 rerun:** `w11-tetra-spring-k-carbon-20261124-001`, job `63281862`.
- **Sources:** all byte-identical to the -20261123 runs (job `prepare-20261124b`).
- **Jobs** (sapphire, 3-day limit, 40 CPUs):
  - w25: `63264367`, `63264369`–`71`.
  - w23: `63264372`–`73`.
  - w24: `63264374`–`77`.
  - Carbon: `63264378`–`79`.

## 2026-11-23 22:50 UTC — Carbon latch-out cancelled for the user's dashboard (infrastructure, rerun queued)
- The results-explorer session (user-approved) cancelled job `63159055` to free a `test` slot for the user's live
  dashboard (job `63206812`). `w20-latch-out-carbon-20261123-001` died at about 56 min of run wall with no submission; its
  `status.json` still says `running`. This is a user-ordered infrastructure cancellation, not a result.
- A fresh `w20-latch-out-carbon-20261123-002` (same config, same job shape) is prepared and launched by
  `experiments/validation/relaunch-latch-carbon.sh` once `63206812` ends. That dashboard job ended after 7 min, so -002
  started (job `63208809`). At the user's request (relayed by the results-explorer session), it was cancelled after 3.4 min
  to give the dashboard a `test` slot again: another infrastructure cancellation. The user reserves one `test` slot for
  the dashboard until they release it (at most 4 strict jobs on `test`), so `-003` runs on sapphire: job `63210168`,
  prepared inside the job (`experiments/validation/prepare-and-run-strict.sh`), 2-day limit.
- **Google latency, 22:50 UTC:** the running Gemini runs' clocks advanced only 0.23–0.55× real time over their first
  3.7 h. Provider holds and slow successful calls (about 3–4 run-minutes per call) dominate, and there are no submissions
  yet. A 6 h run clock then needs 11–26 h real; `test` caps jobs at 12 h, and running jobs cannot be extended. The `test`
  runs continue: a run cut at 12 h is an infrastructure truncation, usable only up to the run minute it reached. The 10
  pending sapphire jobs (waves 23–25) were raised from 14 h to 2 days (`scontrol update`).

## 2026-11-23 18:40 UTC — Neon/Carbon batch launched: Carbon on the hard five, waves 23–26 (24 runs, queued)
- **Model policy (user):** `gemini-4-neon` is the default; Carbon is evaluated in parallel on the same five hard levels.
  All runs: provider holds (`provider_hold_seconds` 14400), accounting-only caps, netscratch root, sapphire, 40 CPUs per job
  (at most two jobs per node), bad nodes excluded. Jobs: `experiments/validation/jobs-20261123.txt`.
- **Carbon** (`experiments/configs/gemini-carbon-20261123/`, pool `gemini-carbon-20261123`): the gemini-20261120 recipe with
  `gemini-4-carbon` (Neon's prices as an accounting placeholder). Sources byte-identical to the Neon runs' (the w11/w19
  copies also hold the new sibling `panda_small` levels; scenes identical). Sapphire jobs `63147154`, `63147204`, `63147208` were cancelled while pending (3,600-job queue) and resubmitted on `test` one per node (`spread-test-20261123.sh`): `63157947`, `63158114`, `63159055` (nodes holy8a24402, 24502, 24302).
  The model id works: the first call (job `63157947`) succeeded and echoes `gemini-4-carbon`.
- **Wave 25** (analyst session's touch harness; user: "Yes, but on Neon"): the four w21 hand levels with
  `tools.local_controller` and `--bash-mode pausable`, each paired in one job with its control, the exact w21 config on
  Neon (these controls also repeat the original hand levels on Neon). Sources identical within each pair. Jobs
  `63147232`, `63147235`, `63147277`, `63147299`.
- **Wave 23** fixture (rigid, rubber, bimodal) and **wave 24** die (grasp/franka × 25/100 frames/s), Neon only; one die
  run per job. Jobs `63147301`, `63147304`, `63147306`, `63147309`, `63147313`, `63147314`. Image `42f70708fdcc` for
  everything above.
- **Wave 26** (user: low-rank model instead of MuJoCo): `cube_spring_block` (point grasp, no force sensing, block mass
  stated) with a `parameters.json` answer (cube mass, spring stiffness, cube COM and world inertia in the initial state),
  scored by the new `lowrank` metric (docs/evaluation.md), against the w20 full-MJCF config, both on Neon, n=2 each.
  New submission kind `parameters` with the predictor texts' estimate substitutions (`examples/sandbox_agent.py`).
  Tests: job `63148030` (36 passed). Image `d9cc983d1fc8…` (job `63148354`, `experiments/images/trusted-image-lowrank.json`;
  the default record still names `42f70708fdcc`). Moved from sapphire (`63149095`, `63149127`, cancelled) to `test`: `63158508`, `63158733` (one lowrank + one MJCF run each).

## 2026-11-23 18:45 UTC — Fixture and die branches merged into main; new trusted image (no runs)
- **Merged** `worktree-fixture-level` (wave 23 positioning fixture) and `worktree-die-level` (biased die, now
  **wave 24**: `experiments/configs/wave24-20261123/`, run ids `w24-die-*`, configs now `entrypoint = "predict.py"`).
  Main's edits made after both branches were copied (pausable bash, `local_controller`, `allowed_actions`) were kept.
- **One predictor runner** (`src/sim2sim/predictor.py`, call and real-time stream modes) and a `PREDICTOR_METRICS` lookup in
  the runtime; one predictor prompt set in the harness; the controller channel also works with `--bash-mode pausable`.
  Neither wave had launched, so their predictor-task prompt texts changed before any run.
- **Image** `42f70708fdcce29393307944b198a284da84d753abe54e853cf046715c42f4b2` (job `63142237`, archive in
  `experiments/images/` on home: the lab image directory hit its quota).
- **Tests** (Slurm `test`): full suite job `63139927` 327 passed, 10 skipped, 2 failed: the inspector's 10-minute check
  (40-minute job limit; passes in a 10-minute job) and a harness channel test bug, fixed; job `63142222` reran both files,
  36 passed. OCI job `63143283` (slurm_step): 8 passed (fixture and die predictor runs, touch harness), the tools-image
  test skipped. `fixture-check.py fixture_rigid` (job `63142622`) and `die-check.py` (job `63143385`) passed;
  outputs in `experiments/validation/merge-20261123/`.

## 2026-11-20 11:30 UTC — Both Gemini batches ended
- **`gemini-20261120`** (with the provider hold): all 10 `completed`, last at 11:25 UTC.
  - Run-clock minutes: w11 neon 217, hydrogen 135; cloth neon 201, hydrogen 256; marble 357/356; hand 354/355; latch neon
    149, hydrogen 349.
  - Provider holds absorbed the outages. For example, latch-hydrogen held about 77 min in its first 90 run-clock minutes.
  - No run hit its Slurm limit.
- **`gemini-20261119`** (no hold), 5 completed: latch-hydrogen 171 min, cloth-neon 182, w11-neon 228, marble-neon 354,
  latch-neon 354.
  - 5 failed on provider timeouts: hand-hydrogen 92 min and hand-neon 154 (both with no submission), w11-hydrogen 95,
    marble-hydrogen 293 (no submission) and cloth-hydrogen 353 (5 submissions).
  - Their scored checkpoints are kept as truncated in the comparison.
- **Spend** (host-priced; no cap): set 1 $252, set 2 $441; this includes the $2 reserve charged per timed-out call.
- **Comparison:** `experiments/validation/model-compare-20261119.py` → `model-compare-20261120.json`; page built by
  `model-compare-page.py` (text in `model-compare-meta.json`) → `experiments/reports/gemini-vs-astra-20261120.html`, published as
  https://claude.ai/artifact/Upe4QRrUr9Zn2o3iDmizzW.

## 2026-11-20 01:45 UTC — Google latency spike killed two Hydrogen runs; provider hold added; second Gemini batch launched
- **Failure:** at about 01:20 UTC every Hydrogen run timed out at the 300 s call limit together.
  - `w21-hand-smooth-mesh-hydrogen-20261119-001` hit the harness's 4 consecutive failures and ended `failed` at 92 min,
    with no submission. It is an infrastructure failure, not a model result.
  - Raw probes of one tiny request took 2–107 s on Hydrogen and 2 s to more than 290 s on Neon. All returned HTTP 200, so
    this is queueing at Google, not rate limiting.
- **Fix:** `[inference] provider_hold_seconds` (runtime.py, test `test_provider_outage_is_held_outside_the_run`).
  - It covers `ProviderUnavailable`: a provider-limited timeout, HTTP 429/5xx after the in-call retries, or a transport error.
  - The failed attempt's time is credited back. The backoff (15 s, doubling to 300 s) runs with the containers paused and
    the clocks stopped, then the identical request is retried.
  - The agent never sees the outage. A successful call's latency still counts, as astra's did. Audit event `provider_hold`.
- **Batch `gemini-20261120`:** the same 10 configs with `provider_hold_seconds = 14400` and accounting-only caps ($1e6).
  - Sources are identical to the astra originals, as before.
  - Jobs on sapphire (12 h / 10 h limits, excluding busy and bad nodes): `62236891`–`94`, `62236907`
    (`experiments/validation/gemini-20261120-jobs.txt`).
  - `w11-tetra-spring-k-hydrogen-20261119-001` also ended `failed` (01:27 UTC, 95 min) for the same reason. The 8 surviving
    `-20261119` runs continue without the hold, as outage-exposed attempts.
  - `w21-hand-smooth-mesh-neon-20261119-001` also ended `failed` (02:26 UTC, 154 min), after provider timeouts as well.
  - All 10 `-20261120` runs were running by 02:05 UTC.
- **Tests:** runtime + sandbox-agent tests passed, except 3 timing-sensitive tests that failed once on sapphire (job
  `62236240`). Those 3 passed on rerun, together with the hold tests (`62236522`, 5 passed).

## 2026-11-20 00:30 UTC — Gemini 4 Neon and Hydrogen vs gpt-6-astra on five hard levels (launched)
- **Question:** how do `gemini-4-neon` and `gemini-4-hydrogen` compare with the existing strict astra runs, at matched wall
  minutes and matched cumulative tokens?
- **Code:** `[inference] api = "gemini"` makes the host broker call Google's native `generateContent` with `GEMINI_API_KEY`
  (docs/protocol.md).
  - The transcript, labelled tool-turn transport, harness, image (`964c615842a1`), task, tools, observations and
    `max_output_tokens` (16384) are identical to astra's. `reasoning_effort = "medium"` becomes `thinkingLevel`.
  - Google reports no cost, so the host prices tokens at AI Studio Standard list prices (USD per 1M tokens, input /
    cached / output incl. thinking): Neon 3.00 / 0.15 / 15.00, Hydrogen 1.50 / 0.075 / 7.50.
  - Transient 429/5xx responses are retried host-side within the call deadline. Neon returned 503 "high demand" during
    the smoke test.
- **Configs:** `experiments/configs/gemini-20261119/`, built by generate-waves.py.
  - Each config equals its astra original except the model/route, the cost lines and no 4 h checkpoint hold. Each run gets
    its original's full wall time: 6 h, or 4 h for w11.
  - Caps: Neon $40, Hydrogen $20 per run, about astra's largest extended spend in tokens. Pool `gemini-20261119` is $200.
  - 2026-11-20 00:25 UTC, user: Gemini spend is unlimited; the key is rate-limited (200 RPM, 2M TPM, 100K RPD).
    - The orchestrator watcher answers any hold with resume at $1e6 caps (`experiments/validation/gemini-20261119-holds.log`).
    - A hold pauses containers and clocks, so this equals uncapped. The comparison stays at matched minutes/tokens (AGENTS.md).
  - First 30 min: per model ~4 RPM and ~150K TPM, far under the limits.
    - Some calls ended `MALFORMED_FUNCTION_CALL` (hand-neon 6 of 14; others 0–2), `MALFORMED_RESPONSE` (≤ 1) or
      `MAX_TOKENS` (≤ 4). These replies carry no tool call, so the harness sends its usual "No tool was called" nudge.
    - hand-neon had two 300 s call timeouts, retried by the harness (each charged a $2 reserve).
- **Prepared on netscratch:** sources are byte-identical to the astra originals. For w11 the only difference is newer
  sibling scene directories in the private `panda_small` folder.
- **Levels and astra baselines:**

  | level | astra baseline runs |
  |---|---|
  | w22-cloth-holes | `-20261116-001` |
  | w11-tetra-spring-k (Panda spring weighing) | `-20261106-001`, `-r2-20261107-001`, `-r3-20261107-001` (image `9bf221a1c5bc`, all stopped at $44) |
  | w21-hand-smooth-mesh | `-20261116-001`, `-r2-20261118-001` |
  | w19-marble-heavy-hint | `-20261114-002`, `-r2-20261118-001` |
  | w20-latch-out | `-20261116-001` |

- **Jobs** (test, two runs per job, ≤ 2 jobs per node): `experiments/validation/gemini-20261119-jobs.txt`, `62209660`,
  `-62`, `-63`, `-92`, `-94`. All 10 loaded the image and made Gemini calls within 2 min.
- **Tests:** suite job `62206530` had 309 passed and 3 failed. The 3 failures then passed in job `62208776` (76 passed).
  One was a test that treated `api` as an unknown setting. The other two came from running in a worktree.

## 2026-11-18 23:30 UTC — 2026-11-18 closed at $184.31; replicates complete except the cylinder
- **Finished:**
  - marble-equal-hint-r2-20261118-001: `finish` at 5.9 h, 6 submissions.
  - hand-smooth-mesh-r2-20261118-001: extended at 4 h, `finish` at 5.9 h, 7 submissions.
- **Spend 2026-11-18:** $35.03 (pool 17 after the boundary) + $149.28 (pool 18) = $184.31.
- **Replicate status after the 2026-11-16 hang losses:**
  - Lighting pairs, second replicates done: tetra (lit and unlit), cube (lit and unlit), msd-spring-visible (lit and
    unlit), spring-out (lit and unlit), cloth-base (lit and unlit).
  - Done: hinted marbles (heavy and equal), cloth colour cues (3), all 4 hand levels.
  - Still to redo: w18-cylinder-r2, on the next day with budget.

## 2026-11-18 17:35 UTC — Today's replicates done; a third node hung at podman init; fail-fast timeouts added
- **Finished:**
  - msd-spring-visible-lit-r2: `finish` at 2.0 h.
  - spring-out-r2: stopped at its $44 hold, 2.5 h.
  - hand-prism-r2 and hand-prism-offset-com-r2: extended at the 4 h checkpoint, stopped at 5.6–5.7 h when pool
    `sim2sim-20261117` reached its $200 cap.
  - marble-heavy-hint-r2: `finish` at 6 h, 7 submissions.
- **Spend 2026-11-18:** $130.17 by 17:20 UTC (post-boundary pool 17 $35.03 + pool 18 $95.14).
- **Filling the day** under the pool-18 cap of $161: marble-equal-hint-r2 and hand-smooth-mesh-r2.
  - The first job (`61866106`, test, holy8a26401) printed nothing for 6 min, stuck at the first podman call (the
    node-wide init `flock`). Cancelled.
  - Resubmitted as `61867881`, excluding holy8a26401, holy8a26105 and holy8a24208; it loaded the image at once on
    holy8a24301.
- **`scripts/podman-job.sh` now fails fast:** `timeout`s on the init `flock` (240 s) and `podman info` (120 s), and on
  `podman load` (900 s). Each prints the hostname, so a bad node shows up in minutes instead of 10–12 h.

## 2026-11-18 11:35 UTC — Strict jobs hung at image load on two sapphire nodes; relaunched on `test`
- **Failure:** the hand-replicate job hung before `trusted_image_id` (i.e. in `podman-job.sh` / `load_trusted_image`)
  until its time limit, twice: `61498406` on holy8a24208 (12 h) and `61675078` on holy8a26105 (10 h).
  - Both runs stayed `prepared`, at no cost.
  - An `srun --overlap` into the second job also hung.
- **Probe** `experiments/validation/image-load-probe.sh` (job `61799796`, test, holy8a24101): fine. The node-wide init
  lock was absent, sha256 took 5 s and `podman load` 8 s. So the hang is node- or state-dependent, possibly debris
  from the jobs killed on 2026-11-17.
- **Workaround:** run on `test` and check `trusted_image_id` within minutes of launch.
- **Relaunched:**
  - `61799979` (test, holy8a24301, 3 runs, 30 CPUs): hand-prism-r2 and hand-prism-offset-com-r2 (`-20261117`, pool
    `sim2sim-20261117`, ≤ $39 left), plus w19-marble-heavy-hint-r2-20261118-001.
  - `61799980` (test, holy8a24102): w20-spring-out-r2 and w20-msd-spring-visible-lit-r2 (`-20261118-001`), which
    complete those lighting pairs.
  - Both loaded the image.
  - Pool `sim2sim-20261118` is capped at $161, so the 2026-11-18 total across both pools stays ≤ $200.
- **Spend so far:** 2026-11-17 total $161.03; 2026-11-18 $0 before these runs.

## 2026-11-17 13:30 UTC — Tetra and cube lighting replicates done ($44 each, no extension); two hand replicates fill the day
- **Done:** tetra-spring, tetra-spring-lit, cube-spring and cube-spring-lit (`-r2-20261117-001`), each stopped at its $44
  hold at about 2.2–2.7 h. No extension under the $200 day. No hang: two jobs on two nodes.
- **Spend:** pool `sim2sim-20261117` at $161.03; the 2026-11-16 pool spent $0 after the boundary.
- **Filling the day:** the touch-only hand runs cost about $8–15 per 6 h. hand-prism-r2 and hand-prism-offset-com-r2
  (`-20261117-001`, job `61498406`) use the remaining ≤ $39. The pool cap of $200 is hard.

## 2026-11-17 10:45 UTC — 12 of 20 replicates hung (three strict jobs per node); 4 relaunched within the $200 day
- **Outcome:** 8 replicates completed (msd-spring-visible-r2, spring-out-lit-r2, hand-spiky-r2, cloth base and base-lit,
  quadrants, quadrants-colored, diagonal-colors). The other 12 stopped producing events 2–16 min in (22:10–22:24 UTC).
  Their jobs were killed at the time limit with the runs still marked `running`.
- **Where each run stopped:** after saving the first `load_reconstruction` checkpoint, i.e. when the controller pauses
  the containers and starts the candidate-change evaluator step. The evaluator step never appears in `sacct`, and no
  `candidate-change-0001` folder exists.
  - The wall watchdog killed the agent steps at 6 h, but the controller never wrote a termination.
- **Pattern, exact so far:**
  - every job on nodes running three strict jobs (six runs) hung: holy8a24302 (`61246660/62/63`, test) and holy8a24106
    (`61246696/97/98`, sapphire);
  - every job on nodes with two jobs (four runs) completed: holy8a24401, holy8a24108.
  - Suspects: the shared per-UID `RLIMIT_NPROC` (AGENTS.md) or rootless Podman contention across jobs on one node.
- **Lost:** w18-cylinder, w19 marble heavy and equal, cube-spring and -lit, msd-spring-visible-lit, spring-out,
  tetra-spring and -lit, hand prism, offset-com and smooth (all `-r2-20261116`). They spent almost nothing, since the
  hang came before further inference.
- **Relaunched within the 2026-11-17 cap:** `rep-20261117` = tetra-spring, tetra-spring-lit, cube-spring and
  cube-spring-lit (`-r2-20261117-001`).
  - Pool `sim2sim-20261117`, cap $200; 2026-11-16 pool spend since the boundary is $0.
  - Two jobs only (`61455403`, `61455417`). Each run stops at its $44 hold, with no extension today.
- **TODO:** keep at most two strict jobs per node until the cause is fixed, and make the controller time out on a
  stalled step launch.

## 2026-11-17 05:00 UTC — Day boundary: 2026-11-16 closed at $1414.50; $200 hard cap enforced for 2026-11-17
- **Pool `sim2sim-20261116`:** settled spend $1414.50 at 05:00 UTC. About $585 of the $2000 day went unused because the
  replicate jobs waited on fairshare priority for 0.5–2.5 h.
- **Reserve `zz-orchestrator-reserve-20261117.json` = $385.50:** sized on settled cost, so the pool (cost + in-flight +
  reserve) reaches its $2000 cap once 2026-11-17 spend reaches $200, including any in-flight reservations. Runs then
  hold ("pool exhausted") and are stopped. Exclude `zz-orchestrator-*` from spend analyses.
- **Reviews of the 20 replicates so far:**
  - Stopped at 4 h: msd-spring-visible-r2 (61% appearance notes, final checks), spring-out-lit-r2 (mechanism exact,
    now the table), cloth-quadrants-diagonal-colors-r2 (shadow tuning).
  - Extended: spring-out-lit-r2 at its cap (2.5 h), cloth-base-lit-r2, hand-spiky-r2 and cloth-quadrants-r2 at 4 h.

## 2026-11-16 21:30 UTC — 20 second replicates launched to use the $2000 day; $200/day hard cap from 2026-11-17
- **User:** use the expanded budget fully today, with repeats where needed. From 2026-11-17, a hard $200 per ET day.
- **Replicates** (`experiments/configs/rep-20261116/`, `-r2`): identical to the originals except pool `sim2sim-20261116`
  with a $2000 cap.
  - Lighting pairs (10): spring-out, tetra-spring, cube-spring, msd-spring-visible and cloth-base, each with its `-lit`
    twin.
  - Box levels whose agents found the contents (3): w19 marble heavy and equal hint, w18 cylinder.
  - Cloth colour cues (3): quadrants, colored, diagonal colors.
  - Touch-only hand (4).
- **Jobs:** 10 jobs, 2 runs each (`experiments/validation/rep-20261116-jobs.txt`). They were pending on fairshare
  priority, so 5 were moved to `test` (12 h limit, that partition's 5-job cap).
- **Pool** `sim2sim-20261116` stood at about $1090 at launch.
- **Day boundary:**
  - At 05:00 UTC, `experiments/validation/day-boundary-reserve.py` adds
    `cost-ledger/sim2sim-20261116/zz-orchestrator-reserve-20261117.json`, sized so the pool exhausts after $200 more.
    Every run then holds, and the cap is never raised past that.
  - It is scheduled as a background job of the orchestrator session. If that session is gone, write the file by hand.
  - Spend analyses must exclude `zz-orchestrator-*` files.

## 2026-11-16 20:10 UTC — w20-msd-spring-visible-001 invalid (controller hang caused by the hand merge); relaunched -002
- **Run `-001`** stopped producing events at 882 s wall (07:01 UTC), after noting "the mechanics are now well
  identified: a 0.5 kg undamped carriage".
- **Cause:** the controller spawns a child for each inference call, and the child re-imports `sim2sim.runtime` from disk.
  It did so in the seconds between applying the hand patch (`runtime.py` imports `sim2sim.robots`) and copying
  `robots.py`, so the child died with `ModuleNotFoundError`. The controller then hung for 13 h and the wall watchdog
  never fired. Job `61027263` was cancelled.
- **Fix:** `scripts/experiment-strict.slurm` freezes `src/sim2sim` into a job-private temp dir on `PYTHONPATH`, so later
  edits cannot reach a running controller. Root-cause hardening is in the TODO.
- **Relaunched** as `-002` (job `61223326`) on the current image `964c615842a1`. Its lit twin ran on `cb84098cc84e`; the
  images differ only in robot, cloth and contract code, which this level does not use.

## 2026-11-16 14:05 UTC — Waves 20–22 hold reviews so far; early lighting-ablation signal
- **Rule:** at a $44 spend hold or the 4 h checkpoint, extend (cap $120, pool $2000) while the agent works on physics or
  the hidden mechanism; stop when it is on appearance, lighting or final validation.
- **Stopped:**
  - w20-msd-damper-visible (2.7 h): mechanics exact, impact traces within 2e-9 N; then rendering.
  - w20-tetra-spring (2.7 h): lighting; it only planned to weigh the cluster.
  - w20-tetra-spring-block (3.0 h): lighting; never used the block.
  - w20-tetra-spring-lit (3.4 h): had weighed the cluster on the spring; then spring strand shape.
  - w22-cloth-quadrants (3.8 h): four-patch model, residuals at floating-point level.
  - w22-cloth-base (4.25 h): 56% of notes on lighting, headlight fitting at the hold.
- **Extended:**
  - w20-spring-out and -lit:
    - both found a 0.15 kg plunger and 200 N/m, exact;
    - both were extended twice: at the cap, then at 4 h.
  - w20-latch-out (cap, then 4 h): spring and hard stop; probing state dependence.
  - w20-cube-spring and -lit (cap, then 4 h): cube 48 g (true 48.1), plate 8 g (true 8), extension stop 30 mm (true).
  - w20-cube-spring-block (cap, then 4 h): calibrated k with the block; plate 11.93 g (true 12).
  - All four w21-hand runs at 4 h: still fitting shape; $8–10 each.
  - w22: diagonal colors (cap, then 4 h; eight bending regions), both centre-hole levels (the stiff-rim run found
    "stiffness concentrated near the opening"), quadrants colored, holes (only 8% of notes on appearance).
- **Lighting ablation, early signal from progress notes (scoring to follow):**
  - **Cloth base:** the unlit run had 56% lighting/appearance notes and was stopped at 4.25 h on the headlight. The lit
    run finished by itself at 3.2 h with 16% such notes, reporting exact force and image matches in every test.
  - **Tetra spring:** the unlit run never weighed the cluster (lighting at its hold). The lit run weighed it, and lasted
    longer on physics (3.4 h).
  - **Spring-out:** no difference so far; both found the mechanism exactly.

## 2026-11-16 UTC — Wave 21 (Shadow hand, touch only) launched; cloth levels merged (wave 22 pending image)
- **Dex hand** (built in worktree `agent-a0d08f5326e924483`, merged into main):
  - The Shadow Dexterous Hand E3M5 (right) from MuJoCo Menagerie at pinned commit `71f066ad`, vendored at
    `environments/shadow_hand/robot/`, palm up.
  - New `src/sim2sim/robots.py` registry; `robot = "shadow_hand"`; `hand_target` with 20 actuator targets.
  - Proprioception: 24 joint positions and velocities, 20 efforts.
  - Tactile: 3×3 grids on each fingertip and a 4×3 grid on the palm.
  - Levels:
    - `hand_prism`: 64×44×30 mm, 42 g;
    - `hand_prism_offset_com`: COM offset (8.9, 4.5, 0) mm;
    - `hand_smooth_mesh`: convex, radius of curvature ≥ 15 mm;
    - `hand_spiky`: two cones.
  - Metrics: chamfer, center_of_mass and dynamics (squeeze and tilt probes, new tactile error). Chamfer and COM now
    accept convex mesh geoms.
  - Model constraint: primitives plus inline convex meshes, fixture unchanged.
  - Check job `61031422`; full suite 308 passed (the inspector expiry assertion fails only in jobs longer than 10 min,
    and passes in a short job).
  - Also fixed: Panda tactile and proprioception contract text had been nested under `reference_interaction = false`,
    so earlier Panda runs never received it.
  - Image `09065bb09510` (build `61032468`, inspection clean); OCI 4 passed, 1 skipped (`61033082`).
  - **Wave 21:** the 4 hand levels with plain `TASK_CONVERGE` plus the reset sentence, `sequence` tool, no images, 6 h
    with the 4 h checkpoint. Jobs `61033320`, `61033322`.
- **Cloth** (built in worktree `agent-a23eddd4c60eb1534`, merged into main):
  - "RoboSim studio" = RoboSimStudio (arXiv 2610.18835): its simplest default cloth `rss.sheet` is a particle grid cut
    into triangles. It is reproduced as a MuJoCo 2D flex: vertex bodies on 3 slide joints, a stiff edge equality for
    stretch, `elastic2d="bend"` for bending; one flex per stiffness region, with seams that act as hinges.
  - Point grasps attach to flex vertices.
  - New `src/sim2sim/cloth.py` metric `cloth`:
    - a 4×4 lift stiffness map;
    - 3 bar drops and 2 cone drops, with per-frame Chamfer and side-by-side mp4s;
    - rest Chamfer.
  - Levels: base, quadrants (E0/4/16/64), quadrants with quadrant colours, quadrants with diagonal colours, holes and
    cuts, centre hole with a stiff rim, centre hole with a soft rim. Check jobs `61032288`–`61032295`.
  - Merge: conflicts in `simulation.py` (graspable-body selection, robot init) resolved by hand.
  - Full suite 310 passed (the same expiry assertion as above; a short job passes it).
  - **Wave 22 launched** (7 levels plus `cloth-base-lit`, point grasp with wrench, RGBD, movable camera, `sequence`,
    cloth model constraint, evaluation timeout 600 s) on image `964c615842a1` (build `61059506`, inspection clean, OCI
    4 passed / 1 skipped `61060611`).
    - Jobs: `61061435` (base-lit, base), `61061438` (soft rim, stiff rim), `61061446` (holes, quadrants-colored),
      `61061447` (quadrants-diagonal-colors, quadrants).
    - Pool $447 at launch.
- **Hold:** w20-spring-out was held at $40 (2.4 h). It had found the spring-loaded stop with an estimated 0.15 kg moving
  mass (true plunger mass 0.15 kg); resumed to a $120 cap, pool $2000. Pool `sim2sim-20261116` stood at $421.

## 2026-11-16 UTC — Wave 20 part 2 launched (point-grasp tetra/cube + ideal spring); budget $2000/day for 11-16, 11-17
- **Levels** `{tetra,cube}_spring{,_block}` (`tetra-spring-scenes.py` point-grasp outputs, check job `61030423`):
  - the panda_small tetra-spring layout without the robot;
  - spring k = 21.7 N/m with damping exactly 0, and stiffer contacts (`solref .004 1`), so free oscillation loses only
    2–3% in 10 s;
  - cube: 5 cm, 0.0481 kg; cluster 0.0572 kg; block 0.0235 kg;
  - weighing on the spring matches m·g/k to 0.03 mm.
  - Metrics: chamfer, center_of_mass, density, image_mse, `dynamics` (lift-swing-drop and spring-hang probes) and
    `expert` (scripted weighing). The evaluation timeout is 600 s (new generator option `eval_timeout`).
- **Prompts:**
  - `TASK_CONVERGE`, plus that the cluster's (or the unknown-mass cube's) inertial properties are important to identify;
  - the spring is ideal (Hooke's law only, no damping), with k stated in the spring levels and the block mass stated
    instead in the block levels;
  - `-lit` twins of tetra-spring and cube-spring add the ground-truth lighting.
- **Conditions:** point grasp without force sensing, RGB + depth, movable camera, `sequence`; 6 h with the 4 h checkpoint.
- **Jobs:**
  - `61031382`: tetra-spring, tetra-spring-lit;
  - `61031384`: tetra-spring-block, cube-spring;
  - `61031386`: cube-spring-lit, cube-spring-block.
- **Budget (user):** $2000 per ET day on 2026-11-16 and 2026-11-17. The generator gives pools `sim2sim-20261116/17` a
  $2000 cap. The 8 part-1 runs were prepared with $1000 and are raised at their holds.
- **Found by the level builder, not changed:** the point-grasp evaluator ignores manifest `object_geoms` and uses every
  non-world geom. So the hidden plunger geoms of the point-grasp tunnel levels count in their Chamfer (waves 16, 17,
  20). Changing it would alter completed scores.

## 2026-11-16 UTC — Wave 20 launched (part 1: outside-start tunnels, 1D spring–mass(–damper), lighting ablation)
- **User request (2026-11-16):**
  - tunnel spring and latch with the rod starting outside the channel;
  - point-grasp tetra/cube spring levels (part 2, pending);
  - 1D mass on an ideal spring that looks / does not look like one, without / with a damper;
  - a ground-truth-lighting twin of each family's basic level;
  - "4 h extendable to 6 h" for all.
- **New infrastructure:** `[inference] checkpoint_wall_seconds`. The agent sees the full 6 h. At 4 h wall the next
  inference call holds exactly like a spend-cap hold (`hold.json`, reason "Wall-time checkpoint reached"), and the
  orchestrator resumes it to 6 h or stops it.
  - Test `test_wall_checkpoint_holds_once_and_stops_on_decision`; runtime and harness tests 44 passed.
  - Host-side only, so image `cb84098cc84e` is unchanged.
- **Levels:**
  - `tunnel_{spring,latch}_out` (`tunnel-scenes.py`, check job `61026358`):
    - the stick starts 240 mm out, in an open frictionless trough that continues as the bore floor;
    - the lip is widened to 29 mm, so nothing is captive;
    - in the probe k = 200.0 / 199.3 N/m, onset at 30 mm, and the latch toggles 18.83 / 0 / 18.83 mm;
    - the mechanism is invisible to camera, depth and grasp rays;
    - the probe inserts 0.24 m first, then repeats the original strokes.
  - `msd_{spring,damper}_{visible,hidden}` (`msd-scenes.py`, check job `61025818`):
    - a 0.5 kg block on a frictionless slide, k = 32 N/m, period 0.785 s; the damper levels add b = 0.8 N·s/m (ζ 0.10);
    - the visible levels have a tendon-cage coil, plus a dashpot in the damper level;
    - the hidden levels are a closed "linear stage" and are pixel-identical to each other;
    - force_profile probe: slow and fast two-sided cycles.
- **Conditions:**
  - All runs: point grasp with wrench, RGB + depth, `sequence` tool, plain `TASK_CONVERGE`.
  - Tunnels use the fixed camera; spring–mass levels use the movable camera.
  - The `-lit` twins (w20-spring-out-lit, w20-msd-spring-visible-lit) add the exact `<light>` elements and headlight
    (`generate-waves.py` `lighting()`).
- **Jobs:**
  - `61027261`: latch-out, msd-damper-hidden;
  - `61027262`: msd-damper-visible, msd-spring-hidden;
  - `61027263`: msd-spring-visible-lit, msd-spring-visible;
  - `61027284`: spring-out-lit, spring-out.
  - Pool `sim2sim-20261116`.

## 2026-11-15 00:30 UTC — Wave 19 complete: with the hint, both marble agents inferred the ball from loads alone
- **marble-equal-hint-003 is valid.**
  - Finished by `finish` at 5.9 h, 7 submissions, on image `cb84098cc84e`.
  - Its interior camera request was rejected (8643 s). It compared one sphere, two spheres and a loose block against
    tilt, shake and pressure-pulse data.
  - Final model: a free sphere of 40 g (true 40) in a 40 g shell (true 40), radius 12 mm (true 9.5), friction 0.3
    (true 0.3).
  - Composite-CoM mean / final error in mm: forward 1/4, side 22/81, toss 4/24, tilt 2/6. Compare the solid box 14/30,
    24/65, 11/24, 12/16 and wave 18 unhinted 18/20, 35/86, 15/11, 12/29.
- **Valid box-level results:**
  - The cylinder (unhinted) and both hinted marbles identified a moving internal body and beat the empty-box baseline
    on most probes.
  - The unhinted marble models are worse than the empty box.
  - The side throw stays hard for every model.
- **Invalid runs:**
  - `w19-*-001` (near-plane clipping);
  - `w19-marble-equal-hint-002` (box carried around a stationary camera).

## 2026-11-14 19:00 UTC — Wave 19 -002: heavy valid and inferred the ball; equal invalid (box carried around the camera)
- **marble-heavy-hint-002 is valid.**
  - Finished by `finish` at 5.9 h, 7 submissions. Its interior camera placements were rejected (notes at 62–1478 s).
  - From settled CoM shifts under controlled tilts, its final model has a free sphere: 67.3 g (true 68 g), radius
    11.6 mm (true 9.5 mm), friction 0.32 (true 0.3).
  - Composite-CoM mean / final error in mm: forward 13/27, side 6/14, toss 10/32, tilt 3/9. Compare the solid box
    24/39, 13/13, 12/9, 22/29 and wave 18 unhinted 34/36, 25/22, 33/44, 21/43.
- **marble-equal-hint-002 is invalid.**
  - At 6185 s: "position the camera at a clear location and move the held box toward it". It carried the box until the
    camera sat inside the cavity, then measured the sphere radius (9.5 mm), the 4 mm walls and the sphere colour from
    "interior views".
  - The aim-time check could not stop this: objects move after the camera is placed.
- **Second fix** (`Simulation.observe`): whenever any surface is within the camera clearance of the observation camera
  at render time, the image is black and the depth is zero, like a real lens covered by an object.
  - This also covers carrying an object to the fixed camera. It uses the same `_surface_within` ray test, now
    `mj_multiRay`, as the aim check.
  - The wrist camera is exempt: the hand's own geoms surround it. It was not enabled in these runs.
  - The test covers a box moved to 1 cm from a clear-aimed camera (black RGB, zero depth). 66 passed (job `60554164`).
- **Wave 18 unaffected:** no cavity, clipping or carry-to-camera notes in any of its three runs.
- **Image `cb84098cc84e`:** build job `60554519`, inspection clean; OCI 4 passed, 1 skipped (job `60555357`'s check job
  `60555092`).
- **Relaunched** as `w19-marble-equal-hint-20261114-003` (job `60555357`). Pool `sim2sim-20261114` stood at $137 at
  relaunch.

## 2026-11-14 13:00 UTC — Wave 19 -001 invalid (camera clipped through the box); fixed and relaunched as -002
- **What happened:** both hinted runs saw inside the sealed box within about 1 min.
  - They parked the movable camera against the box, and the near clip plane (0.01 × extent ≈ 2 cm) cut its 4 mm wall.
  - Notes: "The cavity view reveals a spherical internal weight" (heavy); "a sphere intersected by the camera's near
    plane" (equal).
  - None of the three unhinted wave 18 runs did this (no clipping or interior notes), so wave 18 stands.
- **Runs `-001` are invalid:**
  - equal was stopped at its hold (3.8 h, `budget_suspended`);
  - heavy was cancelled with its job, `60451443`;
  - spend $74.59 in pool `sim2sim-20261114`.
- **Fix** (`Simulation._aim_camera`):
  - The rule: `camera` rejects any `pos` closer than 1.5 × the near-plane frustum corner reach to any surface.
  - How: 512 Fibonacci-sphere rays, all geom groups. `mj_ray` also catches positions inside a geom, or inside a small
    closed cavity.
  - For the box levels the minimum clearance is 3.4 cm.
  - Test `test_movable_camera_cannot_clip_into_a_sealed_box`; simulation, Panda and wrist-camera tests 41 passed.
  - Documented in `docs/protocol.md`.
- **Image `b573545561d7`:** build job `60478907`, inspection clean; OCI 4 passed, 1 skipped (job `60479478`).
- **Relaunched** as `w19-marble-{heavy,equal}-hint-20261114-002`, job `60479567`, with the same configs.

## 2026-11-14 08:30 UTC — Wave 18 results; wave 19 (hinted marble boxes) launched
- **Wave 18 outcomes:**
  - marble-heavy: `finish` at 6.0 h, 7 submissions, never held.
  - cylinder: `finish` at 5.9 h, 8 submissions.
  - marble-equal: held at the cap at 5.4 h and was stopped. It had noticed an off-centre CoM (modelled as fixed offset
    inertia) and briefly tested a hollow-box collision model, but never considered a moving body; it was on shadow
    artifacts when it held.
- **Discovery:**
  - Only the cylinder agent found the contents. From load-centre shifts in tilted grasps it built a fore–aft internal
    slider, and checked that this was not a pose-estimation artifact. Its model slides rather than rolls.
  - Neither marble agent identified a body inside.
- **Composite-CoM error vs a "nothing inside" baseline** (`experiments/validation/box-baseline.py`): a solid uniform 80 g
  box with the level's exterior; results in `wave18-analysis/baseline/box-baseline.json`. Mean error in mm, final model /
  baseline, for forward / side / toss / tilt-shake:

  | Level | Forward | Side | Toss | Tilt-shake |
  |---|---|---|---|---|
  | cylinder | 7 / 21 | 31 / 32 | 8 / 18 | 8 / 15 |
  | marble-heavy | 34 / 24 | 25 / 13 | 33 / 12 | 21 / 22 |
  | marble-equal | 18 / 14 | 35 / 24 | 15 / 11 | 12 / 12 |

  - The cylinder slider is 2–3× better than the empty box on three probes.
  - Both marble models are worse than the empty box: their contact and friction fits compensate for the hidden ball.
  - Total mass was exact (80 g) in every final model. Chamfer is 2.6–2.8 mm, dominated by the hidden body's surface.
- **Wave 19** = the two marble levels rerun with the hint "The closed box is not solid: it has hidden contents that can
  move inside it when the box moves; your reconstruction must reproduce them too."
  - Otherwise identical to wave 18 (`RUNS21`), on image `8887bc5f3292`; only `inspector.py` changed after the build.
  - Job `60451443`, pool `sim2sim-20261114`.

## 2026-11-13 UTC — Wave 18 (sealed boxes with hidden contents) launched
- **Question (user, 2026-11-13):** can an agent reconstruct an object whose geometry is hidden inside a sealed, grippable
  box?
- **Levels** (`environments/panda_small/box_*`, written by `experiments/validation/box-scenes.py`, checked by `box-check.py`;
  check jobs `60337517`, `60336828`):
  - **Box:** outer 57 × 82 × 42 mm, identical in every level; 80 g total; shell friction 1.5.
  - **Friction model:** elliptic cones with impratio 10. The fixture gripper squeezes only about 1.4 N per finger, and with
    the default cones a heavy marble crept out of the grip.
  - **Contents:**
    - `box_marble_heavy`: a 19 mm ball of 68 g in a 12 g shell. The interior is 49 × 74 mm, 2.6× and 3.9× the diameter,
      so the ball rolls.
    - `box_marble_equal`: the same ball, 40 g in a 40 g shell.
    - `box_cylinder`: a 19 mm × 48.4 mm cylinder of 55 g in a 25 g shell. 0.6 mm end play, so it rolls along y only.
  - The hidden body collides only with invisible liners. Hard impacts press it up to about 2 mm into them; it is hidden
    behind 4 mm walls.
- **Conditions:** Franka with joint torques, movable camera, `sequence` tool (code-written action lists), and plain
  `TASK_CONVERGE` with nothing about the contents. 6 h wall, $44 cap with the 36 h hold.
- **Evaluation:**
  - Four arm probes, the same in every level: `throw_forward`, `throw_side`, `toss_up`, `tilt_shake`.
  - Each probe starts with a pick. Release speeds are 0.9–1.2 m/s and every box lands on the table.
  - New in `dynamics`: the composite CoM trajectory of all non-robot free bodies, sampled every 10 ticks, with errors and
    a PNG per probe. It is model-agnostic.
  - Between levels the mean CoM paths differ by 6–30 mm; source vs itself scores 0.
  - Chamfer and `center_of_mass` include the hidden body.
- **Inspector:**
  - Registry-based property editor (opacity, mass, friction; a new property is one entry).
  - Live composite and per-body CoM markers.
  - Tests: inspector job `60335253` passed 22. `test_dashboard_physics_tab_with_live_backend` then timed out twice
    (jobs `60338257`, `60339218`). Fixed in the test: Chromium's virtual time stalls with live Viser viewers, so the check now polls in real time. Inspector and dashboard tests 47 passed (job `60359870`).
- **Validation:**
  - Affected tests: 142 passed and 2 failed (job `60338257`). The runtime timing test then passed on rerun (job
    `60339218`).
  - Image `8887bc5f3292` (build job `60338360`); OCI test 4 passed, 1 skipped (job `60339219`).
- **Jobs:** `60340049` (marble-heavy, marble-equal) and `60340071` (cylinder), sapphire. Pool `sim2sim-20261113`, worst
  case $132 before holds.

## 2026-11-11 23:59 UTC — Wave 17 results: the hint makes the hidden mechanism identifiable
- **All 8 hinted runs found the mechanism**; in wave 16 none of these 8 conditions had one in the final model.
  - Seven ended by `finish` near the 6 h wall after being resumed at the $44 cap.
- **Orchestrator error:** w17-nonlinear-grasp-hint was never reviewed and timed out after 36 h, stopping at 3.1 h.
  - It held 48 s after another run, and the restarted `hold-watch.sh` seeded "seen" from holds already present.
  - Fixed: the watcher now reports any hold without a written decision.
  - The condition is rerun as `-002` (job `59831896`, pool `sim2sim-20261110`):
    - resumed at the cap (2.4 h, while testing a wrong "stateful latch" hypothesis), then finished at 5.9 h with 20
      submissions;
    - its final model has all three zones (linear, the 2 N plateau, the quadratic rise), at 0.15 / 2.2 N;
    - use `-002` for the hinted nonlinear point-grasp condition and keep `-001` as cut short.
- **Probe rule changes:** the probe now attaches at time 0, before `settle_ticks`, and grasps the nearest movable pixel
  within `grasp_radius_px` = 3 of `grasp_pixel`.
  - Why: two hinted curved candidates sat ~1 px off the reference handle, or drifted off it while settling.
  - Effect: all 206 submissions of waves 16–17 now score (job `59840356`).
  - The trusted image still has the earlier probe; rebuild it before any further tunnel wave.
- **Summary** (`experiments/validation/tunnel-summary.py`; `wave17-analysis/tunnel-summary.{json,md}`): final-model RMS
  error over the 30–59 mm working range vs the source magnitude there.
  - **Hinted runs:**
    - damper (both) 0.00–0.03 / 3.4 N, including the fast (damped) cycle;
    - nonlinear-seq 0.01–0.02 / 2.2 N;
    - spring 0.06–0.12 / 3.3 N;
    - nonlinear (3.1 h) 0.2 / 2.2 N;
    - curved 0.4–2.2 / 3.2 N.
  - **Latch-seq hinted:** retract 0.48 / 3.2 N. It reproduces the latch, holding at 0 N until about 48 mm (plot). Its
    push error of 6.75 N is one spurious 48 N spike in the final version; versions 004–005 had 0.86 N overall.
  - **Wave 16 for contrast:**
    - spring-grasp-seq 0.1 and damper-franka 0.14–0.16 are the only near-exact unhinted models;
    - latch-grasp and latch-franka have an exact spring and no latch (1.2 N);
    - the six no-mechanism models score about their source magnitude;
    - curved-grasp is a 77 N collision artifact.
- **Plots:** `experiments/validation/wave1{6,7}-analysis/force-profiles/`. Engagement: `wave17-analysis/engagement.jsonl`
  (every hinted run hit the plunger within 20–340 s, except curved-grasp-hint at 1754 s).

## 2026-11-10 07:45 UTC — Wave 17 (hinted tunnel reruns) launched; wave 16 force profiles
- **Force-profile probe fixed and rescored** (`experiments/validation/rescore-force-profile.py`, jobs `59440676`,
  `59451247`):
  - **What was wrong:** the in-run probe used a fixed world ray and missed most candidates, because point-grasp agents
    build in shifted frames.
  - **Fix 1:** the probe now grasps through a trusted-camera pixel (`grasp_pixel`, emitted by `tunnel-scenes.py`; scenes
    byte-identical).
  - **Fix 2:** `Simulation` collected default graspable bodies from the first `<worldbody>` only. Franka agents include
    the robot first, which hid their bodies; the fix only affects researcher point grasps on Panda scenes.
  - **Median error dropped:** free travel dominates the samples, so a model with no mechanism scored 0.
  - **Coverage:** all 18 finished runs, every submission `ok`.
  - **Plots:** `experiments/validation/wave16-analysis/force-profiles/`. The hard-stop spike (~35 N at 61 mm)
    dominates the RMS, so read the plots.
- **Final model vs truth (slow push at 45 mm / 58 mm, fast push at 45 mm; true spring 2.94 / 5.53 N):**
  - **Exact or close:**
    - spring-grasp-seq: exact spring.
    - latch-grasp: exact spring, no latch (its retract branch is wrong).
    - damper-franka: spring and damping (fast 4.54 vs 4.58 N).
    - damper-franka-seq: spring, damping too low.
  - **Mechanism present, magnitude wrong:**
    - spring-franka, nonlinear-franka: too stiff.
    - spring-franka-seq, nonlinear-franka-seq: too soft.
    - latch-franka-seq: 20 vs 5.6 N.
    - curved-franka-seq: wrong shape.
  - **No mechanism:**
    - zero force: nonlinear-grasp, nonlinear-grasp-seq, curved-grasp-seq, latch-grasp-seq, damper-grasp,
      damper-grasp-seq.
    - geometry artifacts only: spring-grasp (force before the plunger), curved-grasp (340 N spike).
  - latch-franka (finished 6 h, job `59477038`): spring close (2.79 / 5.32 N), no latch.
  - curved-franka: mechanism present but wrong shape (2.10 / 2.32 N; 1.75 N already at 20 mm).
- **Plunger engagement from audit states** (`wave16-analysis/engagement.jsonl`):
  - Never touched: spring-grasp, nonlinear-grasp, damper-grasp, damper-grasp-seq.
  - Every Franka run engaged it.
  - curved-grasp(-seq) and latch-grasp-seq touched it within minutes without noticing.
  - Soft hard stop: compression reached 40–60 mm on a 30 mm stroke.
- **Wave 17** = the 8 "no mechanism" conditions rerun with the hint "The closed housing hides a mechanism at its far end,
  out of view, that the visible parts can interact with; your reconstruction must reproduce its behavior too."
  - Otherwise identical to wave 16 (`experiments/configs/generate-waves.py` `RUNS19`).
  - Image `b986371a2e38`: build job `59451970`, inspection clean, OCI 4 passed / 1 skipped (job `59453491`).
  - Jobs `59454676`, `59454690`, `59454713`, `59454717` (sapphire, 2 runs each).
  - Pool `sim2sim-20261110`; worst case $352 before holds.
  - Agent prompts were checked: run IDs and level names never reach the agent.

## 2026-11-09 08:25 UTC – 2026-11-10 06:12 UTC — Wave 16 hold reviews (in progress)
- **Rule:** at the $44 cap a run held. I read its model-origin progress notes and, where they were unclear, its audit
  states.
  - **Resumed** (cap $120, pool $1000, same run) if the agent had engaged the mechanism and its model of it was still
    incomplete.
  - **Stopped** otherwise.
- **Stopped (7), all `budget_suspended`:**
  - Six never pushed past the free travel. They spent their time pulling the stick out and fitting its collars, inertia
    and the entrance lip: nonlinear-grasp (2.5 h), curved-grasp-seq (2.7 h), curved-grasp (2.8 h), damper-grasp (2.8 h),
    damper-grasp-seq (2.9 h), spring-grasp (3.5 h).
  - latch-grasp-seq (3.3 h) pushed 100 mm in at ~4400 s. Its audit states show the plunger fully compressed and the latch
    toggled, but the agent called it "a strong stop near the housing entrance" and never modelled it.
  - Scene note from that push: under the stiff point grasp the plunger overshot its 30 mm limit to 39 mm (soft joint
    limit, `solreflimit .004`). The force-profile probe stays at 61 mm and slow, so the metric does not reach this.
- **Resumed (7):**
  - latch-grasp: a spring-loaded internal stop found by push + release at ~5160 s.
  - nonlinear-franka: an inward-only self-returning slider, then fitting the onset of nonlinear resistance.
  - spring-grasp-seq: first inward probe at 4.9 h, which showed a mismatch at the far end.
  - spring-franka: a compliant "upper stop" with rebound.
  - damper-franka: soft axial stops fitted to push-and-release.
  - latch-franka and curved-franka: compliant inner stop and spring-like return. Both were held ~20 h while the
    orchestrator session was down; the hold excluded that time.
- **Never held (6):** spring-franka-seq, nonlinear-grasp-seq, nonlinear-franka-seq, latch-franka-seq, damper-franka-seq
  and curved-franka-seq. They finished under $44, the last at the 6 h wall.
- **Pool spend** at 2026-11-10 06:11 UTC: $808.53 (all charged to ET 2026-11-09 except the final two resumes).
- **New tool:** `experiments/validation/mechanism-engagement.py RUN...` prints each run's maximum plunger compression from
  the audit states, for the results report.

## 2026-11-09 05:51 UTC — Wave 16 (occluded tunnel levels) launched
- **Question (user, 2026-11-08):** can an agent use a stick to identify a mechanism it cannot see, at the closed far
  end of a tunnel? The task is plain `TASK_CONVERGE` and never mentions a tunnel, spring or mechanism.
- **Levels** (`environments/tunnel_*`, `environments/panda_small/tunnel_*`, written by
  `experiments/validation/tunnel-scenes.py`; check job `59098236`, plots in `experiments/validation/tunnel-check/r6/`):
  - All levels: a frictionless bore, a 0.38 kg stick with a knob, a 30 mm gap, then a plunger with a 30 mm stroke to a
    hard stop.
  - **spring:** k = 200 N/m, no damping.
  - **nonlinear:** linear to 10 mm, a 2 N plateau to 20 mm, then plateau + 4e4·(x−0.02)² N. Built from deadband
    tendons; within 1% of the design.
  - **curved:** tunnel and stick are an arc of radius 0.25 m.
  - **latch:** a push-push heart cam that latches at 18.8 mm. Reliable over 3 cycles at 10, 50 and 100 mm/s.
  - **damper:** k = 200 N/m plus b = 20 N·s/m.
- **Design:** 5 levels × {point grasp + wrench, Franka + joint torques} × {without, with the new `sequence` tool} =
  20 runs, `w16-<level>-<grasp|franka>[-seq]-20261109-001`.
- **Budgets:** 6 h wall budget, visible to the agent from the start, and a $44 cap.
  - New `budget_hold_seconds` = 36 h: at the cap the run holds frozen for review, then either resumes with a raised cap
    in the same run or stops.
  - Review rules are in experiment-todo.md: rerun with a hint if a run never discovers the mechanism; continue to 6 h
    if runs are still working on it.
  - `audit_bytes` = 16 GiB.
- **Infrastructure changes:**
  - Budget hold:
    - `WallClock.hold` freezes the run, and hold time is excluded from elapsed time.
    - The orchestrator answers through `hold-decision.json`.
    - Inference replies carry `deadline_unix`, which the harness adopts.
  - `[tools] sequence`: the harness runs an agent-written JSON list of up to 2000 physical actions in one world and saves
    each step's non-image outputs.
  - Metric `force_profile` (`src/sim2sim/force_profile.py`) gives source vs candidate resistance along the push path;
    plot with `experiments/validation/plot-force-profiles.py`.
  - **Wrench fix (affects all point-grasp runs from now on):**
    - `Simulation._wrench` now reports the force applied in the last step.
    - It used to re-solve after integration, which read low by about 10 N·s/m × velocity: phantom damping, found while
      building these levels.
    - Earlier wrench runs used the old reading; do not compare their damping estimates with new runs.
- **Validation:**
  - Focused tests: 65 passed (job `59089897`), then 46 passed after the wrench fix (job `59098223`).
  - OCI integration on the intermediate image `6a02a145af5f`: 4 passed, 1 skipped (tools image not opted in; job
    `59092028`).
  - Final image `c50bab6efa8f` (build job `59098387`, inspection clean).
- **Jobs:** `59099991`, `59100005`, `59100009`, `59100011`–`59100017` on sapphire, 2 runs each, 44 h limit. List in
  `experiments/validation/wave16-20261109-jobs.txt`.
- **Budget:** pool `sim2sim-20261109`, initial worst case 20 × $44 = $880.

## 2026-11-07 13:45 UTC — Wave 15 results: SAM-3D use required, no detectable effect
- **Both runs finished** at the $44 cap (`budget_suspended`): boat 5 submissions in 116 min, assembly 5 in 108 min
  (jobs `58615374`/`58615376`). Services `58615373`/`58615375` were released at run end. Scored with tag `-M`.
- **The agents called the tool:** boat 10 `sam3d_call` events, assembly 11, all `ok` (8–20 s each). Most calls came in
  the first 15 minutes, one per part, as instructed. The assembly agent later used a SAM-3D sphere estimate as a
  hypothesis and then checked it with depth measurements.
- **Final Chamfer, no other differences (mass and COM exact except w12 assembly):**

  | Scene | w10 base | w12 offered (0 calls) | w12 directed (0 calls) | w14 movable camera | w15 use required |
  |---|---|---|---|---|---|
  | Boat | 12.5 mm (15.9 at matched tokens) | 11.9 mm | 10.0 mm | 13.8 mm | 14.7 mm |
  | Assembly | 13.9 mm (13.8 at matched tokens) | — | 19.7 mm (COM 26 mm) | 12.7 mm | 12.6 mm |

- **Conclusion:** conditions that differ only in an unused tool already span 10–20 mm, so that is the run-to-run noise at n=1. Calling SAM-3D does not
  measurably change shape accuracy on these scenes. The agents still build primitive unions, so the meshes serve only as
  hints. No replicates queued.
- Pool `sim2sim-20261107`: $482 billed.

## 2026-11-07 11:38 UTC — Wave 15 (SAM-3D use required) queued (done; see 13:45)
- **Question:** wave 12 agents never called `sam3d`. Does the tool help when the task requires calling it?
- **Runs:** `w15-boat-sam3d-required` and `w15-assembly-sam3d-required`.
  - Each equals the w12 directed config except the task sentence: "Before modeling each part, call the sam3d action on a
    camera image with a mask of that part ..., and compare the returned 3D shape with your observations before using
    it." No scene information is added.
  - Sources are identical to `w10-*-base`.
- **Setup:** image `9bf221a1c5bc` (current main, contains the relay). One A100 service per run on
  `kempner_gpu_priority` (`58615373`, `58615375`, 6 h, never H100/H200); strict jobs `58615374`/`58615376` start 5 min
  after their service.
- Pool `sim2sim-20261107` (about $400 billed), worst case +$88. The waiter releases the services when the runs end. List
  in `experiments/validation/wave15-20261107-jobs.txt`.

## 2026-11-07 14:30 UTC — Camera r3 results: movable vs fixed camera on spring levels, n=3
- **All four r3 runs finished** (jobs `58594552`/`53`), each at the $44 cap. Scored with tag `-L`.
- **Final submissions, r1 / r2 / r3:**

  | level | Chamfer, movable | Chamfer, fixed | COM, movable | COM, fixed |
  |---|---|---|---|---|
  | K stated | 0.5 / 0.69 / 4.11 mm | 3.6 / 3.46 / 3.57 mm | 10.3 / 30.3 / 39.3 mm | 22.2 / 13.1 / 48.9 mm |
  | block | 0.7 / 1.03 / 1.03 mm | 1.3 / 3.93 / 3.85 mm | 7.1 / 33.7 / 6.7 mm | 6.7 / 7.3 / 7.2 mm |

  - **Block-level stiffness ratio:** movable ×1.01 / ×1.15 / ×0.92, fixed ×1.38 / ×0.27 / ×4.61.
  - **Exact plate masses (0.012 / 0.008):** movable 4/6 runs, fixed 1/6.
- **Conclusions (n=3):**
  - The movable camera improves geometry: median Chamfer 0.69 vs 3.57 mm (K) and 1.03 vs 3.85 mm (block); better in 5/6
    pairs.
  - It also makes the spring identification reliable. With the block mass stated, stiffness was within 15% in 3/3
    movable runs and off by 38% to 4.6× in 3/3 fixed runs.
  - The cluster COM is not improved (mixed). No run resolved the 1:2:4:8 sphere masses.
- Pool `sim2sim-20261107`: about $400 billed.

## 2026-11-07 10:40 UTC — Wave 14 and camera r2 results; camera r3 launched
- **All six runs finished** (jobs `58553798`/`99`/`801`), each at the $44 per-run cap (about $40 billed). Scored with
  tag `-K`.
- **Geometry scenes** (movable camera vs w10 baseline, n=1). The movecam runs reached the cap faster, with 4 submissions
  each.
  - Boat: final Chamfer 13.8 mm at 16.5M tokens. Baseline 15.9 mm at matched tokens, 12.5 mm final.
  - Assembly: final Chamfer 12.65 mm, COM 0.4 mm at 13.7M tokens. Baseline final 13.94 mm.
  - Both are within the run-to-run spread (the three wave 12 boats ranged 10.0–12.5 mm). **No camera effect is
    detectable on the geometry scenes.**
- **Spring levels, movable vs fixed, final submissions (r1 → r2):**

  | level | Chamfer, movable | Chamfer, fixed | COM, movable | COM, fixed | block-level stiffness, movable | fixed |
  |---|---|---|---|---|---|---|
  | K stated | 0.5, 0.69 mm | 3.6, 3.46 mm | 10.3, 30.3 mm | 22.2, 13.1 mm | — | — |
  | block | 0.7, 1.03 mm | 1.3, 3.93 mm | 7.1, 33.7 mm | 6.7, 7.3 mm | ×1.01, ×1.15 | ×1.38, ×0.27 |

  - Chamfer is better with the movable camera in 4/4 pairs, and so is the block-level stiffness in 2/2.
  - Masses and COM are mixed: the r2 movable runs missed the masses (k-r2 plates ×1.3/×1.2, cluster ×2.1). **The exact
    plate masses of w11-k did not reproduce.**
- **r3 of all four camera conditions launched** (`experiments/configs/rep3-20261107/`). Jobs `58594552`/`58594553`; list
  in `experiments/validation/rep3-20261107-jobs.txt`. Pool `sim2sim-20261107`: $241 billed before launch, worst case
  +$176.

## 2026-11-07 06:30 UTC — Wave 14 (movable camera on geometry scenes) and camera replicates launched
- Pool `sim2sim-20261107` ($1000), $44 per run, convergence prompt, image `9bf221a1c5bc`, netscratch root. Worst case
  $264.
- **Wave 14:** `w14-boat-movecam` and `w14-assembly-movecam`. Each is identical to its `w10-*-base` config except for
  `movable_camera = true` (diffed) and the pool lines. Sources are identical to w10.
- **Replicates (n=2) of the camera pair on the spring levels:** `w11-tetra-spring-{k,block}-r2` and
  `w13-tetra-spring-{k,block}-fixedcam-r2` (`experiments/configs/rep-20261107/`).
- Jobs `58553798`/`58553799`/`58553801` (sapphire); list in `experiments/validation/wave14-rep-20261107-jobs.txt`.
- The agent-visible `public_tool_config` was checked: movable camera on/off exactly as intended; effort and wrench
  unchanged from each twin.

## 2026-11-07 06:00 UTC — Wave 13 results: fixed camera vs movable camera (spring levels, n=1 each)
- **Wave 13 finished.** Job `58492344` COMPLETED. Both runs stopped at the $44 per-run cap: `k-fixedcam` at 190 min with
  10 submissions, `block-fixedcam` at 181 min with 7. Scored with tag `-J`.
- **Scoring-script fixes.** `report-extract-models.py`:
  - Joint-axis inertia is now computed for hinges only. An agent's slide-joint body had child bodies, and the lever-only
    assertion failed.
  - Per-geom masses compile with `boundmass`/`boundinertia`, so a massless geom alone still compiles. This replaces the
    02:30 special case.
  - `compare.py` skips hinge-only quantities on slide joints. Wave 11 was rescored with the same script (`-I`); its
    results are unchanged.
- **Final submissions, movable (w11) → fixed (w13):**

  | level | Chamfer | COM | cluster mass | plates (0.012 / 0.008) | stiffness (21.7) |
  |---|---|---|---|---|---|
  | K stated | 0.5 → 3.6 mm | 10.3 → 22.2 mm | ×1.05 → ×1.40 | exact → exact | stated |
  | block mass stated | 0.7 → 1.3 mm | 7.1 → 6.7 mm | ×1.00 → ×1.00 | exact → 0.010 / 0.010 | ×1.01 → ×1.38 |

  - At matched tokens the ranking is the same.
  - At 120 min both conditions were still far from their final values: cluster mass ×1.4–2.1.
- **Reading:** the movable camera helped geometry (Chamfer) in both levels. In the block level it also helped the spring
  identification: equal plates and a stiffness 38% too high without it. This is n=1 per condition; replicate before
  citing.

## 2026-11-07 02:30 UTC — Wave 11 (spring levels) results; wave 13 (fixed-camera twins) launched
- **Wave 11 finished.** Jobs `58441029`/`58441031` COMPLETED. All four runs stopped at the $44 per-run cap after
  138–152 min, with 7–9 submissions each. Scored with tag `-I` (job `58490059`).
  - The first attempt (`58489539`) crashed: `report-extract-models.py` recompiled the zero-mass coil core alone. Massless
    geoms are now recorded as 0 without recompiling.
- **Final submissions** (masses in kg; source cluster 0.0572 = spheres 0.0038/0.0076/0.0153/0.0305; plates
  0.012/0.008; K 21.7):

  | run | Chamfer | COM | cluster | plates | other | stiffness |
  |---|---|---|---|---|---|---|
  | `k` | 0.5 mm | 10.3 mm | 0.060 (×1.05), 4 equal spheres | 0.012 / 0.008 exact | — | 21.7 (stated) |
  | `k-square` | 2.8 mm | 31.8 mm | 0.100 (×1.75), equal | 0.012 / 0.008 exact | — | 21.7 (stated) |
  | `k-plug` | 0.7 mm | 5.7 mm | 0.0572 exact total, equal | 0.012 / 0.008 exact | plug 0.0318 (true 0.0317) | 21.7 (stated) |
  | `block` | 0.7 mm | 7.1 mm | 0.0572 exact total, equal | 0.012 / 0.008 exact | block 0.0235 (stated) | 22.0 (×1.014) |

- **The exact plate masses were checked for a leak; there is none.**
  - The `k` agent tracked the free two-plate oscillation in images and fitted it with the known K: reduced mass 0.0048,
    split 0.4, hence 0.012 / 0.008.
  - It also used the Panda's joint-angle sag under load ("the robot's known position-controller deflection") as a scale.
    The `k-plug` agent weighed the plug the same way.
  - **Caveat for every Franka "no force sensing" ablation:** hiding `actuator_effort_nm` does not remove load sensing,
    because the position controller's steady-state joint deflection encodes the payload.
- **No agent seated the spring in the hole** (no insertion attempt in any progress note).
- **No agent resolved the 1:2:4:8 sphere masses** of the cluster: all used equal spheres. This is the remaining COM error.
- **Camera use:** 37–51 `camera` actions per run, for face-on plate measurements and orthogonal views.
- **Wave 13** (pool `sim2sim-20261106`, $622 spent before launch, worst case +$88): `w13-tetra-spring-k-fixedcam` and
  `w13-tetra-spring-block-fixedcam`.
  - Each is identical to its w11 twin except that `movable_camera` is absent. Configs were diffed; sources are identical
    to w11.
  - Job `58492344` (sapphire); list in `experiments/validation/wave13-20261106-jobs.txt`.

## 2026-11-07 01:40 UTC — Wave 12 (SAM-3D) results: the tool was never called; baseline scoring fix
- **Wave 12 finished.** Slurm jobs `58391601`/`58391610`/`58391614` all COMPLETED; the GPU services were released.

  | run | end | submissions | wall | final Chamfer | final COM | baseline (w10) final |
  |---|---|---|---|---|---|---|
  | `w12-boat-sam3d` | per-run cap ($44) | 7 | 139 min | 11.9 mm | 0 | boat 12.5 mm, COM 0 |
  | `w12-boat-sam3d-directed` | per-run cap ($44) | 7 | 100 min | 10.0 mm | 0 | boat 12.5 mm, COM 0 |
  | `w12-assembly-sam3d-directed` | `agent_finish` | 7 | 234 min | 19.7 mm | 26.1 mm | assembly 13.9 mm, COM 0 |

- **SAM-3D calls: 0 in all three runs.**
  - Counted from the audit payloads (`sam3d_call` events).
  - The `sam3d` tool definition was in every model request (about 690 blobs per run).
  - The agents' progress notes never mention it, including the two runs whose task says when it helps.
  - Observation PNGs are saved under `/workspace/observations/`, so inputs were available.
  - So the score differences are run-to-run variation, not a tool effect. Assembly is already known to be noisy at n=1
    (COM 4–28 mm across three baselines).
- **Pairs added to `rerun-analysis/compare.py`** (index tag `-H`, job `58474114`).
- **Trivial-baseline scoring:**
  - Jobs `58373539`–`41`: 53/56 ok.
  - The empty baseline failed on the w10 boat, assembly and tetra runs ("object has no positive compiled mass"). Policy
    transfer flings the object, and an empty scene has none.
  - Fix in `sim2sim/baselines.py`: the empty baseline drops `policy_transfer`, which becomes a gap.
  - Rerun `58474633`: all 6 ok.

## 2026-11-06 23:05 UTC — Research handoff PDF (document build only)
- User requested a self-contained PDF summarizing human information-acquisition studies and instructions for further experiment-design research.
- Source: `docs/research/sim2sim-research-handoff-20261106.tex`; build: `experiments/validation/build-research-handoff.slurm` on CPU `test` (1 CPU, 1 GiB, 5 min).
- No simulation, model calls, or experimental batch is planned by this document task.
- Built in jobs `58436665` and `58437038` (final pagination fix). Final PDF: 7 pages, 253,419 bytes; all seven sections retained, link annotations present, and rendered pages visually checked. No LaTeX overflow warnings. Output: `docs/research/sim2sim-research-handoff-20261106.pdf`.

## 2026-11-06 21:00 UTC — SAM-3D validated end to end; wave 12 (first SAM-3D runs) queued
- **SAM-3D test, first attempt (`58373463`) failed:** the model loaded (70.8 s), but the test sent the upstream full-size
  example photo. The relay correctly rejected it as over the agents' 4 MiB PNG limit. The test then didn't stop its service
  thread, so the job idled to its 1 h limit. Fixed in `scripts/sam3d-smoke.slurm`: the image is downscaled to 1024 px
  longest side (mask nearest-neighbour), and the service is stopped in a `finally`.
- **SAM-3D test, rerun (`58390179`, A100, holygpu8a19204) passed:**
  - pipeline loaded in 44 s
  - one request through the spool relay took 36.8 s and returned 19,964 vertices and 40,000 faces
  - camera-frame z −1.94..−1.74 (in front of the camera, as expected)
  - `sam3d-lock.json` written with 228 sha256-pinned files (checkpoints, HF cache and torch cache)
  - Caveat: the first load fetched the DINOv2 hub code (GitHub `main`, unpinned) and its 1.1 GB weights into the netscratch
    torch cache. That cache is now hash-locked; the service verifies the lock and runs with HF offline.
- **Wave 12** (pool `sim2sim-20261106`, $1000; convergence prompt; 4 h; image `trusted-image-sam3d.json`
  `5e99630de861`; netscratch root; jobs in `experiments/validation/wave12-20261106-jobs.txt`):

  | run | factor | baseline |
  |---|---|---|
  | `w12-boat-sam3d` | `sam3d` action available | `w10-boat-base` |
  | `w12-boat-sam3d-directed` | + `SAM3D_HINT` (task says when it helps) | `w12-boat-sam3d` |
  | `w12-assembly-sam3d-directed` | tool + hint | `w10-assembly-base` (assembly is noisy at n=1) |

  - Metrics match the w10 baselines (including friction, expert and policy_transfer, so the evaluation load is the same).
    Prepared sources equal the baselines' (`diff -r`). Config diffs: only the tool, hint and pool.
  - Each run has its own A100 service on `kempner_gpu_priority` (jobs `58391593`, `58391607`, `58391613`; 6 h) and its
    own spool, so runs never queue on each other. The strict jobs start 5 min after their service (`after:+5`).
  - The baselines ran on image `5904fd183fa4`. The sam3d harness change only adds the action when it is enabled.

## 2026-11-06 23:20 UTC — Wave 11 (spring levels) launched; new ablation option `movecam`
- **User:** start the spring levels; let the agent move the camera for these; add camera movement as an ablation option;
  use the stop-when-done prompt (`TASK_CONVERGE`, already in all four configs).
- **New option** `[simulation] movable_camera = true` (`generate-waves.py` option `movecam`):
  - adds a controller action `camera {world, pos, lookat}` that re-aims that world's observation camera (up = +z;
    coordinates within ±4 m; at least 0.05 m apart) and returns an observation
  - no physics time or ticks pass; the pose persists across resets
  - evaluation is unaffected: it builds fresh simulations with the trusted camera
  - it enters the public config and task only when set, so other configs are byte-identical
  - code: `Simulation._aim_camera` (`mj_camlight`), candidate-worker allowlist, runtime contract and routing, harness tool schema
- **Validation:**
  - unit test `tests/test_simulation.py::test_movable_camera…` (pose and forward exact, image changes, no ticks or time,
    persists across reset, bounds, opt-in)
  - full suite `58437252` passed (282 passed, 13 skipped)
  - new main image `9bf221a1c5bc` (`58437250`; previous record kept as `trusted-image-a9482d3ea278.json`)
  - strict OCI `58440163` passed (4 passed, 1 skipped), including the new `MovableCameraOCITests`: the in-container agent
    re-aims both worlds, the returned poses match, and 0 ticks are charged. The first attempt (`58439288`) failed only
    because the test read the observation at the wrong key.
- **Wave 11** (pool `sim2sim-20261106`, at $443.62 before launch; worst case +$176; 4 h; convergence prompt; image
  `trusted-image.json` `9bf221a1c5bc`; netscratch; jobs `58441029`, `58441031` in
  `experiments/validation/wave11-20261106-jobs.txt`):
  - `w11-tetra-spring-k` (round plates, stiffness stated)
  - `w11-tetra-spring-k-square` (square plates, stiffness stated)
  - `w11-tetra-spring-k-plug` (plug of unstated mass in the hole, stiffness stated)
  - `w11-tetra-spring-block` (ideal spring + block mass stated)
  - all: Franka, camera on and movable, `wrench = false`, `effort = false`
  - The agent-visible `public_tool_config` was checked: `effort` false, `movable_camera` true, `wrench` false. Within a
    minute all four agents were using the camera (top and axis-aligned views).
- Wave 12: both boat runs ended (17:33 and 18:11 ET), and their GPU services were released. The assembly run continues.

## 2026-11-07 01:40 UTC — Live dashboard draws visual tendons (spring coil)
- **User:** the coil was missing from the live dashboard. The Viser inspector drew only geoms, never tendons.
- **Change:**
  - `inspector_worker.describe` lists the visible spatial tendons (color and width; fixed tendons such as the Panda's finger
    coupling are excluded). `Session.state` sends each one's polyline: MuJoCo's wrap points, with a cylinder wrap's arc
    filled in as a helix and a sphere wrap's as a great circle.
  - `inspector.Viewer` adds one line-segments node per tendon and moves its points with every state.
- **Checks (srun on `test`):**
  - paths end exactly on the plate sites, and the arc points lie on the 0.039 m core (170° sweep)
  - the hidden core geom is not sent to the viewer
  - a Viser viewer builds four tendon nodes that follow the physics state
  - `tests/test_inspector.py` + `tests/test_dashboard.py`: 45 passed (`58419120`)
- `scripts/serve-dashboard.sh` default roots now include netscratch. The running dashboard (job `58409684`, started
  16:48 ET) has the old viewer code loaded and must be restarted.

## 2026-11-07 01:00 UTC — Spring levels: helical coil visual (user review 4)
- **User:** show visually that this is a spring, without changing any collision properties.
- **Change:** four visual-only tendon strands, from bottom-plate angle 90k° to top-plate angle 90k°+170°. Each wraps
  around an invisible core cylinder on the bottom plate (`spring_coil_core`: `mass="0"`, `contype`/`conaffinity` 0,
  group 3, transparent), so it follows a helix. The result looks like a four-start coil whose pitch tightens under
  compression.
- **Why collision is unchanged:** the core is not in `object_geoms`, so no metric sees it. Tendons do not collide. All
  physics checks give the same numbers as before (masses, flush 0.16 mm, compression 26.4 mm, dim-1 plate–table contacts),
  and the core appears in no contact.
- The four wave-11 runs were re-prepared (sources byte-identical). **Held for the user's review.**

## 2026-11-07 00:30 UTC — Spring levels: square-plate and plug variants; slight damping (user review 3)
- **User:**
  - add square plates whose diagonal equals the round diameter, as a separate level, so the hole fit is less obvious
  - add a level with a cylindrical plug of unknown mass starting in the hole
  - these are the stiffness-stated levels
  - give every spring very slight damping so it settles
- **Levels** (`LEVELS` in `experiments/validation/tetra-spring-scenes.py`; renders in `experiments/validation/tetra-spring-panda/`):

  | level | plates | extra | task statement |
  |---|---|---|---|
  | `tetra_spring` | round, 13 cm | — | stiffness exactly 21.7 N/m |
  | `tetra_spring_square` | square, 9.19 cm sides (13 cm diagonal; fits the hole at any yaw) | — | stiffness exactly 21.7 N/m |
  | `tetra_spring_plug` | round | teal cylinder in the hole, 7.2 cm wide (graspable) × 13 cm tall (4 cm above the table), 0.0317 kg, mass not stated | stiffness exactly 21.7 N/m |
  | `tetra_spring_block` | round | purple block, 0.0235 kg | ideal spring; block mass exactly 0.0235 kg |

  - All springs: damping 0.06 N·s/m, a damping ratio of about 3% with the cluster on top. The cluster's bounce decays from
    ±25.7 mm to ±0.7 mm in 7 s. The statements say "two plates" (was "two round plates").
  - The task text still says "ideal", by the user's earlier wording, although the spring now has this slight damping.
  - On the square plates the cluster footprint (0.052 m) slightly exceeds the 0.046 m half-side along the axes. It still
    rests stably; the sphere centers are 0.027 m from the axis.
- **Checks (srun on `test`), all four levels:**
  - no robot–object penetration at reset
  - top plate flush within 0.16 mm in the hole
  - cluster compression 26.4 mm (expected 25.9)
  - plate–table contacts are frictionless (dim 1)
  - the spring slides down the hole wall to the floor
  - the plug stays upright and centered in the hole
- **Wave 11** is now four configs; differences between them are only the scene and its statement. All four are prepared on
  netscratch with sources byte-identical to the environments. **Held for the user's review.**

## 2026-11-06 23:30 UTC — Spring levels: rigid column removed (user review 2)
- **User:** the rigid piece between the plates must go; the cluster's weight compresses the spring into it. The tube was
  6 cm long in a 7.8 cm gap, so it reached the top plate after 1.8 cm of compression (the cluster's static compression is
  2.6 cm).
- **Now:**
  - Between the plates are eight visual-only coil strands: spatial tendons with no collision and no mass, bottom-plate site
    k to top-plate site k+2 (a quarter-turn twist), 3.5 mm wide. The spring stays massless and ideal.
  - The slide range allows compression until the plates are 2 mm apart (−72 mm).
  - Plates are 8 mm thick. They are frictionless only against the table and hole, through masks (table 2/3, plates 4/1)
    plus 58 explicit `condim = 1` pairs. Against the gripper, cluster and block they have normal friction, so the Franka can
    pinch a rim and the cluster no longer slides on the plate.
- **Checks:**
  - no robot–object penetration at reset
  - in the hole the top plate is 0.17 mm below flush
  - the cluster compresses the spring 25.8 mm (expected 25.9) with an undamped ±24 mm bounce, which stays clear of the stop
  - plate–table contacts are dim 1 only; plate–cluster contacts are dim 3
  - the spring slides down the hole wall to the floor
- Renders are in `experiments/validation/tetra-spring-panda/`. Wave 11 was re-prepared from the new sources (byte-identical);
  configs and stated values are unchanged. **Held for the user's review.**

## 2026-11-06 22:40 UTC — Spring levels rebuilt for the Franka (user review); `effort` option; new main image
- **User review of the first spring levels:**
  - empty between the plates
  - should be manipulable by the Franka
  - hole not visible
  - start the spring on its side

  The point-grasp versions (`environments/tetra_spring*`) and their prepared runs were deleted, unrun.
- **New levels:** `environments/panda_small/tetra_spring{,_block}` (generator `experiments/validation/tetra-spring-scenes.py`,
  checks `tetra-spring-check.py`, renders and close-ups in `experiments/validation/tetra-spring-panda/`).
  - **Base:** the `panda_small` tetrahedral cluster (10 cm, 0.0572 kg), Franka arm and table. The round hole is cut into the table
    top: 13.6 cm wide, 9 cm deep. Its geometry is a 24-gon of hidden tangent collision boxes inside a square frame, plus one
    visual-only ring mesh.
  - **Spring:**
    - stiffness 21.7 N/m, no damping
    - frictionless 13 cm plates (the cluster footprint is 10.4 cm)
    - a telescoping column (3.2 cm orange tube on the bottom plate, yellow rod on the top plate) with normal friction, so the
      gripper (8 cm opening) can hold it
    - lies on its side at the start
  - **Level 2:** adds a 3 cm, 0.0235 kg purple block.
  - **Checks (srun on `test`):**
    - no robot–object penetration at reset
    - the lying spring stays put (2.7 mm settle)
    - upright in the hole the top plate is 0.17 mm below flush (undamped oscillation ±0.56 mm)
    - the cluster on the spring compresses it 26.3 mm (expected 25.9 mm), no overhang (0.052 vs 0.065 m)
    - plate contacts are condim 1, and the spring slides down the hole wall to the floor
  - The fixed camera is moved closer than `panda_small`'s so the hole and spring are visible at 320×240.
- **No force sensing on the Franka:** Panda proprioception always included `actuator_effort_nm` (joint torques), which
  lets an agent weigh what it holds. New `[simulation] effort = false` (Panda only, default true) removes it from reference
  and reconstruction observations and from the observation contract. It enters the public config only when set, so all
  other configs and tasks are unchanged. Wave 11 sets `effort = false` and `wrench = false` (tactile is off by default).
  Unit test in `tests/test_infrastructure.py`.
- **Validation:**
  - full suite `58396489` passed (280 passed, 12 skipped)
  - new main image `a9482d3ea278` (job `58396513`, archive on netscratch); the previous record is kept as
    `trusted-image-5904fd183fa4.json`
  - strict OCI `58398616` passed (3 passed, 1 skipped)
  - The SAM-3D relay OCI test first failed (`58397456`, `58398087`) on its own check that no `/opt` path names sam3d. The
    image legitimately contains the `sim2sim/sam3d.py` module, which the sub-agent had noted. The check now allows only
    that module (and its bytecode) and still forbids any spool path.
- **Wave 11** (`w11-tetra-spring-k`: "ideal linear spring with a stiffness of exactly 21.7 N/m"; `w11-tetra-spring-block`:
  "ideal linear spring" + "purple block … exactly 0.0235 kg"; values asserted against the scenes):
  - convergence prompt, camera on, metrics chamfer, center_of_mass, density, image_mse
  - prepared on netscratch with sources byte-identical to the environments
  - **held for the user's review again**

## 2026-11-06 21:30 UTC — Replicates and wave 10 results; cap raised to $1000/day; SAM-3D weights fetched; spring levels built (held for user review)
- **User (2026-11-06):** daily cap $1000 (`generate-waves.py` default for pools from `sim2sim-20261106`; earlier configs
  byte-identical). Start the new experiments. Build two tetrahedral spring levels without force sensing (below).
- **Replicates done** (n=3 per condition = original + r2 + r3; final submission; ratios predicted ÷ true;
  `rerun-analysis/rep-summary.{py,json}`, analysis job `58373538`):

  | condition | mass | damping | friction loss / stiffness | finding at n=3 |
  |---|---|---|---|---|
  | lever, force | 1.00 ×3 | 1.00 ×3 | 1.00 ×3 | exact every time |
  | lever, no force | 0.32, 0.34, 0.27 | 0.35, 0.39, 0.30 | friction 0, 0.09, 0 | one common scale factor; friction overlooked (3/3) |
  | lever, no force + hinge hint | 0.30, 0.24, 0.18 | 0.30, 0.25, 0.19 | friction 0.26, 0.24, 0.17 | hint recovers friction up to the same unknown scale (3/3) |
  | lever, no force + stated mass + hint | 1.00 ×3 | 1.01, 1.01, 1.00 | friction 0.93, 0.96, 1.00 | near-complete identification (3/3) |
  | lever, no-interaction + stated mass + hint | 1.00 ×3 | 0.98, 1.00, 0.98 | friction 1.18, 0.99, 1.19 | near-complete; friction ±19% (3/3) |
  | spring, force | 1.00 ×3 | 1.00 ×3 | stiffness 1.00 ×3 | exact every time |
  | spring, no force | 1.97, 1.20, 1.24 | same factor | stiffness same factor | scale never right; factor varies 1.2–2.0 |
  | spring, no force + stated mass | 1.00 ×3 | 1.00, 1.00, 1.07 | stiffness 1.00, 1.00, 1.06 | stated mass fixes the scale (3/3) |
  | blocks, force | exact ×3 (COM 0, Chamfer 0) | | | exact every time |
  | blocks, no force | total 0.24, 0.28, 0.53 | | red/blue ratio kept only in r3 | COM 25.6, 11.2, 0 mm: ratio lost in 2/3 |

  - Wall time to the final submission under the convergence prompt varies a lot. The lever with force ran 221, 98 and
    228 min; the spring with force 52, 46 and 37 min.
  - The original `w6-spring-hard-nowrench-mass` reports total mass ×19.5 because the agent added extra bodies. The lever's
    own mass is exact.
- **Wave 10 done** (`validation/results-wave10.json`, analysis `58373537`; n=1 each):
  - **Panda blocks, final Chamfer / COM error:**
    - camera: exact, 0 / 0
    - touch (tactile): 14.4 mm / 34 mm
    - joint positions only: 18.6 mm / 110 mm
    - So touch cut the COM error about 3× relative to joint positions alone.
  - **Blind-run checks:** neither camera-free run reproduced its canary colors (distance ≥0.66). The touch run colored
    its blocks near the original red and blue (0.20, 0.28). The scene's body names contain the colors, but the agent's
    progress notes never mention color, so this looks like a default red/blue prior. Colors are excluded from blind
    scoring. Audit search: "block_red" appears only in the private `run_start` manifest
    (`resolved_source_config`), not in `public_tool_config`, the task or any wire. No leak; the colors are a prior.
  - **Panda hard lever:**
    - planning tool v2: damping and friction exact (12 submissions, $44.34)
    - without the tool: damping ×0.53, friction ×0.08
    - The planning effect is n=1 and not comparable with `w4`.
  - **New-prompt baselines:**
    - tetra: exact (Chamfer 0, density 0)
    - assembly: Chamfer 13.9 mm, COM 0
    - boat: Chamfer 12.5 mm, COM 0
    - Assembly and boat ended at the spend cap.
- **Trivial baselines** for wave 10 and the replicates: jobs `58373539`–`41` (done; see 2026-11-07 01:40).
- **SAM-3D:** user logged in to Hugging Face. Weights fetched at the pinned revision; all 19 pinned files verified (job
  `58373415`). Real-inference smoke test `58373463` (A100) running.
- **New levels** (`environments/tetra_spring`, `environments/tetra_spring_block`, generated by
  `experiments/validation/tetra-spring-scenes.py`, checked by `tetra-spring-check.py`; previews in
  `experiments/validation/tetra-spring/`):
  - **Scene:** the unchanged tetrahedral cluster (1/2/4/8 kg) beside a round hole, 0.754 m across and 0.30 m deep, in
    the surface.
  - **Spring:** frictionless and ideal (slide joint, stiffness 1730 N/m, no damping), with two flat round plates 0.74 m
    in diameter (cluster footprint radius 0.333 m). It starts upright beside the hole.
  - **Why round:** a point grasp cannot control yaw, and a square plate in a square hole with 5 mm clearance jams beyond
    ~0.8°.
  - **Hole construction:** a 24-gon of hidden tangent collision boxes, plus one visual-only mesh with the same hole.
    Overlapping coplanar box tops caused shadow acne.
  - **Checks:**
    - in the hole the top plate sits 0.3 mm below flush
    - spring contacts are condim 1 (frictionless), and the spring slides to the floor when pushed into a wall
    - the cluster on the spring settles 85 mm down (15 g / k)
    - on the frictionless plate the bouncing cluster drifts about 5 cm (a consequence of the user's frictionless spec)
  - **Level 1** (`w11-tetra-spring-k`): task = convergence prompt + "ideal linear spring with a stiffness of exactly
    1730 N/m".
  - **Level 2** (`w11-tetra-spring-block`): adds a 2.35 kg purple block (0.12 m cube); task = "ideal linear spring" +
    "purple block … exactly 2.35 kg".
  - **Both:** `wrench = false`, metrics chamfer, center_of_mass, density, image_mse. The stated values are asserted
    against the scenes. Config diffs against `w10-tetra-base` are only the declared ones.
  - Prepared on netscratch (home has 13 GB free). **Held at the user's request until they review the scenes.**
  - The base prompt is the convergence prompt, not the original wave-1 "use the full budget" prompt (user policy 2026-11-03).

## 2026-11-05 06:30 UTC — SAM-3D tool merged (waiting on the user's HF login); first replicates finished
- **SAM-3D:** built by a sub-agent in a netscratch copy and merged into main via
  `/n/netscratch/qianqian_lab/Lab/aejxie/sam3d/sam3d-merge.patch`. Main's pre-merge versions are in the session scratchpad,
  and all wave configs regenerate byte-identically.
  - **Design:** `sam3d` is a controller action. The controller writes the agent's RGB PNG and mask PNG (bounded size, header-checked)
    to a private spool on netscratch. `scripts/sam3d-service.slurm` (`kempner_gpu_priority`,
    `--gres=gpu:nvidia_a100-sxm4-40gb:1`, account `kempner_qianqian_lab`, QoS `kempner_qianqian_priority`) answers with a
    PLY mesh in the camera frame plus pose and scale. Each call is audited as a `sam3d_call` event. A missing or stale
    service fails the run loudly (`ServiceFailure`). Checkpoint revision and code commit are pinned
    (`experiments/sam3d/checkpoint-pins.json`). Runs without `[tools] sam3d = true` are unchanged: the action is listed only when enabled.
  - **Environment:** `SAM3D_ROOT=/n/netscratch/qianqian_lab/Lab/aejxie/sam3d`, built with CUDA 12.2. Kaolin and flash-attn
    2.8.3 were built from source (Rocky 8 glibc). GPU check: job `57945921` (A100). Use A100 only; the pinned cu121 stack has
    no Blackwell kernels. Node `holygpu8a19102` reported its GPU busy (exclude it). Sub-agent tests: full suite `57917801`
    274 passed, 13 skipped; strict relay test with a fake-model service `57917802`/`57918384` passed. Cross-node relay:
    5 calls, 0.5–1.0 s.
  - **Merged main:** full suite `57946824` passed (277 passed, 10 skipped). SAM-3D agent image `5e99630de861` (job `57947635`
    on `test`; the earlier sapphire submission `57946878` was refused because image builds require `test`). It is recorded
    only in `experiments/images/trusted-image-sam3d.json` (archive on netscratch); inspection found no forbidden files.
    `trusted-image.json` is unchanged (`5904fd183fa4`), so the 11-06 replicates use the same image as the 11-05 ones.
  - **Blocked on the user:**
    1. Accept the license at https://huggingface.co/facebook/sam-3d-objects.
    2. Run `/n/netscratch/qianqian_lab/Lab/aejxie/sam3d/env/bin/hf auth login`.
    3. Then `sbatch --account=qianqian_lab scripts/fetch-sam3d.slurm` (~13 GB).
    4. Then `sbatch --exclude=holygpu8a19102 scripts/sam3d-smoke.slurm full`, which writes the sha256 lock. Check its camera-frame z-range line.
  - **Caveats:**
    - Agent PNGs are decoded by PIL in the service, outside the sandbox (bounded size).
    - Runs that share one service wait on each other; wait and inference times are audited separately, so give each
      compared run its own service or don't overlap them.
    - A run waiting on the service fails if the service job ends.
- **First replicates (11-04 pool, now ~$620 of $700):**
  - lever with force r2: agent finished at 100 min ($17.96)
  - lever no force r2: hit the $44 cap at 132 min, like its original
  - spring with force r2: finished at 51 min ($20.86)

  To be analyzed with the other replicates.
- **Wave 10 and the 11-05 replicates** started 00:05–00:28 ET.
- **Trivial-baseline normalization done:** all 53 reported runs have `baseline-scores.json` with both the empty and
  bounding-box baselines `ok` (106/106; sapphire jobs in `experiments/validation/baselines-20261105-jobs.txt`, 2.3–4.1 h each).
  Not yet in the results report.

## 2026-11-05 03:15 UTC — Replicates queued (n=3 for the key findings); SAM-3D GPU tool being built
- **User (2026-11-05):** replicates up to n=3, only for the important findings; SAM-3D may use `kempner_gpu_priority`, but
  not H100/H200.
- **Replicates.** 10 conditions × 2 more runs = 20 runs (n=3 each with the originals). Configs come from
  `generate-waves.py` `REPLICATES` (`experiments/configs/rep-*`) and equal the originals except the cost-pool lines (diffed).
  Run IDs are `<original>-rN-<pool date>-001`. Each prepared `scene.xml` is byte-identical to its original's
  (`experiments/validation/rep-prepare.sh`).

  | finding | conditions replicated |
  |---|---|
  | force sensing (wrench) fixes the scale | lever hard base vs nowrench; spring hard base vs nowrench; blocks hard base vs nowrench |
  | stated mass fixes the scale (spring) | `w6-spring-hard-nowrench-mass` |
  | hinge hint finds friction; mass + hint near-complete identification without force, even no-interaction | `w8-lever-hard-nowrench-hint`, `w9-lever-hard-nowrench-mass-hint`, `w9-lever-hard-noaction-mass-hint` |

  - **Schedule:** r2 of lever base, lever nowrench and spring base started now on the 11-04 pool (worst case $132, which
    brings 11-04 to ≤$672 of $700; jobs `57914325`, `57914331`). The 7 other r2 runs start 11-05 00:05 ET beside wave 10
    (11-05 worst case $688). All 10 r3 runs start 11-06 00:05 ET (worst case $440). Job lists are in
    `experiments/validation/rep-2026110{4,5,6}-jobs.txt`.
  - **Storage:** replicates run on netscratch (`/n/netscratch/qianqian_lab/Lab/aejxie/sim2sim-experiments`). Home has
    23 GB free and a 4 h run writes about 0.75 GB, so 8 wave-10 runs plus 20 replicates on home would go below the
    10 GB floor. Netscratch purges files after 90 days, so copy finished runs to lab once the user confirms quota headroom.
    The first two runs started correctly there. `compare.py`, `analyze.sh` and `build-results-report.py` search it.
  - **Image:** current main `5904fd183fa4`. The originals used `44064383e99f`. The agent harness (`examples/sandbox_agent.py`,
    `client.py`) and the prepared scenes are unchanged. The only agent-visible difference is that the public tool config now
    also lists `"tactile": false`.
- **Trivial-baseline scoring, first attempt failed:** I submitted `scripts/baselines.slurm` to `sapphire`, but
  `dev_worker.py`/`devrunner.py` only accepted `test` (unlike `candidate_history.require_compute_allocation`). Every
  evaluation was refused ("Agent exited before completing protocol"), and two jobs timed out. Both checks now accept
  `test` or `sapphire` (user rule, 2026-11-03), as does the native skip in `tests/test_development.py`. Strict runs don't
  use these workers. Resubmitted on sapphire (jobs in `experiments/validation/baselines-20261105-jobs.txt`); the first run
  scored `ok` for both baselines.
- **SAM-3D:** a sub-agent is building it in a code copy on netscratch, so live runs are not affected. The design is a GPU service
  on `kempner_gpu_priority` (A100-40GB or RTX Pro 6000 gres, never H100/H200) that the controller relays to; the agent
  container stays CPU-only and offline. The checkpoints are gated on Hugging Face and there is no HF token on the
  cluster, so the user must accept the license and log in.

## 2026-11-04 23:55 UTC — Images rebuilt and validated; wave 10 queued for 2026-11-05 (first use of new infrastructure)
- **Images:** main `5904fd183fa4` (job `57860580`) and tools `6b541c5839a4` on it (job `57860601`).
- **Validation on merged main:**
  - full suite 269 passed, 12 skipped (`57862068`)
  - strict OCI main 2 passed, 1 skipped (tools test; `57862690`)
  - tools OCI 3 passed (`57863422`)
- **Wave 10** (pool `sim2sim-20261105`, $700/day; convergence prompt; 4 h; `--begin` 00:05 ET; jobs in
  `experiments/validation/wave10-20261105-jobs.txt`):

  | question | runs | image |
  |---|---|---|
  | new-prompt baselines scored with the new contact-friction, expert-video and policy-transfer metrics | `w10-assembly-base`, `w10-tetra-base`, `w10-boat-base` | main |
  | tactile vs proprioception vs vision (Panda blocks, random canary colors) | `w10-panda-blocks-proprio`, `w10-panda-blocks-tactile`, `w10-panda-blocks-vision` | main |
  | planning tools v2 (Panda hard lever) | `w10-panda-lever-hard-toolsimg-base`, `w10-panda-lever-hard-planning` | tools |

  - Caps are $44, and $58 for the two Panda lever runs (worst case $380).
  - Panda blocks also score `friction`. Expert and transfer are not defined for Panda.
  - Sources: the paired Panda sources are identical (`diff -r`). The tactile observation was checked on the prepared camera-free
    source (srun `57862329`). Config diffs: only the declared factor.
  - Not comparable with the old planning run (`w4-…-planning`): the tool and its README changed.
- **Trivial-baseline normalization** (empty and bounding-box candidates) is running over the 53 reported runs (jobs in
  `experiments/validation/baselines-20261104-jobs.txt`). It writes `baseline-scores.json` per run, with no API spend.
- **Not queued:**
  - pose jitter (replicates are deferred by the user)
  - SAM-3D (needs a GPU decision from the user)
  - retro-scoring old submissions with the new metrics (no tool for it yet)

## 2026-11-04 23:20 UTC — Wave 9 results; infrastructure merged into main; images rebuilding
- **Wave 9** (job `57800657`; analysis `57860075`, `rerun-analysis/compare-all.json`, `validation/results-wave9.json`). Both
  runs ended `budget_suspended` ($41.17, $40.04). n=1 each; ratios are predicted ÷ true:

  | hard lever | mass | inertia | damping | friction loss |
  |---|---|---|---|---|
  | no force, stated mass (`w7-…-nowrench-mass`) | 1.00 | 1.12 | 1.04 | **0** |
  | no force, stated mass + hint (`w9-…-nowrench-mass-hint`) | 1.00 | 1.00 | 1.01 | 0.93 |
  | no-interaction, stated mass + hint (`w9-…-noaction-mass-hint`) | 1.00 | 0.99 | 0.98 | 1.18 |

  - **Stated mass plus hint gives near-full identification without force sensing.** Friction converges over submissions
    (×0.71 → 0.77 → 0.82 → 0.93). It was not exact at the cap (prediction "fully exact": close, not met).
  - **Video alone is nearly as good.** The no-interaction run was within 2.5% on damping and 20% on friction from submission 3
    (106 min). Friction ended at ×1.18; its first submission was ×1.10 at 38 min.
- **Pool:** `sim2sim-20261104` closed at $539.90 of $700.
- **Infrastructure merge:** the `infra-tools` worktree work (tactile mode, planning tools v2, contact-friction, expert-video
  and policy-transfer metrics, trivial-baseline normalization, pose jitter) was copied into main.
  - 29 files. Every file main also has was unchanged since the worktree's copy base, so it was a plain copy; main's previous
    versions are backed up in the session scratchpad.
  - Validation before the merge: worktree full suite 269 passed, 12 skipped (`experiments/validation/infra-full-wt-57823123.log`).
    The earlier failures in that agent's logs pass on rerun in both trees (`infra-recheck-*`).
- **Images:** the main image (job `57860580`) and tools image (`57860601`, afterok) are rebuilding. Previous records are kept
  as `trusted-image-44064383e99f.json` and `trusted-image-tools-c3926b9bb997.json`.

## 2026-11-04 17:10 UTC — Daily limit raised to $700; wave 9 moved to today and launched
- **Limit:** the user raised the daily spend limit to $700, effective today. `generate-waves.py` defaults pools after
  `sim2sim-20261104` to $700. Today's waves pass `DAY700`. Earlier configs are byte-identical (regenerated and diffed).
- **Wave 9 change:** the 16:40 queue (job `57754657`, pool `sim2sim-20261105`) was cancelled before start and its prepared
  directory removed. It now runs on today's pool (at $458.69 before this wave):
  - `w9-lever-hard-nowrench-mass-hint`: no force sensing, task states the lever mass and names damping and friction loss.
  - `w9-lever-hard-noaction-mass-hint`: same task, no-interaction. Is video alone enough on a friction hinge? (The spring was
    exact no-interaction with a stated mass.)
- **Setup:** job `57800657`, sapphire, home root, image `44064383e99f`, 4 h, $44 cap. Both sources are byte-identical to the
  wave-7 lever (srun `57800575`). Configs differ from `w7-lever-hard-nowrench-mass` only by the hint sentence, the pool
  ceiling and (no-interaction) the interaction flag.
- **Predictions:** the no-force run is fully exact. The no-interaction run gets exact mass and close damping and friction, if
  free decay alone separates viscous from Coulomb damping.
- **Results report:** extended with waves 7–8 in place (`experiments/reports/results-report-20261104.zip`, 51 runs, spec `report-spec-20261104b.py`; previous zip kept as `report-work-20261104/results-report-20261104-v1.zip`).

## 2026-11-04 16:30 UTC — Wave 8 results: lever friction under the convergence prompt
- **Runs** (job `57722092`; analysis `57754234`, `rerun-analysis/compare-all.json`, `validation/results-wave8.json`; n=1 each;
  ratios are predicted ÷ true):

  | run | end | result |
  |---|---|---|
  | `w8-lever-hard-base` (force sensing) | finish at 235 min (limit 240) | exact from submission 2 (40 min, 2.7M tokens): mass, inertia, damping, friction ×1.00 |
  | `w8-lever-hard-nowrench-hint` | cap, 17.2M tokens | mass ×0.301, inertia ×0.304, damping ×0.303, **friction ×0.257** (from submission 2, 67 min) |
  | `w7-lever-hard-nowrench` (reference, no hint) | cap | mass ×0.316, inertia ×0.343, damping ×0.346, friction 0 |

- **Predictions held:**
  - With force sensing, the new prompt gets friction exactly, so the prompt alone does not suppress friction.
  - Without force sensing, naming friction in the task gets it fitted, near the common factor. Friction ÷ inertia is 0.85 of
    the true value.
  - Without the hint, both new-prompt runs left friction at 0; the old-prompt run added it late (submission 5).
  - Reading: without force sensing, Coulomb friction is identifiable but easy to overlook. Whether an agent fits it looks like
    a matter of attention (n=1 per arm; old vs new prompt is confounded with run-to-run noise).
- **Behavior:** the force-sensing base was exact at 40 min, yet it kept testing through 15 submissions and finished 6 minutes
  before the limit. The new-prompt spring base finished at 56 min. The convergence prompt does not reliably end runs early.
- **Pool:** `sim2sim-20261104` is at $458.69 of $500 (wave 8: $36.77 + $41.22). No further launches today; a lever run costs
  about $40.

## 2026-11-04 12:40 UTC — Wave 7 results; wave 8 launched (lever friction under the convergence prompt)
- **Wave 7 results** (n=1 each; ratios are predicted ÷ true; analysis job `57721741`, `rerun-analysis/compare-all.json`,
  `validation/results-wave7.json`):

  | run | end | result |
  |---|---|---|
  | `w7-blocks-hard-base` (force sensing) | finish, 74 min (last submission 58 min) | exact (Chamfer 0.0 mm, COM 0.0 mm, both masses ×1.00) |
  | `w7-spring-hard-noaction` | finish, 79 min | mass, inertia, damping and stiffness all ×1.812 (same factor as the old-prompt run) |
  | `w7-spring-hard-noaction-mass` | finish, 114 min | exact (damping and stiffness rel. error < 1e-5) |
  | `w7-lever-hard-nowrench` | cap, 179 min (last submission 133 min) | mass ×0.316, inertia ×0.343, damping ×0.346, **friction loss 0** |
  | `w7-lever-hard-nowrench-mass` | cap, 185 min | mass ×1.00, inertia ×1.12, damping ×1.04, **friction loss 0** |

- **Predictions:** the blocks and spring predictions held. The lever prediction failed on friction. Mass, inertia and damping
  follow the stated scale, but friction loss stayed 0 in both runs at every submission.
  - The old-prompt no-force lever (`w2-lever-hard-nowrench`) added friction at submission 5 (133 min, 11.6M tokens), reaching
    ×0.288 next to its ×0.245 common factor. The new runs had reached 13.0M and 16.8M tokens at their last submission.
  - The no-force blocks failure (`w6-blocks-hard-nowrench`, red pinned at a round 0.1 kg) has a new-prompt baseline now.
    With force sensing, the new prompt recovers both blocks exactly.
- **No-interaction spring:** one stated mass is enough for full identification from video alone (exact at 114 min, 6.3M tokens).
- **Footnote:** the `w6-spring-hard-nowrench-mass` total-mass ratio ×19.5 comes from a fixed 6.9 kg `support_body` the agent
  added. The lever body itself is ×1.00.
- **Both lever runs ended `budget_suspended`** (about $40 each), not by `finish`.
- **Pool:** `sim2sim-20261104` stood at $380.69 of $500 after wave 7.
- **Wave 8 launched** (job `57722092`, sapphire, home root, image `44064383e99f`, 4 h, $44 cap; worst case $469 of $500):
  - `w8-lever-hard-base`: convergence prompt with force sensing. Does the new prompt miss friction even with force?
  - `w8-lever-hard-nowrench-hint`: no force sensing, the task names damping and friction loss. Does naming friction recover it?
  - Both prepared sources are byte-identical to the `w7-lever-hard-nowrench` source (srun `57722007`). Config diffs vs wave 7
    are only the declared factor.
  - Predictions: the base run gets exact friction (force makes it directly measurable). The hint run gets nonzero friction near
    its mass factor.

## 2026-11-04 10:10 UTC — Cumulative results report; corrections from compiled submissions
- **Report:** `experiments/reports/results-report-20261104.zip` (2.07 MB, 44 runs). The builder `experiments/validation/build-results-report.py`
  is now the reusable template, driven by a spec file (`report-spec-20261104.py`). Sections are organized by question, with
  matched final / 120 min / equal-token tables, convergence plots, 3D overlays and the taxonomy. New runs have no stills or videos.
- **Count correction:** 22 new runs finished on 11-04 (10 batch A + spring no-interaction + 6 batch B + 5 wave 6), not 26.
- **Corrections to the 05:20 entry:**
  - No-interaction is an exact common factor only for the spring (×1.812) and the Panda lever (×0.183). The hard lever is
    ×0.226 / 0.242 / 0.249 (mass / inertia / damping) with friction dropped.
  - "Geometry not hurt" was too strong. Tetra no-interaction shape is exact, but its COM is 98 mm off (vs 0.3 mm), because the
    spheres got round masses 0.2/0.3/0.4/0.6 kg. Boat Chamfer is +38%, with no noise band available.
- **Cause of the spring ×1.97 (resolved):** both no-force runs set the ball to a round 0.1 kg (rod 0.638 kg fitted to the
  ratios). The no-interaction ×1.81 rod is 0.587 kg, MuJoCo's default density 1000 kg/m³ for the fitted capsule. Unidentifiable
  scale is filled by round masses or default density (both verified in models.json).
- **Tool uptake (bash calls touching the tool):** segmentation 0 of 170 (offered) and 0 of 125 (task says when it helps);
  planning 18 of 143 (first use at minute 3). The segmentation-directed result is therefore not a tool effect.

## 2026-11-04 09:30 UTC — Wave 7 launched: stated-mass follow-ups (convergence prompt)
- **Questions:**
  1. Does the blocks ratio failure come from the new prompt? Adds the missing new-prompt blocks baseline with force sensing.
  2. Does a stated mass fix the scale on a second hinge scene with friction (hard lever)?
  3. Is a video plus one stated mass enough for full identification (no-interaction spring, with and without the mass)?
- **Runs** (4 h, $44 cap, pool `sim2sim-20261104` at $257.45 of $500; sapphire, home root; image `44064383e99f`; jobs in
  `experiments/validation/wave7-20261104-jobs.txt`):
  - `w7-blocks-hard-base`
  - `w7-lever-hard-nowrench` and `w7-lever-hard-nowrench-mass` (1.3290 kg)
  - `w7-spring-hard-noaction` and `w7-spring-hard-noaction-mass` (0.37442 kg)

  Stated masses were asserted against the compiled prepared sources (srun job `57699077`). Config diffs: only the declared factor.
- **Predictions:**
  - blocks base exact
  - lever with mass exact (damping and friction), without mass a common factor
  - no-interaction spring with mass exact, without mass a common factor

## 2026-11-04 09:10 UTC — Batch B reruns and wave 6 complete
- All 15 runs completed (`agent_finish`, except `w4-assembly-seg-directed` which hit its $44 cap at 151 min with 4 submissions).
  Analysis job `57697978`: `rerun-analysis/checkpoints-B.json`, `models-B.json`, `compare-all.json`; scores in
  `results-rerunB-w3.json`, `results-rerunB-w4.json`, `results-wave6.json`. Spend: 11-03 pool $1081.21 (ceiling $2000),
  11-04 pool $257.45 (of $500). n=1 per arm; ratios are predicted ÷ true.
- **Stated mass (wave 6, new prompt):** spring is exact with the stated lever mass, against ×1.971 without it. Blocks:
  the stated red mass was kept exact, but blue came out ×1.41 (COM 28 mm). Without it, red was ×0.263 (= 0.1 kg) and blue ×0.19
  (COM 26 mm). **A stated mass anchors only the body it names; the mass ratio between separate blocks was not recovered
  in either new-prompt run.** (The old-prompt run without force sensing did recover the ratio: common ×0.525, COM exact.)
- **Round-number default masses when the scale is unidentifiable:** red block = 0.1000 kg (new-prompt no-force run), 0.1/0.1 kg
  (no-interaction), 0.2000 kg (old-prompt no-force run). This explains the "common factors" on blocks.
- **New-prompt finish times:** spring base 56 min, spring stated mass 112 min, spring no force sensing 150 min. Both blocks runs
  used about 233 min.
- **Hinge hint without force sensing (hard lever):** the scale stays wrong: ×0.49 on mass, inertia and damping (vs ×0.24
  without the hint); frictionloss −43% (vs −71%).
- **Directed segmentation tool (assembly):** COM 0.0 mm and Chamfer 16.1 mm, against 4.1 and 16.9 mm for the undirected run.
  The progress notes never mention the tool; count actual calls from traces before crediting it. Within the assembly noise band.
- **Planning tool (Panda hard lever):** damping −74% vs −12% without the tool; frictionloss lost in both.
- **Wrist camera (Panda small, assembly):** Chamfer 3.7 vs 6.3 mm (better), COM 14.5 vs 4.0 mm and total mass ×0.80 vs
  ×1.00 (worse). Mixed result.
- **Depth (boat):** Chamfer 13.3 vs 10.1 mm and COM 0.3 vs 0.0 mm: no gain. Equal at 120 min (13.3 vs 13.3 mm).

## 2026-11-04 05:45 UTC — Wave 6 launched: does one stated mass fix the absolute scale? (new prompt)
- **Question:** without force sensing (and when only watching), every hinge parameter was off by one common factor. If the
  task states one true mass, does the agent propagate it to fix the scale of everything else?
- **First wave under the convergence prompt** (`TASK_CONVERGE`: finish when nothing differs) and main image `44064383e99f`.
  Baselines are rerun under this prompt; compare only within wave 6. Time to finish is now an outcome.
- **Runs** (hard variants, 4 h, $44 per-run cap, pool `sim2sim-20261104` with $118.92 already spent of $500; sapphire, home root;
  jobs in `experiments/validation/wave6-20261104-jobs.txt`):

  | run | condition |
  | --- | --- |
  | `w6-spring-hard-base` | grasp + force |
  | `w6-spring-hard-nowrench` | no force sensing |
  | `w6-spring-hard-nowrench-mass` | no force sensing, task states the lever's total mass (0.37442 kg) |
  | `w6-blocks-hard-nowrench` | no force sensing |
  | `w6-blocks-hard-nowrench-mass` | no force sensing, task states the red block's mass (0.3808 kg) |

  The stated masses were asserted against the compiled prepared sources (srun job `57655157`). Config diffs: only the declared factor.
- **First result, new-prompt spring baseline:** exact damping and stiffness from the first submission. The agent called
  finish at 56 of 240 min (3 submissions, 4.5M tokens, $11.22), after naming occluded surfaces as the only unvalidated aspect.
  Under the old prompt the same scene used 229 min and 9.1M tokens for the same exact answer.
- **Stated mass fixes the spring scale:** `w6-spring-hard-nowrench-mass` reached damping and stiffness within 0.001% (submission 5
  of 6; +5% at submission 1), keeping the stated mass. It called finish at 112 min (10.2M tokens, $27.07), and used the mass
  both as a scale anchor and to infer arm/ball density and radius. `w6-spring-hard-nowrench` (no stated mass) finished at
  150 min with ×1.971 on damping and stiffness (6 submissions, 13.1M tokens, $33.01). The old-prompt run reached ×1.972:
  two independent runs land on the same factor: both set the ball to a round 0.1 kg (see the 10:10 entry).
- **Prediction:** with a stated mass, spring and blocks reach the exact scale. Without it, they repeat the common factor
  (spring ×1.97, blocks ×0.525 under the old prompt).

## 2026-11-04 05:20 UTC — Batch A rerun results: no-interaction, no force sensing (assembly, tetra), Panda lever
- All 10 batch-A runs completed (9 `agent_finish`; `w3-boat-base-…-002` hit its $58 per-run cap at 224 of 240 min,
  9 submissions). Old prompt and images, n=1 each. Spend at 05:20 UTC: 11-03 pool $876.20.
- Analysis: `experiments/validation/rerun-analysis/` (`analyze.sh` job `57653373`: `checkpoints.json`, `models.json`;
  `compare.py` → `compare-A.json`, the final checkpoint, the 120 min mark and equal tokens). Scores also in
  `results-rerunA-w5.json`, `results-rerunA-w3.json`, `results-w5-spring-001.json`. Ratios below are predicted ÷ true.
- **No-interaction (reference world can only be watched and reset): absolute scale is lost; ratios survive on hinges.**

  | scene | baseline (grasp + force) | no-interaction |
  | --- | --- | --- |
  | hard spring | exact | mass, inertia, damping, stiffness all ×1.81 |
  | hard lever | exact | mass ×0.23, inertia ×0.24, damping ×0.25, friction lost (0) |
  | Panda hard lever | lever mass ×0.92, damping ×0.88, friction lost | mass, inertia, damping, friction all ×0.18 |
  | hard blocks | exact | masses ×0.26 / ×0.47 (ratio lost), COM 48 mm |
  | assembly | total mass ×1.00, COM 4.6 mm, Chamfer 13.8 mm | total mass ×0.16, COM 24 mm, Chamfer 19.9 mm |
  | tetra | total mass ×1.00, Chamfer 12.2 mm | total mass ×0.10, Chamfer 0.0 mm |
  | boat | total mass ×1.00, COM 0.0 mm, Chamfer 10.1 mm | total mass ×0.05, COM 5.5 mm, Chamfer 13.9 mm |

  The same conclusions hold at 120 min and at equal tokens; no-interaction runs used fewer tokens (5–16M vs 9–24M).
  The hinge results are an exact common factor, so watching passive motion pins only ratios to inertia: gravity torque
  scales with mass too, so it cannot fix the scale. That matches the earlier runs without force sensing (spring ×1.97,
  lever ×0.23–0.29, blocks ×0.525): **grasping without force sensing adds no absolute scale either.** It does add
  mass *ratios* on blocks: COM was exact without force sensing but 48 mm off when only watching.
  Geometry is not hurt by no-interaction (tetra exact, boat and assembly within the assembly noise band). Total mass is
  the one quantity that only force sensing provides.
- **No force sensing, assembly and tetra:** assembly total mass ×1.36, COM 40.6 mm (baseline band over three runs
  4.1–27.7 mm); tetra total mass ×0.067, Chamfer 9.4 mm vs 12.2 mm.
- **Panda arm vs point grasp, hard lever:** the arm run reached damping −12% and lost frictionloss (0), against exact
  with the point grasp (−26% at 120 min).
- Caveats: n=1 per arm. Panda total mass includes robot links in the candidate, so the lever body mass is used.

## 2026-11-04 01:00 UTC — Reruns of the 16 interrupted experiments (old prompt), home storage
- **User decisions:** rerun under the old "use the full budget" prompt and images (`c81c28c7d470`; tools `ea7d56785aed`),
  so they compare with the existing baselines. Home is the hot storage (cap about 10 concurrent runs); finished runs are
  copied to lab storage later.
- **Configs:** `experiments/configs/rerun-20261103/` are the original configs with only the pool lines changed (pool
  `sim2sim-20261103`, ceiling $2000; diff-verified), so no run is suspended by the 11-04 $500 pool. IDs are `<name>-002`.
- **Batch A, running on sapphire** (`experiments/validation/rerun-20261103-jobs.txt`): no-interaction lever-hard, blocks-hard,
  assembly, tetra, boat and Panda lever-hard, plus baselines/ablations boat-base, Panda lever-hard, assembly-nowrench and
  tetra-nowrench. `w5-spring-hard-noaction-20261103-001` (lab root) survived the incident and continues.
- **First no-interaction result (survived the incident):** `w5-spring-hard-noaction-20261103-001` completed (`agent_finish`,
  6 submissions, sealed audit, no write errors). Final damping 0.3313 vs true 0.1828 (+81%), against exact with
  grasp and force sensing. That factor of 1.81 is close to the 1.97 common scale factor of spring-hard without force
  sensing, which fits the scale ambiguity: watching motion pins only ratios to inertia. Stiffness and masses are not
  scored by this image; compile the submissions to check.
- **Batch B, launched 04:45 UTC on sapphire** (home had 37 GB free; finished batch-A runs are 1–2 GB): boat-depth +
  lever-hard-nowrench-hint (`57652457`), ps-assembly-fixed/wrist (`57652458`), both on main image `c81c28c7d470`;
  assembly-seg-directed + panda-lever-hard-planning (`57652459`) on the old tools image `ea7d56785aed`.
- **New tools image built:** job `57593904`, `c3926b9bb997` on main `44064383e99f` (new finish policy). Only for
  future conditions under the new prompt; not used by any rerun.
- **Equal-time and equal-token comparison (user):** wall time is a poor proxy for compute, since 4 h runs used 9.2M to 18.8M tokens.
  `experiments/validation/checkpoint-index.py` gives every scored checkpoint its elapsed minutes and cumulative model
  tokens (old runs: from inference replies; cost-blind runs: from the host-only `inference_accounting` event), and
  `at(checkpoints, wall_min=120)` / `at(..., tokens=X)` return the result available at a budget. Verified on an old,
  a cost-blind and a lab-root run (`checkpoint-index-test.json`).

## 2026-11-03 23:44 UTC — INCIDENT: lab storage quota exceeded; 16 of 17 running experiments died
- **Cause (orchestrator error):** my parallel copy of the 60 home runs to lab storage (job `57589769`, 24 rsyncs) filled the
  `qianqian_lab` quota. Every controller writing there failed with `[Errno 122] Disk quota exceeded`, about 80 min
  into the runs: 3b (`57337196-8`), 4 (`57561965`) and 5 (`57583488`, `57583490`, `57583709`, plus the lever run in `57583486`).
  The copy was cancelled and its partial lab copies deleted; the home originals were untouched. Only
  `w5-spring-hard-noaction` may still be alive. These runs are invalid (truncated by infrastructure, not by the agent),
  so they must be rerun with new IDs. Spend wasted is about $150 (both pools).
- **Lesson:** never bulk-copy into the shared lab quota while experiments write there. Check headroom first.

## 2026-11-03 — Correction from the report rebuild (compiled every submission)
- **Hard blocks without force sensing were NOT exact.** Final masses 0.2000/0.1116 kg vs true 0.3808/0.2124 (47% low);
  only the mass *ratio* is right, which is all the COM metric checks. The wave-2/3a claim "free blocks still exact
  without the wrench" is wrong. The scale ambiguity (motion fixes only ratios) now appears on three scenes: lever,
  spring and blocks.
- **Hard lever without force sensing:** lever mass is also wrong (0.33 vs 1.33 kg). Factors range 0.23–0.29, not one
  common factor, and even the inertia ratios are off (damping 4%, friction 17%), whereas the spring's ratios are exact.
- **Depth, first submission:** 2.4× better than the mean of the three RGB assembly runs (the earlier 3× was vs one run).
- Each no-force agent kept one mass at a round first guess (0.2/0.1/0.2 kg) and scaled everything else to it, and none
  noted that the absolute scale is unidentifiable from motion alone.
- Report: `/n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-experiments/results-report-20261103.zip`
  (builder `experiments/validation/build-results-report.py`).

## 2026-11-03 ~23:30 UTC — Finish policy changed; images rebuilt
- **User:** agents must not be told to use up their time; they finish when they believe nothing about their
  reconstruction differs from the reference world. The harness system prompt no longer says "ending early earns nothing"
  or "finish only when little time remains". The one-time finish guard now asks for a list of unvalidated aspects and
  executes a second finish. New conditions use `TASK_CONVERGE`; existing configs are unchanged.
- **Images:** main `44064383e99f` (job `57593108`; harness tests passed, job `57593107`). It also carries the hinge-stiffness
  metric. The tools image is rebuilt on top of it (job `57593904`). Previous records are kept as
  `experiments/images/trusted-image-c81c28c7d470.json` and `trusted-image-tools-ea7d56785aed.json`. All runs to date
  (including the 17 running now) used the old "use the full budget" prompt; baselines must be rerun before
  comparing new-prompt ablations.

## 2026-11-03 ~21:00 UTC — Results viewer and combined dashboard
- **Results viewer** (one-off, user request): `/n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-experiments/results-viewer-20261103.zip`
  (1.8 MB; unzip and open `index.html`). It has plots for each conclusion and 11 side-by-side source/reconstruction videos. Each
  video pairs the last reconstruction episode that repeats a reference episode's leading actions, using recorded frames only.
  Builder: `experiments/validation/build-results-viewer.py` + `results-viewer.html`.
- **Dashboard:** `sim2sim dashboard` / `inspect` accept repeated `--root`, and `serve-dashboard.sh` accepts a colon-separated
  `SIM2SIM_EXPERIMENTS`. The combined snapshot is at `.../sim2sim-experiments/dashboard`
  (see [dashboard.md](dashboard.md)).

## 2026-11-03 22:25 UTC — Wave 5: no-action ablation (no-interaction reference world), on sapphire
- **Question (user, 2026-11-03):** how informative are the agent's interactions with the "real" world? The reference
  world is no-interaction (`[simulation] reference_interaction = false`). grasp/move/release/arm_target are rejected there
  with an explicit error, while observe and reset still work. Gravity acts as usual, and the reconstruction world keeps every action.
  The public task says so in `worlds.reference` and marks those actions "reconstruction only". Runtime test
  `test_observe_only_reference_rejects_interaction`; runtime/harness tests passed (job `57582939`).
- **Runs** (each changes one factor from its baseline; sources verified byte-identical to the baseline's):
  lever-hard (vs w2-lever-hard-base), spring-hard (vs w3-spring-hard-base), blocks-hard (vs w2-blocks-hard-base),
  assembly (vs the three RGB assembly baselines), tetra (vs w1-tetra-base), boat (vs w3-boat-base), and Panda lever-hard
  (vs w3-panda-lever-hard). `w5-*-noaction-20261103-001`, lab root, jobs in
  `experiments/validation/wave5-20261103-jobs.txt`. $58 cap each; pool `sim2sim-20261103` with its ceiling raised to $2000.
- **Interpretation note:** passive observation is not static. The no-interaction lever agent saw its arm fall and swing
  from the initial pose under gravity. Wave 5 therefore measures what actions add *beyond* passive gravity-driven motion
  (and settling), not what dynamics information is worth overall. Scenes that start at rest (blocks, assembly) bound
  the purely visual case.
- **Partition:** sapphire (same `holy8a` hardware, QoS `normal`, no 5-job limit). The strict OCI tests passed there
  (job `57583004`: 2 passed; the tools-image test is skipped by design). Job `57583489` failed at startup: two of my jobs
  on one node raced to initialize rootless Podman's shared SHM lock segment. `scripts/podman-job.sh` now serializes
  the first podman call per node with `flock`; the retry is `57583709`.

## 2026-11-03 22:24 UTC — Daily cap lifted; waves 3b and 4 started early
- The user lifted the $500 cap for 2026-11-03 and asked for more parallel jobs. Waves 3b (`57337196-8`) and 4
  (`57561965`) were released to start at 22:24 UTC instead of 05:00. Their configs still name pool
  `sim2sim-20261104`, but they are tallied as 11-03 spend. New runs on 11-04 will use a fresh pool name.
- 17 runs are now concurrent: 10 on test and 7 on sapphire. That exceeds the original 10-run guideline, as the request for more parallel jobs implies.

## 2026-11-04 00:00 ET (05:00 UTC) — Wave 4: directed tool use (started early: see the 22:24 UTC entry)
- **Question:** wave 3c's agent ignored an available segmentation tool. Does it use a tool, and gain from it, when the
  task says when the tool helps?
- **Runs:** job `57561965` (resubmitted; the first submission, `57490834`, was cancelled before it started, $0) (2 runs, 20 CPUs; with wave 3b that makes 10 concurrent runs and 100 of 112 CPUs).
  Tools image `ea7d5678…`, lab root, pool `sim2sim-20261104`. Cost-blind task.
  - `w4-assembly-seg-directed`: segmentation tool plus a task sentence on per-part masks. $44 cap. Compare with
    w3-assembly-seg (undirected, unused) and the three assembly baselines.
  - `w4-panda-lever-hard-planning`: planning tool (IK/RRT) plus a task sentence. $58 cap, same as its baseline
    w3-panda-lever-hard (wave 3b, same hard source).
- **Outcomes:** tool calls per run (uptake) as well as the usual metrics. The undirected planning condition is not run.
- **Budget:** 3b caps total $464 and wave 4 $102, more than the $500 pool. Past runs averaged about $35, so
  the expected total is about $360. If the pool runs out, the host suspends the next caller (`budget_suspended`),
  and that run can be resumed on 11-05.
- **Evaluator:** the `hinges` metric now also scores `stiffness` (absolute/relative error, N·m/rad), and plots and dashboard
  show it (focused tests: 50 passed, job `57489759`). This takes effect only in images built after this change.
  Waves 3b/4 use the existing images, which have no spring scenes. w3-spring-hard stiffness was already checked
  statically (log entry for wave 3a).

## 2026-11-03 14:42 UTC — Wave 3c: segmentation tool on random assembly (completed 18:35 UTC)
- **Runs:** `w3-assembly-{seg,toolsimg-base}-20261103-001`, job `57381749`, lab root. Tools image `ea7d5678…`
  for both, so the only difference is the mounted MobileSAM tool and one task paragraph describing it.
  $44 cap each, pool `sim2sim-20261103`. These ran under the old spend exposure (see the entry below).
- **Results** ([results-wave3c.json](../experiments/validation/results-wave3c.json)); per checkpoint, Chamfer / COM error (m):

  | Run | Checkpoints | Final | Subs, cost |
  |---|---|---|---|
  | seg | .023/.090 → .020/.014 → .025/.038 → .017/.0041 (×4) | 0.0169 / 0.0041 | 7, $39.21 |
  | toolsimg-base | .032/.093 → .017/.018 → .018/.027 → .018/.027 → .019/.027 → .019/.028 | 0.0188 / 0.0277 | 6, $33.80 |

- **The tool was never used.** No call to `segment.py` (not even `--help`). The only reference to the tool
  outside the task text is the host's mount record. So this pair is **not** a tool ablation: it is two
  runs of effectively the same condition.
- **Conclusions:**
  1. **Making an optional tool available does not mean the agent uses it.** Tool-access ablations need
     tool uptake as a primary outcome, and a condition where the task says when the tool helps (prompt
     study) or makes it the only route to some information.
  2. **First estimate of run-to-run spread on assembly (n=3 baselines: w1-base, these two):** final Chamfer
     0.0138–0.0188 m; final COM 0.0041–0.0277 m (7×). Earlier single-run assembly COM contrasts are
     within this spread: depth's 0.0006 vs 0.0046 and all-angles' 0.015. Depth's final Chamfer 0.0115 lies
     below the baseline range, and its first-submission advantage (3×) is larger than the spread of
     first submissions (0.023–0.040), so "depth helps geometry" survives as suggestive. "All-angles hurts"
     (Chamfer 0.019) does not.
  3. **Regression after a good checkpoint.** toolsimg-base reached COM 0.018 at submission 2 and drifted to
     0.028 by the end. The agent cannot see scores, and its own fit did not catch this. Report best and
     final, and treat "final" as the agent's chosen answer.

## 2026-11-03 10:42 UTC — Evaluator and scene fixes, trusted image `c81c28c7d470`
- **Density:** per-primitive density is now *unavailable*, not an error, when geom densities don't determine
  the mass distribution: explicit body `<inertial>`, massless geom, or massless visual/collision twins.
  This fixes the w1-tetra artifact.
- **Canaries:** canary colors stay at least 0.35 RGB from a stereotypical palette as well as from the
  originals. `run-summary.py` now scores canaries per object, matched by geometry; it compiles, so run it
  on `test`. Per-object recheck of w1-panda-blind: the agent's colors were 0.46 and 0.24 from the
  canaries and 0.93 from the originals, so no leak
  ([check](../experiments/validation/results-canary-check-20261103.json)).
- **Panda tetra scene:** the cluster moved +0.2 m in y, so the hand no longer starts inside a sphere.
  Physics is unchanged, and the hard variant now prepares.
- **Tests:** 58 OCI/runtime tests passed on the new image (job `57322631`), plus 65 focused tests (job
  `57300325`).
- **Image provenance:** waves 1–2 and the continuations used `cd3b0ae00868`
  (`trusted-image-cd3b0ae00868.json`). Wave 3a onwards uses `c81c28c7d470`. Same harness and Containerfile;
  only evaluator code differs.

## 2026-11-03 ~19:20 UTC — Spend caps hidden from the model
- **What the agents saw:** never their running spend, tokens or remaining calls (the harness shows only
  remaining wall time). The first task message did include the static caps `max_cost_usd`, `max_calls`
  and `max_total_tokens`: $58 in w1, $44 in w2/w3a/w3c, $35 for continuations.
- **Effect:** a scan of every model progress note found no mention of money or API budget. Every
  "spending"/"expensive" note is about time or compute. Within-wave contrasts shared one cap. The cross-wave
  pairs (w2-tetra-gray, w2-assembly-allangles and w2-panda-vision vs wave-1 baselines) differ in the
  visible cap. Treat that as a small, unmeasured confound.
- **Change (user requirement: the agent must see no mention of dollars, spend or caps; spend is
  orchestrator-only).** From wave 3b on:
  1. The public task's `inference` block omits every spend cap.
  2. Inference replies entering the container carry only `content`, `tool_calls` and context-size usage
     (`promptTokens`/`cacheReadTokens`/`cacheWriteTokens`, used for compaction). Cost and remaining-budget
     fields are now a host-only `inference_accounting` audit event. Before this change they sat in the
     harness's memory, which the model's `bash` could in principle read (`ptrace_scope=0`).
  3. A budget stop reaches the container as `{"budget_stop": true, "reason": "host stop"}`. The harness
     submits any unsubmitted work and finishes without another model call. The run is recorded as
     `budget_suspended` (host `inference_suspended` event), and the orchestrator may restart it from
     that checkpoint with `[continuation]`.
  - Tests passed in job `57440219`, including a runtime test asserting that no cost, USD or cap field
    reaches the agent side.
  4. Continuation carry-over keeps only the parent's model-origin notes; adapter notes, which can record a
     budget stop, are dropped. Preparation refuses any carried text that mentions spend. Tests passed in
     job `57441542`.
  - The wave-3c segmentation pair started before this change. Both runs saw the old $44 cap and had cost
    fields in harness memory, so the pair is consistent with itself.
  - Spend-visibility history: waves 1–3a and 3c ran under the old exposure; wave 3b is the first fully
    cost-blind wave.

## 2026-11-04 00:00 ET (05:00 UTC) — Wave 3b (scheduled)
- **Jobs:** `57337196`, `57337197`, `57337198` with Slurm `--begin` (3+3+2 runs). Pool `sim2sim-20261104`, $58 cap
  each. Image `c81c28c7d470`. Configs: `experiments/configs/wave3b-20261104/`.
- **Storage:** runs live under the lab-storage root `/n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-experiments`
  (`SIM2SIM_EXPERIMENTS_ROOT` / `--root`).
  - Home was at 94% and a 4 h strict run writes about 3 GB of audit data.
  - The earlier jobs `57323673/5/6` were cancelled before start ($0) and re-prepared there.
  - A strict start under that root was smoke-tested (`smoke-labroot-20261103-001`, job `57337121`).
  - Trusted image archives also moved to lab storage; their records point there and are sha256-checked.

  | Run | Paired with | Question |
  |---|---|---|
  | w3-boat-base / w3-boat-depth | each other | boat profile; does depth help on a second scene? |
  | w3-assembly-nowrench | w1-assembly-base | is force sensing needed for COM of a complex rigid body? |
  | w3-tetra-nowrench | w1-tetra-base, w2-tetra-gray | …for the mass distribution of a cluster? |
  | w3-ps-assembly-fixed / -wrist | each other | multiview: fixed camera vs fixed + wrist (Panda small) |
  | w3-panda-lever-hard | w2-lever-hard-base, -nowrench | do Panda joint efforts substitute for the wrench? Same hinge values |
  | w3-lever-hard-nowrench-hint | w2-lever-hard-nowrench | does naming the parameters help when the baseline fails? Same source |

## 2026-11-03 10:43 UTC — Wave 3a: hard spring, wrench on vs off (completed 14:40 UTC)
- **Results** ([results-wave3a.json](../experiments/validation/results-wave3a.json); stiffness checked
  statically from the submissions because the `hinges` metric omits it):

  | Run | Damping (true 0.1828) | Stiffness (true 2.823) | Arm / ball mass (true 0.3237 / 0.0507) | Subs, cost |
  |---|---|---|---|---|
  | base | 0.1828 | 2.823 | exact | 6, $26.47 |
  | no-wrench | 0.360 (1.97×) | 5.566 (1.97×) | 0.638 / 0.100 (1.97×) | 5, $40.59 (budget stop at 173 min) |

- **Conclusion:**
  - **The wrench effect replicates on a second hinge scene, and it has a clear mechanism.** Without
    force sensing, every parameter was off by one common factor, so the trajectories matched while
    the absolute scale was wrong.
  - Under position-controlled grasping, motion alone identifies only ratios: stiffness, damping and
    gravity torque relative to inertia.
  - The wrench pins the absolute scale. The hard-lever no-wrench errors (0.23× damping, 0.29×
    frictionloss) are consistent with the same unidentifiability.
  - Free blocks were still exact without the wrench, perhaps because contact and constraint dynamics
    leak mass information there. Untested.

- **Question:** does the hinge wrench effect seen in w2 generalize to a second hinge scene, one with
  stiffness and damping?
- **Runs:** `w3-spring-hard-{base,nowrench}-20261103-001` in job `57322768`.
  - Identical hard source (hash `2fa1e579…`): damping 0.1828, stiffness 2.823; frictionloss stays 0.
  - $44 cap each, pool `sim2sim-20261103`.
- **Image:** first runs on rebuilt trusted image `c81c28c7d470`, which carries the evaluation-only density
  fix. The harness is unchanged (`sandbox_agent.py` sha `c6e65f…`). The old record is preserved as
  `experiments/images/trusted-image-cd3b0ae00868.json`.

## 2026-11-03 05:02 UTC — Sandboxed wave 2 and wave-1 continuations (running)
- **Wave 2 completed** (all 8 runs, about 230 min each). Results:
  [results-wave2.json](../experiments/validation/results-wave2.json).
  Two runs, `lever-hard-hint` and `lever-hard-nowrench`, ended by graceful budget stop near $40.

  | Contrast | Result (final submission) |
  |---|---|
  | lever-hard base / hint / no-wrench (true damping 0.1157, frictionloss 0.07767) | base and hint exact; **no-wrench 0.0271 / 0.0224, about 75% error** |
  | blocks-hard base vs no-wrench | both exact (COM error < 1e-7 m) |
  | tetra-gray vs w1-tetra-base | gray exact (Chamfer, COM, density) |
  | panda-vision vs w1-panda-blind | both exact geometry and COM; only vision matched colors |
  | assembly-allangles vs w1-assembly-base | **worse**: Chamfer 0.019 vs 0.014 m, COM 0.015 vs 0.0046 m, 5 vs 7 submissions |

- **Conclusions** (directional, n=1):
  1. **Grasp force sensing is decisive for hinge damping and dry friction, not for mass.**
     - Without the wrench, the agent could not recover the hard lever's hinge parameters (about 75% error).
     - Free-body masses were still recovered exactly without it. Wave 3a tests a second hinge scene.
  2. **Non-round values alone do not defeat simulation-based fitting.** The agent fits parameters by
     simulating its own candidate and matching trajectories, so hard variants were recovered to near
     machine precision whenever the right signal (force) was observable.
     - Discriminative scenes need *information-limited* observations, or richer hidden physics, rather
       than just non-round numbers.
  3. **Naming parameters in the prompt has no effect when the baseline already succeeds** (twice now).
  4. **Losing color identity did not hurt the tetra cluster.** The agent distinguished balls by geometry
     and dynamics.
  5. **The explicit "match from every angle" instruction did not help assembly, and may have hurt.**
     It spent effort on appearance and submitted less. n=1, so treat this as a hypothesis.
  6. **The arm adds no difficulty for blocks.** Ideal grasp, Panda blind and Panda vision were all
     exact on geometry and mass.
- **Continuations completed** (2 h each, from the 4 h submissions;
  [results-continuations.json](../experiments/validation/results-continuations.json)):
  - RGB: Chamfer 0.0138 → 0.0135 m, COM 0.0046 → 0.0025 m (5 submissions, $22.17).
  - RGB + depth: plateaued at Chamfer 0.0115 m, COM 0.0006 m (8 submissions, $27.89).
  - **Conclusion:** depth's advantage persists. At 6 h, RGB is still worse than depth was at 4 h
    (Chamfer about 17% worse, COM about 4× worse), so depth adds information rather than just
    speeding things up.
- **Jobs:** `57226610`, `57262770`, `57262773`, `57263851`, plus continuations in `57264622`
  ([job list](../experiments/validation/wave2-20261103-jobs.txt)). The launcher now takes up to
  3 runs per job, with `--cpus-per-task=30 --mem=72G`.
- **Invalid attempts:**
  - `w2-lever-hard-base-...-001` was killed at 58 min by a 10 s `scancel` timeout during resume. Freeze and
    thaw now re-send until the freezer reports the target state (180 s bound); 56 tests passed in job
    `57259601`. Relaunched as `-002`.
  - `w2-tetra-gray-...-001` was superseded by `-002`, which adds the `center_of_mass` metric.
- **Continuations** (2 h, $35 cap, from each parent's last submission):
  - `c1-assembly-base` and `c1-assembly-depth` continue their wave-1 runs, testing whether RGB-only
    catches up with depth.
  - `-001` was refused before launch ($0) because prepared-output validation rejected the `prior/`
    directory. Fixed, with a regression test.
  - `-002` is waiting on the `test` QoS CPU cap (112 per user) and starts when job `57226610` ends.
- **Design:** see [the plan](../experiments/plan-20261102.md#wave-2--8-runs-prepared-2026-11-03-experimentsconfigswave2-20261103-44-cap-each).
  - Hard non-round variants for the scenes wave 1 saturated.
  - Original scenes for unsaturated contrasts, which pair with wave-1 baselines.
- **Runs:** `w2-*-20261103-001`, prepared on a `test` node. Groups share identical sources:
  - lever-hard ×3: hash `626039be…`
  - blocks-hard ×2: hash `a4e01b43…`
- **Infrastructure added overnight:**
  - strict continuation (`[continuation]`; 49 tests passed, job `57202011`)
  - non-round hard variants (`physics = "nonround"`; 21 of 22 scenes; job `57224580`)

## Budget tally (API spend)
Daily cap is $500 per US-Eastern day, with one cost pool per day. Amounts are the gateway's
reported `cost.total`, which is a catalog estimate. Check live spend with
`sim2sim cost --pool <pool>`. Reserves are $2 per call that was in flight when a run was
cancelled. They are counted against the pool but may not have been billed.

| Day (ET) | Pool | Batch | Calls | Spend (USD) | Reserves | Status |
|---|---|---|---:|---:|---:|---|
| 2026-11-02 | `sim2sim-500` | smoke tests (-001, -002) | 52 | 2.16 | 0 | done |
| 2026-11-02 | `sim2sim-500` | wave 1 attempt -001 (invalid) | 156 | 5.11 | 10 | cancelled |
| 2026-11-02 | `sim2sim-500` | wave 1 attempt -002 (invalid) | 298 | 12.99 | 8 | cancelled |
| 2026-11-02 | `sim2sim-500` | wave 1 attempt -003 (8 runs) | 2295 | 263.55 | 2 | done |
| **2026-11-02** | | **day total (pool)** | | **283.82** | 20 | cap $500; runs capped at $58 each |
| 2026-11-03 | `sim2sim-20261103` | wave 2 (incl. invalid lever-hard-base-001), 2 continuations, wave 3a (2), lab-root smoke | 3283 | 397.20 | 0 | done |
| 2026-11-03 | `sim2sim-20261103` | wave 3c (2) | 618 | 73.00 | 0 | done |
| **2026-11-03** | | **day total (pool)** | 3901 | **470.20** | 0 | cap $500 |
| 2026-11-03 (cap lifted) | `sim2sim-20261104` (name kept) | wave 3b (8 × $58 cap) | | | | running since 22:24 UTC |
| 2026-11-03 (cap lifted) | `sim2sim-20261104` (name kept) | wave 4 (2; caps $44, $58) | | | | running since 22:24 UTC |
| 2026-11-03 (cap lifted) | `sim2sim-20261103` (ceiling $2000) | wave 5 no-action (7 × $58 cap) | | | | running since 22:25 UTC |

Pre-API development runs (gpt-6-astra through Codex subagents) are not included: their
cost is unknown.

## 2026-11-03 02:47 UTC — Sandboxed wave 1 (attempt -003, completed 06:48 UTC)
- **Status:** all 8 completed. 7 ended with `agent_finish` at about 230 min; `lever-base` reached the 240 min
  `wall_deadline` and was recorded as completed with its 7 submissions. Results:
  [results-wave1-003.json](../experiments/validation/results-wave1-003.json), from `run-summary.py 'w1-*-20261102-003' wave1-003`.
- **Results** (n=1 per cell; first → final submission; private scores):

  | Run | Budget used | Subs | Cost | Key metrics |
  |---|---:|---:|---:|---|
  | assembly-base | 97% | 7 | $35.24 | Chamfer 0.040→0.014 m, COM 0.058→0.0046 m |
  | assembly-depth | 96% | 11 | $48.13 | Chamfer 0.013→0.012 m, COM 0.017→0.0006 m |
  | blocks-base | 96% | 4 | $24.35 | exact (Chamfer 0, COM 0, image MSE 0) |
  | spring-base | 96% | 4 | $17.29 | exact damping 0.12, frictionloss 0 |
  | lever-base | 100% | 7 | $32.42 | exact damping 0.12, frictionloss 0.05 |
  | lever-hint | 96% | 9 | $35.06 | exact, same as base |
  | tetra-base | 96% | 12 | $45.77 | Chamfer 0.015→0.012 m; mass 15.007/15 kg, COM error 0.3 mm (static check) |
  | panda-blind | 97% | 7 | $25.29 | Chamfer 0.086→0, COM 0.23→0 m |

- **Notes on reading the results:**
  - Tetra's density metric (rel. error 5.4) is an artifact: the agent used a body `<inertial>`; see TODO.
  - Blind colors were a generic blue/orange guess, and the agent noted "appearance unconstrained". The values
    are not the canary's, but this seed's canary (cyan / orange-red) was close to stereotypical colors, so
    it discriminated weakly.
- **Conclusions** (directional, n=1):
  1. **The anti-early-finish prompt and guard work.** Every run used 96–100% of its budget, against 9–38%
     in development mode. No guard intercept ever fired, so the prompt alone was enough.
  2. **Round-valued scenes saturate.** blocks, spring and lever were recovered exactly. That includes the
     Panda arm with proprioception only and no vision. Contrasts on these scenes, such as the hinge hint,
     show no effect because of the ceiling, not because the factor is inert. Hard non-round variants
     replace them in wave 2.
  3. **Depth helps geometry and mass distribution early.** First submission: Chamfer 3× and COM 3.4×
     better. The gap narrows with time: final Chamfer 0.012 vs 0.014, COM 0.0006 vs 0.0046. Depth also cost
     37% more ($48 vs $35). The 2 h continuations test whether RGB catches up.
  4. **Naming hinge parameters had no measurable effect at the ceiling.** The baseline recovered friction
     loss unprompted. In development mode it had stayed at 0, which suggests the anti-early-finish task's
     mention of untried interactions already elicits it. Retested off the ceiling in wave 2.
  5. **Blind Panda shows no leak.** The agent recovered geometry and mass exactly by touch, but not colors.
  6. **The evaluator must handle equivalent parameterizations.** A rigid cluster with explicit inertia
     scored as a density failure despite being physically correct.
 Jobs `57193339`, `57193341`, `57193342`, `57193343`, 2 runs each
  ([job list](../experiments/validation/wave1-20261102-jobs.txt)).
- **Runs:** `w1-{blocks-base,spring-base,lever-base,lever-hint,assembly-base,assembly-depth,tetra-base,panda-blind}-20261102-003`.
- **Validation before launch:** 36 tests passed with the `slurm_step` OCI backend (job `57192888`), including a real freeze/thaw with no podman calls.
- **Attempt -002 is invalid** (jobs `57188767`, `57188783`, `57188784`, `57188794`; launched 02:28, cancelled 02:44:07 UTC).
  - Resuming a paused container after private evaluation went through `podman inspect` and failed in `w1-panda-blind-...-002`.
  - That container's step had been paused about 8 s after it started. About 1 in 19 pauses failed, so runs would have died at random.
  - Resume now uses each container's Slurm step, recorded once at launch verification. About $13 was spent.
- **Attempt -001 is invalid** (jobs `57186236`–`57186239`; launched 02:17, cancelled 02:26 UTC).
  - The shared cost-ledger reader failed closed on a transient NFS replace race and stopped `w1-lever-hint-...-001` after 487 s.
  - The other 7 were cancelled before they hit it. About $5 was spent.
  - Fixed with a bounded re-read and a regression test (34 passed, job `57188532`).
- **Configs:** `experiments/configs/wave1-20261102/`. Trusted image `cd3b0ae00868`.
- **Cost:** pool `sim2sim-500` ($500); per-run cap $58 (graceful stop near $54).
- **Smoke test:** `smoke-blocks-20261102-002` (job `57181952`).
  - Completed: 2 submissions, both evaluated `ok`; $2.16 over 52 calls in 14 min.
  - The agent recovered both block masses (0.35 and 0.25 kg) from grasp-force ratios.
  - Smoke `-001` (job `57180717`) failed on a cost-pool bug in the inference subprocess. It was fixed and cost $0.
- **Transport fix before launch:** about a quarter of smoke-run turns imitated tool calls as text. The broker now reports each call alongside its result instead of in the model's own messages.
- **Plan:** [experiments/plan-20261102.md](../experiments/plan-20261102.md). It includes:
  - 8 one-off conditions (no replicates for now, per user)
  - a 4 h wall limit each
  - strict OCI sandbox enforced through Slurm steps
  - a tool-using in-container harness
  - `gpt-6-astra` at medium reasoning
  - an identical anti-early-finish prompt in every condition
- **Gate 0:**
  - sandbox enforcement backend
  - harness and gateway broker
  - canary and uniform-gray color perturbation: **done**, `[perturbation] seed, colors`; 44/44 env×mode verified (job `57164701`), 14 tests passed (job `57165052`)
  - a 20-minute smoke run

## 2026-11-02 — Leak audit of development-mode blind runs
- **Status:** done. Report: [leak-audit-20261102.md](../experiments/validation/leak-audit-20261102.md).
- **Result:** no source read found.
  - Blind blocks colors were approximate guesses that the agent labeled as assumptions.
  - Blind fixed-cube color was wrong.
- **Strict-path gaps found (being fixed):**
  - the Panda fixture isn't staged
  - the agent sees no proprioception
  - `scripted_agent.py` in the image contains color priors
  - run IDs reveal the scene and condition

## 2026-11-02 — Panda arm, 6 h (development mode)
- **Status:** completed. Record: [panda-six-hour-20261102-batch.json](../experiments/validation/panda-six-hour-20261102-batch.json).
- **Mode:** `development_shared_filesystem` (not OS isolated), `gpt-6-astra` medium, 21600 s wall limit, Panda arm.
- **Runs:** 8. Six vision runs (blocks, spring, damped lever, tetrahedral, random assembly, boat) and two proprio-only runs (blocks, fixed cube).
- **Outcome:** all 8 ended with `agent_finish`; 9 checkpoints, all evaluated `ok`.
  - Budget used was only 9–25% (32–91 min of 360).

## 2026-10-30 → 11-02 — Medium, 2 h, six environments (development mode)
- **Status:** completed. Record: [astra-medium-20261029-batch.json](../experiments/validation/astra-medium-20261029-batch.json).
- **Mode:** development mode, `gpt-6-astra` medium, 7200 s, ideal point grasp.
- **Budget used:**

  | Environment | Used | Accepted checkpoints |
  | --- | --- | --- |
  | random assembly | 118 min | 3 |
  | tetrahedral | 85 min | 3 |
  | boat | 69 min | 2 |
  | damped lever | 46 min | 2 |
  | spring | 41 min | 2 |
  | blocks | 23 min | 1 |

- **Failed attempt:** the first damped-lever attempt expired without a model spawn; it's preserved, and the retry was `-20261101-001`.

## 2026-10-28 — Development pilot, random assembly, 30 min
- **Status:** completed (`astra-development-20261028-002`, job `56069945`).
- **Outcome:**
  - 3 submissions
  - 28 of 30 min used
  - 3000 ticks
- **Failed attempt:** an earlier attempt, `-001`, failed before completion.
