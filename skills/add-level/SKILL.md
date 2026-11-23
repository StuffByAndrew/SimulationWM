---
name: add-level
description: Design and add a new sim2sim level (environment scene, private manifest, check script and strict experiment configs). Interviews the user for the task semantics, then walks through the agent-capability settings (camera, RGB/RGB-D, force sensing, reference interaction, resets, code execution, tools, hints, metrics). Use when the user wants a new level, environment, scene family or level variant.
---

A level is a MuJoCo scene plus everything the evaluator and the agent see of it. Adding one touches, in order:
a generator script, `environments/[panda_small/]<level>/{scene.xml,manifest.toml}`, a check script, configs
via `experiments/configs/generate-waves.py`, and `docs/environments.md`. Running it is the
[run-experiment](../run-experiment/SKILL.md) skill. Read [environment authoring](../../docs/environments.md) and
the metric sections of [evaluation](../../docs/evaluation.md) you will use before writing anything.

## 1. Interview: semantics first (one open question)

Ask the user for a free-form description, prompting for whatever of this they did not say:

- **What is in the scene** and what is visible vs. hidden (e.g. tunnel mechanism out of view, contents of a sealed box).
- **What the agent should have to discover**: geometry, mass/COM, friction, a joint, a spring/damper law, a latch...
  This is the scientific point of the level; every later choice should make it discoverable but not free.
- **Variants**: a family of levels that differ in one hidden property (precedent: `tunnel_*`, `box_*`), and what is
  held identical across them.
- **Closest existing level** to copy defaults from (list `environments/` and `environments/panda_small/`).

Restate the semantics in 3–5 lines and get a yes before the settings round.

## 2. Settings walkthrough (at most two AskUserQuestion rounds)

Do not ask what the description already decided. Pre-select the closest level's value as the recommended option and
offer **"both (ablation pair)"** where it makes sense: each pair becomes two configs differing in exactly that one
factor against a shared baseline, never two factors at once.

Round A — embodiment and sensing (`[simulation]`):

| Decision | Options → config | Notes |
| --- | --- | --- |
| Actuator | point grasp (`environments/<level>`) / Franka (`environments/panda_small/<level>`, `robot = "panda"`, `arm_target`) / both | Franka levels include the shared `robot/` fixture and use `PANDA_CONSTRAINTS`; the fixture gripper pinches only ~1–1.4 N, so design light objects or form closure. |
| Camera | fixed / `movable_camera = true` / + `wrist_camera = "wrist"` (Franka only) | The scene still needs a worldbody perspective camera named `fixed`; evaluation always uses the trusted fixed camera. |
| Images | none (proprio-only: `rgb = calibration = false`) / RGB / RGB-D (`depth = true`); resolution (default 320×240) | |
| Force sensing | point grasp `wrench`; Franka `effort` (joint torques, default on), `tactile` | `wrench` defaults false; disabling `effort` means no force sensing at all on Franka. |
| Time | lock-step (default: physics advances only on action ticks) / real time (`realtime = true`, `frame_rate = N`; generate-waves `o["realtime"] = N`) | Generic, any level or embodiment: the world runs at wall-clock rate, also while the agent thinks; frames only every 1/N s; no pause, ticks-advance, reconstruction world or state access; needs `wall_only`. Pair with `[tools] channel` so code can act on the live world. An optional private manifest `[launcher]` adds perturbed resets and `throw` protocols (docs/protocol.md). |

Round B — interaction, budget and agent capability:

| Decision | Options → config | Notes |
| --- | --- | --- |
| Reference interaction | allowed / observe-only (`reference_interaction = false`) | |
| Reset budget | unlimited (current behaviour) / **capped N resets — not implemented** | Today only `[budget]` ticks/submissions/active seconds exist, and only when `wall_only = false` (`physics_ticks`, `submissions`, `active_seconds`). Recent waves are `wall_only = true`. |
| Code execution | yes (current: `bash` + python with mujoco/numpy/scipy in the sandbox) / **no code — not implemented** | A no-code condition removes `bash` and changes the harness `SYSTEM` prompt, which is otherwise identical across conditions; its baseline must be rerun with the same harness image. |
| Optional tools | `[tools]` `sequence`, `channel` (code reaches the controller during bash), `planning` (Franka), `segmentation` (tools image), `sam3d` (GPU service) | Each adds a fixed sentence from `AGENT_TOOLS`; omit for the control. |
| Task hints | none / stated values / mechanism hint | Hints are public text appended to `TASK_CONVERGE`. Never state hidden values unless that disclosure *is* the condition. |
| Budget | wall seconds (recent: 21600 + `hold` 129600), per-run cost cap | Costs and caps never appear in agent-visible text. |
| Perturbation | none / `colors`, `physics = "nonround"`, `pose = "jitter"` | See run-experiment. |

Then pick metrics from what the level is about, and note what each needs in the manifest:

| Metric | Manifest requirement |
| --- | --- |
| `chamfer`, `center_of_mass`, `density` | `object_bodies`, `object_geoms`, `[reconstruction]` (+ `[evaluation.reconstruction]` in config) |
| `image_mse`, `appearance` | none beyond the camera |
| `dynamics` | `[[probes]]` (point grasp) or `[[arm_probes]]` (Franka) |
| `hinges` | `[hinge_correspondence]` or config `hinge_matching = "single"` |
| `force_profile` | `[force_profile]` (see `src/sim2sim/force_profile.py`, tunnel manifests) |
| `die` (predictor submission, real time) | `[launcher]` and `[die]` (see `src/sim2sim/die.py`, `environments/die_biased`) |
| `friction`, `expert`, `policy_transfer` | see `docs/evaluation.md`; `expert` needs `[[expert_trajectories]]`, `policy_transfer` is point-grasp only with `[[transfer_tasks]]` |

Finish the round with a one-screen summary: levels × conditions, each condition's single factor, metrics, wall
budget and cost per run × count. Get approval before building.

## 3. Unimplemented capabilities

If the user chose a capped reset budget, no-code, or any other setting absent from `runtime.py` (`sim_config`
defaults, `public_config`, task `action_schema`) and `examples/sandbox_agent.py`, implement it as its own change
before the level configs:

- One config key, default preserving current behaviour so every existing config and prepared run is unchanged.
- Coordinate the controller task contract, enforcement, harness tool list, `docs/protocol.md` and tests per
  AGENTS.md. Enforce on the host (a reset cap refuses the action loudly; resets still never refund ticks).
- A harness change means rebuilding the trusted image (`scripts/build-image.slurm`) and rerunning any baseline the
  new condition is compared against with that image.

## 4. Build

1. **Generator** `experiments/validation/<family>-scenes.py` (precedents `tunnel-scenes.py`, `box-scenes.py`): named
   constants with units and a reason, deterministic output, writes every variant's `scene.xml` and `manifest.toml`.
   Panda-small levels reuse `TOP, TABLE` and the shared camera; if it computes probes with IK, run it on Slurm.
2. **Scene rules**: named bodies and geoms, standard primitives/joints, 2 ms timestep unless justified, no `_grasp_`
   prefix, fixed camera framing the whole workspace. Hidden parts must be invisible to RGB, depth and the grasp ray
   and unreachable if that is the design. Physical claims live in MJCF values, not text.
3. **Manifest**: `name` (`panda_small_<level>` for Franka), `scene`, `camera`, `graspable_bodies`, `object_bodies`,
   `object_geoms`, `[correspondence]`, plus the metric tables chosen above. All of it is private.
4. **Check script** `experiments/validation/<family>-check.py OUTDIR` on a Slurm `test` node: files equal the generator
   output; each scene compiles and rests; hidden parts stay hidden; every source probe grasps (a source miss fails
   evaluation) and is deterministic; the designed property is measurable through the agent's permitted channels; a
   scripted Franka can grasp/move the target. Save previews and numbers under `experiments/validation/`. Inspect the
   Slurm output and record the actual result.
5. **Configs**: add a `RUNS<n>` dict and `WAVES["wave<k>-<date>"]` entry to `generate-waves.py`, extending `config()`
   only for a genuinely new knob. Keep these as plain top-level assignments: the dashboard parses them for each
   environment's base prompt. Run `.venv/bin/python experiments/configs/generate-waves.py`; `git status
   experiments/configs` must show only new files (existing outputs stay byte-identical).
6. **Docs**: a `docs/environments.md` section (design, hidden property, variants table, probes, check job ID and
   result); an entry in `docs/experiment-todo.md` (planned) or `docs/experiment-log.md` (launched).
7. **Smoke**: prepare one config with `--prepare-only` on a `test` node and confirm the level appears in the dashboard
   Environments view. Then hand off to run-experiment.

Keep scene values, manifests, check outputs and the physical-parameter text out of the task, image and any
agent-visible file.
