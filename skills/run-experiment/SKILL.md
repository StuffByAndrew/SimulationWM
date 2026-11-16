---
name: run-experiment
description: Configure, prepare, or run a sim2sim experiment, including requested physical parameter changes and private evaluation choices. Use for experiment work in this repository, not ordinary simulation explanations.
---

Use the repository CLI and read [experiment configuration](../../docs/experiments.md).
Keep configurations under `experiments/configs/` and each attempt under
`experiments/runs/<id>/`. Use a fresh ID for another attempt.

When the user specifies physical values, make them real scene changes:

1. Prepare with `sim2sim experiment CONFIG --id ID --prepare-only`.
2. Edit the requested MJCF attributes in `experiments/runs/ID/source/`, preserving
   the original environment. Resolve units, inherited defaults and overrides:
   geom `mass` overrides `density`; hinge damping and friction loss are distinct.
3. Update `experiment.json` physical-parameter descriptions to describe the actual
   configured scene. Verify requested attributes directly; use Slurm for native
   compilation or dynamics checks. Text metadata alone does not apply values.
4. Start with `sim2sim experiment --start ID`, adding `--development` for the
   explicitly selected development mode. The runner records the resolved source
   hash. If a requested value remains unresolved, resolve it before launch.

For a seeded object-color leak canary (`colors = "random"`) or a color-free
ablation (`colors = "uniform_gray"`), or a seeded hard variant with non-round
object sizes, masses/densities and hinge parameters (`physics = "nonround"`), or a
replicate with a rigidly jittered object layout (`pose = "jitter"`, free objects, no Panda), add a
private `[perturbation]` table instead of hand edits; see
[seeded color perturbations](../../docs/experiments.md#seeded-color-perturbations).
Prepare physics variants on a `test` node (it compiles the scene). Preparation
records exact values; `--start` verifies the compiled source first.

To be able to give a run more time later as if it had it from the start, give it the full wall budget up front
and a lower cost cap with `[inference] budget_hold_seconds`; at the cap it holds until you write
`RUN/hold-decision.json` (resume with raised caps, or stop). See
[budget hold](../../docs/experiments.md#continuing-a-run-at-its-spend-cap-budget-hold).

The older alternative: to continue a finished strict run, copy its configuration, set the new wall budget
and add `[continuation] parent = "RUN_ID"`, `submission = "last"` (or a number);
prepare with a fresh ID and start it as a strict run. Keep the source, `[simulation]`
and task unchanged; preparation refuses otherwise. Only the selected submission
project and model/adapter progress notes reach `/opt/public/prior`; see
[continuing a finished attempt](../../docs/experiments.md#continuing-a-finished-attempt).

For unchanged environments, `sim2sim experiment CONFIG --id ID` prepares and runs
in one step. Preserve the selected model and explicit budgets; never substitute
another model to make a run succeed. The wall budget includes preflight,
initialization, inference and private evaluation.

All physics, rendering, container builds and model experiments run on Slurm's
`test` partition. Build and inspect the trusted image on the same compute host
used for the run when images are node-local. For `strict_oci`, CPU/memory enforcement and private file boundaries must pass
before any agent runs. A blocked attempt is a valid recorded outcome; never
silently bypass it or substitute a different backend.

For the user-selected `development_shared_filesystem` mode, pass
`--development` deliberately. This mode is not OS isolated and cannot guarantee
that a spawned agent cannot read sibling private files. Use it only within the
user's explicit development-mode authorization. After the Slurm bridge is ready,
spawn the selected model with no inherited conversation history and provide only
the public task/RPC paths. This is a usage convention, not a security boundary.
Stop the spawned agent at the controller deadline/STOPPED signal; the native
controller cannot kill an independently spawned tool agent by itself.

Before an external model spawn, save the exact launch message in the run's
`experiment.json` as `orchestrator_prompt`, along with the actual model,
reasoning effort and fork mode. Append verbatim later steering messages and
their timestamps to `orchestration-messages.jsonl` in that run. Preserve the
controller's original `public/task.json` and audited task message; never replace
historical prompts with a revised preset. These records cover authored messages,
not an unavailable full external context or transcript. Mark missing historical
messages as missing, and label retrospective notes separately.

Strict broker credentials are checked by presence only. Never copy host credentials
into the image, configuration, task or logs. Physical-parameter descriptions,
source XML, manifests and correspondence stay private. Only the explicit goal,
model constraints and permitted observations go to the agent.

Record genuine timestamped progress and completion status. Provider-returned
summaries are labeled summaries, never hidden reasoning or a full external tool
transcript. After partial failure, leave unknown model-call and execution fields
unknown rather than asserting zero. Finish by refreshing
`sim2sim dashboard --root experiments` and reporting actual outcomes, artifacts
and unresolved infrastructure blockers.
