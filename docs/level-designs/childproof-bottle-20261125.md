# Childproof bottle (design, 2026-11-25)

Designed with the user in the level-designer session; every setting below was confirmed by the user (2026-11-25).
Builders: follow `skills/add-level/SKILL.md` steps 3–7 from this document; do not re-ask the settings. Values marked
*builder* may be tuned, keeping the stated constraint.

## Hypotheses

| ID | Claim | Test |
| --- | --- | --- |
| H1 | Agents can discover a force-gated hidden mechanism (push to engage, then turn) by interaction, without a semantic cue | mechanism probes, all conditions (neutral look only) |
| H2 | A second simultaneous point grasp helps or is needed (hold the bottle while working the cap) | 1 vs 2 grasps |
| H3 | When the bottle holds unpredictable contents, agents open it to complete their model, rather than modelling a closed or guessed bottle | contents vs empty |

The empty variant is H3's control: its opening rate shows how often agents open the cap with no contents to find.

## Scene `childproof_bottle` (point grasp; precedents `tunnel_latch`, cable two-grasp)

- A **free** bottle standing on the table, **neutral look**: a plain grey cylinder with a grey cap, no medicine colours,
  labels or bottle semantics (user: neutral only). Body ~Ø 45 × 80 mm, cap ~Ø 50 mm (*builder*).
- **Push-and-turn bayonet cap**, all of it plain-MJCF contacts of primitives:
  - an **outer shell** that spins freely over an **inner cap**, held up by a hidden spring (preload sets the engage force,
    ~8 N, *builder*);
  - pushing the shell down past the preload engages hidden **teeth** between shell and inner cap;
  - turning the engaged shell rotates the inner cap through a **quarter-turn bayonet** track on the neck (hidden lugs),
    after which the cap **lifts off**.
- Spring, teeth and lugs are inside the opaque cap: invisible to RGB-D and the grasp ray. Graspable: bottle body and
  cap shell.
- The check script (Slurm) must show:
  - twist without push leaves the bottle closed (the shell spins);
  - a push below the preload, then twist, leaves it closed;
  - a push above the preload, then a quarter turn, then lift, opens it with two scripted grasps, deterministically;
  - whether one grasp can open it (press so table friction holds the bottle, then twist). **Record the result either
    way**: one-grasp infeasibility is acceptable and is part of what H2 measures;
  - the hidden parts stay hidden while closed.

## Variants

| Variant | Contents |
| --- | --- |
| `childproof_bottle` (empty) | none |
| `childproof_bottle_contents` | several loose objects of distinct shapes and colours, e.g. a 12 mm cube, a Ø 8 × 20 mm cylinder and a Ø 10 mm sphere, ~15 g in total (*builder*) |

The contents rattle (shaking reveals "something loose" and the wrench gives total mass), but their count, shapes and
colours cannot be inferred while the bottle is closed; they become visible to RGB-D only once the cap is off (looking
in, or tipping them out). Bottle and cap are identical across variants.

## Sensing and settings (all conditions)

Point grasp(s) with `wrench = true`; RGB-D with `movable_camera = true`; lock-step; reference interaction allowed;
unlimited resets; `sequence` tool; no hints; no perturbation. The usual stop-when-done task (`TASK_CONVERGE`) and
point-grasp model constraints; the agent submits an **MJCF model** as usual. The task never mentions a cap mechanism,
contents or opening. gemini-4-neon, **n = 1**, **4 h wall** (14400 s, `wall_only`), no checkpoint hold,
`provider_hold_seconds` as in recent Gemini waves, accounting-only caps. Two grasps use `[simulation] point_grasps = 2`
(the capability built for the cable levels).

## Conditions (4 runs)

| Condition | Variant | Grasps |
| --- | --- | --- |
| B-empty-1, B-empty-2 | empty | 1, 2 |
| B-contents-1, B-contents-2 | contents | 1, 2 |

## Metrics (private) and what must be built

| Metric | Use | Status |
| --- | --- | --- |
| Mechanism probes on the submitted model: twist only; push below threshold + twist; push above + quarter turn + lift. World-frame cap rotation about the bottle axis and cap lift vs the source | primary (H1) | `[[probes]]` with a rotation/lift comparison (as built for the gear train) |
| Engage force and opening angle implied by the model, vs truth (from probe sweeps) | secondary | new, small |
| Chamfer / COM of bottle, cap and contents in the initial state; contents per-object found + count (geometry-only assignment) | H3 | existing Chamfer/COM + the per-object breakdown built for the tabletop levels |
| Behaviour: opened in the reference world (cap lifted off), time of first opening, looked inside / tipped contents out | H2, H3 | new, from the audited reference-world states |

A model that omits the contents leaves the contents metrics as misses (counted, not zero error); unavailable stays a gap.

## Evals card

- **Quantitative**: mechanism probe agreement (primary); engage-force and angle errors; contents count and per-object
  Chamfer (contents variant); opening success and time in the reference world; all at matched wall minutes and tokens.
- **Qualitative**: whether the agent tries pushing at all or only twists; whether it reads the wrench to find the
  threshold; how it holds the bottle with one grasp; whether it opens the bottle to look inside and what it does with
  contents it cannot see (guess, omit, state uncertainty); failure modes: twisting forever, tipping the bottle.
- **Human visuals**:
  - *run check*: dashboard video with both grasp points, action timeline, physics inspector on submitted versions.
  - *report*: probe-outcome table (truth vs model) by condition; push force vs cap rotation achieved; a clip of the
    first opening; contents reconstruction overlay (contents variant).
  - *dashboard*: existing views suffice.

## Queueing notes

Needs the two-grasp capability (cable branch) merged first. Image rebuild (isolated agent/evaluator images). Record the
batch in `docs/experiment-log.md`.

## Revisions (user, 2026-11-25; supersede the text above where they conflict)

- **Weld grasps**: both point grasps are weld grasps (AGENTS.md): the agent translates each grasp and rotates the grasped
  part about its grasp point; each wrench reports full force and torque. Twisting the cap is a rotation of the cap grasp
  about the bottle axis (no longer only a tangential translation). Two welds plus the cap's own joints form a closed
  loop, so pose mismatch appears as constraint force in the wrenches: the check must show the scripted opening is
  stable (no jitter or blow-up) with welds, and redo the one-grasp feasibility test with a weld.
