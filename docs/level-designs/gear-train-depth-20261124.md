# Explanation depth: gear train (design, 2026-11-24)

Designed with the user in the level-designer session; every setting below was confirmed by the user (2026-11-24).
Builders: follow `skills/add-level/SKILL.md` steps 3–7 from this document; do not re-ask the settings. Values marked
*builder* may be tuned, keeping the stated constraint.

## Question

When a mechanism admits two explanations with **identical predictive power** on everything the agent can probe, does
the agent build the deeper (mechanistically faithful, more expensive) model, the shallower (lumped, cheap) one, or both
to decide, and does the stated downstream use change that choice?

| ID | Claim | Test |
| --- | --- | --- |
| H1 | Unprompted, agents prefer one depth (mechanistic fidelity vs parsimony) | depth class in G-recon |
| H2 | A stated use where cost matters moves agents to the shallow model | G-recon vs G-cheap |
| H3 | A stated use that needs internal structure moves agents to the deep model | G-recon vs G-deep |
| H4 | Agents compare depths (build and time both) before choosing | trace classification, all conditions |

H2 alone could mean "a task was mentioned, so do less"; H3 is the control that shows whether agents match depth to the
task in both directions.

## Scene `gear_train` (point grasp; precedent `damped_lever`)

- A vertical panel: a **crank** (hinge, handle at ~40 mm radius) on the left, a **dial hand** (hinge) on the right, and
  between them a visible train of **~20 gears** (idlers and compound pairs; overall ratio about −1:7.5, *builder*).
  Gears are thin cylinders with non-colliding tooth visuals, axles at their centres (balanced: no gravity torque).
- The mesh couplings are joint equality constraints (ideal gears). All gears are one material (uniform density) and
  every axle has the same viscous damping, so a deep model built from the visible geometry with uniform values is
  exactly right; only the lumped sum is identifiable from crank/hand data.
- Only the crank handle and the dial hand are graspable. Gears are visible to RGB-D and the movable camera, but the grasp
  refuses them and nothing the agent can move can contact them. No transparent pane (it would occlude the depth image).
- **Shallow explanation**: two hinges, one coupling with the overall ratio and sign, the gears' inertia
  (Σ I_i r_i²) and damping (Σ b_i r_i²) lumped into the crank joint (armature, damping).
  **Deep explanation**: every gear its own body and hinge, pairwise couplings.
- **Check script (Slurm) must prove the premise**: we author both reference models; they score within noise of each
  other on every probe, and the deep one costs **≥ 10×** the shallow one in step time on the evaluation worker
  (raise the gear count if not). It must also show gears render in RGB and depth, the grasp refuses gears, and the
  removal interventions below are well defined.

## Sensing and settings (all conditions)

Point grasp with `wrench = true`; **RGB-D** with `movable_camera = true` (RGB-D is now the default for new levels);
lock-step time; reference interaction allowed; `sequence` tool; no hints; no perturbation. gemini-4-neon, **n = 1**,
**4 h wall** (14400 s, `wall_only`), no checkpoint hold, `provider_hold_seconds` as in recent Gemini waves,
accounting-only caps, a new Gemini pool.

## Conditions (3 runs)

All share a **neutral base task** (not `TASK_CONVERGE`): "Build a MuJoCo model of this mechanism. Name the crank body
`crank` and the dial hand body `hand`." plus the point-grasp model constraints and submission instructions, and a plain
finish rule (finish when done). The only factor is one appended sentence:

| Condition | Appended sentence (public) |
| --- | --- |
| G-recon | none |
| G-cheap | "Your model will be used by a planner that runs many thousands of simulations of how the hand moves when the crank turns; it is scored on prediction accuracy and simulation speed." |
| G-deep | "Your model will be used to predict how the crank and hand move after individual gears are taken out of the train; we remove gears from your model by their axle position, and it is scored on those predictions." |

Never say "simple", "lumped", "detailed" or "every gear" in any condition.

## Metrics (private) and what must be built

| Metric | Use | Status |
| --- | --- | --- |
| `structure`: rotating bodies and couplings between `crank` and `hand`; candidate gears matched to source gears by axle position (geometry only); depth class shallow / partial / deep (≥ 80% of gears matched) | primary, all conditions | new |
| `sim_cost`: median step time per simulated second over 5 runs in the evaluation worker, plus body, constraint and contact counts | all; G-cheap score | new |
| Probe accuracy: crank turns (slow, fast, step) and hand pushes; unwrapped world-frame rotation of the `crank` and `hand` bodies about the panel normal, relative to their initial orientation (not joint qpos, not image masks, which would penalize visibly different gears) | accuracy, all; continuous error always recorded | `[[probes]]` with a rotation comparison |
| G-cheap score: accuracy gate = the worst probe error of the shallow reference over {armature ±10%} × {damping ±10%} (one manifest value; the exact reference itself has ≈ 0 error); over the gate is "fails accuracy gate" with no score (not zero), under it the score is step cost relative to the shallow reference | G-cheap | new, small |
| G-deep score: for each of a few gears, remove it from source and candidate (matched by axle position), then the same probes; a candidate without that gear is "cannot answer" (a gap, never a zero or fabricated error) | G-deep (reported for all) | new |
| Depth comparison in traces: did the agent build ≥ 2 model variants of different depth, simulate and time them | H4 | new researcher-side classifier over the audit (never agent-visible) |

## Evals card

- **Quantitative**: depth class of every submission and of the final one (primary); probe accuracy (expected equal
  across depths); simulation cost vs both references; the G-cheap and G-deep scores; all at matched wall minutes and
  tokens.
- **Qualitative**: explicit reasoning about which depth is needed and why; whether the stated use is cited; timing of own
  models; building both before choosing; how lumped inertia/damping were identified.
- **Human visuals**:
  - *run check*: dashboard video, model-context and candidate diff, physics inspector on submitted versions (which gears
    exist).
  - *report*: scatter of simulation cost vs probe error per submission with the deep and shallow references marked;
    side-by-side renders of each condition's final model; table of depth choice by condition.
  - *dashboard*: a small per-version model-structure summary (bodies, joints, couplings, step time); new, small.

## Queueing notes

Image rebuild required (new metrics, neutral task). Record the batch in `docs/experiment-log.md`.
