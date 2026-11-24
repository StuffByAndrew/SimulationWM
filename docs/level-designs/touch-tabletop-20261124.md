# Touch-only tabletop levels (design, 2026-11-24)

Designed with the user in the level-designer session; every setting below was confirmed by the user (two settings
rounds, 2026-11-24). Builders: follow `skills/add-level/SKILL.md` steps 3–7 from this document; do not re-ask the
settings. Physical values marked *builder* may be tuned for stability or reach, keeping the stated constraint.

## Hypotheses

| ID | Claim about the agents | Test |
| --- | --- | --- |
| H1 | They can find and reconstruct several free objects on a table from touch alone with an arm-mounted hand | L1 absolute scores (floor reference: w21 single object in palm, final Chamfer 2.5–7.3 mm) |
| H2 | Recognizable shapes are reconstructed better than matched abstract shapes, without being told what they are | L1 vs L2-unstated (identical task text) |
| H3 | Being told the object categories improves reconstruction (and may bias the unusual dimensions toward priors) | L2-unstated vs L2-stated |
| H4 | They can tell a spiky ball from smooth ones by touch (identification) and grab it (manipulation) | L3, scored separately |
| H5 | The wave 25 touch tooling helps on these levels | harness pair on every level |

Predictions: H2 holds if L2-unstated beats L1 on per-object Chamfer and labels are right; prior over-trust shows as the
unusual dimensions reverting to typical values. H3 holds if L2-stated beats L2-unstated. If the identification in L3
succeeds while the grab fails, the gap is dexterity, not sensing.

## Embodiment and sensing (all levels)

- **Shadow hand mounted on the Franka** in place of the Panda gripper (new fixture: `robots.py` entry combining the
  `panda` arm and the Menagerie `shadow_hand` E3M5 right hand, both verified byte for byte; actions `arm_target` (7) and
  `hand_target` (20)). **Every object and grasp pose must be reachable**: the check script must show IK solutions
  (Slurm) with the palm facing down over every object position, the whole bin interior and 15 cm above it.
  Arm gains must hold the hand's weight without sag beyond a few mm (*builder*).
- **Touch only**: arm and hand joint proprioception and efforts, fingertip/palm taxels (`tactile = true`); no images,
  no point grasp, no wrench. Lock-step time. Reference interaction allowed, unlimited resets (reset restores the scene).
- Table: the panda_small table (`TOP, TABLE` constants), all objects free rigid bodies (no welds), friction high enough
  that a light touch does not slide them (*builder*, e.g. μ 0.8).

## Levels

### L1 `panda_shadow_table_shapes` (abstract control)
Three free abstract objects at the same three table poses as L2, each **matched to one L2 object** in bounding box (±10%),
primitive count (±1), mass (±10%) and density layout, but with no recognizable category. Suggested (*builder*): a stepped
tower of three offset boxes (mug match), a zigzag chain of five capsules with alternating bends (banana match), two
boxes crossing at 60° with a sphere at one end and a dense part matching the hammer's head (hammer match). Objects ≥ 8 cm
apart, each resting stably.

### L2 `panda_shadow_table_objects` (everyday objects)
- **Mug**: hollow ceramic (2300 kg/m³), Ø 75 mm, **120 mm tall** (unusually tall; typical ~95 mm), wall and base built
  from primitives, handle of capsules on one side.
- **Banana**: chain of 5–6 capsules along an arc, Ø ~35 mm, ~190 mm along the arc, **radius of curvature 90 mm**
  (unusually strong curve), 950 kg/m³.
- **Hammer**: wooden handle (600 kg/m³) **160 mm long** (unusually short) Ø 22 mm, steel head (7800 kg/m³) 100 × 25 × 25 mm
  lying flat; its COM sits near the head.
- The three bolded dimensions are the *prior-defeating dimensions*, scored separately.

### L3 `panda_shadow_bin_spiky`
A rigid open-top bin (inner ~24 × 24 cm, walls 30 mm so the forearm clears the rim with a pitched wrist, *builder*)
holding four free balls of equal mass (~60 g), **equal outer envelope (~90 mm) and equal inertia** (explicit inertials):
one **spiky** (55 mm core plus 6–8 cones 15–20 mm tall, bases at least one palm taxel cell wide) and three smooth spheres
as wide as the spike-tip envelope. Seeded, deterministic positions on the bin floor. Revised 2026-11-24 (user) after the
first check (jobs `63317646`, `63319921`, `63320350`) found the fine spikes unreadable by the taxel grid and a size
shortcut (spiky envelope 75 mm vs 55 mm smooth). The check is mandatory before configs: scripted palm presses give
clearly different taxel patterns; top height and contact envelope agree within a few mm; inertia agrees within
tolerance; a scripted grab lifts a 90 mm ball.

## Conditions (8 runs, gemini-4-neon, n = 1)

| Condition | Level | Task addition (public) |
| --- | --- | --- |
| L1 | L1 | name each object body after what the object is |
| L2-unstated | L2 | name each object body after what the object is (identical to L1) |
| L2-stated | L2 | same, plus "The objects are a mug, a banana and a hammer." |
| L3 | L3 | find the spiky ball among the balls in the bin, report its initial position in `parameters.json` (`spiky_ball_xyz`, world m), model the scene, and finish holding the spiky ball above the bin rim |

Each condition × harness {wave 21: `sequence` tool; wave 25: `sequence` + `local_controller` + pausable bash} = 8 runs.
Task base: the robot family's stop-when-done task (`TASK_CONVERGE`) and model constraints (primitives plus inline convex
meshes, robot fixture unchanged). Never state masses, sizes or positions. Budget: **4 h wall** (14400 s, `wall_only`),
no checkpoint hold, `provider_hold_seconds` as in recent Gemini waves, accounting-only cost caps, a new Gemini pool.

## Metrics (private) and what must be built

| Metric | Levels | Status |
| --- | --- | --- |
| Per-object found + centroid error (initial state; geometry-only assignment) | L1, L2, L3 | new or extend `chamfer` with a per-object breakdown |
| Per-object Chamfer (primary for L1/L2) | L1, L2 | `chamfer` with `object_bodies`; check per-body assignment |
| Prior-defeating dimensions (candidate extent vs truth along the three bolded axes) | L2 | new, small |
| Prior-only baseline: canonical mug/banana/hammer (typical sizes) at the true poses, scored by the same metrics | L2 | new baseline entry |
| Labels: candidate body names vs a private per-object synonym list, matched through the geometric assignment | L1 (unscored, reported), L2 | new, small |
| `center_of_mass` / `density` (secondary) | L1, L2 | existing |
| Identification: nearest initial ball to `spiky_ball_xyz` is the spiky one; distance error | L3 | new, reads the `parameters` submission kind |
| Grab success at finish on the reference world: spiky ball COM ≥ rim + 2 cm, in hand contact, all smooth balls in the bin | L3 | new |

## Evals card

- **Quantitative**: L1/L2 primary per-object Chamfer, then found rate, centroid error, labels, prior-defeating
  dimensions vs the prior-only baseline; L3 identification correct/incorrect + distance, grab success. Compare at matched
  wall minutes and matched cumulative tokens; report malformed-call and timeout rates.
- **Qualitative**: exploration strategy (systematic sweep, edge following, random poking), whether agents pick a
  category template then verify by touch or build from contacts, use of `local_controller`, knocking objects over and
  using reset, how the spiky ball is told apart (press patterns, rolling).
- **Human visuals**:
  - *run check*: dashboard researcher video of the reference world, action timeline, model context, physics inspector
    on submitted versions.
  - *report*: 3D true-vs-reconstructed overlay per object; a **contact coverage map** (every taxel contact in world
    coordinates over the true scene); label table (L2, L1 for reference); L3 identification/grab table plus the final
    grab video clip.
  - *dashboard*: the contact coverage map is a **new view** to build (researcher-only, from the audited
    observations; no new agent-visible data).

## Queueing notes

Image rebuild required (new fixture, metrics, task texts). Record the batch in `docs/experiment-log.md`. Wave 25 (touch
harness on the w21 hand levels) is running; the user chose to queue these without waiting for it.

## Revisions (user, 2026-11-24; supersede the text above where they conflict)

- **L3 ball size**: the spiky ball (6 octahedral cones, 20 mm proud, 29 mm bases, 22 mm core; check job `63323774`) rests
  on three tips and is wider than tall, so no sphere matches both. Smooth balls match its **resting height** (Ø 48.5 mm);
  the 14–20 mm width difference is accepted as the spiky shape itself (it needs sideways probing to notice). Mass and
  inertia stay equal.
- **L3 taxel check**: passes if a scripted procedure with up to 3 palm presses per ball identifies the spiky ball in
  ≥ 95% of trials (single presses: smooth 1 taxel in 9/9, spiky 2–3 in 6/9).
- **L3 grab**: kept hard (scripted: spiky 1/72 open-loop tries, smooth 6/15; jobs `63324572`, `63324769`); this is a
  dexterity finding, and identification is scored separately. Walls 40 mm to cut escapes over the rim. All balls keep
  equal friction.
