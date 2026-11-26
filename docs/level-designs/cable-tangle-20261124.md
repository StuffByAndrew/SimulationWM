# Cable tangles (design, 2026-11-24)

Designed with the user in the level-designer session; every setting below was confirmed by the user (2026-11-24).
Builders: follow `skills/add-level/SKILL.md` steps 3–7 from this document; do not re-ask the settings. Values marked
*builder* may be tuned, keeping the stated constraint.

## Hypotheses

| ID | Claim | Test |
| --- | --- | --- |
| H1 | Agents can tell one tangled cable from two when vision alone cannot decide | piece count, C2 (correct = 2) vs C1 (correct = 1; false-positive control) |
| H2 | Agents can identify a cable's material properties by manipulation | per-property errors vs a typical-cable prior |
| H3 | A second simultaneous point grasp helps (hold one strand, pull another; real untangling) | 1 vs 2 grasps |
| H4 | Requiring a full MJCF scene changes how well the tangle is understood | parameters only vs parameters + MJCF |

## Levels (point grasp; same material, colour, diameter and ball size in both)

- **C1 `cable_knot_single`**: one open cable, 1.6 m, with an overhand knot, coiled into a compact ball (~12–15 cm,
  *builder*) on the table.
- **C2 `cable_tangle_pair`**: one open cable, 1.0 m, with the same knot, entangled with a **closed loop** of the
  identical cable, 0.6 m, **not topologically linked** (it can come off). Total length equals C1's.
- Both levels show exactly **two free ends**, so counting ends cannot reveal the second piece; the agent must trace,
  straighten or separate.
- Cable model (*builder*): a chain of capsules (~1.5 cm segments, Ø ~8 mm, so a strand is graspable from a close
  movable-camera view) with ball joints carrying bending stiffness and damping; the loop closed by a connect equality.
  Inextensible. Same matte colour for both pieces. The table's friction is lower than the cable's, so the cable's μ
  governs cable–table and cable–cable contacts.
- The tangled state must be the scene's **compiled initial state** (bodies placed at their settled poses, joints at
  zero; no keyframe). The generator ties the knot, threads the loop, settles on a Slurm node and writes the settled
  poses.
- **Hidden by design**: the check script renders the initial scene from sampled movable-camera poses with a
  researcher-only segmentation. The loop's visible fraction must stay low (≤ ~30%, *builder*), and no view may show it
  as a clearly closed ring.

## Sensing and settings (all conditions)

Point grasp with `wrench = true`; RGB-D with `movable_camera = true`; lock-step; reference interaction allowed;
unlimited resets; `sequence` tool; no hints; no perturbation. gemini-4-neon, **n = 1**, **4 h wall** (14400 s,
`wall_only`), no checkpoint hold, `provider_hold_seconds` as in recent Gemini waves, accounting-only caps, a new Gemini
pool.

## New capability: two simultaneous point grasps

`[simulation] point_grasps = 1 | 2`, default 1 so every existing config is unchanged. With 2, grasp/move/release take a
grasp index, each grasp keeps its own connect constraint, and `wrench` reports one force per active grasp under the
existing wrench rules (only that grasp's object-on-handle force, zero torque about its grasp point). Coordinate per
AGENTS.md: controller task contract, enforcement, tick accounting (unchanged per tick), harness tool text,
`docs/protocol.md`, tests, image rebuild. AGENTS.md's "one ideal point grasp" scope statement becomes "one or two
(configured)", approved by the user 2026-11-24.

## Conditions (6 runs; one factor per pair against the shared baseline)

| Condition | Grasps | Submission |
| --- | --- | --- |
| C1-base, C2-base | 1 | `parameters.json` only (JSON submission) |
| C1-two, C2-two | 2 | `parameters.json` only |
| C1-mjcf, C2-mjcf | 1 | scene project + `parameters.json` in the same ZIP |

Public task (identical across C1 and C2): the table holds a tangle of cable; report in `parameters.json` how many separate
pieces of cable there are and, for each piece, whether it is a closed loop, its length (m), diameter (m), mass (kg),
bending stiffness (N·m²), surface friction coefficient and damping. The exact JSON format is stated in the task.
The MJCF conditions add: also submit a MuJoCo scene of the cable(s) in their initial state. Never state any value.

## Metrics (private) and what must be built

| Metric | Use | Status |
| --- | --- | --- |
| Piece count correct; per-piece open/closed correct | primary (H1) | new `cable` answer metric (generalize `lowrank`'s JSON answer to a manifest-declared schema) |
| Per-piece length, diameter, mass, bending stiffness EI, friction μ, damping: log-ratio errors; pieces matched by open/closed then length | H2 | same metric |
| Typical-cable prior baseline (generic values, e.g. a USB cable) scored by the same metric | baseline | new baseline entry |
| MJCF conditions: Chamfer of the cable surfaces in the initial state, `dynamics` probes (lift each free end), and the piece count implied by the scene (connected bodies and equalities) vs the stated one | H4 | existing Chamfer/probes + small count check |

A missing or malformed `parameters.json` makes the answer metrics unavailable (a gap), not zero. (Superseded 2026-11-25: with the weld grasp, twist stiffness is identifiable and is added to the answer; see Revisions.) Twist stiffness and
stretch are excluded: a point grasp cannot apply torque, and the cable is inextensible.

## Evals card

- **Quantitative**: piece-count correctness (C2) and false-positive rate (C1); time of the first correct count in the
  submission history; per-property errors vs the prior; MJCF Chamfer/probe errors; compared at matched wall minutes and
  tokens.
- **Qualitative**: tracing ends, straightening by pulling the ends, using the second grasp as an anchor, separating the
  loop, weighing and hanging tests; failure modes: "two ends, so one cable", pulling at the knot without progress.
- **Human visuals**:
  - *run check*: dashboard RGB video with the grasp point(s) marked, action timeline, physics inspector on MJCF
    versions.
  - *report*: initial-to-final filmstrip of the reference world with each piece in its own colour; property error bars
    against the prior; piece-count table by condition.
  - *dashboard*: a researcher-only per-piece colouring render (new); it never reaches the agent.

## Queueing notes

Image rebuild required (two grasps, cable metric, tasks). Record the batch in `docs/experiment-log.md`.

## Revisions (user, 2026-11-24; supersede the text above where they conflict)

The first scene build (jobs `63327823`–`63328679`) found that a released tangle (EI 1e-4–1e-3 N·m², μ 0.6–1.0) sprawls
flat (17–30 cm × ~2 cm), creeps, and leaves C2's loop ~96% visible. Decision: **no container and no elastic memory**
(the rest shape stays straight). Instead lower the bending stiffness (e.g. toward 1e-6 N·m²) and raise the surface
friction (e.g. toward ~2) until the tangle stays compact (≲ 15 cm) and at rest; elliptic friction cones or noslip are
allowed against creep. C2's loop counts as hidden when its visible fraction is ≤ ~30% **and** no sampled camera view
shows its whole closed path. The tuned values become the truth the agent must report; flag values outside plausible
cable ranges. Wave 30 (`experiments/configs/wave30-20261124/`, pool `gemini-20261124-cable`).
- **Weld grasp (user, 2026-11-25)**: the point grasps are weld grasps (AGENTS.md): the agent rotates the grasped segment
  about the grasp point and the wrench reports full force and torque. Because the agent can now twist the cable, **twist
  stiffness (N·m²) is added to the per-piece answer** and to the `cable` metric; stretch stays excluded.
- **Hidden-loop criterion (designer, 2026-11-25; supersedes "≤ ~30% visible")**: with EI 1e-5 N·m², μ 1.2–2.0 and
  anti-creep settings the tangle is compact (10–14 cm) and at rest, but the loop stays 43–85% visible on average: ~0.9 m
  of 8 mm cable cannot cover the outside of a 0.6 m loop. What reveals a second piece is a traceable closed path, so the
  loop counts as hidden when, in **every** sampled view, its longest contiguous visible run is **≤ 50% of its length**,
  the mean visible fraction over views is ≤ ~0.6, and no view shows the whole closed path. If no seed passes at 0.6 m,
  the loop becomes **0.4 m with a 1.2 m main cable** (total still 1.6 m; same diameter, material and colour). A thinner
  loop is not allowed (it breaks the same-diameter control).
