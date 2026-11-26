# Environment authoring

Each directory in `environments/` contains a MuJoCo `scene.xml` and a
`manifest.toml`. The XML must define a fixed camera named `fixed` (or specify
another camera in configuration). Use named movable bodies and standard MuJoCo
joints, masses, contact parameters, springs and damping. Assets resolve relative
to the scene directory (or compiler meshdir/texturedir). Includes are expanded
recursively before movable-body discovery; nested include paths resolve relative
to their containing file. Cyclic includes fail. The selected camera must be a
fixed, direct worldbody perspective camera using fovy; attached/tracking cameras,
orthographic projection and custom sensor/focal intrinsics are unsupported.

The manifest identifies `name`, `scene`, `camera`, and `graspable_bodies`. A
`[correspondence]` table maps source body names to submitted body names. Each
`[[probes]]` entry identifies a movable `body`, a `local_point` in its frame,
a world `displacement` in meters and a positive integer `ticks` duration. These
are privileged evaluation descriptions, not observation fields.

`blocks` supplies two free blocks on a table. `spring` supplies a hinged lever
with joint stiffness and damping. Both use 2 ms integration steps and the same
fixed perspective camera. The spring returns toward its rest pose after release;
its motion also depends on gravity.

`random_assembly` (seed `20261028`) is an irregular connected assembly of
randomized boxes, spheres, capsules, ellipsoids and cylinders. `primitive_boat`
(seed `20261029`) arranges randomized primitive dimensions and colors into a
boat-like hull, deck, cabin and fittings. Each scene has exactly ten object geoms
joined on one free body, plus a separate ground plane. Both omit the gravity
attribute to use MuJoCo's default gravity and retain a 2 ms timestep. Their fixed
oblique cameras center the initial object and are designed to hide some rear or
central parts; rendering and occlusion validation run on Slurm compute nodes.

At load time the interaction engine adds an invisible mocap body and inactive
connect equality for each graspable body. A grasp activates exactly one connect
constraint at the visible ray hit point. Moving interpolates the mocap target
through the requested world displacement across physics steps. The original
scene's joints and contact dynamics remain active. Avoid naming authored bodies
or equalities with the reserved `_grasp_` prefix.

## Sliding friction

The blocks scene sets `condim="3"` and `friction="0.8 0.02 0.002"` in its shared
geom default. Thus the table and both blocks have sliding coefficient μ = 0.8.
`condim=3` enables normal and sliding contact forces, without torsional/rolling
friction constraints. For dynamically generated contacts at equal geom priority,
MuJoCo takes the elementwise maximum of the two geom friction vectors. A
higher-priority geom supplies its coefficients instead; an explicit contact pair
supplies its own parameters. See [MuJoCo contact parameter rules](https://mujoco.readthedocs.io/en/latest/modeling.html#contact-parameters).

Reducing only a block's μ below the table's 0.8 therefore leaves the effective
sliding coefficient at 0.8. The friction regression changes the shared default,
so both surfaces change. To study object μ separately, author a known table
coefficient below the tested object range, or deliberately use geom priority or
explicit pair parameters. These choices expose different parameter contracts;
they do not establish unique material inference from motion. Drag/release probes
measure contact behavior, while privileged parameter scores compare declared
geom coefficients.

Validation: Slurm `test` job `56034580` confirmed both new scenes contain ten
object geoms plus a plane, with one free body, effective default gravity
`[0, 0, -9.81]`, and stable motion over 1000 ticks (2 seconds). Random assembly
parts `part_03` and `part_05`, and boat `rear_fender`, are hidden initially;
alternate views reveal them. Each environment directory contains `preview.png`.


## Per-primitive density in the new scenes

Each of the ten object primitives has an explicit homogeneous volumetric density
in kg/m³ and no geom `mass` override. The densities are independent seeded draws
from a uniform range of 200–900 kg/m³, in XML geom order: seed `2026102801` for
`random_assembly` (realized range 212.206–877.077), and seed `2026102901` for
`primitive_boat` (realized range 387.520–876.272). Colors and geometry are retained;
density may differ between primitives. Slurm `test` job `56041119` validated the density variants for 1000 ticks
(2 seconds), preserved initial occlusions, and regenerated their previews.

MuJoCo infers the rigid body's mass, center of mass and inertia by adding the
individual homogeneous geom contributions. Overlapping regions therefore count
for each participating geom: these are additive primitive solids, not a Boolean
union with a single material-density field. Explicit geom mass would override
density, so the scenes omit it. See the [MuJoCo geom density and inertia
reference](https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-geom).


## Damped lever and unequal-mass tetrahedron

`damped_lever` provides a separate unsprung hinged arm and handle on a fixed
standing fixture. Its named `hinge` has damping `0.12 N·m·s/rad`, dry joint
friction loss `0.05 N·m`, and a ±65° range; stiffness is left at MuJoCo's default
zero. Moving geometry remains clear of the floor throughout that range. The
fixture is world geometry; its visual axle has collision disabled. The manifest
automatically matches a single source hinge to a single candidate hinge without
requiring its name, and probes upward/downward pulls followed by release.

`tetrahedral_cluster` joins four equal-radius (`0.16 m`) spheres on one free body
at regular tetrahedron vertices, with edge length `0.30 m`. The `0.02 m` overlaps
connect the primitives. Four colors distinguish explicit homogeneous sphere
masses of **1, 2, 4 and 8 kg**, totaling **15 kg**; implied densities follow from
the common sphere volume. This additive mass convention includes the overlaps.
Three distinct surface pulls/lifts/drags followed by release probe the uneven
mass distribution. The ground plane is separate from the four object geoms.

Both scenes keep default gravity and fixed world cameras, and enable the shared
Chamfer, per-primitive density and COM metrics through `[reconstruction]`.
Reconstruction sampling seeds are `20261030` (lever) and `20261031` (tetrahedron).
Their post-addition physics, probes and previews are validated on Slurm.

Slurm `test` job `56047534` validated both additions for 1000 ticks (2 seconds)
and all five manifest probes. The tetrahedron compiled to 15 kg with world COM
`[0.02, 0.023094, 0.295639] m`; reducing its 8 kg sphere to 1 kg preserved zero
Chamfer error while producing `0.103531 m` COM error. Lever damping/friction-loss
perturbations produced the expected parameter and held-out motion errors; exact
copies scored zero. Previews are saved beside each scene.

## Panda arm variants

`environments/panda/` contains Panda versions of `blocks`, `spring`,
`damped_lever`, `tetrahedral_cluster`, `random_assembly`, and `primitive_boat`.
It also contains a separate, proprioception-only `blocks_proprio` level and a
proprioception-only `fixed_cube_proprio` level. The latter has a jointless cube
body fixed to the table. Existing nonrobot environments remain separate.

All eight scenes share the public `robot/` fixture, derived from the
[MuJoCo Menagerie Panda model](https://github.com/google-deepmind/mujoco_menagerie/tree/b0655057a9df2abe39ea7fc989914ba6f248301c/franka_emika_panda).
Its kinematics, inertials, joint limits, actuator gains and gripper coupling are
retained. Official collision meshes also supply compact visual geometry; the
detailed upstream visual meshes are omitted. The pinned revision, changes and
Apache-2.0 license are recorded in `robot/IMPORT.md` and `robot/LICENSE`.

The robot has seven position-controlled joints and a coupled two-finger
gripper. Public observations are limited to these joint and finger positions,
velocities and arm actuator efforts. Six scenes additionally enable RGB and
camera calibration. The two proprioception-only scenes disable public RGB,
depth, calibration and point-grasp wrench. The fixed camera remains in each
scene for trusted researcher uses. `arm_target` controls the gripper and arm
through physical contact; pixel grasp actions are unavailable in Panda levels.
Source manifests contain private target-body and target-geom selectors and arm
probe trajectories.

`panda/tetrahedral_cluster` change, 2026-11-03: the cluster body moved from
`pos="0 0 0.565"` to `pos="0 0.2 0.565"`. At the Panda home pose the hand and left
finger previously started about 50 mm inside the top 8 kg sphere, so non-round hard
variants refused the scene. Masses, radii, sphere layout, height, table and robot
are unchanged; the `reach_close_retreat` arm probe still descends onto the cluster.
Runs prepared earlier keep their private copies of the old layout.

## Small Panda variants

`environments/panda_small/` provides eight separate smaller-object variants of
all Panda levels, leaving `environments/panda/` unchanged. Each target retains
its primitive types, colors, relative poses and connectivity. The robot keeps
its original dimensions and dynamics. Target placements move toward the robot,
centered around world y = -0.15 m; the tabletop is 0.76 × 0.68 m at z = 0.4 m.
The fixed cube remains a jointless body.

| Levels | Uniform target scale | Representative grasp dimension |
| --- | ---: | --- |
| Blocks, blocks proprio, fixed cube proprio | 0.375 | Red/cube 45 mm; blue 60 × 37.5 × 37.5 mm |
| Spring, damped lever | 0.45 | Handle diameter 49.5 mm |
| Tetrahedral cluster | 0.15625 | Each sphere diameter 50 mm |
| Random assembly | 0.12 | Primitive widths approximately 17–43 mm |
| Primitive boat | 0.15 | Hull width 55.1 mm; cabin width 35.2 mm |

Density values are preserved; explicit geom masses scale by the cube of the
length scale. Lever supports scale with their target, and hinge stiffness,
damping and dry friction torque scale by the fourth power of the length scale,
matching the scaling of gravity torque. This defines new physical scenes rather
than changing only a descriptive parameter. Private point-probe distances scale
with the target; private arm probes use physical approach, close and lift steps.

A common closer global camera frames the full arm and compact workspace. The
public robot fixture adds an offset camera named `wrist` on the hand, pointing
along hand +Z toward the fingertips. The six vision presets
`experiments/configs/astra-medium-panda-small-*-6h.toml` request both cameras at
640 × 480. Both proprio-only presets keep RGB, depth and calibration disabled
and omit `simulation.wrist_camera`. All eight retain the explicit development
backend, Astra medium and the six-hour wall-only budget. These presets do not
start experiments automatically.

Slurm `test` validation job `57149940` compiled all eight variants and exercised
their physical arm probes. Free targets remained lifted at the end of the closed
gripper step: blocks 106–112 mm, tetrahedron 111 mm, random assembly 74 mm and
boat 105 mm, with sustained bilateral finger contact above 20 mm. Both hinged
handles and the fixed cube received actual finger contacts; hinge excursions
were 0.43 rad (spring) and 1.20 rad (damped lever, including soft-limit contact
transients). At home and each probe phase endpoint, the full robot mesh, table
and pedestal stayed inside the global image with at least 6.26% normalized
frame margin. Researcher-only global/wrist previews and the full measurements
are in `experiments/validation/panda-small/`; wrist previews for proprio levels
are private validation artifacts, not enabled public observations.

## Shadow hand tactile shape levels

`experiments/validation/hand-scenes.py` writes four `environments/shadow_hand/` levels (2026-11-16). The public fixture
`environments/shadow_hand/robot/` is MuJoCo Menagerie's Shadow Hand E3M5 right hand (commit `71f066ad`), mounted as in
Menagerie's demo: palm up, fingers along +x, forearm welded. At reset the hand holds its home cradle pose and one free
object rests on the palm; the floor is 0.12 m below the palm. The agent gets `hand_target`, joint proprioception and
efforts, and fingertip/palm taxels (`tactile = true`), with no images and no point grasp. Object contact uses
Menagerie's demo object settings (condim 6, priority 1) with sliding μ 0.5: the object stays on the level palm, and full
wrist flexion (WRJ1 = −0.698) slides it off toward the wrist onto the floor, after which only `reset` recovers it.
Densities are 500 kg/m³.

| Level | Object | Mass | COM (body frame) |
| --- | --- | --- | --- |
| `hand_prism` | 64 × 44 × 30 mm box | 42.2 g | centre |
| `hand_prism_offset_com` | same box; a 20 × 20 × 22 mm non-colliding block of extra density 3880 kg/m³ inside one corner | 76.4 g | (8.94, 4.47, 0) mm |
| `hand_smooth_mesh` | inline convex mesh: an irregular 6-vertex core polytope dilated by a 15 mm ball (68 × 55 × 44 mm); every radius of curvature ≥ 15 mm, larger than the ~8.5 mm fingertip | 39.7 g | (−1.5, −0.5, −4.2) mm |
| `hand_spiky` | 56 × 40 × 30 mm box plus two sharp inline-mesh cones (base radius 8 mm): one 30 mm up toward the fingers, one 26 mm out toward the little finger. The creases where a cone meets the box are narrower than a fingertip | 35.5 g | (0.1, 0.7, 0.4) mm |

The generator settles each object on the home-pose hand for 3 s and stores the settled pose relative to the palm, so
reset starts it resting without penetration (deepest contact ≤ 0.21 mm); hand and object then sag together about 4 mm
under gravity in the first 0.5 s. Two private probes per level: `squeeze` (full finger curl with the thumb raised, hold,
reopen) and `tilt` (full wrist flexion, which drops the object, then back to home). Metrics: `dynamics`, `chamfer`,
`center_of_mass`. `experiments/validation/hand-check.py OUTDIR` measures hold, taxel response, the scripted drop,
reset and the identity/wrong-candidate scores (Slurm job `61031422`; results in `experiments/validation/hand-levels/`).

## Touch-only tabletop levels (Franka + Shadow hand)

Design: `docs/level-designs/touch-tabletop-20261124.md` (hypotheses H1–H5, user-approved 2026-11-24). Generator
`experiments/validation/touch-scenes.py` writes the public fixture `environments/panda_shadow/robot/` (the Panda arm with
Menagerie's Shadow Hand E3M5 in place of its gripper; actions `arm_target` (7) and `hand_target` (20); verified byte for
byte) and three levels. Touch only: joint proprioception and efforts plus fingertip/palm taxels (`tactile = true`); no
images, no point grasp, no wrench; lock-step; unlimited resets. Reset pose: palm down 25 cm above the table centre.
With the palm down and the wrist straight the forearm hangs about 4 cm below the palm face; flexing the wrist
(WRJ1 ≈ −0.6) lifts it, which reaching into the bin requires.

| Level | Objects |
| --- | --- |
| `table_shapes` (L1) | abstract tower (17 primitives, 339 g), zigzag of capsules (300.7 g) and two crossed boxes with a dense end (540 g), each matched to one L2 object: mass ratio 0.9999 / 1.0000 / 1.0256, bounding box within 4.1%, primitive count ±1 |
| `table_objects` (L2) | mug 120 mm tall (typical ~95), banana with 90 mm curvature radius, hammer with a 160 mm handle and steel head; these three unusual dimensions are scored as prior-defeating dimensions. A private `prior.xml` holds typical-size versions at the true poses (prior-only baseline) |
| `bin_spiky` (L3) | a static bin (24 cm inner, 40 mm walls) with four 60 g balls: one spiky (22 mm core + six octahedral cones 20 mm proud with 29 mm bases, at least one palm taxel cell; it rests on three tips with three tips up) and three 48.5 mm smooth spheres, each with explicit inertials equal to the spiky ball's compiled inertia |

L3 removes the size and dynamics cues (user decision 2026-11-24): equal mass (60.000 g), isotropic inertia equal within
0.6%, equal top height (0.4585 m). A width difference remains (spiky 62–68 mm by yaw against 48.5 mm): with few cones the
tip hull rests as a polyhedron, wider than tall, so no sphere can match both height and width. The original 14 small cones
on a 55 mm ball with 80 mm walls failed the discriminability and reach checks (jobs `63317646`, `63319921`, `63320350`).

Metrics: L1/L2 `objects` (primary: per-object Chamfer; found, centroid, labels, prior-defeating dimensions), `chamfer`,
`center_of_mass`, `density`; L3 `objects` and `identification` (from the submitted `parameters.json`), plus post-run grab
success and the contact coverage map (`python -m sim2sim.tabletop RUN_DIR`; see `docs/evaluation.md`).

Check `experiments/validation/touch-check.py OUTDIR` (Slurm job `63460653`, results in `experiments/validation/touch-check/`):
generated files match; all objects rest (< 0.4 mm in 2 s); palm-down IK over every table object (12/12 collision free) and
the bin interior (75/75 solved, 57 collision free: the rest touch balls or walls at floor level); held-pose error
≤ 9.2 mm.

L3 is an explicitly hard identification task (user, 2026-11-24): the palm's 4 × 3 taxel grid rarely spans two of the spiky
ball's tips, and there is no pass bar. The check measures scripted identification procedures (3 palm presses per ball at
random ±15 mm offsets, 40 seeded trials each) as reference floors for reports, stored privately in
`environments/panda_shadow/bin_spiky/reference_floors.json`:

| Procedure | Spiky ball named |
| --- | --- |
| ≥ 2 palm taxels, fingers curled | 25% |
| ≥ 2 palm taxels, fingers straight | 47.5% |
| most presses with first contact ≥ 14 mm above the ball top (the fingers meet the side tips) | 97.5% |

Scripted grab with 40 mm walls: an open-loop sweep (12 hand yaws × 3 grip heights × 3 closing poses, job `63458855`) lifts
the spiky ball in 1 of 108 tries (5.5 cm above the rim, all smooth balls in the bin); the check pins that grasp. With
30 mm walls it was 1 of 72 (job `63324769`); smooth balls lift in 6 of 15. Metrics on reference
candidates: identity scores zero error and label accuracy 1; the prior-only baseline errs on all three prior-defeating
dimensions (mug −25 mm, hammer +120 mm, banana bow −18 mm); a single-box candidate finds no object; L3 identification is
correct for the true position, wrong for a smooth ball's, and `unavailable` for a missing, malformed or non-JSON file.

Evals card: as in the design document. Run checking uses the dashboard action timeline, model context and physics
inspector (no camera); the report uses per-object 3D overlays, the contact coverage map (`RUN/touch-view.html`, linked
from the run page), the label table and the L3 identification/grab table.

## Sealed-box Franka levels

`experiments/validation/box-scenes.py` writes three `environments/panda_small/` levels
(2026-11-13): `box_marble_heavy`, `box_marble_equal` and `box_cylinder`. A closed
opaque box (outer 57 × 82 × 42 mm, 4 mm walls, interior 49 × 74 × 34 mm) rests on
the table at (0, −0.22) m and is gripped across its 57 mm side. The outside, the
shell friction (`1.5 0.01 0.0001`, condim 4) and the 80 g total object mass are
identical in all three; only the hidden contents and the shell/content mass split
differ:

| Level | Hidden body | Inner / shell mass |
| --- | --- | --- |
| `box_marble_heavy` | sphere, 19 mm diameter | 68 g / 12 g (5.7×) |
| `box_marble_equal` | same sphere | 40 g / 40 g |
| `box_cylinder` | cylinder, 19 mm diameter, 48.4 mm long along x (0.6 mm end clearance): rolls along y only | 55 g / 25 g (2.2×) |

The hidden body touches only invisible liner slabs behind the walls (friction 0.3),
never the robot, table or visible shell. The fixture gripper squeezes only about
1.4 N per finger, so the objects are light and the scenes use elliptic friction
cones with `impratio = 10`; with the default pyramidal cone the heavy-marble box crept
out of the grip while shaken. Contacts use `solref = 0.004 1` (the stiffest stable
value at the 2 ms step); throw impacts press the hidden body up to about 2–3.6 mm
into the liners, so the walls are 4 mm thick.

Four private arm probes, identical in all three manifests: `throw_forward`,
`throw_side`, `toss_up` (pick, lift, swing, open the gripper mid-swing, hold while
the box lands) and `tilt_shake` (wrist tilts about y, shakes along x and y, set down,
release). `experiments/validation/box-check.py` checks every source probe (grasp
held, release speed, landing, inner motion, determinism) and the composite-COM
comparison between all level pairs.

## Biased-die levels (real time)

`experiments/validation/die-scenes.py` writes `environments/die_biased` (point grasp) and
`environments/panda_small/die_biased` (Franka with the unchanged fixture; 2026-11-23). Both share the tray, die,
camera, lighting and launcher; only the actuator differs.

**Die.** A regular dodecahedron (d12), 40 mm across parallel faces, of resin (1150 kg/m³), with twelve colored face
plates. Face k has outward normal n_k, given in the task. Face 1 is +z in the die frame and opposite faces sum to 13.
The concealed bias is a 12 mm tungsten slug sealed 11 mm off centre toward faces 12 and 8. Its geom density is the excess
over the displaced resin, it has no contacts, sits in hidden geom group 3 and is enclosed by the opaque die. The die
weighs 67.5 g, and its centre of mass is 2.68 mm off the geometric centre ((0, −1.04, −2.47) mm in the die frame).

**Tray and contacts.**

- The die rests in a 300 mm square tray with 80 mm walls, 0.48 m in front of the Franka base.
- Die–tray contacts are explicit pairs: friction 0.45, `solref 0.004 0.3`, a bouncy contact with restitution about 0.35.
  This is why the scene steps at 1 ms (at least 4 steps per impact).
- The die's own geom is grippy (friction 1.2, condim 4, elliptic cones with impratio 10) for the weak fixture gripper,
  as on the sealed boxes.
- The fixed camera looks down at the tray from the front, with the Franka behind it.

**Real time and launcher.** The levels set `[simulation] realtime = true` (`docs/protocol.md`). The private `[launcher]`
(seed 20261123) restores a perturbed rest pose on face 1 at every reset: 10 mm in x and y, 0.3 rad of yaw. `throw`
releases the die with one of four public protocols, perturbed by Gaussian noise of 3 mm, 0.05 rad, 0.03 m/s and
0.8 rad/s per axis:

| Protocol | Nominal release (above the tray floor) |
| --- | --- |
| `drop` | 130 mm, tilted 0.35 rad about x, at rest |
| `roll` | 60 mm, 0.9 m/s along +x with 30 rad/s about +y |
| `toss` | 120 mm, lob (−0.5, −0.4, 0.6) m/s with tumble (10, −14, 6) rad/s |
| `spin` | 100 mm, (0.3, 0.9, 0.4) m/s with (30, 10, 40) rad/s into the walls |

`[die]` sets the prediction deadline (0.15 s after release), 30 evaluation trials per protocol, 200 empirical-baseline
throws per protocol, and the mechanics baseline's measurement noise (`docs/evaluation.md`).

**Variants and conditions** (`experiments/configs/wave24-20261123/`):

- Point grasp without wrench, or Franka with joint efforts.
- RGB-D 320×240, movable camera, `sequence` and `channel` tools.
- 25 frames/s, with a single-factor control at 100 frames/s: more precise observations of the same throws.

**Check** (`experiments/validation/die-check.py`, 150 throws per protocol). Slurm job `62998950` passed both levels and
the metric sanity check; job `62998993` re-ran the Franka level with an in-tray toss. Results for both levels:

- Files equal the generator output.
- The die rests on face 1. The slug changes neither RGB nor depth at the same states (0 mismatches) and is never hit by
  any of 726 grasp rays through die pixels.
- Launches are bit-identical in separately built simulations.
- No die left the tray, and every throw came to rest within 6 s. Die-at-rest times are 0.68–0.98 s; 0–2 of 30 rested
  cocked against a wall.
- The die is still moving at the deadline in 30/30 trials of every protocol. The face up at the deadline matched the
  final outcome in only 9, 0, 3 and 6 of 30 (drop, roll, toss, spin).

Outcome counts over faces 1–12, 150 throws each, biased vs fair die (same scene without the slug):

| Protocol | Biased | Fair | Bias χ² p |
| --- | --- | --- | --- |
| drop | 38 10 5 15 34 19 6 10 2 2 2 7 | 26 7 11 14 15 16 9 9 7 10 16 10 | 5.5e−4 |
| roll | 28 11 8 5 25 38 0 0 17 12 4 2 | 22 20 4 4 4 44 7 6 23 8 4 4 | 1.8e−4 |
| toss | 25 5 5 28 47 12 5 0 0 7 14 2 | 12 2 10 29 14 6 14 12 7 11 16 17 | 2.0e−9 |
| spin | 31 12 10 21 37 16 7 1 4 7 4 0 | 15 15 10 13 11 14 9 12 14 10 15 12 | 5.1e−8 |

The distributions differ strongly between protocols (homogeneity χ² p = 5e−16). The bias shows on every protocol
(pooled against the fair die, p = 2e−22). The Franka scene gives the same distributions to within one or two throws;
its arm at home is never touched.

**Actuation.**

- Point grasp: the die is grasped at its projected pixel, lifted 113 mm, moved and released at 0.50 m/s, and comes to
  rest in the tray.
- Franka: the scripted planner IK grip uses fingertips down, with the jaws closing horizontally 8 mm above the die centre
  so the pads press on the upward-tilted upper faces. It lifted the die 134 mm. A 0.2 rad base-yaw toss, with the
  gripper opening mid-swing, released it at 0.55 m/s, and it landed in the tray.
- Tilted antipodal grasps put one finger into the tray floor, and the gripper closes slowly at the 1 ms step (ramp
  ≥ 1 s).

**Metric sanity** (10 trials per protocol, local predictor processes through the real-time stream):

| Predictor or baseline | Brier score | Note |
| --- | --- | --- |
| Uniform predictor | 0.917 | 40/40 on time |
| Empirical predictor (frequencies learned from 150 session-stream throws per protocol) | 0.825 | |
| Always-late predictor | 0.917 | 40/40 late, scored uniform |
| Evaluator baseline: empirical | 0.837 | |
| Evaluator baseline: mechanics | 1.40 | noisy release; the throws are chaotic |
| Evaluator baseline: oracle | 0 | |

Trial generation and baselines take seconds (point grasp) to about half a minute (Franka) per run.

## Cloth levels

`experiments/validation/cloth-scenes.py` writes seven point-grasp cloth levels (2026-11-16), `environments/cloth_*`.
The ground truth is the simplest default cloth of RoboSimStudio (arXiv 2610.18835; its `rss.sheet` / Newton
`add_cloth_mesh`: a flat grid of particles triangulated into a sheet with stretch `tri_ke`, bending `edge_ke`, damping and
areal density, run by VBD) written as plain MuJoCo flex: a 0.40 m square, 16 × 16 cells in a "union jack" triangulation
(cell diagonals point at the centre, so the axes and both diagonals are mesh edges), lying flat on a 0.72 m table at
z = 0.402 m. One body per vertex moves on three slide joints with mass = 0.30 kg/m² × a third of its triangles' area (48 g
for the full square). Each material region is a 2D `<deformable><flex>` over those shared vertex bodies, made inextensible
by a flex edge equality (`solref="0.005 1"`: hung by a corner it stretches 0.3% on average, 10% at the grasp) and given bending stiffness by `<elasticity young=… thickness="0.001" poisson="0"
elastic2d="bend" damping="0.001"/>` (edge damping 0.02 N s/m, friction 0.8, radius 2 mm). MuJoCo has one elasticity per
flex, so a region of different stiffness is its own flex: an agent can write exactly this MJCF. Bending acts only between
triangles of one flex, so a seam between region flexes is a free hinge (a sewn seam); it is part of the ground truth.
Region flexes do not collide with one another (contype 2, conaffinity 1) and no flex collides with itself. MuJoCo 3.17
requires `integrator="discrete"` for flex elasticity; the scenes use the CG solver (tolerance 1e-6), several times faster
than Newton with the ~1500 cloth-table contacts of a resting cloth. Bending stiffness D = E t³/12; base E0 = 3e5 Pa
(elastic length (D/(ρ_A g))^(1/3) ≈ 2 cm).

| Level | Flexes (Young's modulus) | Colours |
| --- | --- | --- |
| `cloth_base` | one, E0 | one blue |
| `cloth_quadrants` | four quadrants: E0, 4 E0, 16 E0, 64 E0 for (+x+y), (−x+y), (−x−y), (+x−y) | one blue |
| `cloth_quadrants_colored` | same quadrants | red, yellow, green, blue by quadrant (aligned with stiffness) |
| `cloth_quadrants_diagonal_colors` | same quadrants (eight flexes: quadrant × diagonal sector) | the same four colours by the sectors between the diagonals (right, top, left, bottom): misaligned with stiffness |
| `cloth_holes` | one, E0 | one blue; seeded (20261116) wobbly outline, three wobbly holes and two slits cut in from the edge; largest connected piece |
| `cloth_center_hole_stiff_rim` | base E0 plus a rim flex 64 E0 (triangles within 0.11 m of the centre) | one blue; 0.07 m circular hole (boundary vertices moved onto the circle) |
| `cloth_center_hole_soft_rim` | one, E0 (the rim is simply base cloth, no seam) | same mesh as the stiff rim |

Manifests set `graspable_bodies = []`: the point grasp attaches to the cloth itself (the vertex of the hit flex nearest
the ray hit; `Simulation` adds one shared flex connect for any scene with a flex) and the wrench reports that vertex's
pull. Each manifest carries a private `[cloth]` probe table for the `cloth` metric (`docs/evaluation.md`): a 4 × 4 lift
grid, 8192 surface samples, and five seeded drops (three on a bar, tilt ≤ 20°; two over a cone, tilt ≤ 15°; 1.5 s each).
Suggested metrics: `image_mse`, `cloth`. `experiments/validation/cloth-check.py OUTDIR LEVEL ... [--eval SOURCE:CANDIDATE ...]`
checks compilation, rest, step cost, a scripted pixel grasp + lift + drag of a corner, and the metric.

Validation (2026-11-16, Slurm `test`): job `61032288` compiled all seven levels (step 2.6–3.8 ms resting), rest drift
≤ 0.2 mm over 0.5 s, and a camera-pixel grasp lifted a corner 15 cm in each (wrench reported 0.09–0.17 N while part of
the cloth still lay on the table). `cloth` metric, about 2 min per evaluation on 2 CPUs: source vs source is exactly zero
(`61032290`); `cloth_quadrants` vs a `cloth_base` candidate gives lift-force MAE 0.015 N (11% of the mean source force),
stiffness MAE 0.23 N/m, drop Chamfer 1.3–3.3 cm at rest against a 2.2 mm rest floor (same flat cloth, different sampling)
(`61032293`); stiff vs soft rim gives 7% lift-force error and 1.7–2.8 cm final drop Chamfer (`61032294`). Outputs, videos
and maps: `experiments/validation/cloth-check/`. Tests: `61032295` (63 passed; the one failure, a too-coarse 3 × 3 lift
grid in the new test, was fixed) and `61032724` (`tests/test_cloth.py`, 2 passed).

## Spring-powered positioning-fixture levels

`experiments/validation/fixture-scenes.py` writes three `environments/panda_small/` levels (2026-11-23): `fixture_rigid`,
`fixture_rubber` and `fixture_bimodal`. The task is calibration, not reconstruction: the agent characterizes the fixture,
picks dial settings for two marked target regions and submits a predictor program (metric `fixture`,
[evaluation](evaluation.md#positioning-fixture-fixture); runner [protocol](protocol.md#predictor-submissions-srcsim2simpredictorpy)).

Shared fixture (all values in the generator):

- A 40 × 30 × 24 mm carriage (80 g, plus a 20 g, 5 mm thick blade trailing 0.36 m behind it on its top) rides a rail on a
  slide joint (bearing drag: damping 0.05 N s/m, dry friction 0.03 N), with an end bumper (joint limit) at 245 mm. Its red
  index line reads the ruler: zero (red tick) where the relaxed spring leaves it; ticks every 10 mm, majors every 50 mm,
  −50 to 250 mm. Target bands: `target` 70–100 mm (green) and `narrow_target` 18–22 mm (orange).
- Launcher: a 20 g plunger on a 300 N/m spring (joint stiffness, stop at its free length). `reset_carriage` latches
  plunger and carriage (joint equalities) at compression `8 + 3.2 · dial` mm (8–40 mm), sets the pad preload, lifts the pad
  holder by a uniform 0–2 mm (rigid 0.5–2 mm) with the pad hinges perturbed by N(0, σ) about their rest angles, then settles
  200 ticks (0.4 s). The perturbation comes from a private per-run seed (`simulation.fixture_seed`, drawn by the runtime);
  evaluation draws trials from the same distribution with its own seeds.
- Brake: a pad holder (30 g, slide z, damping 20 N s/m) inside a closed caliper housing presses on the blade through a fixed
  tendon spring `L = pad_z + s · pad_preload`; the hidden preload nut is latched at the pressure dial's position by each
  cocking cycle. The housing is 110 mm long, its side walls reach to 0.5 mm above the carriage and the blade passes through
  slots in its end walls, so no camera pose sees the pad. Blade friction 0.1 (the pad surface sets every contact).
- Dials: two sliders (40 g blocks with a 16 mm vertical peg) on tracks 0.35 m in front of the robot, 0.1 m of travel = 0–10
  scale divisions, dry friction 1.5 N and damping 3 N s/m; they carry no load. Compression (yellow) at x −0.20…−0.10 m,
  pressure (blue) at x −0.02…0.08 m. The open gripper straddles a peg along x and closes on it (the coupled fingers centre
  it: form closure), so the ~1.4 N grip is irrelevant; the arm pushes against the dial friction.
- Public: RGB-D, movable camera, the `fixture` readings (both dials and `carriage_cocked`), Panda proprioception with
  joint efforts. Private: pad state, seed, compression/pressure mappings, the carriage position (read it from images).

| Level | Pad | Pad force over the pressure dial | Behaviour |
| --- | --- | --- | --- |
| `fixture_rigid` | rigid shoe, μ 0.5 | 1.0–5.0 N (tendon 400 N/m) | the lifted shoe settles to one equilibrium: releases repeat (sd < 0.5 mm) |
| `fixture_rubber` | 3-segment rubber flap (7 × 2 × 14 mm segments, hinge stiffness 0.004 N m/rad) hanging from the holder's front edge, pre-bent to tuck backward under it; μ 0.9; holder rocks (pitch, 0.05 N m/rad) | 0.8–3.2 N (400 N/m) | the moving blade pushes the tucked strip into its root: it buckles and sticks and slips; how it lands in the settle sets holder height and pitch: a continuous spread |
| `fixture_bimodal` | 3-segment flap (10 mm segments) hanging from the holder's rear edge, straight; one face grippy rubber (μ 1.1), the other slick fabric (μ 0.25); holder bare face μ 0.3 | 1.0–3.5 N (250 N/m) | the reset tilt decides the fold: tucked forward under the holder (rubber face on the blade, short stops, about 70 %) or folded back behind it (holder steel on the blade, the holder propped higher, long stops) |

Held-out evaluation settings: 25 seeded points (seed 2026112301), one per cell of a 5 × 5 grid over both dials;
evaluation seed 2026112302. `experiments/validation/fixture-check.py OUTDIR [LEVEL ...]` (Slurm) checks generator
equality, the cocked rest, the public observation keys, pad invisibility from up to 128 camera poses around the caliper,
closed-loop scripted Franka dial setting, the per-level behaviour (rigid repeatability; rubber spread, settle-height
correlation and flap stick→slip transitions; bimodal fold-separated ranges) and metric sanity (an empirical predictor vs a
mean-only predictor, each with its own choice of settings). `experiments/validation/fixture-explore.py` is the tuning aid.

Validation (2026-11-23, Slurm `test`): jobs `62995910` (rigid), `62995912` (rubber), `62995913` (bimodal) passed every
check; outputs, previews and evaluator views in `experiments/validation/fixture-check/<level>/`. All levels: step 22–31 µs,
cocked carriage drift < 0.005 mm over 1 s, pad pixels 0 in all 79 accepted camera poses, the scripted Franka set both dials
(8.07 / 2.98 for 8 / 3, one closed-loop stroke each) and its release matched the evaluator's direct setting to 0.01 mm.

| Level | Release spread over the 25 held-out settings (16 each) | Pad dynamics while sliding (median of 12 releases) |
| --- | --- | --- |
| `fixture_rigid` | sd 0.0 mm at every setting (stops 4–245 mm over the dials) | holder excursion 0.64 mm, 0 snaps, brake-force CV 0.24 |
| `fixture_rubber` | median sd 16 mm (max 45); settled holder height vs stop correlation, median over settings 0.68 | holder excursion 2.7 mm, 1 buckle-snap (> 1 mm), brake-force CV 0.42 |
| `fixture_bimodal` | tucked fold 46.5 % of releases; at 20 of 25 settings every tucked release stops short of every folded-back one (e.g. 17.5 ± 1.6 vs 80 ± 24 mm at 7.3 / 5.4) | holder excursion 1.65 mm, brake-force CV 0.41 |

Metric sanity (empirical predictor from independent releases vs a mean-only predictor; each picks its own settings on an
11 × 11 search grid): rubber CRPS 12.0 vs 18.4 mm (oracle 11.8, settings-blind 27.7); bimodal 20.1 vs 31.0 mm (oracle 19.0,
settings-blind 27.6). On the bimodal narrow target the mean-only choice (4, 8; mean inside the band, modes on either side)
hit 0 of 40 releases while predicting 100 %; the empirical choice hit 9 of 40 (predicted 38 %). The rigid level is exactly
predictable (CRPS 0, all hits). Earlier tuning (jobs `62987997`–`62994769`, `experiments/validation/fixture-explore-*.log`)
found that column-loaded flaps jam, a 2 mm blade lets a dropped shoe tunnel through, the original short housing showed a
folded flap, and a loaded dial creeps under MuJoCo's soft dry friction; the design above fixes each.

## Explanation-depth gear train (`gear_train`)

Design: [level-designs/gear-train-depth-20261124.md](level-designs/gear-train-depth-20261124.md). Hypothesis: when two
explanations predict everything the agent can probe equally well, the stated use of the model decides whether the agent
builds the deep one (every gear) or the shallow one (crank and hand with lumped inertia and damping). Conditions (wave 28)
differ only in one appended sentence: none (G-recon), a planner that values speed (G-cheap), gear-removal predictions
(G-deep). Metric `gear_train` ([evaluation](evaluation.md#gear-train-gear_train)).

`experiments/validation/gear-scenes.py` writes `environments/gear_train/{scene.xml, shallow.xml, manifest.toml}` (all values
in the generator):

- A vertical panel with a crank (left; handle knob at 40 mm radius), a dial hand (right; pointer 70 mm) and 20 gears between
  them: idlers and two compound pairs on two layers, overall ratio hand:crank = −7.5, zigzag train 0.69 m long. Gears are
  aluminium cylinders (2700 kg/m³, 6 mm thick) with massless non-colliding tooth visuals; crank and hand carry
  counterweights, so no body has a gravity torque. Couplings are joint equalities (solref 0.001, solimp 0.9999); every axle
  has damping 1.0e-4 N m s/rad (crank coast time constant about 0.12 s). No moving geom collides; only crank and hand are
  graspable (gears refuse the grasp).
- Timestep 0.5 ms and Newton tolerance 1e-12 (the default 1e-8 stops a stiff 21-constraint train early); RGB-D, movable
  camera, point grasp with wrench, `sequence` tool.
- Public grip slip `simulation.grasp_slip_m = 0.002` ([protocol](protocol.md)): the grip lets go when stretched more than
  2 mm. Without it an over-pull (dragging the handle past its reach) stretches the soft grasp until MuJoCo's
  inertia-scaled constraint softness winds up the deep train and makes it spring back after release, which the lumped model
  cannot do (up to 3.39 rad hand RMS difference). The grasp softness stays at the default.
- `shallow.xml` (private): crank and hand, one coupling with the overall ratio, crank armature Σ I_i r_i² = 5.20e-3 kg m²
  and damping b (1 + Σ r_i²) = 0.0460 N m s/rad, at 5 ms, the largest step whose probe error stays within the gate.
- Probes (private): crank turns of 120° in 3.0 s and 1.8 s and −60° in 0.9 s; hand pushes of 30° and −40° in 0.8 s; each
  follows the handle's or tip's circle in 5° chords, then release and 0.8 s of coasting. `[gear_train]`: removals
  `gear_03`, `gear_10`, `gear_14`; axle tolerance 15 mm; G-cheap gate `cheap_gate_rad = 0.02838` (the worst probe error of
  `shallow.xml` with armature and damping each ×0.9 or ×1.1); timing 5 runs × 1 s with a 0.02 N m crank torque.

Evals card: primary, the depth class of every submission and the final one; secondary, probe accuracy (expected equal across
depths), cost per simulated second against both references, G-cheap and G-deep scores, at matched wall minutes and tokens.
Qualitative: reasoning about which depth the use needs, citing the stated use, timing own models, building both depths
before choosing, how lumped inertia and damping were identified. Run check: dashboard video, candidate diff, physics
inspector on submitted versions (which gears exist). Report: cost vs probe error per submission with both references,
side-by-side final models, depth choice by condition. Not built: the H4 trace classifier and a dashboard per-version
structure summary (read `scores.gear_train.structure` and `sim_cost` from the private reports).

Validation (2026-11-24, Slurm `test`): `experiments/validation/gear-check.py` job `63463232` passed, outputs in
`experiments/validation/gear-check/`. The generator output matched the files. Bodies rest (0 rad/s), and the lumped crank
inertia equals the deep train's (7.6454e-3 kg m²). All 20 gears render (104–320 px, 8–19 mm in front of the panel in
depth) and refuse the grasp. Source probes are deterministic.

| Check | Result |
| --- | --- |
| Shallow at 0.5 ms vs deep, every probe | mean 0.00017 rad, worst probe/body 0.00087 rad (3 % of the gate) |
| `shallow.xml` at 5 ms | 0.0059 rad, passes the gate |
| Cost per simulated second | deep 40.7 ms (22 bodies, 21 equalities) vs shallow 0.80 ms: 51× |
| In-reach sweeps, 0.5–1.5× probe speed, deep and both shallow | never slip; peak stretch 1.0 mm at probe speed, 1.6 mm at 1.5× |
| Over-pulls (crank 5 cm slow and fast, hand 8 cm) | all slip in both models; afterwards agree within 0.006 rad (21 % of the gate); without the slip up to 3.39 rad |
| Evaluator on copies | source copy and a renamed copy with flipped crank/hand axes: 0 rad, class deep 20/20, removals answer with 0 rad |
| Evaluator on `shallow.xml` | class shallow, G-cheap ok, all 3 removals `cannot_answer`; the unmodified shallow model misses each removed-gear source by 2.97–2.98 rad |

Tests: same job, 49 passed (`test_gear_train`, `test_simulation`, `test_metric_selection`, `test_evaluation`,
`test_realtime`, `test_candidate`). Tuning history: jobs `63323565`–`63325726` (2 ms: deep and shallow diverged under crank
grasps), `63454396`–`63455903` (0.5 ms with stiffer grasps: still diverged under over-pulls), `63460340`–`63461841` (slip
threshold, probe speeds, solver tolerance; `gear-explore.py`, `gear-diag.py`). Smoke: `--prepare-only` of `w28-gear-cheap`
(public `grasp_slip_m = 0.002`, neutral task plus the G-cheap sentence) and the dashboard lists the level.

## Childproof bottle (`bottle_empty`, `bottle_contents`)

Design: [level-designs/childproof-bottle-20261125.md](level-designs/childproof-bottle-20261125.md). Hypotheses: agents discover
a force-gated hidden mechanism (push to engage, then turn) by interaction without a semantic cue (H1); a second simultaneous
grasp helps or is needed (H2: 1 vs 2 weld grasps); with unpredictable contents agents open the bottle to complete their model
(H3: contents vs empty). Wave 32: {empty, contents} × {1, 2 weld grasps}, n = 1. Metrics `bottle` ([evaluation](evaluation.md#childproof-bottle-bottle)),
`objects`, `chamfer`, `center_of_mass`.

`experiments/validation/bottle-scenes.py` writes `environments/bottle_{empty,contents}/{scene.xml, manifest.toml}` (all
values in the generator); bottle and cap compile identically in both variants:

- A free hollow bottle (Ø 45 mm, 87 mm to its collar; base, wall, shoulder, 24 mm neck and a collar under the cap, all box
  rings) with a Ø 51 mm cap, plain neutral grey, on a table. Plastic 1100 kg/m³: bottle 44 g, cap 24 g.
- Push-and-turn bayonet, plain-MJCF primitive contacts and joint springs, all inside the opaque cap shell. The visible
  `cap_shell` sits on the hidden inner `cap` on a frictionless hinge and a slide whose 400 N/m spring holds it up with a
  7.6 N preload. Pushed past the preload (3 mm stroke), six hidden teeth under the shell mesh with six on the inner cap
  (explicit contact pairs), so turning the shell turns the inner cap; unengaged, the shell spins and turns nothing. The
  inner cap's two hooks sit in 1 mm notches under two neck lugs, pressed up by a 4 N spring washer (`liner`): the cap
  cannot turn until pushed down out of the notches (a detent: without it a quick spin of the bottle turned the light inner
  cap by inertia), and lifts off once turned about 50° or more (a quarter turn opens it).
- Contents variant: a 12 mm aluminium cube (red), an upright Ø 8 × 20 mm steel rod (blue) and a Ø 10 mm glass ball (yellow),
  13.8 g, loose inside; visible only once the cap is off. Graspable: `bottle` and `cap_shell` only.
- 1 ms timestep, contact and stop time constant 2 ms (the 20 ms default let gram-scale parts sink ~10 mm under a few
  N), `multiccd`; preloaded contacts and the shell's stop start 0.03 mm engaged so the springs act from the first step.
- Configs: weld grasp(s) (`grasp = "weld"`, `point_grasps = 2` in the two-grasp runs) with wrench, RGB-D, movable camera,
  `sequence` tool, `TASK_CONVERGE` (never mentions a cap, contents or opening), MJCF answer, gemini-4-neon, 4 h wall.

Evals card: primary, mechanism-probe outcome agreement (`bottle.outcome_agreement`); secondary, engage-force and opening-angle
errors, contents found and per-object Chamfer (contents variant; omitted contents are unassigned gaps), opening in the
reference world and its time (`python -m sim2sim.bottle RUN_DIR`), at matched wall minutes and tokens. Qualitative: pushing at
all or only twisting, reading the wrench to find the threshold, holding the bottle with one grasp, opening to look inside,
what the agent does with contents it cannot see (guess, omit, state uncertainty); failure modes twisting forever and tipping
the bottle. Run check: dashboard video with both grasp overlays, action timeline, physics inspector on submitted versions.
Report: probe-outcome table (truth vs model) by condition, push force vs cap rotation, a clip of the first opening, contents
overlay. No new dashboard view.
- Private `[bottle]` probes: cap grasp at the shell's top centre, bottle grasp on the wall facing the camera; pushes 0, 2, 4,
  5, 6 and 8 mm of the cap weld (0.3 s), a quarter turn (0.5 s), unpush, release the bottle, lift 20 mm; one cap-only
  probe at 8 mm; opening-angle sweep 30–120°.

Validation (2026-11-25, Slurm `test`, weld grasps of the merged weld branch): `experiments/validation/bottle-check.py` job
`63752922` passed in 2 min 47 s (outputs `experiments/validation/bottle-check/`). Files match the generator; bottle and cap
compile identically in both variants; both scenes rest (2 s: every body within 0.25 mm, speeds ≤ 0.0045).

| Premise | Result (two welds unless stated) |
| --- | --- |
| Hidden while closed | 66 views per variant, upright and lying on its side: 0 pixels of the 78/81 hidden geoms (inner cap, washer, teeth, neck, lugs, contents) in segmentation (so in RGB and depth); 0 of 43 200 grasp rays reach them |
| Contents not inferable from outside | every view of the empty and the contents bottle renders bit-identical RGB and depth |
| Twist only (0 mm, 0.41 N) | closed; the shell turns 90°, the bottle 0.04° |
| Pushes below the preload, then a quarter turn | 2, 4, 5 mm (4.2, 7.2, 7.7 N): closed |
| Push past the preload, quarter turn, lift | 6 mm (8.05 N) and 8 mm (12.3 N): open, cap 18.9 mm above the bottle; repeat bit-identical |
| Stability of the scripted probes | grasps hold; weld translation residual ≤ 3.2 mm, rotation residual ≤ 0.165 rad (engaged turn against friction); bottle and cap speeds ≤ 3.1 rad/s closed, ≤ 19 rad/s as the freed cap pops off the neck; no jitter or blow-up |
| **One weld grasp** (cap only) | **opens**: 6 and 8 mm open with the bottle held by table friction alone (it turns 1.3–1.9°); 0–5 mm closed |
| Opening angle (8 mm push) | 60° closed, 70° open |

Metric on reference candidates (real evaluator, `bottle`, `objects`, `chamfer`, `center_of_mass`):

| Candidate | Outcome agreement | Engage force error | Opening angle | Contents |
| --- | --- | --- | --- | --- |
| Source copy, empty / contents | 1.0 / 1.0 | 0 N (engage 8.05 N, bracket 7.72 N) | 70° / 70°, error 0 | all found, Chamfer 0, COM 0 |
| No teeth (shell never engages) | 0.571 | gap (`never_opens`) | gap (`never_opens`) | — |
| No lugs (cap free on the neck) | 0.429 | 5.43 N (opens at 0 mm) | 30°, error 40° | — |
| Wrong contents (cube → sphere) | 1.0 | 0 | 0 | cube Chamfer 1.4 mm, scene Chamfer 2.0 mm, COM 1.4 mm |
| Missing contents (empty bottle) | 1.0 | 0 | 0 | cube, rod, ball unassigned with a reason (gaps), found 0.4, COM 9.2 mm |

Tuning history: raw-MuJoCo weld exploration jobs `63737259` (default 20 ms constraint softness: parts sank 12 mm into their
stops), `63738174` (stiff contacts at exactly touching starts: springs fired into inactive constraints), `63739635`
(mechanism works; a 90° bottle spin in 40 ms leaves it closed), rest jobs `63739180`–`63740690` (lying steel cylinder rolled:
upright rod, `multiccd`); check job `63742170` before the weld (static sections passed).

Image and integration: `experiments/images/trusted-image-isolated-bottle.json` (build job `63754732`; agent `2e7cd4413663`
from the same agent-side sources as the weld record, evaluator `053bb9233a8a` adding `bottle.py`). OCI suite with it, job
`63755828`: 12 passed, 1 skipped (tools image), including `BottleOCITests`: through the strict controller the agent holds
the bottle with weld 1, pushes the cap 8 mm with weld 0 (wrench 12.25 N), turns it a quarter turn, lifts it off and looks
in; `python -m sim2sim.bottle` on the run finds the first opening at action 7 and the contents seen, and the evaluator
container scores the submission with `bottle` and `objects`. Smoke: `--prepare-only` of `w32-bottle-contents-2grasp`
(job `63760019`); the dashboard lists `bottle_empty` and `bottle_contents`.
