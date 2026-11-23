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
