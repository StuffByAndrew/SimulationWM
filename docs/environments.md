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
