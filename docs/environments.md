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
