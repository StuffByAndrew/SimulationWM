# Environment authoring

Each directory in `environments/` contains a MuJoCo `scene.xml` and a
`manifest.toml`. The XML must define a fixed camera named `fixed` (or specify
another camera in configuration). Use named movable bodies and standard MuJoCo
joints, masses, contact parameters, springs and damping. Assets resolve relative
to the scene directory. Includes are supported.

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
