# Compact Franka Emika Panda fixture

Imported from Google DeepMind MuJoCo Menagerie, directory
`franka_emika_panda`, pinned commit
`b0655057a9df2abe39ea7fc989914ba6f248301c`.
Upstream: https://github.com/google-deepmind/mujoco_menagerie/tree/b0655057a9df2abe39ea7fc989914ba6f248301c/franka_emika_panda
The upstream Apache-2.0 license is included as `LICENSE`.

The kinematic tree, explicit inertials, joint ranges and damping/armature,
collision geometry, finger pads, tendon/equality coupling, actuator gains,
force bounds and contact exclusion are retained. Detailed visual OBJ meshes
are omitted; copies of the official collision meshes supply off-white visual
geometry. This is a compact visual approximation, not a visually exact robot.
The base is mounted at world `(0, -0.65, 0.4)`, facing positive world y.
A transparent `panda_ee` site is added at hand-local `(0, 0, 0.1034)`.
Compiler/options and the upstream home keyframe are omitted from the shared
include: the enclosing scene supplies radians and mesh directory, and the
runtime initializes the robot joints independently of target free joints.

A candidate project can copy the public `robot/` directory unchanged, include
`robot/panda.xml` from its root `scene.xml`, and set
`<compiler angle="radian" autolimits="true" meshdir="robot/assets"/>`.
The fixture is about 218 kB including this attribution and license, comfortably
below the 8 MiB expanded submission bound. Research manifests, source objects
and experiment configurations are not part of the public fixture.

This copy additionally attaches a `wrist` camera to `hand` at hand-local
`(0.06, 0, 0.01)`, looking toward the fingertips in the hand +Z direction. The lateral
offset clears the hand shell. Robot geometry and physics are unchanged.
