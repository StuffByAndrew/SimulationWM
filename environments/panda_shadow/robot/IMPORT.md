# Franka arm with Shadow hand fixture

Composed from two Google DeepMind MuJoCo Menagerie imports, both under the Apache-2.0 license included as `LICENSE`:

- the compact Franka Emika Panda arm (`franka_emika_panda`, pinned commit `b0655057a9df2abe39ea7fc989914ba6f248301c`): links 0 to 7 with their inertials, joints, collision meshes and visual copies, mounted at
  world `(0, -0.65, 0.4)` facing positive world y. The Panda gripper (hand, fingers, their tendon and equality, actuator8) is
  removed. Arm position gains and damping are 4 times the upstream values (force bounds unchanged), so the arm
  holds the heavier hand with a static error of a few millimetres.
- the Shadow Dexterous Hand E3M5 right hand (`shadow_hand`, pinned commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`): every body, inertial, joint, tendon, actuator, contact exclusion, geom and mesh unchanged,
  with its `<option cone="elliptic" impratio="10"/>`. Its forearm is attached to the arm's flange where the Panda hand was
  (`link7`-local `(0, 0, 0.107)`, rotated -45 degrees about the flange axis), the forearm axis along the flange axis.

Actions: `arm_target` drives actuators `actuator1` to `actuator7` (Panda joints 1 to 7); `hand_target` drives the 20 hand
actuators. A candidate project copies the public `robot/` directory unchanged, includes `robot/panda_shadow.xml` once from
its root `scene.xml`, and sets `<compiler angle="radian" autolimits="true" meshdir="robot/assets"/>`. Target-object meshes are
written inline (`<mesh name="..." vertex="x y z ..."/>`, collided as the convex hull of the vertices).

The sibling scenes in this repository hold private target objects and are not part of the public fixture.
