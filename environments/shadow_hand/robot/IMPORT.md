# Shadow Dexterous Hand fixture

Imported from Google DeepMind MuJoCo Menagerie, directory `shadow_hand`
(Shadow Hand E3M5, right hand), pinned commit
`71f066ad0be9cd271f7ed58c030243ef157af9f4`.
Upstream: https://github.com/google-deepmind/mujoco_menagerie/tree/71f066ad0be9cd271f7ed58c030243ef157af9f4/shadow_hand
The upstream Apache-2.0 license is included as `LICENSE`; the original assets
were provided by the Shadow Robot Company under that license.

`shadow_hand.xml` is the upstream `right_hand.xml` with only its `<compiler>`
line removed: the enclosing scene supplies `angle="radian"`,
`autolimits="true"` and the mesh directory. Every body, explicit inertial,
joint, tendon coupling, position actuator, contact exclusion, collision and
visual geom, the `<option cone="elliptic" impratio="10"/>` setting and all
mesh files under `assets/` are unchanged. The forearm keeps the upstream
mounting (`quat="0 1 0 1"` at the world origin): the palm faces world +z and
the fingers point along world +x. Each mesh geom collides as its convex hull.

A candidate project copies the public `robot/` directory unchanged, includes
`robot/shadow_hand.xml` once from its root `scene.xml`, and sets
`<compiler angle="radian" autolimits="true" meshdir="robot/assets"/>`.
Because that mesh directory holds the fixture assets, target-object meshes are
written inline (`<mesh name="..." vertex="x y z ..."/>`, collided as the
convex hull of the vertices) rather than as extra files.

The sibling scenes in this repository hold private target objects and are not
part of the public fixture.
