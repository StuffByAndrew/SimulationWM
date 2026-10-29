# Simulation action and observation payloads

The Python core exposes `Simulation(scene_path, config=None)`,
`action(action_dict)`, `step(op, args=None)`, `observe()`, and `close()`.
The runtime transports these payloads; consult its entrypoint for the transport
envelope and budget enforcement.

Actions use an `op` field (`action` is an alias):

| op | Additional fields | Behavior |
| --- | --- | --- |
| `observe` | none | Render the current state without advancing physics. |
| `grasp` | `pixel: [u, v]` | Attach to the first visible graspable object at the pixel. A miss leaves the grasp inactive. |
| `move` | `displacement: [dx, dy, dz]`, optional `ticks` | Move the grasp target by world meters over the specified physics ticks. Requires a grasp. |
| `release` | optional `ticks` | Disable the grasp and optionally advance physics. |
| `wait` | optional `ticks` | Advance physics without moving the target. `advance` is an alias. |
| `reset` | none | Restore initial physical state and release the grasp. |

There is one grasp at a time. A second grasp requires release. Pixel origin is
the image's top left; coordinates must lie within the image. The ray and hit
identity remain private. A surface behind another visible surface cannot be
grasped through it. Ticks default to 60 and must be integers from 1 to 100000;
the runtime may apply tighter budgets. Invalid actions raise `ValueError`.

Every successful action returns the following JSON serializable observation:

- `rgb`: base64 encoded PNG, RGB8, with no data URL prefix.
- `depth`: base64 encoded PNG, unsigned 16 bit millimeters of camera depth,
  saturated at 65535; no data URL prefix. PNG decoders may expose it as int32.
- `rgb_format`: `png_rgb8`; `depth_format`: `png_uint16_mm`.
- `width`, `height`: image dimensions, default 640 by 480.
- `time`: simulation seconds since reset.
- `ticks`: physics steps since reset.
- `total_ticks`: physics steps across resets, never refunded.
- `grasped`: whether the single connect constraint is active.

Reset, observe and grasp consume no integration ticks. Release consumes only its
explicit optional ticks. Simulation configuration accepts `width`, `height`,
`camera`, `timestep`, `default_ticks`, `graspable_bodies`, and an internal
`recording_callback(sim)` called after each physics step. If graspable bodies
are omitted, named bodies containing a joint or freejoint are eligible.

`observe` also accepts optional positive `ticks` to settle the simulation before
rendering. Configuration flags `rgb` and `depth` independently select image
modalities (both default true); `calibration` selects camera calibration (default
false). `observe(full=True)` is an internal privileged rendering path for video
and evaluation, independent of those agent settings. Calibration contains a 3x3
`intrinsics` matrix with pixel center coordinates and a 4x4
`world_from_camera` pose. Camera axes are x right, y up, negative z forward;
image v increases downward. Depth is axial camera distance, not ray length.

`grasp_solref` and `grasp_solimp` configure MuJoCo connect constraint softness.
An internal `grasp_controller_factory(sim)` can replace the `PointGrasp`
controller; its interface is `attach(body, world_point)`, `move(delta)`, and
`release()`. The default controls only position and applies no orientation target.
