---
name: object-localization
description: Interpret Triskel's native, unaligned RGB/depth images and camera calibration to estimate positions from observations.
---

# Native RGB and depth

`humanoid_observe` supplies PNG attachments and metadata for both cameras. RGB
and depth are **not pixel-aligned**. Never use an RGB pixel coordinate directly
to index the depth image. Depth is uint16; multiply by `depth_scale` for meters.
Zero depth is invalid. Read each attachment's own intrinsics and frame ID.

For a valid depth pixel `(u, v)` with depth `z`, the point in its optical frame is
`[(u-cx)*z/fx, (v-cy)*z/fy, z]`. Optical axes are +X right, +Y down, +Z forward.
Apply that attachment's `optical_to_base` rigid transform to express the point
in robot-base coordinates. Transform through the inverse RGB optical-to-base
pose and project with RGB intrinsics to associate depth with an RGB detection.
Handle occlusion, invalid samples and image bounds explicitly; prefer robust
interior samples and report uncertainty rather than inventing missing depth.

Use measured `base_pose_world` to map base-frame points into world coordinates.
Check timestamps and observe again after driving; do not reuse stale camera
geometry as though the robot had remained stationary. The camera is fixed on
the base, so turning or translating the base changes what it sees.

Use image evidence to discover objects and free space. The task does not expose
private scene coordinates or the number of remaining objects. Learn a coverage
and memory strategy that can justify completion from observations.
