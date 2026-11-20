# Camera-guided object pickup

Use **Start pick policy → enter an object → Run pick** in the operator UI.
`humanoid.policy.autonomy.pick.PickPolicy` uses persistent models from
[`humanoid.serving`](../serving/README.md). Segmentation and grasp poses are learned;
only jaw width is estimated geometrically. GraspGenX uses diffusion only, excluding
analytic OBB candidates. Preview-only scripts and per-request model workers have
been replaced by the managed policy and model service.

## Geometry and execution

Native optical depth is registered to RGB using each stream's intrinsics and
URDF extrinsics. The unmodified learned mask selects object points; RGB-occluded
and unsegmented points remain in the scene cloud. Unsupported nonzero distortion
and sparse/empty targets abort explicitly.

GraspGenX conditions on the inner-finger volume at open and half-open positions,
derived from the Finray meshes and URDF mimic joints. Its native frame shares the
tool origin, with +Z along the fingers and +X closing. Predicted root poses map
back to the existing tool frame without adding an insertion-depth heuristic.
The rigid meshes describe the gripper; Finray deformation is not modeled.

GraspGenX predicts no jaw width. Following the projection approach in upstream's
[`compute_close_value_for_object`](https://github.com/NVlabs/GraspGenX/blob/44eb1d77d6d27bc97ee361e3afb717a2115e30a9/end2end/e2e_grasp_demo.py),
we transform all segmented object points into each predicted gripper frame and
measure their extent along its closing axis. Unlike that helper, this needs no
known object mesh. Because Triskel closes symmetrically, the required gap is twice
the largest distance from the URDF-derived finger center; this accounts for an
off-center object without moving the learned grasp pose. Points must span both
sides of that center. Sparse, non-finite, too-small and oversized estimates are
rejected, never clipped into a valid opening. This whole-object extent can reject
grasps on a narrow part of a larger object.

The estimate describes **visible** geometry: missing/occluded surfaces can make it
too small, and an off-center object may contact one finger first. It is not a force
measurement. The planner converts the width into a joint target using the configured
gap-to-angle calibration and 4 mm squeeze preload, then lifts 80 mm vertically.
There is no full-close fallback; unresolved widths remain non-executable. Physical
closure/preload still need validation, and contact-aware motor protection is not
implemented in this pickup policy.

GraspGenX candidates are checked along the open-gripper approach against a 2 mm
occupancy octree of **all** observed depth points. Per-piece convex hulls of the
actual gripper meshes conservatively fill internal ribs. This filters grasps;
it never creates or adjusts them.
The planner rejects workspace violations and runs the existing controller's IK
and self-collision preflight before motion.

Fresh depth, appearance and robot state are checked again before starting the approach.
Commands stream at 30 Hz with bounded pose/gripper speeds. Stale feedback,
tracking failure, moved base, timeout or operator takeover aborts the trial.
Completion/interruption requests Idle without opening; operator takeover is preserved.

## Validation and limits

The shared perception helpers register camera depth, estimate width, filter gripper
collisions and perform controller IK preflight. The managed `PickPolicy` owns the
realtime trajectory; inference and preflight run in a background task with separate
Pinocchio data. A cancelled or superseded result cannot resume motion. Robot state,
controller output and mode heartbeat freshness are checked on every active tick.

The observed depth cloud cannot represent hidden obstacles. The scene filter checks
only the open-gripper approach, not the entire arm, closure or carried-object lift.
The scene must remain stationary during planning and pickup. No force/vision check
confirms retention: completion means the commanded lift finished. Physical pickup
remains untested.
