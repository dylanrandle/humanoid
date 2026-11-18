# Pose records

Record the observation timestamp, measured base world pose, tool pose relative
to the base, gripper joint positions, and the evidence supporting any estimated
object pose. Preserve the frame name and units with each value; quaternions are
x, y, z, w. Include the source image/attachment and camera intrinsics when a pose
came from pixels. Distinguish an estimated object pose from a commanded tool
pose and a measured tool pose. Record uncertainty and re-observe after motion.

A saved successful pose is evidence from that episode, not a target to replay
blindly in another layout. Re-localize objects from current observations.
