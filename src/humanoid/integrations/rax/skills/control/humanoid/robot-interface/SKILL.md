---
name: robot-interface
description: Triskel's measured state, mobile base, arm, gripper, bounded asynchronous commands and feedback. Read before commanding motion.
---

# Triskel robot interface

`humanoid_status` reports measured joint state, tool pose relative to the robot
base, base pose in the world, gripper limits, and all command limits. Positions
are meters, joint angles are radians, and quaternions use **x, y, z, w** order.
The tool frame is `gripper_base_link`; the fingers extend along its local +Y.
Tool-frame position is not the fingertip position. Infer the contact location
from camera observations and measured response.

`state.arm_joints` reports each arm joint by name, with its measured `position`,
`velocity`, and URDF `lower_position_limit` / `upper_position_limit`. Null limits
mean the joint is unbounded. The raw `joint_positions` array is a Pinocchio
configuration, which can include cosine/sine pairs rather than scalar angles;
use the named feedback when checking joint limits. Cartesian workspace bounds
do not guarantee every orientation is reachable at every position. A persistent
pose error near a joint limit may require a different approach configuration.

The robot and camera run continuously. Tool calls submit commands to the normal
`RemoteControlPolicy` and wait for measured results; they do not advance physics by a
fixed number of steps. Model reasoning also consumes wall-clock time. Commands
are bounded to at most two seconds; use the current limits from status.

- `humanoid_move_tool(position, quaternion, duration)` takes an absolute tool
  target in robot-base coordinates. Respect workspace and per-call limits.
  Translation and rotation interpolate from the measured starting pose over
  `duration`, subject to the robot's tool velocity limits. A longer duration or
  smaller displacement reduces the requested speed. Duration includes the motion;
  it does not add a settling interval after reaching the target.
- `humanoid_move_base(delta, duration)` takes local planar `[dx, dy, yaw]`
  displacement relative to the measured base at command start. Positive yaw is
  counterclockwise viewed from above. Check measured world pose afterward.
  Duration bounds execution time; base speed comes from the feedback controller
  and reported velocity limits, rather than displacement divided by duration.
- `humanoid_gripper(positions, duration)` takes absolute gripper joint targets in
  the order returned by status. Respect the reported URDF joint limits.
  For a grasp, command a small additional closure beyond estimated first contact
  so the position servo maintains pressure on the object. Matching its estimated
  width exactly can leave no squeeze. Choose the closing direction from observed
  response, stay within joint limits, and retain the closing target during lift
  and carry instead of resetting it to the contact-blocked measured position.
  These are position commands, not calibrated force commands; verify retention
  visually rather than treating a target error as a force measurement.
- `humanoid_hold(duration)` holds the measured arm/base pose and keeps the last
  gripper target for that interval.

Each command reports `reached`, position/orientation/gripper errors and elapsed
seconds. A successful tool transport is not proof the target was reached. Use
fresh measured state for subsequent commands. Completion anchors the arm/base at
their measured poses. The gripper keeps its last rate-limited target between
commands so contact does not remove the squeeze; short commands may need further
gripper commands to finish closing. A blocked gripper can report `reached=false`
while maintaining its target, which is not proof of possession. Stale feedback,
a mode change, or operator Stop interrupts execution and clears the held target.
Do not retry an interrupted command without rechecking status.

There are no object poses, object counts, contacts or simulator handles in these
tools. A closed gripper is not proof of possession. Verify consequences visually.
Task-specific goals belong in TASK.md and the generated program.
