# Architecture

## Runtime boundary

ROS 2 is the only runtime, middleware, hardware abstraction, and composition layer. Runtime
features must use native ROS topics, services, actions, parameters, TF, lifecycle semantics,
diagnostics, and `ros2_control` resource ownership.

## Sources of truth

| Fact | Owner |
|---|---|
| Links, joints, axes, geometry, limits, meshes | `triskel_description` URDF |
| Feetech IDs, operating modes, command/state interfaces | `triskel_hardware` xacro |
| Controller selection, wheel geometry, odometry, timeouts | `triskel_control` YAML |
| Planning groups, named poses, collision exemptions | `triskel_moveit_config` SRDF |
| IK, planning, and trajectory execution | `triskel_moveit_config` YAML |
| Process composition and hardware selection | `triskel_bringup` launch |
| Operator modes, device adaptation, recording/replay | `triskel_operator` ROS node |
| Measured, constrained-command, and task-target browser visualization | `triskel_visualization` ROS node |

Triskel is the only supported robot, and its configuration lives in the `triskel_*` ROS
package family.

`robot_state_publisher` owns link transforms. The omni controller owns wheel kinematics,
command timeout behavior, odometry, and `odom -> base_link`. MoveIt owns collision-aware arm
planning and named configurations. Every consumer uses the canonical URDF, whose links pair
detailed visual meshes with simplified planning collision primitives.

Meta Quest support is split at a normal ROS device boundary. `meta_quest_bridge`
converts headset SDK samples to `PoseStamped` and `Joy`; all clutching, freshness checks,
axis mapping, Servo commands, gripper control, homing, and recording are ROS-side behavior.
The bridge starts with the composite bringup, treats missing hardware as a waiting state, and
accepts a wireless ADB address at the launch boundary.

`triskel_visualization` loads the canonical detailed description three times for clearly
separated views. `/joint_states` drives the solid measured robot; arm and gripper trajectory
topics drive a translucent commanded ghost; and integrated Servo twist input drives an orange
task-space gripper target. The ghost therefore shows the posture accepted by the constrained
controller while the target remains free to move beyond it. It owns no robot state or command
path; the browser scene only observes standard ROS interfaces.

The operator samples feedback, command, visualization, and Meta Quest topic rates in sliding
windows. The dashboard marks continuously required or currently active streams healthy or
unhealthy against explicit minimum rates, while valid inactive command streams are shown idle.

## Local tooling

The root `./triskel` shell command owns host-side stack lifecycle and development tasks. It
does not participate in the ROS graph. Robot behavior and reusable runtime functionality live
in ROS packages under `ros_ws/src`.
