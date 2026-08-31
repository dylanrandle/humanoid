# Triskel

Triskel is a custom omni-wheel mobile manipulator built on ROS 2 Jazzy. ROS launch owns
composition, `ros2_control` owns hardware resources, standard controllers own base and arm
commands, and MoveIt consumes the same canonical robot description.

## Packages

The ROS workspace contains one robot-specific package family:

- `triskel_description`: canonical URDF, collision model, meshes, and RViz configuration.
- `triskel_hardware`: Feetech motor IDs, command/state interfaces, and validation gates.
- `triskel_control`: standard controller configuration.
- `triskel_bringup`: mock-first launch composition.
- `triskel_moveit_config`: planning groups, named poses, limits, and controller actions.
- `triskel_operator`: dashboard, keyboard teleop, Meta Quest input bridge, and
  rosbag2 recording/replay.
- `triskel_visualization`: layered browser-native Viser view of measured state, constrained
  controller commands, and the unconstrained task-space gripper target.

See [ARCHITECTURE.md](ARCHITECTURE.md) for ownership boundaries.

## Test in Docker

With Docker Desktop running, the complete mock stack is one command:

```bash
./triskel start
```

The command builds the image when needed, starts it in the background, waits for ROS and
Viser to become healthy, then prints the dashboard URL. The everyday lifecycle is:

```bash
./triskel status
./triskel logs
./triskel stop
```

Use `./triskel start --foreground` to keep logs attached or `./triskel start --no-build` to reuse
the current image. Open <http://127.0.0.1:8765>; Viser is embedded in the dashboard and is
also available full-screen at <http://127.0.0.1:8080>. Both ports bind only to the local
machine.

After completing the physical validation checklist, start the same stack against the STS
hardware. The standard adapter path is selected automatically:

```bash
./triskel start --hardware
```

The default serial device is `/dev/ttyACM0`; use `--serial-port /dev/ttyUSB0` only when the
adapter appears at a different path, and use `--baud-rate` only when the bus is configured for
a value other than `1000000`. The hardware service maps the selected host device to
`/dev/triskel` inside the container and launches with `use_mock_hardware:=false`. Simulation
and hardware are mutually exclusive; starting one stops the other. Dashboard confirmation is
still required before motion.

Docker Desktop does not directly pass USB devices through on macOS. Attach the adapter to
Docker's Linux VM with the documented
[USB/IP workflow](https://docs.docker.com/desktop/features/usbip/) first. If the device path
created inside that VM is not `/dev/ttyACM0`, pass it with `--serial-port`. A native Linux
robot computer can pass its `/dev/...` device directly.

The self-testing image is based on ROS 2 Jazzy and Ubuntu 24.04. It imports the pinned
Feetech driver, resolves dependencies with `rosdep`, builds the colcon workspace, expands
both hardware variants, and launches the complete mock stack. The smoke test drives the
validated home/rest poses, keyboard and Meta Quest teleoperation, gripper and base commands,
MoveIt Servo, odometry/TF, Viser visualization, rosbag2 recording, and replay.

```bash
docker compose -f docker/compose.ros2.yaml --profile test build ros2-test
docker compose -f docker/compose.ros2.yaml --profile test run --rm ros2-test
```

The convenient equivalent is `./triskel smoke`. The suite has a thin shared harness and
focused modules under `docker/tests/` for description expansion, runtime interfaces, named
poses, keyboard control, Meta Quest control, and recording/replay.

The equivalent raw Compose command is:

```bash
docker compose -f docker/compose.ros2.yaml up --build ros2-sim
```

For a sourced ROS shell:

```bash
docker compose -f docker/compose.ros2.yaml run --rm ros2-shell
```

The test and shell services do not receive a robot device. Physical hardware remains
explicitly opt-in through the `hardware` profile or `./triskel start --hardware`.

## Build on ROS 2 Jazzy

```bash
sudo apt install \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  ros-jazzy-desktop \
  ros-jazzy-moveit \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers

source /opt/ros/jazzy/setup.bash
python3 -m pip install --user --break-system-packages \
  -r ros_ws/src/triskel_visualization/requirements.txt
vcs import ros_ws/src < triskel.repos
rosdep install --from-paths ros_ws/src --ignore-src --rosdistro jazzy -r -y
cd ros_ws
colcon build --symlink-install
source install/setup.bash
```

## Run

The complete mock runtime is the default:

```bash
ros2 launch triskel_bringup operator.launch.py \
  use_mock_hardware:=true \
  start_rviz:=false
```

Open <http://127.0.0.1:8765>, leave `Triskel` selected, and choose Home, Rest, Keyboard, or
Meta Quest. The embedded Viser scene shows the measured URDF as the solid robot, the
latest controller-limited posture as a translucent green ghost, and the freely integrated
task-space gripper target in orange. When Cartesian input is inactive, that target follows
the measured gripper and is re-seeded from the current pose when input resumes. The
dashboard also measures topic rates in a sliding live window: required streams turn green
when healthy and red when stale or slow;
inactive command streams remain gray. This behaves the same in simulation and real-hardware
modes. The mock runtime exercises the same ROS controllers and operator paths as real
hardware; it is deterministic controller simulation rather than contact-rich physics.

The validated Home, Rest, Open, and Closed presets are defined once in the MoveIt SRDF. The
operator reads those values directly for homing and gripper command limits.

### Keyboard teleoperation

Select **Keyboard teleop** in the dashboard and hold the displayed keys or buttons. Browser
heartbeats and controller timeouts act as dead-man protection. Tool motion goes through
MoveIt Servo, the base uses `/cmd_vel`, and the gripper uses its trajectory controller.

### Meta Quest teleoperation

Install and configure the maintained Meta Quest reader package and its headset APK/ADB link,
then start the ROS hardware-edge adapter in a separately sourced terminal:

```bash
ros2 run triskel_operator meta_quest_bridge
```

Select **Meta Quest** in the dashboard. The bridge publishes only standard ROS
interfaces, so another OpenXR bridge can be substituted without changing Triskel control:

- `/triskel/teleop/meta_quest/right_controller_pose` (`geometry_msgs/PoseStamped`)
- `/triskel/teleop/meta_quest/joy` (`sensor_msgs/Joy`)

Either grip is the motion dead-man. The right controller commands the tool, the left stick
drives/strifes the base, and right-stick X controls yaw. Hold A to close, hold B to open,
press X to home the arm while preserving the gripper, and press Y to toggle rosbag2 capture.
Input freshness is enforced at 300 ms, so a sleeping headset or lost link stops commands.

### ROS interfaces and recording

The base accepts stamped velocity commands and publishes odometry plus `odom -> base_link`:

```bash
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/TwistStamped \
  "{twist: {linear: {x: 0.05}}}"
```

Real hardware requires completion of
`ros_ws/src/triskel_hardware/config/hardware_validation.yaml` and an explicit selection:

```bash
ros2 launch triskel_bringup operator.launch.py \
  use_mock_hardware:=false \
  serial_port:=/dev/ttyACM0 \
  start_rviz:=true
```

The dashboard requires a confirmation before every real-hardware mode transition and replay.
Recordings are MCAP rosbag2 sessions containing command, feedback, odometry, TF, Servo, and
Meta Quest input topics. Replay is restricted to ROS command topics.

## Development

Development tooling targets Python 3.12 or newer. ROS-facing adapters must use the system
Python 3.12 interpreter shipped with Jazzy on Ubuntu 24.04.

```bash
uv sync --group dev
./triskel check
```

`./triskel check` runs shell syntax validation, formatting, linting, static type analysis,
unit tests, and ROS workspace structural checks. Ruff formats and lints every Python file
under `ros_ws/src`.
The normal type pass is strict for dependencies available on macOS; a second pass includes
the ROS-dependent nodes and launch files while ignoring only imports unavailable outside a
sourced ROS installation. Colcon and the simulation smoke suite validate those imports in
the Jazzy environment.

## Repository layout

```text
ros_ws/src/                 ROS packages and the sole runtime
ros_ws/build/               generated colcon build trees (ignored)
ros_ws/install/             generated ROS overlay (ignored)
ros_ws/log/                 generated colcon logs (ignored)
triskel                     local stack and development command
tests/ros/                  ROS package and Docker contract tests
docker/tests/               modular ROS simulation smoke tests
docker/                     Jazzy image, Compose stack, and smoke harness
triskel.repos               pinned ROS source dependencies
```
