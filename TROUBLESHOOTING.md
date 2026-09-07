# Troubleshooting

## Package not found

Build and source the workspace in the same shell:

```bash
source /opt/ros/jazzy/setup.bash
cd ros_ws
colcon build --symlink-install
source install/setup.bash
```

Then verify the package index:

```bash
ros2 pkg prefix triskel_bringup
```

## Controllers do not activate

Start with mock hardware and inspect the manager:

```bash
ros2 launch triskel_bringup operator.launch.py use_mock_hardware:=true start_rviz:=false
ros2 control list_controllers -c /controller_manager
```

All four controllers should be `active`: `joint_state_broadcaster`,
`omni_base_controller`, `arm_controller`, and `gripper_controller`.

## Physical serial device unavailable

Physical hardware is never selected unless `--hardware` is present. Confirm the device and
permissions, complete `ros_ws/src/triskel_hardware/config/hardware_validation.yaml`, and
start the hardware stack:

```bash
./triskel start --hardware
```

The standard device is `/dev/ttyACM0`. If the adapter has another name, pass it with
`--serial-port /dev/ttyUSB0`. On native Linux, confirm the device exists and is accessible to
Docker. Docker Desktop does not directly pass through USB devices on macOS; first attach the
adapter to its Linux VM with Docker's
[USB/IP workflow](https://docs.docker.com/desktop/features/usbip/), then override the device
path only if necessary.

## Home or Rest reaches the wrong pose

Home and Rest are absolute joint angles from
`ros_ws/src/triskel_moveit_config/config/triskel.srdf`; they do not calibrate motor zero.
Mode 0 uses a position command. Its velocity value is an unsigned speed cap, with direction
determined by the position target, so the arm and gripper controllers claim only `position`.
`proportional_vel_max=1000` scales positive speed caps from joint position errors for
coordinated arrival.

The pinned STS driver mixed units in that proportional path: it compared a raw steps/s cap
with each joint's rad/s limit, reducing 1000 steps/s to roughly 5. Docker and the documented
native build apply `sts-proportional-velocity-units.patch` before compiling the dependency.
`proportional_vel_deadband=0` keeps the configured cap active for every nonzero position error.

`position_center_steps=2048` maps feedback as
`q = (2048 - raw_position) * 2*pi/4096`. If a pose is consistently offset after commands are
moving correctly, inspect the motor's mechanical zero and stored encoder offset. The
`/one_key_calibration` service redefines the current physical position as the encoder
midpoint; it does not move the arm to Home or Rest.

Inspect the active stack from the robot checkout:

```bash
./triskel shell ros2 control list_controllers
./triskel shell ros2 control list_hardware_interfaces
./triskel shell ros2 topic echo --once /arm_controller/controller_state
./triskel shell ros2 topic echo --once /gripper_controller/controller_state
./triskel shell ros2 topic echo --once /dynamic_joint_states
./triskel logs
```

The arm and gripper should claim only `position` commands. Compare controller reference,
output, and feedback positions. If output is right but feedback does not follow, inspect bus
errors, motor IDs, power, torque, and the stored speed cap.

After changing package resources, rebuild the image; `--no-build` keeps the previous baked
configuration.

## Meta Quest stays in Waiting

The dashboard supports the device even when no headset is connected, but it reports the input
online only while both pose and Joy samples are fresh. The bridge starts with the stack. For
Docker Desktop, confirm the Wi-Fi address passed to `--quest-ip`, then inspect the process and
its ROS topics:

```bash
./triskel logs
docker compose -f docker/compose.ros2.yaml run --rm ros2-shell
```

From the sourced ROS shell opened by the second command:

```bash
ros2 topic hz /triskel/teleop/meta_quest/right_controller_pose
ros2 topic hz /triskel/teleop/meta_quest/joy
```

If no samples arrive, connect the headset to the Mac once, verify `adb devices`, accept USB
debugging inside the headset, run `adb tcpip 5555`, and restart with the address reported by
`adb shell ip route`. The image contains the pinned reader, ADB client, and APK. Releasing both
grip buttons or losing either input stream intentionally stops motion after 300 ms.

## Dashboard unavailable

Launch the composite operator stack, wait for all four controllers to become active, then
open <http://127.0.0.1:8765>. Use `dashboard_host:=0.0.0.0` only on a trusted robot network.
When the stack is running on a Raspberry Pi, leave its localhost-only binding in place and
forward both operator ports from the Mac:

```bash
./triskel dashboard dylan@triskel.local
```

If forwarding fails, verify ordinary SSH access first and check that ports 8765 and 8080 are
not already occupied on the Mac.

For the local Docker simulation, the simplest diagnostics are:

```bash
./triskel status
./triskel logs
```

`./triskel stop` cleanly removes the stack so `./triskel start` can launch a fresh instance.

## Topic rate is red

Red means a stream required by the current mode is stale or below the minimum rate shown on
its card. Joint feedback, odometry, and the visualization heartbeat are always expected.
Quest input becomes required after selecting Meta Quest mode; motion command streams are only
required while the corresponding dead-man input is active. Gray is therefore normal for an
idle command topic. Inspect a red stream directly with `ros2 topic hz <topic>`.
When using Docker Desktop on macOS, run that ROS command from `ros2-shell` as shown in the
Meta Quest section above.

## Browser visualization unavailable

The Viser ROS node serves its browser client on port 8080. With Docker, use the friendly
launcher so both dashboard and visualization ports are published and readiness is checked:

```bash
./triskel start
```

Check <http://127.0.0.1:8080> directly and verify the ROS heartbeat:

```bash
docker compose -f docker/compose.ros2.yaml run --rm ros2-shell
ros2 topic echo --once /triskel/visualization/ready
```

For a host ROS installation, install the pinned Python visualization dependencies from
`ros_ws/src/triskel_visualization/requirements.txt` into the same Python environment used by
ROS.

## Reproduce the supported environment

Use the Docker smoke test when the host ROS installation is uncertain:

```bash
docker compose -f docker/compose.ros2.yaml --profile test build ros2-test
docker compose -f docker/compose.ros2.yaml --profile test run --rm ros2-test
```
