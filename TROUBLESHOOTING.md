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

## Meta Quest stays in Waiting

The dashboard supports the device even when no headset is connected, but it reports the input
online only while both pose and Joy samples are fresh. Start the bridge and inspect its ROS
topics:

```bash
ros2 run triskel_operator meta_quest_bridge
ros2 topic hz /triskel/teleop/meta_quest/right_controller_pose
ros2 topic hz /triskel/teleop/meta_quest/joy
```

If the bridge cannot import its reader dependency, install the maintained Meta Quest reader
package into the same ROS Python environment. If no samples arrive, verify `adb devices`,
accept USB debugging inside the headset, and confirm the Meta Quest teleoperation APK is
running. Releasing both grip buttons or losing either input stream intentionally stops motion
after 300 ms.

## Dashboard unavailable

Launch the composite operator stack, wait for all four controllers to become active, then
open <http://127.0.0.1:8765>. Use `dashboard_host:=0.0.0.0` only on a trusted robot network.

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

## Browser visualization unavailable

The Viser ROS node serves its browser client on port 8080. With Docker, use the friendly
launcher so both dashboard and visualization ports are published and readiness is checked:

```bash
./triskel start
```

Check <http://127.0.0.1:8080> directly and verify the ROS heartbeat:

```bash
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
