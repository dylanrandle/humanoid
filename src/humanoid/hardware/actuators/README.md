# Actuator Hardware

The actuator layer exposes joint-keyed runtime interfaces while keeping vendor
protocols inside their own packages. Each entry in `RobotConfig.joint_drives` identifies
a driven URDF joint. Its `control_mode` selects position or velocity commands
for both the hardware driver and simulator. For a hardware robot,
`hardware.actuators.joints` maps the same joint names to `HardwareActuatorBinding`
instances with the controller name, actuator ID, and direction. Controller
definitions live under `hardware.actuators.controllers`. Vendor-specific bindings
hold settings such as the Feetech motor’s `position_pid` registers.

Vendor-specific setup and maintenance documentation lives with each driver. See
the [Feetech documentation](feetech/README.md) for Feetech utilities.
