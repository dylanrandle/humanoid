# Hardware

Physical integrations provide device access behind the same robot interfaces used
by simulation. Robot definitions select the hardware for each platform.

## Actuators

`RobotConfig.joint_drives` identifies driven URDF joints; each drive's `control_mode`
selects position or velocity commands in both runtimes. For physical robots,
`hardware.actuators.joints` maps joint names to controller names, actuator IDs, and
directions. `hardware.actuators.controllers` defines the connected buses.

Vendor-specific settings belong to their typed bindings. See
[Feetech actuators](feetech.md) for bus setup and maintenance utilities.
Robots without physical drivers, such as Panda, use `hardware=None`.

## Cameras

The robot's `cameras` mapping names its streams and selects hardware and simulation
settings. See [Cameras](cameras.md) for RealSense setup and depth interpretation.

State estimation consumes measurements but remains separate from device lifecycle
and robot kinematics. See [State estimation](../state-estimation.md).

## See also

- [Documentation index](../README.md)
- [Robot configurations](../../src/humanoid/config/robot/)
- [Hardware implementations](../../src/humanoid/hardware/)
- [Actuator types](../../src/humanoid/types/actuator.py)
