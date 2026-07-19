# Actuator Hardware

The actuator layer exposes joint-keyed runtime interfaces while keeping vendor
protocols inside their own packages. `RobotConfig.actuator_control_modes` defines
the runtime-independent control surface used by simulation. Optional physical
bindings live under `hardware.actuators.joints`, and their controller definitions
live under `hardware.actuators.controllers`.

Vendor-specific setup and maintenance documentation lives with each driver. See
the [Feetech documentation](feetech/README.md) for Feetech utilities.
