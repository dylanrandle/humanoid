# Hardware

Hardware integrations are grouped by device type:

- [Actuators](actuators/README.md) provide joint-keyed command and feedback interfaces.

Robot definitions select their physical hardware implementations. Simulation
implementations are selected by the runtime and do not require physical hardware
configuration.

`RobotHardwareConfig` keeps each device category at the top level. Actuator
controllers and joint bindings live under `hardware.actuators`. Robots without
physical drivers, such as Panda, use `hardware=None`.

Root-state estimators and estimation algorithms are intentionally separate in
[`state_estimation`](../state_estimation/README.md). They may consume hardware
measurements, but they are not hardware interfaces themselves.
