# Feetech Actuators

The Feetech package includes maintenance utilities for directly connected
actuators. Stop the robot stack before using them.

Every utility accepts `--port`, `--baud-rate`, and `--servo-type`. Specify the
port when more than one Feetech controller is connected so the command cannot
target an actuator with the same ID on another bus. For example:

```bash
uv run python -m humanoid.hardware.actuators.feetech.scripts.scan \
  --port /dev/ttyUSB0 --baud-rate 1000000 --servo-type sts
```

Scan for connected actuators:

```bash
uv run python -m humanoid.hardware.actuators.feetech.scripts.scan
```

Jog a position-controlled actuator:

```bash
uv run python -m humanoid.hardware.actuators.feetech.scripts.jog --actuator-id 1
```

Spin a velocity-controlled actuator:

```bash
uv run python -m humanoid.hardware.actuators.feetech.scripts.spin --actuator-id 1
```

Change an actuator ID:

```bash
uv run python -m humanoid.hardware.actuators.feetech.scripts.set_id --current-id 1 --new-id 2
```

Set the middle position:

```bash
uv run python -m humanoid.hardware.actuators.feetech.scripts.zero --actuator-id 1
```

Read configured gains:

```bash
uv run python -m humanoid.hardware.actuators.feetech.scripts.read_gains --actuator-id 1
```

Robot configs can set `port`, `baud_rate`, `servo_type`, and
`feedback_read_retries` on each `FeetechActuatorControllerConfig`. A single controller may
omit `port` to use SDK auto-detection. Multiple Feetech controllers require distinct
explicit ports. Feedback sync reads retry missing responses and transient bus failures
twice by default; servo-reported status faults are surfaced immediately without retrying.

Position-controlled actuators may also declare persistent `position_pid` gains in
their `FeetechActuatorConfig`. On connection, the driver reads the servo first and
only unlocks/writes EEPROM when a configured gain differs, then locks EEPROM and
verifies the result before enabling torque. Triskel declares each position-controlled
actuator's gains explicitly by physical actuator ID so they can be tuned independently.
