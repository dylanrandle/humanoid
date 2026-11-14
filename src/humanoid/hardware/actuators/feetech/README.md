# Feetech Actuators

## HLS bench test

Test one HLS3930M using the installed Python SDK. Stop the robot stack, secure
the motor, and support any attached load: this utility disables output on exit.
Specify the actual USB port and actuator ID; it never broadcasts commands.

```bash
uv run python -m humanoid.hardware.actuators.feetech.scripts.hls_test position --port /dev/cu.usbserial-XXXX --actuator-id 1 --offset-deg 10
uv run python -m humanoid.hardware.actuators.feetech.scripts.hls_test velocity --port /dev/cu.usbserial-XXXX --actuator-id 1 --velocity 0.2
uv run python -m humanoid.hardware.actuators.feetech.scripts.hls_test torque --port /dev/cu.usbserial-XXXX --actuator-id 1 --torque 50
```

Each test lasts three seconds by default (`--duration` changes this). Negative
velocity or torque reverses direction. Torque values are raw HLS register units,
not calibrated N·m. The temporary raw torque limit defaults to 100/1000; this may
be insufficient to overcome friction. Position tests are relative to the startup
position and do not cross the single-turn boundary. The script assumes the selected
ID is an HLS motor; a successful ping does not validate the model family.

Telemetry prints approximately every 50 ms plus bus time. Travel, speed, and
temperature thresholds terminate the test; these software checks cannot stop a
motor instantly or protect against a disconnected bus. This is a constant-command
bench test, not a high-rate torque controller. Ctrl+C also cleans up. Cleanup
disables output, clears torque/speed targets, and restores the original mode and
torque limit without re-enabling output. Mode changes may write EEPROM; avoid
repeatedly cycling modes in a control loop. See `hls_test --help` via the module
command for all limits and options.

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

### arm_3 HLS gravity compensation (standalone experiment)

With the HLS3930M installed as ID 3, stop the robot stack and support the arm.
Keep the base upright. This script reads IDs 1–8 on the shared bus and writes only
ID 3; other joints must already be held or mechanically supported. Do not run
another bus-owning process alongside it. The new motor must have the same encoder
zero and positive direction as the old arm_3 actuator. Check the displayed angle
against the physical pose before enabling. The URDF mass model is unchanged;
replacement hardware and payload differences introduce gravity error.

Read-only preview (no mode changes or current commands):

```bash
uv run python -m humanoid.hardware.actuators.feetech.scripts.gravity
```

Bounded five-second test, initially with partial gravity compensation:

```bash
uv run python -m humanoid.hardware.actuators.feetech.scripts.gravity --enable --gravity-scale 0.25
```

Default enabled behavior uses `tau = gravity_scale*g(q) + torque_nm - damping*dq`,
a one-second startup ramp, and a 0.4 N·m command cap. Use `--gravity-scale 0
--torque-nm 0.2 --damping 0` for a constant-torque test. Positive torque follows
model joint coordinates. `--output-limit 100` separately caps register 48 at 10%;
this may prevent requested current being reached. It is not a current or N·m unit.
The nominal conversion is 6.5 mA/count and 1.22583125 N·m/A at the output shaft;
friction and calibration error mean actual torque differs.

The loop targets 50 Hz and reports achieved rate; this is not a real-time 500 Hz
controller. It stops on communication errors, feedback acquisition over 100 ms,
15 degrees of travel, 1 rad/s speed, or 60°C. On exit it clears commands, disables
ID 3, and restores its original mode/output limit while leaving it disabled.
Support the arm before stopping: it will not hold position afterward. A process
kill or USB failure can prevent cleanup; there is no independent hardware watchdog.
No production robot configuration is changed by this experiment.

### Main-driver gravity compensation

Triskel's hardware configuration now selects HLS current mode for arm_3 (ID 3).
The other servos retain their existing position/velocity modes on the same port.
The common STS/HLS register protocol is used for mode, current, and feedback;
ID 3 is excluded from STS position packets. No second serial connection is opened.

Run the normal real hardware driver, or run it alone for gravity testing:

```bash
uv run python -m humanoid.nodes.robot.driver
```

Only one process may own the bus. Support the arm at startup/shutdown and verify
its zero/direction. The base must be upright; planar root estimation does not
measure roll or pitch. Initial feedback is required from every configured actuator,
including the wheels. Unlike the bench script, the main driver has no fixed test
duration and configures the other servos to hold their startup positions.

The hardware-only `gravity_compensation` configuration overrides arm_3's logical
position mode: incoming arm_3 position targets are ignored, including homing and
teleoperation targets. This is gravity-only assistance, not position tracking;
simulation retains its existing position-controller behavior. Do not expect an
automatic homing sequence or coordinated arm trajectory to complete with this
experimental override enabled.

At the existing 50 Hz driver rate, each complete measured robot configuration
feeds Pinocchio's gravity calculation. The arm_3 command is `g(q) - 0.05*dq`,
ramped over one second and clamped to +/-0.4 N·m. Parameters live in
`src/humanoid/config/robot/triskel.py`; register 48 remains separately limited to
100 (10% output). These conservative caps can produce only partial compensation.
Setting the hardware `gravity_compensation` field to `None` removes this override;
HLS position-packet support on a mixed bus is not provided by this change, so do
not use that as a position-control fallback for the installed HLS motor.

Missing/nonfinite feedback, acquisition longer than 250 ms, joint-limit violations,
speed >=1 rad/s, or temperature >=60 C abort the driver and attempt zero current.
Gravity runs locally before any controller commands arrive. Once commands arrive,
the normal 250 ms command watchdog also stops current on command loss; a fresh
command restarts the ramp. Shutdown clears current before disabling the HLS output.
Hardware communication failure or a killed process can prevent cleanup; this is
not an independent hardware watchdog or a real-time 500 Hz loop.
