"""Short, bounded HLS position, velocity, and raw torque bench tests.

Stop the robot stack and support the mechanism before running: output is disabled
on exit. Torque units are servo register units, not calibrated N·m.
"""

import argparse
import math
import time
from typing import Any

import scservo_sdk as scs

from humanoid.hardware.actuators.feetech.config import FEETECH_SPEED_UNIT_RAD_S

MAX_ID = 253
MAX_POSITION = 4095
MAX_TORQUE = 1000

MODES = {"position": 0, "velocity": 1, "torque": 2}
POSITION_SCALE = 4096 / (2 * math.pi)
TORQUE_ENABLE = 40
MODE = 33
LOCK = 55
TORQUE_LIMIT = 48
GOAL_TORQUE = 44
GOAL_SPEED = 46


def checked(result: tuple, operation: str) -> Any:
    """Check both transport and servo errors before using returned data."""
    *values, communication, error = result
    if communication != 0 or error != 0:
        raise RuntimeError(f"{operation}: communication={communication}, servo_error={error}")
    return values[0] if values else None


def signed_word(value: int) -> int:
    """HLS uses a sign bit, not two's complement."""
    return abs(value) | (0x8000 if value < 0 else 0)


def set_mode(handler: Any, servo_id: int, mode: int) -> None:
    if checked(handler.read1ByteTxRx(servo_id, MODE), "read mode") == mode:
        return
    checked(handler.write1ByteTxRx(servo_id, LOCK, 0), "unlock EEPROM")
    try:
        checked(handler.write1ByteTxRx(servo_id, MODE, mode), "write mode")
    finally:
        checked(handler.write1ByteTxRx(servo_id, LOCK, 1), "lock EEPROM")
    if checked(handler.read1ByteTxRx(servo_id, MODE), "verify mode") != mode:
        raise RuntimeError("Operating mode verification failed")


def cleanup(
    handler: Any, servo_id: int, original_mode: int | None, original_limit: int | None
) -> None:
    """Attempt every cleanup action even if an earlier bus operation fails."""
    errors = []
    # Goal writes can re-enable output on HLS firmware; disable after clearing them.
    for address, value, width in ((GOAL_TORQUE, 0, 2), (GOAL_SPEED, 0, 2), (TORQUE_ENABLE, 0, 1)):
        try:
            write = handler.write1ByteTxRx if width == 1 else handler.write2ByteTxRx
            checked(write(servo_id, address, value), f"cleanup register {address}")
        except Exception as error:
            errors.append(str(error))
    # Do not change mode unless disabled output is confirmed.
    try:
        if checked(handler.read1ByteTxRx(servo_id, TORQUE_ENABLE), "verify disabled") != 0:
            raise RuntimeError("Output is still enabled")
        if original_mode is not None:
            set_mode(handler, servo_id, original_mode)
        if original_limit is not None:
            checked(
                handler.write2ByteTxRx(servo_id, TORQUE_LIMIT, original_limit),
                "restore torque limit",
            )
        checked(handler.write1ByteTxRx(servo_id, TORQUE_ENABLE, 0), "final disable")
        if checked(handler.read1ByteTxRx(servo_id, TORQUE_ENABLE), "final verification") != 0:
            raise RuntimeError("Output is still enabled after restoring settings")
    except Exception as error:
        errors.append(str(error))
    if errors:
        raise RuntimeError("Cleanup incomplete: " + "; ".join(errors))


def run(args: argparse.Namespace) -> None:  # noqa: PLR0915 - explicit bench-test lifecycle
    port: Any = scs.PortHandler(args.port)
    handler: Any = scs.hls(port)
    original_mode = original_limit = None
    selected = False
    try:
        if not port.setBaudRate(args.baud_rate):
            raise RuntimeError(f"Cannot open {args.port} at {args.baud_rate} baud")
        model = checked(handler.ping(args.actuator_id), "ping")
        print(f"ID {args.actuator_id} responded (model code {model}); using HLS protocol.")
        selected = True
        checked(handler.write1ByteTxRx(args.actuator_id, TORQUE_ENABLE, 0), "disable output")
        original_mode = checked(handler.read1ByteTxRx(args.actuator_id, MODE), "read mode")
        original_limit = checked(
            handler.read2ByteTxRx(args.actuator_id, TORQUE_LIMIT), "read torque limit"
        )
        initial = checked(handler.ReadPos(args.actuator_id), "read initial position")
        target = initial + round(math.radians(args.offset_deg) * POSITION_SCALE)
        if args.mode == "position" and not 0 <= target <= MAX_POSITION:
            raise ValueError("Position target crosses the single-turn boundary; use another offset")
        set_mode(handler, args.actuator_id, MODES[args.mode])
        checked(
            handler.write2ByteTxRx(args.actuator_id, TORQUE_LIMIT, args.torque_limit),
            "set temporary torque limit",
        )
        # Stage every command field before enabling; never replay an old goal.
        checked(handler.WritePosEx(args.actuator_id, initial, 1, 10, 0), "stage safe goal")
        checked(handler.write1ByteTxRx(args.actuator_id, TORQUE_ENABLE, 1), "enable output")
        speed = max(1, round(abs(args.velocity) / FEETECH_SPEED_UNIT_RAD_S))
        if args.mode == "position":
            checked(
                handler.WritePosEx(args.actuator_id, target, speed, 10, args.torque_limit),
                "position command",
            )
        elif args.mode == "velocity":
            checked(
                handler.WriteSpec(
                    args.actuator_id, speed if args.velocity > 0 else -speed, 10, args.torque_limit
                ),
                "velocity command",
            )
        else:
            checked(
                handler.write2ByteTxRx(args.actuator_id, GOAL_TORQUE, signed_word(args.torque)),
                "torque command",
            )

        start = time.monotonic()
        previous = initial
        displacement = 0
        while time.monotonic() - start < args.duration:
            position = checked(handler.ReadPos(args.actuator_id), "read position")
            velocity = checked(handler.ReadSpeed(args.actuator_id), "read velocity")
            current = checked(handler.read2ByteTxRx(args.actuator_id, 69), "read current")
            temperature = checked(handler.read1ByteTxRx(args.actuator_id, 63), "read temperature")
            displacement += (position - previous + 2048) % 4096 - 2048
            previous = position
            degrees = math.degrees(displacement / POSITION_SCALE)
            velocity *= FEETECH_SPEED_UNIT_RAD_S
            print(
                f"delta={degrees:+7.2f} deg  velocity={velocity:+6.3f} rad/s  "
                f"current_raw={handler.scs_tohost(current, 15):+6d}  temp={temperature} C"
            )
            if abs(degrees) >= args.travel_limit_deg or abs(velocity) >= args.speed_limit:
                raise RuntimeError("Travel or speed limit reached")
            if temperature >= args.temperature_limit:
                raise RuntimeError("Temperature limit reached")
            time.sleep(0.05)
    finally:
        try:
            if selected:
                cleanup(handler, args.actuator_id, original_mode, original_limit)
        finally:
            if port.is_open:
                port.closePort()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--port", required=True)
    parser.add_argument("--actuator-id", type=int, required=True)
    parser.add_argument("--baud-rate", type=int, default=1_000_000)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument(
        "--offset-deg",
        type=float,
        default=10.0,
        help="Position mode: offset from measured startup position",
    )
    parser.add_argument(
        "--velocity", type=float, default=0.2, help="rad/s; also sets position-mode speed"
    )
    parser.add_argument("--torque", type=int, default=50, help="Signed raw HLS units, NOT N·m")
    parser.add_argument(
        "--torque-limit", type=int, default=100, help="Temporary raw limit (1-1000)"
    )
    parser.add_argument("--travel-limit-deg", type=float, default=30.0)
    parser.add_argument(
        "--speed-limit", type=float, default=2.0, help="Measured rad/s stop threshold"
    )
    parser.add_argument("--temperature-limit", type=float, default=60.0)
    args = parser.parse_args()
    positive = (args.duration, args.travel_limit_deg, args.speed_limit, args.temperature_limit)
    if not all(math.isfinite(value) and value > 0 for value in positive):
        parser.error("Duration and stop thresholds must be finite and positive")
    if not 1 <= args.actuator_id <= MAX_ID or args.baud_rate <= 0:
        parser.error("Specify an actuator ID in 1-253 and a positive baud rate")
    if not 1 <= args.torque_limit <= MAX_TORQUE or abs(args.torque) > args.torque_limit:
        parser.error("Torque limit must be 1-1000; torque magnitude must not exceed it")
    if not math.isfinite(args.velocity) or not 0 < abs(args.velocity) < args.speed_limit:
        parser.error("Velocity must be finite, nonzero, and below the speed stop threshold")
    if not math.isfinite(args.offset_deg) or abs(args.offset_deg) >= args.travel_limit_deg:
        parser.error("Position offset must be finite and inside the travel stop threshold")
    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted; output disabled and original settings restored.")
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
