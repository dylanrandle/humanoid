"""Bounded arm_3 gravity/current experiment; default is read-only preview.

Stop the robot stack, support the arm, and keep the base upright. Other joints
must already be held or supported. Only ID 3 is written; it is disabled on exit.
"""

import argparse
import math
import time
from pathlib import Path
from typing import Any

import pinocchio as pin
import scservo_sdk as scs

from humanoid.config.actuator import HLS3930M_CURRENT_CALIBRATION
from humanoid.config.robot.triskel import ACTUATOR_CONFIGS, GRIPPER_ID, JOINT_IDS
from humanoid.hardware.actuators.feetech.config import FEETECH_SPEED_UNIT_RAD_S
from humanoid.hardware.actuators.feetech.ports import detect_servo_port
from humanoid.hardware.actuators.feetech.scripts.hls_test import (
    GOAL_TORQUE,
    MODE,
    TORQUE_ENABLE,
    TORQUE_LIMIT,
    checked,
    cleanup,
    set_mode,
    signed_word,
)

TARGET_ID = 3
TARGET_JOINT = "arm_3"
LOG_INTERVAL = 0.25
MAX_SAMPLE_SECONDS = 0.1
RATED_TORQUE_NM = 0.85
MAX_OUTPUT_LIMIT = 1000


def torque_to_raw(torque: float, limit: float, *, inverted: bool = False) -> int:
    """Clamp in N·m, then quantize toward zero to respect the requested limit."""
    if not math.isfinite(torque) or not math.isfinite(limit) or limit <= 0:
        raise ValueError("Torque must be finite and its limit positive")
    calibration = HLS3930M_CURRENT_CALIBRATION
    torque = max(-limit, min(limit, torque))
    raw = int(torque / (calibration.amperes_per_unit * calibration.effort_per_ampere))
    raw = max(-2047, min(2047, raw))
    return -raw if inverted else raw


def load_model() -> Any:
    urdf = Path(__file__).resolve().parents[4] / "robots/assets/triskel/urdf/triskel.urdf"
    return pin.buildModelFromUrdf(str(urdf), mimic=True)


def read_configuration(handler: Any, model: Any) -> Any:
    """Read the whole arm, using the same zero/sign convention as the driver."""
    q = pin.neutral(model)
    for name in [*JOINT_IDS, GRIPPER_ID]:
        actuator = ACTUATOR_CONFIGS[name]
        raw = checked(handler.ReadPos(actuator.actuator_id), f"read {name}")
        angle = (raw - 2047.5) / 2047.5 * math.pi
        if actuator.inverted:
            angle = -angle
        joint = model.joints[model.getJointId(name)]
        q[joint.idx_q] = angle
    return q


def run(args: argparse.Namespace) -> None:  # noqa: PLR0912, PLR0915 - explicit hardware lifecycle
    model = load_model()
    data = model.createData()
    joint = model.joints[model.getJointId(TARGET_JOINT)]
    actuator = ACTUATOR_CONFIGS[TARGET_JOINT]
    port_name = args.port or detect_servo_port()
    port: Any = scs.PortHandler(port_name)
    handler: Any = scs.hls(port)
    modified = False
    original_mode = original_limit = None
    try:
        if not port.setBaudRate(args.baud_rate):
            raise RuntimeError("Cannot open servo adapter")
        print(f"Using {port_name}; {'ENABLED' if args.enable else 'READ-ONLY preview'}")
        checked(handler.ping(TARGET_ID), "ping arm_3")
        q = read_configuration(handler, model)
        initial = float(q[joint.idx_q])
        if args.enable:
            original_mode = checked(handler.read1ByteTxRx(TARGET_ID, MODE), "read mode")
            original_limit = checked(handler.read2ByteTxRx(TARGET_ID, TORQUE_LIMIT), "read limit")
            modified = True
            checked(handler.write1ByteTxRx(TARGET_ID, TORQUE_ENABLE, 0), "disable")
            if checked(handler.read1ByteTxRx(TARGET_ID, TORQUE_ENABLE), "verify disabled") != 0:
                raise RuntimeError("Output did not disable")
            set_mode(handler, TARGET_ID, 2)
            # Register 48 is percent output, independent of current/N·m limits.
            checked(
                handler.write2ByteTxRx(TARGET_ID, TORQUE_LIMIT, args.output_limit), "output limit"
            )
            checked(handler.write2ByteTxRx(TARGET_ID, GOAL_TORQUE, 0), "zero current")
            checked(handler.write1ByteTxRx(TARGET_ID, TORQUE_ENABLE, 1), "enable")
        start = time.monotonic()
        last_log = start - 1
        cycles = 0
        while time.monotonic() - start < args.duration:
            cycle_start = time.monotonic()
            q = read_configuration(handler, model)
            position = float(q[joint.idx_q])
            speed = checked(handler.ReadSpeed(TARGET_ID), "read speed") * FEETECH_SPEED_UNIT_RAD_S
            if actuator.inverted:
                speed = -speed
            temperature = checked(handler.read1ByteTxRx(TARGET_ID, 63), "temperature")
            if (
                not model.lowerPositionLimit[joint.idx_q]
                < position
                < model.upperPositionLimit[joint.idx_q]
            ):
                raise RuntimeError("arm_3 outside model joint limits; check calibration")
            if abs(position - initial) >= math.radians(args.travel_limit_deg):
                raise RuntimeError("Travel limit reached")
            if abs(speed) >= args.speed_limit or temperature >= args.temperature_limit:
                raise RuntimeError("Speed or temperature limit reached")
            gravity = float(pin.computeGeneralizedGravity(model, data, q)[joint.idx_v])
            elapsed = time.monotonic() - start
            requested = args.gravity_scale * gravity + args.torque_nm - args.damping * speed
            requested *= min(1.0, elapsed / args.ramp_seconds)
            raw = torque_to_raw(requested, args.max_torque_nm, inverted=actuator.inverted)
            if time.monotonic() - cycle_start > MAX_SAMPLE_SECONDS:
                raise RuntimeError("Feedback acquisition exceeded 100 ms; stopping")
            if elapsed >= args.duration:
                break
            if args.enable:
                checked(
                    handler.write2ByteTxRx(TARGET_ID, GOAL_TORQUE, signed_word(raw)),
                    "current command",
                )
            cycles += 1
            if time.monotonic() - last_log >= LOG_INTERVAL:
                print(
                    f"q={position:+.3f} rad g={gravity:+.3f} Nm requested={requested:+.3f} Nm "
                    f"current_raw={raw:+d} rate={cycles / max(elapsed, 0.001):.1f} Hz"
                )
                last_log = time.monotonic()
            time.sleep(max(0.0, 1 / args.rate_hz - (time.monotonic() - cycle_start)))
    finally:
        try:
            if modified:
                cleanup(handler, TARGET_ID, original_mode, original_limit)
        finally:
            port.closePort()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port")
    parser.add_argument("--baud-rate", type=int, default=1_000_000)
    parser.add_argument("--enable", action="store_true", help="Actually command arm_3 current")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--gravity-scale", type=float, default=1.0)
    parser.add_argument(
        "--torque-nm",
        type=float,
        default=0.0,
        help="Additional joint torque; use gravity-scale 0 for constant torque",
    )
    parser.add_argument("--max-torque-nm", type=float, default=0.4)
    parser.add_argument(
        "--output-limit", type=int, default=100, help="Register 48, 0.1%% units; 100 means 10%%"
    )
    parser.add_argument("--damping", type=float, default=0.05, help="N·m per rad/s")
    parser.add_argument("--ramp-seconds", type=float, default=1.0)
    parser.add_argument("--travel-limit-deg", type=float, default=15.0)
    parser.add_argument("--speed-limit", type=float, default=1.0)
    parser.add_argument("--temperature-limit", type=float, default=60.0)
    args = parser.parse_args()
    positive = (
        args.duration,
        args.rate_hz,
        args.max_torque_nm,
        args.ramp_seconds,
        args.travel_limit_deg,
        args.speed_limit,
        args.temperature_limit,
    )
    if not all(math.isfinite(value) and value > 0 for value in positive):
        parser.error("Timing and limits must be positive and finite")
    if not all(
        math.isfinite(value) for value in (args.gravity_scale, args.torque_nm, args.damping)
    ):
        parser.error("Controller parameters must be finite")
    if not 0 <= args.gravity_scale <= 1 or args.damping < 0:
        parser.error("Gravity scale must be 0..1 and damping nonnegative")
    if (
        args.max_torque_nm > RATED_TORQUE_NM
        or not 1 <= args.output_limit <= MAX_OUTPUT_LIMIT
        or args.baud_rate <= 0
    ):
        parser.error("Maximum torque must be <=0.85 Nm, output limit 1..1000, baud rate positive")
    try:
        run(args)
    except KeyboardInterrupt:
        print("Stopped.")
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
