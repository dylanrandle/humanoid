import argparse
import sys

import numpy as np
from pynput import keyboard

from humanoid.logger import get_logger
from humanoid.motors.feetech.configurator import ID_MAX, ID_MIN
from humanoid.motors.feetech.controller import FeetechMotorController

logger = get_logger(__name__)

DEFAULT_STEP = 0.5  # rad/s per keypress
DEFAULT_MAX_SPEED = 10.0  # rad/s


def spin_motor(motor_id: int, step: float = DEFAULT_STEP, max_speed: float = DEFAULT_MAX_SPEED):
    controller = FeetechMotorController(servo_ids=[motor_id])
    controller.connect()

    try:
        if not controller.ping(motor_id):
            raise RuntimeError(f"Motor {motor_id} not found.")

        logger.info(f"Switching motor {motor_id} to velocity mode...")
        if not controller.set_operating_mode(motor_id, 1):
            raise RuntimeError(f"Failed to set motor {motor_id} to velocity mode.")

        logger.info("Controls:")
        logger.info("  UP    - increase velocity")
        logger.info("  DOWN  - decrease velocity")
        logger.info("  SPACE - stop (velocity = 0)")
        logger.info("  ESC or 'q' - quit")
        logger.info(f"Step: {step} rad/s  |  Max: ±{max_speed} rad/s\n")

        current_velocity = 0.0

        def send(v: float):
            nonlocal current_velocity
            current_velocity = v
            controller.write_velocity({motor_id: current_velocity})
            logger.info(
                f"Velocity: {current_velocity:+.2f} rad/s  "
                f"({np.rad2deg(current_velocity):+.1f} deg/s)"
            )

        def on_press(key):
            try:
                if key == keyboard.Key.up:
                    send(min(current_velocity + step, max_speed))
                elif key == keyboard.Key.down:
                    send(max(current_velocity - step, -max_speed))
                elif key == keyboard.Key.space:
                    send(0.0)
                elif key == keyboard.Key.esc or (hasattr(key, "char") and key.char == "q"):
                    send(0.0)
                    logger.info("Exiting.")
                    return False
            except Exception as e:
                logger.error(f"Error: {e}")

        send(0.0)
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    finally:
        controller.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Send velocity commands to a Feetech motor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m humanoid.motors.feetech.scripts.spin --motor-id 1
  python -m humanoid.motors.feetech.scripts.spin --motor-id 3 --step 1.0 --max-speed 5.0
        """,
    )
    parser.add_argument("--motor-id", type=int, required=True, help="Motor ID (1-253)")
    parser.add_argument(
        "--step",
        type=float,
        default=DEFAULT_STEP,
        help=f"Velocity step per keypress in rad/s (default: {DEFAULT_STEP})",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=DEFAULT_MAX_SPEED,
        help=f"Maximum speed magnitude in rad/s (default: {DEFAULT_MAX_SPEED})",
    )
    args = parser.parse_args()

    if not (ID_MIN <= args.motor_id <= ID_MAX):
        logger.error(f"Invalid motor ID {args.motor_id}. Must be {ID_MIN}-{ID_MAX}.")
        sys.exit(1)

    if args.step <= 0:
        logger.error("--step must be positive.")
        sys.exit(1)

    if args.max_speed <= 0:
        logger.error("--max-speed must be positive.")
        sys.exit(1)

    spin_motor(args.motor_id, args.step, args.max_speed)


if __name__ == "__main__":
    main()
