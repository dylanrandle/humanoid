import argparse
import sys

import numpy as np
from pynput import keyboard

from humanoid.logger import get_logger
from humanoid.motors.feetech.configurator import ID_MAX, ID_MIN
from humanoid.motors.feetech.controller import FeetechMotorController

logger = get_logger(__name__)

# Jog step size in radians
DEFAULT_STEP_SIZE = 0.05  # ~2.86 degrees
DEFAULT_ACCELERATION = 15


def jog_motor(
    motor_id: int, step_size: float = DEFAULT_STEP_SIZE, acceleration: int = DEFAULT_ACCELERATION
):
    """Jog a motor using arrow keys with global keyboard capture."""
    logger.info(f"Starting jog mode for motor {motor_id}")
    logger.info(f"Step size: {step_size:.4f} rad (~{np.rad2deg(step_size):.2f} degrees)")
    logger.info(f"Acceleration: {acceleration}")
    logger.info("Controls:")
    logger.info("  LEFT ARROW  - Move motor counter-clockwise")
    logger.info("  RIGHT ARROW - Move motor clockwise")
    logger.info("  ESC or 'q'  - Quit")
    logger.info("\nNote: Keyboard input works globally (terminal doesn't need focus)")

    controller = FeetechMotorController(servo_ids=[motor_id])
    controller.connect()

    try:
        # Check if motor exists
        if not controller.ping(motor_id):
            raise RuntimeError(f"Motor {motor_id} not found. Exiting.")

        # Read initial position
        current_pos = controller.read_position(motor_id)
        if current_pos is None:
            raise RuntimeError(f"Failed to read position from motor {motor_id}")

        logger.info(f"Initial position: {current_pos:.4f} rad ({np.rad2deg(current_pos):.2f} deg)")
        logger.info("Ready! Use arrow keys to jog the motor.\n")

        def on_press(key):
            nonlocal current_pos
            try:
                if current_pos and key == keyboard.Key.left:
                    # Move counter-clockwise (decrease angle)
                    new_pos = current_pos - step_size
                    controller.write_position({motor_id: new_pos}, acceleration=acceleration)
                    current_pos = new_pos
                    logger.info(
                        f"← Position: {current_pos:.4f} rad ({np.rad2deg(current_pos):.2f} deg)"
                    )
                elif current_pos and key == keyboard.Key.right:
                    # Move clockwise (increase angle)
                    new_pos = current_pos + step_size
                    controller.write_position({motor_id: new_pos}, acceleration=acceleration)
                    current_pos = new_pos
                    logger.info(
                        f"→ Position: {current_pos:.4f} rad ({np.rad2deg(current_pos):.2f} deg)"
                    )
                elif key == keyboard.Key.esc or (hasattr(key, "char") and key.char == "q"):
                    logger.info("Exiting jog mode...")
                    return False  # Stop listener
            except Exception as e:
                logger.error(f"Error during jog: {e}")

        # Start listening for keyboard events
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

        logger.info(
            f"Final position: {current_pos:.4f} rad ({current_pos * 180 / 3.14159:.2f} deg)"
        )

    finally:
        controller.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Jog a Feetech motor using arrow keys",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Jog motor with ID 1 using default step size
  python -m humanoid.motors.feetech.scripts.jog --motor-id 1

  # Jog motor with ID 5 using custom step size
  python -m humanoid.motors.feetech.scripts.jog --motor-id 5 --step-size 0.1
        """,
    )
    parser.add_argument(
        "--motor-id",
        type=int,
        required=True,
        help="ID of the motor to jog (1-253)",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=DEFAULT_STEP_SIZE,
        help=f"Step size in radians (default: {DEFAULT_STEP_SIZE})",
    )
    parser.add_argument(
        "--acceleration",
        type=int,
        default=DEFAULT_ACCELERATION,
        help=f"Acceleration value (default: {DEFAULT_ACCELERATION})",
    )
    args = parser.parse_args()

    # Validate motor ID
    if not (ID_MIN <= args.motor_id <= ID_MAX):
        logger.error(f"Invalid motor ID: {args.motor_id}. Must be between 1 and 253.")
        sys.exit(1)

    # Validate step size
    if args.step_size <= 0:
        logger.error(f"Invalid step size: {args.step_size}. Must be positive.")
        sys.exit(1)

    # Validate acceleration
    if args.acceleration <= 0:
        logger.error(f"Invalid acceleration: {args.acceleration}. Must be positive.")
        sys.exit(1)

    jog_motor(args.motor_id, args.step_size, args.acceleration)


if __name__ == "__main__":
    main()
