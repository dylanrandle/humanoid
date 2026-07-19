import argparse
import sys

import numpy as np

from humanoid.hardware.actuators.feetech.config import (
    FEETECH_ACCELERATION_MAX,
    FEETECH_ACCELERATION_MIN,
    FEETECH_ACTUATOR_ID_MAX,
    FEETECH_ACTUATOR_ID_MIN,
    FeetechActuatorControllerConfig,
)
from humanoid.hardware.actuators.feetech.driver import FeetechActuatorDriver
from humanoid.hardware.actuators.feetech.scripts.common import (
    add_connection_arguments,
    controller_config_from_args,
)
from humanoid.logger import get_logger

logger = get_logger(__name__)

# Jog step size in radians
DEFAULT_STEP_SIZE = 0.05  # ~2.86 degrees
DEFAULT_ACCELERATION = 15


def get_jog_target(current_position: float, direction: str, step_size: float) -> float | None:
    """Return the target for an arrow-key direction, including at zero."""
    if direction == "left":
        return current_position - step_size
    if direction == "right":
        return current_position + step_size
    return None


def jog_actuator(
    actuator_id: int,
    step_size: float = DEFAULT_STEP_SIZE,
    acceleration: int = DEFAULT_ACCELERATION,
    controller_config: FeetechActuatorControllerConfig | None = None,
) -> None:
    """Jog an actuator using arrow keys with global keyboard capture."""
    from pynput import keyboard  # noqa: PLC0415

    logger.info(f"Starting jog mode for actuator {actuator_id}")
    logger.info(f"Step size: {step_size:.4f} rad (~{np.rad2deg(step_size):.2f} degrees)")
    logger.info(f"Acceleration: {acceleration}")
    logger.info("Controls:")
    logger.info("  LEFT ARROW  - Move actuator counter-clockwise")
    logger.info("  RIGHT ARROW - Move actuator clockwise")
    logger.info("  ESC or 'q'  - Quit")
    logger.info("\nNote: Keyboard input works globally (terminal doesn't need focus)")

    driver = FeetechActuatorDriver.for_actuator_ids(
        [actuator_id],
        controller_config=controller_config,
        configure_on_connect=True,
    )
    driver.connect()

    try:
        # Check if actuator exists
        if not driver.ping(actuator_id):
            raise RuntimeError(f"Actuator {actuator_id} not found. Exiting.")

        # Read initial position
        initial_position = driver.read_position(actuator_id)
        if initial_position is None:
            raise RuntimeError(f"Failed to read position from actuator {actuator_id}")
        current_pos = float(initial_position)

        logger.info(f"Initial position: {current_pos:.4f} rad ({np.rad2deg(current_pos):.2f} deg)")
        logger.info("Ready! Use arrow keys to jog the actuator.\n")

        def on_press(key):
            nonlocal current_pos
            try:
                key_name = getattr(key, "name", "")
                new_pos = get_jog_target(current_pos, key_name, step_size)
                if new_pos is not None:
                    driver.write_position({actuator_id: new_pos}, acceleration=acceleration)
                    current_pos = new_pos
                    direction = "←" if key_name == "left" else "→"
                    logger.info(
                        f"{direction} Position: {current_pos:.4f} rad "
                        f"({np.rad2deg(current_pos):.2f} deg)"
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
        driver.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jog a Feetech actuator using arrow keys",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Jog actuator with ID 1 using default step size
  uv run python -m humanoid.hardware.actuators.feetech.scripts.jog --actuator-id 1

  # Jog actuator with ID 5 using custom step size
  uv run python -m humanoid.hardware.actuators.feetech.scripts.jog \
      --actuator-id 5 --step-size 0.1
        """,
    )
    parser.add_argument(
        "--actuator-id",
        type=int,
        required=True,
        help="ID of the actuator to jog (1-253)",
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
    add_connection_arguments(parser)
    args = parser.parse_args()

    # Validate actuator ID
    if not (FEETECH_ACTUATOR_ID_MIN <= args.actuator_id <= FEETECH_ACTUATOR_ID_MAX):
        logger.error(f"Invalid actuator ID: {args.actuator_id}. Must be between 1 and 253.")
        sys.exit(1)

    # Validate step size
    if args.step_size <= 0:
        logger.error(f"Invalid step size: {args.step_size}. Must be positive.")
        sys.exit(1)

    # Validate acceleration
    if not FEETECH_ACCELERATION_MIN <= args.acceleration <= FEETECH_ACCELERATION_MAX:
        logger.error(
            f"Invalid acceleration: {args.acceleration}. Must be between "
            f"{FEETECH_ACCELERATION_MIN} and {FEETECH_ACCELERATION_MAX}."
        )
        sys.exit(1)

    jog_actuator(
        args.actuator_id,
        args.step_size,
        args.acceleration,
        controller_config_from_args(args),
    )


if __name__ == "__main__":
    main()
