import argparse
import sys

import numpy as np

from humanoid.hardware.actuators.feetech.config import (
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
from humanoid.types.actuator import ActuatorControlMode

logger = get_logger(__name__)

DEFAULT_STEP = 0.5  # rad/s per keypress
DEFAULT_MAX_SPEED = 10.0  # rad/s


def spin_actuator(
    actuator_id: int,
    step: float = DEFAULT_STEP,
    max_speed: float = DEFAULT_MAX_SPEED,
    controller_config: FeetechActuatorControllerConfig | None = None,
) -> None:
    from pynput import keyboard  # noqa: PLC0415

    driver = FeetechActuatorDriver.for_actuator_ids(
        [actuator_id],
        ActuatorControlMode.VELOCITY,
        controller_config,
        configure_on_connect=True,
    )
    driver.connect()

    try:
        if not driver.ping(actuator_id):
            raise RuntimeError(f"Actuator {actuator_id} not found.")

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
            driver.write_velocity({actuator_id: current_velocity})
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
        driver.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send velocity commands to a Feetech actuator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python -m humanoid.hardware.actuators.feetech.scripts.spin --actuator-id 1
  uv run python -m humanoid.hardware.actuators.feetech.scripts.spin \
      --actuator-id 3 --step 1.0 --max-speed 5.0
        """,
    )
    parser.add_argument("--actuator-id", type=int, required=True, help="Actuator ID (1-253)")
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
    add_connection_arguments(parser)
    args = parser.parse_args()

    if not (FEETECH_ACTUATOR_ID_MIN <= args.actuator_id <= FEETECH_ACTUATOR_ID_MAX):
        logger.error(
            f"Invalid actuator ID {args.actuator_id}. Must be "
            f"{FEETECH_ACTUATOR_ID_MIN}-{FEETECH_ACTUATOR_ID_MAX}."
        )
        sys.exit(1)

    if args.step <= 0:
        logger.error("--step must be positive.")
        sys.exit(1)

    if args.max_speed <= 0:
        logger.error("--max-speed must be positive.")
        sys.exit(1)

    spin_actuator(
        args.actuator_id,
        args.step,
        args.max_speed,
        controller_config_from_args(args),
    )


if __name__ == "__main__":
    main()
