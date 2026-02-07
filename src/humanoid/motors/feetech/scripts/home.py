import argparse

from humanoid.logging import get_logger
from humanoid.motors.feetech.controller import FeetechController

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Home a Feetech servo to its midpoint position")
    parser.add_argument(
        "--servo-id",
        type=int,
        required=True,
        help="ID of the servo to home",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=2000.0,
        help="Maximum speed in position units per second (default: 2000.0)",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=100,
        help="Update frequency in Hz (default: 100)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5.0,
        help="Position tolerance to consider homing complete (default: 5.0)",
    )
    args = parser.parse_args()

    logger.info(f"Homing servo {args.servo_id} to midpoint position")
    with FeetechController(servo_ids=[args.servo_id]) as controller:
        success = controller.home(
            servo_id=args.servo_id,
            speed=args.speed,
            update_frequency_hz=args.frequency,
            tolerance=args.tolerance,
        )
        if success:
            logger.info("Homing completed successfully")
        else:
            logger.error("Homing failed")


if __name__ == "__main__":
    main()
