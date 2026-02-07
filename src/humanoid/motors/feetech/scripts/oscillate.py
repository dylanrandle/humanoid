import argparse

from humanoid.logging import get_logger
from humanoid.motors.feetech.controller import FeetechController

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test Feetech servo by oscillating it")
    parser.add_argument(
        "--servo-id",
        type=int,
        required=True,
        help="ID of the servo to oscillate",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=100,
        help="Update frequency in Hz (default: 100)",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=6,
        help="Oscillation period in seconds (default: 6)",
    )
    args = parser.parse_args()

    logger.info(f"Starting oscillation test for servo {args.servo_id}")
    with FeetechController(servo_ids=[args.servo_id]) as controller:
        controller.oscillate(
            servo_id=args.servo_id,
            update_frequency_hz=args.frequency,
            period_s=args.period,
        )


if __name__ == "__main__":
    main()
