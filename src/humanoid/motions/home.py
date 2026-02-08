import argparse
import time

from humanoid.logger import get_logger
from humanoid.middleware.lcm import Publisher
from humanoid.motors.feetech.controller import POS_MAX, POS_MIN
from humanoid.types.robot import RobotCommand

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Home servo to midpoint by publishing RobotCommand messages"
    )
    parser.add_argument(
        "-i",
        "--servo-id",
        type=str,
        required=True,
        help="ID of the servo to home (as string)",
    )
    args = parser.parse_args()

    logger.info(f"Homing servo {args.servo_id} to midpoint position")
    publisher = Publisher()
    midpoint = (POS_MAX - POS_MIN) / 2
    command = RobotCommand(timestamp=time.perf_counter(), joint_positions={args.servo_id: midpoint})
    publisher.publish(command)


if __name__ == "__main__":
    main()
