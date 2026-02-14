import argparse
import time

from humanoid.constants import ROBOT_CONFIG
from humanoid.logger import get_logger
from humanoid.middleware.lcm import Publisher
from humanoid.types.robot import RobotJointCommand

logger = get_logger(__name__)


def home(servo_ids: list[str] | None = None) -> None:
    """Home one or more servos to position 0.0.

    Args:
        servo_ids: List of servo IDs to home. If None, homes all servos.
    """
    # Default to all servos if none specified
    if servo_ids is None:
        servo_ids = [str(sid) for sid in ROBOT_CONFIG.servo_ids]
        logger.info(f"Homing all servos for {ROBOT_CONFIG.name}: {servo_ids}")
    else:
        logger.info(f"Homing servos: {servo_ids}")

    # Create joint positions dict with all servos set to 0.0
    joint_positions = dict.fromkeys(servo_ids, 0.0)

    publisher = Publisher()
    command = RobotJointCommand(timestamp=time.perf_counter(), joint_positions=joint_positions)
    publisher.publish(command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--servo-ids",
        type=str,
        nargs="+",
        help="IDs of the servos to home. If not specified, homes all servos.",
    )
    args = parser.parse_args()

    home(servo_ids=args.servo_ids)


if __name__ == "__main__":
    main()
