import argparse

from vassar_feetech_servo_sdk import ServoController

from humanoid.logging import get_logger

logger = get_logger(__name__)


def verify_motor_at_id(controller: ServoController, servo_id: int) -> bool:
    try:
        pos = controller.read_position(servo_id)
        if pos is None:
            logger.error(f"No motor found at ID {servo_id}. Check connections.")
            return False
        logger.info(f"Motor found at ID {servo_id} (position: {pos})")
        return True
    except Exception as e:
        logger.error(f"Failed to read motor at ID {servo_id}: {e}")
        return False


def set_motor_id(current_id: int, new_id: int) -> bool:
    if not (1 <= new_id <= 253):
        logger.error(f"Invalid new ID: {new_id}. Must be between 1 and 253.")
        return False

    if current_id == new_id:
        logger.warning(f"Current ID and new ID are the same ({current_id}). No change needed.")
        return True

    logger.info(f"Attempting to change motor ID from {current_id} to {new_id}...")

    with ServoController(servo_ids=[current_id]) as controller:
        if not verify_motor_at_id(controller, current_id):
            return False

        success = controller.set_motor_id(current_id, new_id)

        if success:
            logger.info(f"✓ Successfully changed motor ID from {current_id} to {new_id}")
        else:
            logger.error(f"Failed to change motor ID from {current_id} to {new_id}")

        return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-id", type=int, default=1, help="Current ID of the motor (default: 1)"
    )
    parser.add_argument(
        "--new-id",
        type=int,
        required=True,
        help="New ID to assign to the motor (1-253)",
    )
    args = parser.parse_args()

    set_motor_id(args.current_id, args.new_id)


if __name__ == "__main__":
    main()
