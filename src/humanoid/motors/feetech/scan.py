from vassar_feetech_servo_sdk import ServoController

from humanoid.logging import get_logger

logger = get_logger(__name__)


def scan_servos():
    with ServoController(servo_ids=[1]) as controller:
        found_servos = []
        for servo_id in range(254):
            try:
                # Try to read the position. If it answers, it exists.
                # A timeout error means no servo at that ID.
                pos = controller.read_position(servo_id)

                # If we get a valid number (not None or error), we found one
                if pos is not None:
                    logger.info(f"[+] FOUND Servo at ID: {servo_id}")
                    found_servos.append(servo_id)

            except Exception as e:
                logger.error(f"Caught exception: {e}")

        if not found_servos:
            logger.warning("No servos found. Check power and connections.")
        else:
            logger.info(f"Scan complete. Found IDs: {found_servos}")


if __name__ == "__main__":
    scan_servos()
