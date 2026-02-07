import math
import time

from alive_progress import alive_it
from vassar_feetech_servo_sdk import ServoController

from humanoid.logging import get_logger
from humanoid.loop import loop_at_rate

logger = get_logger(__name__)

ID_MIN = 1
ID_MAX = 253

POS_MIN = 0
POS_MAX = 4095


class FeetechController(ServoController):
    def __init__(self, servo_ids: list[str]):
        super().__init__(servo_ids=servo_ids)

    def ping(self, servo_id: int) -> bool:
        try:
            pos = self.read_position(servo_id)
            if pos is None:
                logger.error(f"No motor found at ID {servo_id}. Check connections.")
                return False
            logger.info(f"Motor found at ID {servo_id} (position: {pos})")
            return True
        except Exception as e:
            logger.error(f"Failed to read motor at ID {servo_id}: {e}")
            return False

    def oscillate(
        self, servo_id: int, update_frequency_hz: float = 100, period_s: float = 6
    ) -> bool:
        midpoint = (POS_MAX - POS_MIN) / 2
        angular_freq = 2 * math.pi / period_s

        def work():
            target = midpoint * math.sin(time.time() * angular_freq) + midpoint
            cmd = {servo_id: int(target)}

            logger.info(f"Sending command: {cmd=}")
            result = self.write_position(
                cmd,
            )
            logger.info(f"Result: {result=}")

        try:
            loop_at_rate(work, update_frequency_hz)
        except KeyboardInterrupt:
            logger.info("Shutting down")


class FeetechConfigurator:
    @classmethod
    def scan(cls) -> list[int]:
        found_ids = []
        possible_ids = list(range(ID_MIN, ID_MAX + 1))
        for id in alive_it(possible_ids):
            with FeetechController(servo_ids=[id]) as controller:
                if controller.ping(id):
                    found_ids.append(id)
        return found_ids

    @classmethod
    def set_id(self, current_id: int, new_id: int) -> bool:
        if not (ID_MIN <= new_id <= ID_MAX):
            logger.error(f"Invalid new ID: {new_id}. Must be between {ID_MIN} and {ID_MAX}.")
            return False

        if current_id == new_id:
            logger.warning(f"Current ID and new ID are the same ({current_id}). No change needed.")
            return True

        logger.info(f"Attempting to change motor ID from {current_id} to {new_id}...")

        with FeetechController(servo_ids=[current_id]) as controller:
            if not controller.ping(current_id):
                logger.error(f"Unable to find motor {current_id}")
                return False

            success = controller.set_motor_id(current_id, new_id)

            if success:
                logger.info(f"✓ Successfully changed motor ID from {current_id} to {new_id}")
                controller.servo_ids = [new_id]
            else:
                logger.error(f"Failed to change motor ID from {current_id} to {new_id}")

            return success
