from alive_progress import alive_it

from humanoid.logger import get_logger
from humanoid.motors.feetech.controller import (
    FeetechController,
)

logger = get_logger(__name__)

ID_MIN = 1
ID_MAX = 253

ADDR_P_GAIN = 21
ADDR_D_GAIN = 22
ADDR_I_GAIN = 23
ADDR_LOCK = 48


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

    @classmethod
    def set_zero(cls, servo_id: int):
        logger.info(f"Setting zero (middle) position for {servo_id}")
        with FeetechController(servo_ids=[servo_id]) as controller:
            controller.set_middle_position([servo_id])

    @classmethod
    def read_gains(cls, servo_id: int) -> None:
        with FeetechController(servo_ids=[servo_id]) as controller:
            for addr, name in zip(
                [ADDR_P_GAIN, ADDR_I_GAIN, ADDR_D_GAIN], ["P", "I", "D"], strict=True
            ):
                curr, res, err = controller.packet_handler.read1ByteTxRx(servo_id, addr)
                if res != 0 or err != 0:
                    raise RuntimeError(f"Problem reading {name} gain for {servo_id}")
                logger.info(f"{name} gain for {servo_id}: {curr}")
