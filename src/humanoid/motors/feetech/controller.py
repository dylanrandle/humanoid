import math

from alive_progress import alive_it
from vassar_feetech_servo_sdk import ServoController

from humanoid.logger import get_logger

logger = get_logger(__name__)

ID_MIN = 1
ID_MAX = 253

POS_MIN = 0
POS_MAX = 4095
POS_MID = (POS_MAX + POS_MIN) / 2

ADDR_P_GAIN = 21
ADDR_D_GAIN = 22
ADDR_I_GAIN = 23
ADDR_LOCK = 48
ADDR_TEMPERATURE = 63


class FeetechController(ServoController):
    def __init__(self, servo_ids: list[str]):
        super().__init__(servo_ids=servo_ids)

    @staticmethod
    def angle_to_position(angle: float) -> int:
        angle = max(-math.pi, min(math.pi, angle))
        position = POS_MID + (angle / math.pi) * (POS_MAX - POS_MID)
        return int(position)

    @staticmethod
    def position_to_angle(position: int) -> float:
        return ((position - POS_MID) / (POS_MAX - POS_MID)) * math.pi

    def write_position(self, positions: dict[int, float]) -> None:
        raw_positions = {
            servo_id: self.angle_to_position(angle) for servo_id, angle in positions.items()
        }
        super().write_position(raw_positions)

    def read_position(self, servo_id: int) -> float | None:
        raw_position = super().read_position(servo_id)
        if raw_position is None:
            return None
        return self.position_to_angle(raw_position)

    def read_all_positions(self) -> dict[int, float]:
        raw_positions = super().read_all_positions()
        return {servo_id: self.position_to_angle(pos) for servo_id, pos in raw_positions.items()}

    def ping(self, servo_id: int) -> bool:
        try:
            pos = self.read_position(servo_id)
            if pos is None:
                logger.error(f"No motor found at ID {servo_id}. Check connections.")
                return False
            logger.info(f"Motor found at ID {servo_id} (angle: {pos:.3f} rad)")
            return True
        except Exception as e:
            logger.error(f"Failed to read motor at ID {servo_id}: {e}")
            return False

    def read_temperature(self, servo_id: int) -> int:
        temp, res, err = self.packet_handler.read1ByteTxRx(servo_id, ADDR_TEMPERATURE)
        if res != 0 or err != 0:
            raise RuntimeError(f"Problem reading temperature for servo {servo_id}")
        return temp


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
