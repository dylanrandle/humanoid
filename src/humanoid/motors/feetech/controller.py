import math

from vassar_feetech_servo_sdk import ServoController

from humanoid.logger import get_logger

logger = get_logger(__name__)

POS_MIN = 0
POS_MAX = 4095
POS_MID = (POS_MAX + POS_MIN) / 2

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

    def read_all_temperatures(self) -> dict[int, float]:
        return {servo_id: self.read_temperature(servo_id) for servo_id in self.servo_ids}
