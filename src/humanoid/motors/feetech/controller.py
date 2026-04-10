import math

from vassar_feetech_servo_sdk import ServoController

from humanoid.logger import get_logger
from humanoid.motors.base import MotorController

logger = get_logger(__name__)

POS_MIN = 0
POS_MAX = 4095
POS_MID = (POS_MAX + POS_MIN) / 2
MAX_ACCELERATION = 254
ADDR_TEMPERATURE = 63


class FeetechMotorController(ServoController, MotorController):
    def __init__(
        self,
        servo_ids: list[int],
        max_acceleration: int = MAX_ACCELERATION,
        inverted_servo_ids: list[int] | None = None,
    ):
        self.max_acceleration = max_acceleration
        self.inverted_servo_ids = set(inverted_servo_ids or [])
        ServoController.__init__(self, servo_ids=servo_ids)
        MotorController.__init__(self, servo_ids=servo_ids)

    def angle_to_position(self, angle: float, servo_id: int) -> int:
        angle = max(-math.pi, min(math.pi, angle))
        # Invert angle if this servo is marked as inverted
        if servo_id in self.inverted_servo_ids:
            angle = -angle
        position = POS_MID + (angle / math.pi) * (POS_MAX - POS_MID)
        return int(position)

    def position_to_angle(self, position: int, servo_id: int) -> float:
        angle = ((position - POS_MID) / (POS_MAX - POS_MID)) * math.pi
        # Invert angle if this servo is marked as inverted
        if servo_id in self.inverted_servo_ids:
            angle = -angle
        return angle

    def write_position(self, positions: dict[int, float], **kwargs):  # ty:ignore[invalid-method-override]
        raw_positions = {
            servo_id: self.angle_to_position(angle, servo_id)
            for servo_id, angle in positions.items()
        }
        super().write_position(raw_positions, acceleration=self.max_acceleration, **kwargs)

    def read_position(self, servo_id: int) -> float | None:  # ty:ignore[invalid-method-override]
        raw_position = super().read_position(servo_id)
        if raw_position is None:
            return None
        return self.position_to_angle(raw_position, servo_id)

    def read_all_positions(self) -> dict[int, float]:  # ty:ignore[invalid-method-override]
        raw_positions = super().read_all_positions()
        return {
            servo_id: self.position_to_angle(pos, servo_id)
            for servo_id, pos in raw_positions.items()
        }

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
        assert self.packet_handler, "Unable to read temperature"
        temp, res, err = self.packet_handler.read1ByteTxRx(servo_id, ADDR_TEMPERATURE)
        if res != 0 or err != 0:
            raise RuntimeError(f"Problem reading temperature for servo {servo_id}")
        return temp

    def read_all_temperatures(self) -> dict[int, float]:
        return {servo_id: self.read_temperature(servo_id) for servo_id in self.servo_ids}
