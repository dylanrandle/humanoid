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
ADDR_GOAL_SPEED = 46
ADDR_PRESENT_SPEED = 58
# 0.732 RPM per raw unit, BIT15 encodes direction
SPEED_UNIT_RAD_S = 0.732 * 2 * math.pi / 60


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
        acceleration = kwargs.pop("acceleration", self.max_acceleration)
        super().write_position(raw_positions, acceleration=acceleration, **kwargs)

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

    def write_velocity(self, velocities: dict[int, float]):
        # Requires servo to be in velocity mode (mode 1); call set_operating_mode(id, 1) first.
        assert self.packet_handler, "Not connected"
        for servo_id, velocity in velocities.items():
            v = -velocity if servo_id in self.inverted_servo_ids else velocity
            magnitude = int(min(abs(v) / SPEED_UNIT_RAD_S, 32767))
            raw = (magnitude | 0x8000) if v < 0 else magnitude
            self.packet_handler.write2ByteTxRx(servo_id, ADDR_GOAL_SPEED, raw)

    def read_velocity(self, servo_id: int) -> float | None:
        assert self.packet_handler, "Not connected"
        raw, res, err = self.packet_handler.ReadSpeed(servo_id)
        if res != 0 or err != 0:
            return None
        direction = -1 if (raw & 0x8000) else 1
        velocity = direction * (raw & 0x7FFF) * SPEED_UNIT_RAD_S
        if servo_id in self.inverted_servo_ids:
            velocity = -velocity
        return velocity

    def read_all_velocities(self) -> dict[int, float]:
        return {
            servo_id: v
            for servo_id in self.servo_ids
            if (v := self.read_velocity(servo_id)) is not None
        }
