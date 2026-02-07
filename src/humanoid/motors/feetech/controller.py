import math
import time

import numpy as np
from alive_progress import alive_it
from scipy.interpolate import CubicSpline
from vassar_feetech_servo_sdk import ServoController

from humanoid.logging import get_logger
from humanoid.loop import loop_at_rate

logger = get_logger(__name__)

ID_MIN = 1
ID_MAX = 253

POS_MIN = 0
POS_MAX = 4095

ADDR_P_GAIN = 21
ADDR_D_GAIN = 22
ADDR_I_GAIN = 23
ADDR_LOCK = 48


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

        # Read current position to calculate phase offset
        current_pos = self.read_position(servo_id)

        # Calculate phase offset so oscillation starts from current position
        # current_pos = midpoint * sin(phase_offset) + midpoint
        # Solving for phase_offset:
        normalized_pos = (current_pos - midpoint) / midpoint
        # Clamp to [-1, 1] to handle positions outside normal range
        normalized_pos = max(-1.0, min(1.0, normalized_pos))
        phase_offset = math.asin(normalized_pos)

        start_time = time.time()

        def work():
            elapsed = time.time() - start_time
            target = midpoint * math.sin(elapsed * angular_freq + phase_offset) + midpoint
            cmd = {servo_id: int(target)}

            logger.debug(f"Sending command: {cmd=}")
            result = self.write_position(
                cmd,
            )
            logger.debug(f"Result: {result=}")

        try:
            loop_at_rate(work, update_frequency_hz)
        except KeyboardInterrupt:
            logger.info("Shutting down")

    def home(
        self,
        servo_id: int,
        speed: float = 2000.0,
        update_frequency_hz: float = 100,
        tolerance: float = 5.0,
    ):
        midpoint = (POS_MAX - POS_MIN) / 2

        # Read current position
        current_pos = self.read_position(servo_id)

        # Calculate total distance to travel
        total_distance = abs(midpoint - current_pos)

        # If already at home, return immediately
        if total_distance <= tolerance:
            logger.info(f"Servo {servo_id} already at home position (within {tolerance} units)")
            return True

        # Calculate trajectory duration based on distance and speed
        # Use trapezoidal velocity profile assumption: average speed is ~2/3 of max speed
        duration = total_distance / (speed * 0.67)

        logger.info(
            f"Homing servo {servo_id} from position {current_pos} to {midpoint} "
            f"(distance: {total_distance:.1f}, duration: {duration:.2f}s)"
        )

        # Create cubic spline trajectory with zero velocity at endpoints
        # Time points: start, end
        # Position points: current_pos, midpoint
        # Boundary conditions: zero velocity at both ends (bc_type='clamped')
        time_points = np.array([0.0, duration])
        position_points = np.array([current_pos, midpoint])

        # Create cubic spline with zero velocity boundary conditions
        trajectory = CubicSpline(
            time_points,
            position_points,
            bc_type=((1, 0.0), (1, 0.0)),  # Zero first derivative (velocity) at both ends
        )

        start_time = time.time()

        def work():
            elapsed = time.time() - start_time

            # Get position from spline trajectory
            if elapsed >= duration:
                target = midpoint
                progress = 1.0
            else:
                target = float(trajectory(elapsed))
                progress = elapsed / duration

            cmd = {servo_id: int(target)}

            logger.debug(f"Homing progress: {progress:.2%}, target: {int(target)}")
            result = self.write_position(cmd)
            logger.debug(f"Result: {result=}")

            # Stop when we've reached the target
            if progress >= 1.0:
                logger.info(f"Homing complete for servo {servo_id}")
                return False  # Signal to stop the loop

            return True  # Continue the loop

        try:
            loop_at_rate(work, update_frequency_hz)
            return True
        except KeyboardInterrupt:
            logger.info("Homing interrupted")
            return False


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
