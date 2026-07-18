import time

from humanoid.config import ROBOT_CONFIG
from humanoid.logger import get_logger
from humanoid.motors.base import MotorController
from humanoid.robots.base import Robot
from humanoid.types.robot import RobotConfig

logger = get_logger(__name__)

DEFAULT_TEMPERATURE = 25.0
TEMPERATURE_VARIANCE = 5.0


class SimulatedMotorController(MotorController):
    """Simulated motor controller for testing without hardware."""

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
    ):
        """Initialize simulated motor controller.

        Args:
            robot_config: Robot definition used to select simulated servos and
                initialize them at the configured home position.
        """
        super().__init__(robot_config.servo_ids)

        # Set initial positions
        initial_positions = {}
        home_pos = robot_config.home_position
        joint_idx_to_servo_id = robot_config.joint_idx_to_servo_id

        robot = Robot(robot_config)

        for joint_idx, servo_id in joint_idx_to_servo_id.items():
            initial_positions[servo_id] = float(
                home_pos[robot.joint_idx_to_position_idx(joint_idx)]
            )

        # Initialize positions with provided or default values
        self._positions: dict[int, float] = {}
        self._target_positions: dict[int, float] = {}
        for servo_id in robot_config.servo_ids:
            pos = initial_positions[servo_id]
            self._positions[servo_id] = pos
            self._target_positions[servo_id] = pos

        self._velocities: dict[int, float] = dict.fromkeys(robot_config.servo_ids, 0.0)

        self._temperatures: dict[int, float] = dict.fromkeys(
            robot_config.servo_ids, DEFAULT_TEMPERATURE
        )
        self._connected = False
        self._start_time = time.perf_counter()
        self._last_integration_time = self._start_time

    def _integrate_positions(self) -> None:
        """Advance positions by integrating each servo's velocity over elapsed time."""
        now = time.perf_counter()
        dt = now - self._last_integration_time
        self._last_integration_time = now
        for servo_id, velocity in self._velocities.items():
            self._positions[servo_id] += velocity * dt

    def connect(self) -> None:
        """Establish connection to the simulated motor controller."""
        logger.info("Connecting to simulated motor controller...")
        self._connected = True
        logger.info(f"Connected to {len(self.servo_ids)} simulated servos: {self.servo_ids}")

    def disconnect(self) -> None:
        """Close connection to the simulated motor controller."""
        logger.info("Disconnecting from simulated motor controller...")
        self._connected = False
        logger.info("Disconnected")

    def write_position(self, positions: dict[int, float]) -> None:
        """Write target positions (in radians) to simulated servos.

        Args:
            positions: Dictionary mapping servo_id to target angle in radians
        """
        if not self._connected:
            logger.warning("Cannot write positions: controller not connected")
            return

        for servo_id, angle in positions.items():
            if servo_id in self._target_positions:
                self._target_positions[servo_id] = angle
                # In simulation, instantly move to target position
                # In a more realistic simulation, you could add velocity limits
                self._positions[servo_id] = angle
                logger.debug(f"Servo {servo_id} moved to {angle:.3f} rad")
            else:
                logger.warning(f"Unknown servo ID: {servo_id}")

    def read_position(self, servo_id: int) -> float | None:
        """Read current position (in radians) from a simulated servo.

        Args:
            servo_id: ID of the servo to read from

        Returns:
            Current angle in radians, or None if read fails
        """
        if not self._connected:
            logger.warning("Cannot read position: controller not connected")
            return None

        if servo_id not in self._positions:
            logger.error(f"Unknown servo ID: {servo_id}")
            return None

        return self._positions[servo_id]

    def read_all_positions(self) -> dict[int, float]:
        """Read current positions (in radians) from all simulated servos.

        Returns:
            Dictionary mapping servo_id to current angle in radians
        """
        if not self._connected:
            logger.warning("Cannot read positions: controller not connected")
            return {}

        return self._positions.copy()

    def ping(self, servo_id: int) -> bool:
        """Check if a simulated servo is responding.

        Args:
            servo_id: ID of the servo to ping

        Returns:
            True if servo exists, False otherwise
        """
        if servo_id not in self.servo_ids:
            logger.error(f"No motor found at ID {servo_id} (simulated)")
            return False

        pos = self.read_position(servo_id)
        if pos is None:
            return False

        logger.info(f"Motor found at ID {servo_id} (angle: {pos:.3f} rad) [SIMULATED]")
        return True

    def read_temperature(self, servo_id: int) -> int:
        """Read temperature from a simulated servo.

        Args:
            servo_id: ID of the servo to read from

        Returns:
            Simulated temperature value
        """
        if not self._connected:
            raise RuntimeError("Cannot read temperature: controller not connected")

        if servo_id not in self._temperatures:
            raise RuntimeError(f"Unknown servo ID: {servo_id}")

        # Add some time-based variation to make it more realistic
        elapsed = time.perf_counter() - self._start_time
        variation = (elapsed % 10) / 10 * TEMPERATURE_VARIANCE
        return int(self._temperatures[servo_id] + variation)

    def read_all_temperatures(self) -> dict[int, float]:
        """Read temperatures from all simulated servos.

        Returns:
            Dictionary mapping servo_id to simulated temperature
        """
        if not self._connected:
            logger.warning("Cannot read temperatures: controller not connected")
            return {}

        return {servo_id: self.read_temperature(servo_id) for servo_id in self.servo_ids}

    def write_velocity(self, velocities: dict[int, float]) -> None:
        """Write target velocities (in rad/s) to simulated servos.

        Args:
            velocities: Dictionary mapping servo_id to target velocity in rad/s
        """
        if not self._connected:
            logger.warning("Cannot write velocities: controller not connected")
            return

        # Integrate at the previous velocity before applying the new one,
        # so position reflects motion under the command that was just superseded.
        self._integrate_positions()
        for servo_id, velocity in velocities.items():
            if servo_id in self._velocities:
                self._velocities[servo_id] = velocity
                logger.debug(f"Servo {servo_id} velocity set to {velocity:.3f} rad/s")
            else:
                logger.warning(f"Unknown servo ID: {servo_id}")

    def read_velocity(self, servo_id: int) -> float | None:
        """Read current velocity (in rad/s) from a simulated servo.

        Args:
            servo_id: ID of the servo to read from

        Returns:
            Current velocity in rad/s, or None if read fails
        """
        if not self._connected:
            logger.warning("Cannot read velocity: controller not connected")
            return None

        if servo_id not in self._velocities:
            logger.error(f"Unknown servo ID: {servo_id}")
            return None

        return self._velocities[servo_id]

    def read_all_velocities(self) -> dict[int, float]:
        """Read current velocities (in rad/s) from all simulated servos.

        Returns:
            Dictionary mapping servo_id to current velocity in rad/s
        """
        if not self._connected:
            logger.warning("Cannot read velocities: controller not connected")
            return {}

        return self._velocities.copy()
