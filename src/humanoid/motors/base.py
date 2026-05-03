from abc import ABC, abstractmethod


class MotorController(ABC):
    """Base class for all motor controllers."""

    def __init__(self, servo_ids: list[int]):
        self.servo_ids = servo_ids

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the motor controller."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the motor controller."""

    @abstractmethod
    def write_position(self, positions: dict[int, float]) -> None:
        """Write target positions (in radians) to servos.

        Args:
            positions: Dictionary mapping servo_id to target angle in radians
        """

    @abstractmethod
    def read_position(self, servo_id: int) -> float | None:
        """Read current position (in radians) from a single servo.

        Args:
            servo_id: ID of the servo to read from

        Returns:
            Current angle in radians, or None if read fails
        """

    @abstractmethod
    def read_all_positions(self) -> dict[int, float]:
        """Read current positions (in radians) from all servos.

        Returns:
            Dictionary mapping servo_id to current angle in radians
        """

    @abstractmethod
    def ping(self, servo_id: int) -> bool:
        """Check if a servo is responding.

        Args:
            servo_id: ID of the servo to ping

        Returns:
            True if servo responds, False otherwise
        """

    @abstractmethod
    def read_temperature(self, servo_id: int) -> int:
        """Read temperature from a single servo.

        Args:
            servo_id: ID of the servo to read from

        Returns:
            Temperature value
        """

    @abstractmethod
    def read_all_temperatures(self) -> dict[int, float]:
        """Read temperatures from all servos.

        Returns:
            Dictionary mapping servo_id to temperature
        """

    @abstractmethod
    def write_velocity(self, velocities: dict[int, float]) -> None:
        """Write target velocities (in rad/s) to servos.

        Args:
            velocities: Dictionary mapping servo_id to target velocity in rad/s
        """

    @abstractmethod
    def read_velocity(self, servo_id: int) -> float | None:
        """Read current velocity (in rad/s) from a single servo.

        Args:
            servo_id: ID of the servo to read from

        Returns:
            Current velocity in rad/s, or None if read fails
        """

    @abstractmethod
    def read_all_velocities(self) -> dict[int, float]:
        """Read current velocities (in rad/s) from all servos.

        Returns:
            Dictionary mapping servo_id to current velocity in rad/s
        """
