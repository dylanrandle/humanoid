"""Low-level interface implemented by actuator controller drivers."""

from abc import ABC, abstractmethod


class ActuatorDriver(ABC):
    """Driver for a homogeneous set of actuators on one controller."""

    def __init__(self, actuator_ids: list[int]):
        self.actuator_ids = actuator_ids

    @abstractmethod
    def connect(self) -> None:
        """Establish the controller connection."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the controller connection."""

    @abstractmethod
    def write_position(self, positions: dict[int, float]) -> None:
        """Write target positions in radians, keyed by actuator ID."""

    @abstractmethod
    def read_position(self, actuator_id: int) -> float | None:
        """Read one actuator position in radians."""

    @abstractmethod
    def read_all_positions(self) -> dict[int, float]:
        """Read positions in radians, keyed by actuator ID."""

    @abstractmethod
    def ping(self, actuator_id: int) -> bool:
        """Return whether an actuator responds on this controller."""

    @abstractmethod
    def read_temperature(self, actuator_id: int) -> int:
        """Read one actuator temperature."""

    @abstractmethod
    def read_all_temperatures(self) -> dict[int, float]:
        """Read temperatures, keyed by actuator ID."""

    @abstractmethod
    def write_velocity(self, velocities: dict[int, float]) -> None:
        """Write target velocities in rad/s, keyed by actuator ID."""

    @abstractmethod
    def read_velocity(self, actuator_id: int) -> float | None:
        """Read one actuator velocity in rad/s."""

    @abstractmethod
    def read_all_velocities(self) -> dict[int, float]:
        """Read velocities in rad/s, keyed by actuator ID."""
