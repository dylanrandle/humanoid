"""Joint-keyed actuator system used by the robot driver."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from humanoid.hardware.actuators.config import ActuatorConfig, ActuatorControlMode
from humanoid.hardware.actuators.driver import ActuatorDriver


@dataclass(frozen=True)
class ActuatorState:
    """Latest feedback for one logical robot actuator."""

    position: float | None = None
    velocity: float | None = None
    temperature: float | None = None


class ActuatorSystem(ABC):
    """Application-facing actuator interface keyed by URDF joint name."""

    @abstractmethod
    def connect(self) -> None:
        """Connect every controller used by the system."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect every controller used by the system."""

    @abstractmethod
    def write_positions(self, positions: dict[str, float]) -> None:
        """Write joint positions in radians."""

    @abstractmethod
    def write_velocities(self, velocities: dict[str, float]) -> None:
        """Write joint velocities in rad/s."""

    @abstractmethod
    def read_states(self) -> dict[str, ActuatorState]:
        """Read all available actuator feedback."""

    @abstractmethod
    def stop(self) -> None:
        """Stop every velocity-controlled actuator while position actuators hold."""


class CompositeActuatorSystem(ActuatorSystem):
    """Routes logical joint commands across one or more controller drivers."""

    def __init__(
        self,
        actuator_configs: dict[str, ActuatorConfig],
        control_modes: dict[str, ActuatorControlMode],
        drivers: dict[str, ActuatorDriver],
    ):
        self.actuator_configs = actuator_configs
        self.control_modes = control_modes
        self.drivers = drivers
        self._joint_by_address: dict[tuple[str, int], str] = {}
        for joint_name, actuator in actuator_configs.items():
            if actuator.controller not in drivers:
                raise ValueError(
                    f"Actuator {joint_name!r} has no driver for controller {actuator.controller!r}."
                )
            self._joint_by_address[(actuator.controller, actuator.actuator_id)] = joint_name

    def connect(self) -> None:
        connected: list[ActuatorDriver] = []
        try:
            for driver in self.drivers.values():
                driver.connect()
                connected.append(driver)
        except Exception as connection_error:
            rollback_errors: list[Exception] = []
            for driver in reversed(connected):
                try:
                    driver.disconnect()
                except Exception as rollback_error:
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                raise connection_error from ExceptionGroup(
                    "Actuator connection rollback failed",
                    rollback_errors,
                )
            raise

    def disconnect(self) -> None:
        first_error: Exception | None = None
        for driver in reversed(tuple(self.drivers.values())):
            try:
                driver.disconnect()
            except Exception as exc:  # pragma: no cover - defensive hardware cleanup
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def write_positions(self, positions: dict[str, float]) -> None:
        grouped = self._group_commands(positions, ActuatorControlMode.POSITION)
        for controller, commands in grouped.items():
            self.drivers[controller].write_position(commands)

    def write_velocities(self, velocities: dict[str, float]) -> None:
        grouped = self._group_commands(velocities, ActuatorControlMode.VELOCITY)
        for controller, commands in grouped.items():
            self.drivers[controller].write_velocity(commands)

    def read_states(self) -> dict[str, ActuatorState]:
        states: dict[str, ActuatorState] = {}
        for controller, driver in self.drivers.items():
            positions = driver.read_all_positions()
            velocities = driver.read_all_velocities()
            temperatures = driver.read_all_temperatures()
            actuator_ids = positions.keys() | velocities.keys() | temperatures.keys()
            for actuator_id in actuator_ids:
                joint_name = self._joint_by_address.get((controller, actuator_id))
                if joint_name is None:
                    continue
                states[joint_name] = ActuatorState(
                    position=positions.get(actuator_id),
                    velocity=velocities.get(actuator_id),
                    temperature=temperatures.get(actuator_id),
                )
        return states

    def stop(self) -> None:
        errors: list[Exception] = []
        for joint_name, mode in self.control_modes.items():
            if mode is not ActuatorControlMode.VELOCITY:
                continue
            actuator = self.actuator_configs[joint_name]
            try:
                self.drivers[actuator.controller].write_velocity({actuator.actuator_id: 0.0})
            except Exception as error:
                error.add_note(f"Failed to stop actuator joint {joint_name!r}.")
                errors.append(error)
        if errors:
            raise ExceptionGroup("Failed to stop all velocity actuators", errors)

    def _group_commands(
        self,
        commands: dict[str, float],
        expected_mode: ActuatorControlMode,
    ) -> dict[str, dict[int, float]]:
        grouped: dict[str, dict[int, float]] = {}
        for joint_name, value in commands.items():
            try:
                actuator = self.actuator_configs[joint_name]
            except KeyError as exc:
                raise ValueError(f"Unknown actuator joint {joint_name!r}.") from exc
            control_mode = self.control_modes[joint_name]
            if control_mode is not expected_mode:
                raise ValueError(
                    f"Actuator {joint_name!r} uses {control_mode.value} control, "
                    f"not {expected_mode.value} control."
                )
            grouped.setdefault(actuator.controller, {})[actuator.actuator_id] = value
        return grouped
