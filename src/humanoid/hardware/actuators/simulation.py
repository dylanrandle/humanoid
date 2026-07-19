"""Joint-keyed actuator simulation."""

import time

from humanoid.hardware.actuators.config import ActuatorControlMode
from humanoid.hardware.actuators.system import ActuatorState, ActuatorSystem
from humanoid.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TEMPERATURE = 25.0
TEMPERATURE_VARIANCE = 5.0


class SimulatedActuatorSystem(ActuatorSystem):
    """In-process actuator state model used by simulation and replay."""

    def __init__(
        self,
        control_modes: dict[str, ActuatorControlMode],
        initial_positions: dict[str, float],
    ):
        if control_modes.keys() != initial_positions.keys():
            raise ValueError("Every simulated actuator requires an initial position.")
        self.control_modes = control_modes
        self._positions = dict(initial_positions)
        self._velocities = dict.fromkeys(control_modes, 0.0)
        self._temperatures = dict.fromkeys(control_modes, DEFAULT_TEMPERATURE)
        self._connected = False
        self._start_time = time.perf_counter()
        self._last_integration_time = self._start_time

    def connect(self) -> None:
        logger.info("Connecting simulated actuator system")
        self._connected = True

    def disconnect(self) -> None:
        logger.info("Disconnecting simulated actuator system")
        self._connected = False

    def write_positions(self, positions: dict[str, float]) -> None:
        self._require_connected()
        self._validate_commands(positions, ActuatorControlMode.POSITION)
        self._positions.update(positions)

    def write_velocities(self, velocities: dict[str, float]) -> None:
        self._require_connected()
        self._integrate_positions()
        self._validate_commands(velocities, ActuatorControlMode.VELOCITY)
        self._velocities.update(velocities)

    def read_states(self) -> dict[str, ActuatorState]:
        self._require_connected()
        self._integrate_positions()
        elapsed = time.perf_counter() - self._start_time
        variation = (elapsed % 10) / 10 * TEMPERATURE_VARIANCE
        return {
            joint_name: ActuatorState(
                position=self._positions[joint_name],
                velocity=self._velocities[joint_name],
                temperature=self._temperatures[joint_name] + variation,
            )
            for joint_name in self.control_modes
        }

    def stop(self) -> None:
        self.write_velocities(
            {
                joint_name: 0.0
                for joint_name, mode in self.control_modes.items()
                if mode is ActuatorControlMode.VELOCITY
            }
        )

    def _integrate_positions(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_integration_time
        self._last_integration_time = now
        for joint_name, velocity in self._velocities.items():
            self._positions[joint_name] += velocity * dt

    def _validate_commands(
        self,
        commands: dict[str, float],
        expected_mode: ActuatorControlMode,
    ) -> None:
        for joint_name in commands:
            try:
                control_mode = self.control_modes[joint_name]
            except KeyError as exc:
                raise ValueError(f"Unknown actuator joint {joint_name!r}.") from exc
            if control_mode is not expected_mode:
                raise ValueError(
                    f"Actuator {joint_name!r} uses {control_mode.value} control, "
                    f"not {expected_mode.value} control."
                )

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Actuator system is not connected.")
