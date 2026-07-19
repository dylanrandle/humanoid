import pytest

from humanoid.hardware.actuators.config import ActuatorControlMode
from humanoid.hardware.actuators.driver import ActuatorDriver
from humanoid.hardware.actuators.feetech.config import FeetechActuatorConfig
from humanoid.hardware.actuators.system import CompositeActuatorSystem

EXPECTED_POSITION = 0.5
EXPECTED_VELOCITY = 0.25
EXPECTED_TEMPERATURE = 31.0
EXPECTED_STOP_FAILURES = 2


class StubActuatorDriver(ActuatorDriver):
    def __init__(
        self,
        actuator_ids: list[int],
        *,
        fail_connect: bool = False,
        fail_disconnect: bool = False,
        fail_velocity_ids: set[int] | None = None,
    ):
        super().__init__(actuator_ids)
        self.position_writes: list[dict[int, float]] = []
        self.velocity_writes: list[dict[int, float]] = []
        self.positions = dict.fromkeys(actuator_ids, 0.0)
        self.velocities = dict.fromkeys(actuator_ids, 0.0)
        self.temperatures = dict.fromkeys(actuator_ids, 25.0)
        self.connected = False
        self.fail_connect = fail_connect
        self.fail_disconnect = fail_disconnect
        self.fail_velocity_ids = fail_velocity_ids or set()
        self.disconnect_calls = 0

    def connect(self) -> None:
        if self.fail_connect:
            raise ConnectionError("controller unavailable")
        self.connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        if self.fail_disconnect:
            raise RuntimeError("controller cleanup failed")
        self.connected = False

    def write_position(self, positions: dict[int, float]) -> None:
        self.position_writes.append(positions)

    def read_position(self, actuator_id: int) -> float | None:
        return self.positions.get(actuator_id)

    def read_all_positions(self) -> dict[int, float]:
        return self.positions

    def ping(self, actuator_id: int) -> bool:
        return actuator_id in self.actuator_ids

    def read_temperature(self, actuator_id: int) -> int:
        return int(self.temperatures[actuator_id])

    def read_all_temperatures(self) -> dict[int, float]:
        return self.temperatures

    def write_velocity(self, velocities: dict[int, float]) -> None:
        self.velocity_writes.append(velocities)
        if self.fail_velocity_ids.intersection(velocities):
            raise RuntimeError("velocity write failed")

    def read_velocity(self, actuator_id: int) -> float | None:
        return self.velocities.get(actuator_id)

    def read_all_velocities(self) -> dict[int, float]:
        return self.velocities


def _actuator(
    controller: str,
) -> FeetechActuatorConfig:
    return FeetechActuatorConfig(
        actuator_id=1,
        controller=controller,
    )


def test_routes_duplicate_ids_by_controller():
    left = StubActuatorDriver([1])
    right = StubActuatorDriver([1])
    system = CompositeActuatorSystem(
        {
            "left_joint": _actuator("left"),
            "right_joint": _actuator("right"),
        },
        {
            "left_joint": ActuatorControlMode.POSITION,
            "right_joint": ActuatorControlMode.POSITION,
        },
        {"left": left, "right": right},
    )

    system.write_positions({"left_joint": 0.1, "right_joint": 0.2})

    assert left.position_writes == [{1: 0.1}]
    assert right.position_writes == [{1: 0.2}]


def test_translates_controller_feedback_to_joint_names():
    left = StubActuatorDriver([1])
    left.positions[1] = EXPECTED_POSITION
    left.velocities[1] = EXPECTED_VELOCITY
    left.temperatures[1] = EXPECTED_TEMPERATURE
    system = CompositeActuatorSystem(
        {"joint": _actuator("left")},
        {"joint": ActuatorControlMode.POSITION},
        {"left": left},
    )

    state = system.read_states()["joint"]

    assert state.position == EXPECTED_POSITION
    assert state.velocity == EXPECTED_VELOCITY
    assert state.temperature == EXPECTED_TEMPERATURE


def test_rejects_commands_for_the_wrong_control_mode():
    driver = StubActuatorDriver([1])
    system = CompositeActuatorSystem(
        {"joint": _actuator("main")},
        {"joint": ActuatorControlMode.VELOCITY},
        {"main": driver},
    )

    with pytest.raises(ValueError, match="uses velocity control"):
        system.write_positions({"joint": 1.0})


def test_stop_sends_zero_to_every_velocity_controlled_joint():
    driver = StubActuatorDriver([1, 2])
    system = CompositeActuatorSystem(
        {
            "position_joint": FeetechActuatorConfig(actuator_id=1, controller="main"),
            "velocity_joint": FeetechActuatorConfig(actuator_id=2, controller="main"),
        },
        {
            "position_joint": ActuatorControlMode.POSITION,
            "velocity_joint": ActuatorControlMode.VELOCITY,
        },
        {"main": driver},
    )

    system.stop()

    assert driver.velocity_writes == [{2: 0.0}]


def test_connect_failure_rolls_back_previously_connected_drivers():
    first = StubActuatorDriver([1])
    second = StubActuatorDriver([2], fail_connect=True)
    system = CompositeActuatorSystem(
        {
            "first_joint": FeetechActuatorConfig(actuator_id=1, controller="first"),
            "second_joint": FeetechActuatorConfig(actuator_id=2, controller="second"),
        },
        {
            "first_joint": ActuatorControlMode.POSITION,
            "second_joint": ActuatorControlMode.POSITION,
        },
        {"first": first, "second": second},
    )

    with pytest.raises(ConnectionError, match="controller unavailable"):
        system.connect()

    assert first.connected is False
    assert second.connected is False


def test_stop_attempts_every_actuator_and_aggregates_failures():
    first = StubActuatorDriver([1, 2], fail_velocity_ids={1})
    second = StubActuatorDriver([3], fail_velocity_ids={3})
    system = CompositeActuatorSystem(
        {
            "first_joint": FeetechActuatorConfig(actuator_id=1, controller="first"),
            "second_joint": FeetechActuatorConfig(actuator_id=2, controller="first"),
            "third_joint": FeetechActuatorConfig(actuator_id=3, controller="second"),
        },
        dict.fromkeys(
            ("first_joint", "second_joint", "third_joint"),
            ActuatorControlMode.VELOCITY,
        ),
        {"first": first, "second": second},
    )

    with pytest.raises(ExceptionGroup, match="Failed to stop all") as exc_info:
        system.stop()

    assert first.velocity_writes == [{1: 0.0}, {2: 0.0}]
    assert second.velocity_writes == [{3: 0.0}]
    assert len(exc_info.value.exceptions) == EXPECTED_STOP_FAILURES


def test_connect_rollback_continues_and_preserves_original_failure():
    first = StubActuatorDriver([1])
    second = StubActuatorDriver([2], fail_disconnect=True)
    third = StubActuatorDriver([3], fail_connect=True)
    system = CompositeActuatorSystem(
        {
            "first_joint": FeetechActuatorConfig(actuator_id=1, controller="first"),
            "second_joint": FeetechActuatorConfig(actuator_id=2, controller="second"),
            "third_joint": FeetechActuatorConfig(actuator_id=3, controller="third"),
        },
        dict.fromkeys(
            ("first_joint", "second_joint", "third_joint"),
            ActuatorControlMode.POSITION,
        ),
        {"first": first, "second": second, "third": third},
    )

    with pytest.raises(ConnectionError, match="controller unavailable") as exc_info:
        system.connect()

    assert first.disconnect_calls == 1
    assert first.connected is False
    assert second.disconnect_calls == 1
    assert isinstance(exc_info.value.__cause__, ExceptionGroup)
