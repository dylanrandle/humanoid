from unittest.mock import Mock, patch

import numpy as np
import pytest

from humanoid.config.robot.elrobot_mobile import ELROBOT_MOBILE_CONFIG
from humanoid.hardware.actuators.system import ActuatorState, ActuatorSystem
from humanoid.nodes.robot.driver import RobotDriverNode
from humanoid.state_estimation.root.base import (
    RootState,
    RootStateEstimator,
)
from humanoid.state_estimation.root.wheel_dead_reckoning import (
    WheelDeadReckoningRootStateEstimator,
)
from humanoid.types.actuator import (
    ActuatorControlMode,
)
from humanoid.types.homing import HomingPreset
from humanoid.types.robot import (
    RobotConfig,
    RobotJointCommand,
    RobotName,
    RobotToolConfig,
)


class StubActuatorSystem(ActuatorSystem):
    def __init__(self):
        self.position_writes: list[dict[str, float]] = []
        self.velocity_writes: list[dict[str, float]] = []
        self.states: dict[str, ActuatorState] = {}
        self.connected = False
        self.stop_calls = 0

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def write_positions(self, positions: dict[str, float]) -> None:
        self.position_writes.append(positions)

    def write_velocities(self, velocities: dict[str, float]) -> None:
        self.velocity_writes.append(velocities)

    def read_states(self) -> dict[str, ActuatorState]:
        return self.states

    def stop(self) -> None:
        self.stop_calls += 1


class StubRootStateEstimator(RootStateEstimator):
    def __init__(self, state: RootState):
        self.state = state
        self.updates: list[tuple[np.ndarray, np.ndarray]] = []

    def update(self, q: np.ndarray, v: np.ndarray) -> RootState:
        self.updates.append((q.copy(), v.copy()))
        return self.state


def _robot_config(modes: list[ActuatorControlMode]) -> RobotConfig:
    return RobotConfig(
        name=RobotName.PANDA,
        tool=RobotToolConfig(frame="tool"),
        homing_presets={
            HomingPreset.HOME: np.zeros(len(modes)),
            HomingPreset.REST: np.ones(len(modes)),
        },
        actuator_control_modes={f"joint_{index}": mode for index, mode in enumerate(modes)},
        hardware=None,
    )


def _make_driver(
    robot_config: RobotConfig,
    velocity_limit: np.ndarray | None = None,
    *,
    clock=None,
    command_timeout_seconds: float = 0.25,
) -> RobotDriverNode:
    """Build a RobotDriverNode with mocked middleware, model, and actuators."""
    joint_names = list(robot_config.actuator_control_modes)
    number_of_joints = len(joint_names)
    if velocity_limit is None:
        velocity_limit = np.full(number_of_joints, 5.0)

    actuator_system = StubActuatorSystem()
    with (
        patch("humanoid.nodes.robot.driver.Subscriber"),
        patch("humanoid.nodes.robot.driver.Publisher"),
        patch("humanoid.nodes.robot.driver.Robot") as mock_robot_cls,
    ):
        mock_robot = Mock()
        mock_robot.config = robot_config
        mock_robot.actuator_joint_names = joint_names
        mock_robot.model.lowerPositionLimit = np.full(number_of_joints, -3.0)
        mock_robot.model.upperPositionLimit = np.full(number_of_joints, 3.0)
        mock_robot.model.velocityLimit = velocity_limit
        mock_robot.joint_name_to_idx.side_effect = joint_names.index
        mock_robot.joint_idx_to_position_idx.side_effect = lambda index: index
        mock_robot.joint_idx_to_velocity_idx.side_effect = lambda index: index
        mock_robot.joint_position_from_q.side_effect = lambda q, index: float(q[index])
        mock_robot.get_root_q_slice.return_value = None
        mock_robot_cls.return_value = mock_robot

        kwargs = {}
        if clock is not None:
            kwargs["clock"] = clock
        return RobotDriverNode(
            robot_config=robot_config,
            actuator_system=actuator_system,
            command_timeout_seconds=command_timeout_seconds,
            **kwargs,
        )


@pytest.fixture
def robot_driver():
    driver = _make_driver(_robot_config([ActuatorControlMode.POSITION] * 7))
    driver.joint_lower_limits = np.array(
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    )
    driver.joint_upper_limits = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    return driver


def _written_positions(driver: RobotDriverNode) -> dict[str, float]:
    return _actuator_system(driver).position_writes[-1]


def _actuator_system(driver: RobotDriverNode) -> StubActuatorSystem:
    assert isinstance(driver.actuator_system, StubActuatorSystem)
    return driver.actuator_system


def test_position_clipping_within_limits(robot_driver):
    positions = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.0, 0.0])
    robot_driver.subscriber.receive = Mock(
        return_value=RobotJointCommand(timestamp=0.0, joint_positions=positions)
    )

    robot_driver.receive()

    written = _written_positions(robot_driver)
    for index, position in enumerate(positions):
        assert written[f"joint_{index}"] == pytest.approx(position)


def test_position_clipping_above_upper_limits(robot_driver):
    positions = np.array([3.0, 2.0, 3.0, 0.0, 3.0, 4.0, 3.0])
    robot_driver.subscriber.receive = Mock(
        return_value=RobotJointCommand(timestamp=0.0, joint_positions=positions)
    )

    robot_driver.receive()

    written = _written_positions(robot_driver)
    for index in range(len(positions)):
        assert written[f"joint_{index}"] <= robot_driver.joint_upper_limits[index]


def test_position_clipping_below_lower_limits(robot_driver):
    positions = np.array([-3.0, -2.0, -3.0, -4.0, -3.0, -1.0, -3.0])
    robot_driver.subscriber.receive = Mock(
        return_value=RobotJointCommand(timestamp=0.0, joint_positions=positions)
    )

    robot_driver.receive()

    written = _written_positions(robot_driver)
    for index in range(len(positions)):
        assert written[f"joint_{index}"] >= robot_driver.joint_lower_limits[index]


def test_position_clipping_mixed_violations(robot_driver):
    positions = np.array([-3.0, 2.0, 0.0, -4.0, 3.0, -1.0, 0.5])
    robot_driver.subscriber.receive = Mock(
        return_value=RobotJointCommand(timestamp=0.0, joint_positions=positions)
    )

    robot_driver.receive()

    written = _written_positions(robot_driver)
    for index in range(len(positions)):
        lower = robot_driver.joint_lower_limits[index]
        upper = robot_driver.joint_upper_limits[index]
        assert lower <= written[f"joint_{index}"] <= upper


def test_position_only_actuators_skip_velocity(robot_driver):
    command = RobotJointCommand(
        timestamp=0.0,
        joint_positions=np.zeros(7),
        joint_velocities=np.full(7, 0.5),
    )
    robot_driver.subscriber.receive = Mock(return_value=command)

    robot_driver.receive()

    assert _actuator_system(robot_driver).velocity_writes == [{}]


def _make_mixed_driver() -> RobotDriverNode:
    config = _robot_config([ActuatorControlMode.VELOCITY] * 3 + [ActuatorControlMode.POSITION] * 8)
    return _make_driver(config, velocity_limit=np.full(11, 4.0))


def test_mixed_modes_route_position_and_velocity_separately():
    driver = _make_mixed_driver()
    positions = np.linspace(-0.5, 0.5, 11)
    velocities = np.linspace(-1.0, 1.0, 11)
    driver.subscriber.receive = Mock(  # ty: ignore[invalid-assignment]
        return_value=RobotJointCommand(
            timestamp=0.0,
            joint_positions=positions,
            joint_velocities=velocities,
        )
    )

    driver.receive()

    written_positions = _actuator_system(driver).position_writes[-1]
    written_velocities = _actuator_system(driver).velocity_writes[-1]
    assert set(written_velocities) == {"joint_0", "joint_1", "joint_2"}
    assert set(written_positions) == {f"joint_{index}" for index in range(3, 11)}
    assert written_positions["joint_5"] == pytest.approx(positions[5])
    assert written_velocities["joint_1"] == pytest.approx(velocities[1])


def test_velocity_clamping():
    driver = _make_mixed_driver()
    driver.subscriber.receive = Mock(  # ty: ignore[invalid-assignment]
        return_value=RobotJointCommand(
            timestamp=0.0,
            joint_positions=np.zeros(11),
            joint_velocities=np.array([10.0, -10.0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0]),
        )
    )

    driver.receive()

    written = _actuator_system(driver).velocity_writes[-1]
    assert written["joint_0"] == pytest.approx(4.0)
    assert written["joint_1"] == pytest.approx(-4.0)
    assert written["joint_2"] == pytest.approx(4.0)


def test_no_velocities_in_command_stops_every_velocity_actuator():
    driver = _make_mixed_driver()
    driver.subscriber.receive = Mock(  # ty: ignore[invalid-assignment]
        return_value=RobotJointCommand(timestamp=0.0, joint_positions=np.zeros(11))
    )

    driver.receive()

    assert _actuator_system(driver).velocity_writes == [
        {"joint_0": 0.0, "joint_1": 0.0, "joint_2": 0.0}
    ]


def test_position_only_command_stops_previous_mobile_velocity():
    driver = _make_mixed_driver()
    driver.subscriber.receive = Mock(  # ty: ignore[invalid-assignment]
        side_effect=[
            RobotJointCommand(
                timestamp=0.0,
                joint_positions=np.zeros(11),
                joint_velocities=np.array([1.0, -2.0, 3.0, *([0.0] * 8)]),
            ),
            RobotJointCommand(timestamp=0.1, joint_positions=np.zeros(11)),
        ]
    )

    driver.receive()
    driver.receive()

    assert _actuator_system(driver).velocity_writes[-1] == {
        "joint_0": 0.0,
        "joint_1": 0.0,
        "joint_2": 0.0,
    }


def test_command_watchdog_stops_after_command_loss():
    now = 0.0
    driver = _make_driver(
        _robot_config([ActuatorControlMode.VELOCITY]),
        clock=lambda: now,
        command_timeout_seconds=0.1,
    )
    driver.subscriber.receive = Mock(  # ty: ignore[invalid-assignment]
        side_effect=[
            RobotJointCommand(
                timestamp=0.0,
                joint_positions=np.zeros(1),
                joint_velocities=np.ones(1),
            ),
            None,
            None,
        ]
    )

    driver.receive()
    now = 0.05
    driver.receive()
    assert _actuator_system(driver).stop_calls == 0

    now = 0.11
    driver.receive()
    assert _actuator_system(driver).stop_calls == 1


def test_command_watchdog_stops_only_once_until_another_nonzero_command():
    now = 0.0
    driver = _make_driver(
        _robot_config([ActuatorControlMode.VELOCITY]),
        clock=lambda: now,
        command_timeout_seconds=0.1,
    )
    driver.subscriber.receive = Mock(  # ty: ignore[invalid-assignment]
        side_effect=[
            RobotJointCommand(
                timestamp=0.0,
                joint_positions=np.zeros(1),
                joint_velocities=np.ones(1),
            ),
            None,
            None,
        ]
    )

    driver.receive()
    now = 0.2
    driver.receive()
    now = 0.3
    driver.receive()

    assert _actuator_system(driver).stop_calls == 1


def test_no_command_does_nothing(robot_driver):
    robot_driver.subscriber.receive = Mock(return_value=None)

    robot_driver.receive()

    assert _actuator_system(robot_driver).position_writes == []
    assert _actuator_system(robot_driver).velocity_writes == []


def test_rejects_non_positive_command_timeout():
    with pytest.raises(ValueError, match="timeout must be positive"):
        _make_driver(
            _robot_config([ActuatorControlMode.VELOCITY]),
            command_timeout_seconds=0.0,
        )


def test_rejects_incompatible_position_dimensions(robot_driver):
    robot_driver.subscriber.receive = Mock(
        return_value=RobotJointCommand(timestamp=0.0, joint_positions=np.zeros(6))
    )

    with pytest.raises(ValueError, match=r"6 positions.*requires 7"):
        robot_driver.receive()

    assert _actuator_system(robot_driver).position_writes == []
    assert _actuator_system(robot_driver).velocity_writes == []


def test_rejects_incompatible_velocity_dimensions(robot_driver):
    robot_driver.subscriber.receive = Mock(
        return_value=RobotJointCommand(
            timestamp=0.0,
            joint_positions=np.zeros(7),
            joint_velocities=np.zeros(6),
        )
    )

    with pytest.raises(ValueError, match=r"6 velocities.*requires 7"):
        robot_driver.receive()

    assert _actuator_system(robot_driver).position_writes == []
    assert _actuator_system(robot_driver).velocity_writes == []


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_positions_before_actuator_writes(robot_driver, invalid_value):
    positions = np.zeros(7)
    positions[2] = invalid_value
    robot_driver.subscriber.receive = Mock(
        return_value=RobotJointCommand(timestamp=0.0, joint_positions=positions)
    )

    with pytest.raises(ValueError, match="positions must all be finite"):
        robot_driver.receive()

    assert _actuator_system(robot_driver).position_writes == []
    assert _actuator_system(robot_driver).velocity_writes == []


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_velocities_before_actuator_writes(robot_driver, invalid_value):
    velocities = np.zeros(7)
    velocities[2] = invalid_value
    robot_driver.subscriber.receive = Mock(
        return_value=RobotJointCommand(
            timestamp=0.0,
            joint_positions=np.zeros(7),
            joint_velocities=velocities,
        )
    )

    with pytest.raises(ValueError, match="velocities must all be finite"):
        robot_driver.receive()

    assert _actuator_system(robot_driver).position_writes == []
    assert _actuator_system(robot_driver).velocity_writes == []


def test_mobile_driver_uses_root_state_estimator():
    actuator_system = StubActuatorSystem()
    estimated_root = RootState(
        position=np.array([1.0, 2.0, 0.0, 1.0]),
        velocity=np.array([0.2, 0.0, 0.0]),
    )
    root_state_estimator = StubRootStateEstimator(estimated_root)
    with (
        patch("humanoid.nodes.robot.driver.Subscriber"),
        patch("humanoid.nodes.robot.driver.Publisher"),
    ):
        driver = RobotDriverNode(
            robot_config=ELROBOT_MOBILE_CONFIG,
            actuator_system=actuator_system,
            root_state_estimator=root_state_estimator,
        )

    assert driver._root_q_slice is not None
    assert driver._root_v_slice is not None
    measured_wheel_velocities = {
        "wheel_1": 4.0,
        "wheel_2": -2.0,
        "wheel_3": -2.0,
    }
    actuator_system.states = {
        joint_name: ActuatorState(
            position=0.0,
            velocity=measured_wheel_velocities.get(joint_name, 0.0),
        )
        for joint_name in driver.actuator_joint_names
    }

    driver.publish()

    measured_q, measured_v = root_state_estimator.updates[-1]
    for joint_name, velocity in measured_wheel_velocities.items():
        joint_idx = driver.joint_indices[joint_name]
        assert measured_v[driver.robot.joint_idx_to_velocity_idx(joint_idx)] == pytest.approx(
            velocity
        )
    assert measured_q.shape == (driver.robot.model.nq,)
    published = driver.publisher.publish.call_args.args[0]  # ty: ignore[unresolved-attribute]
    np.testing.assert_allclose(
        published.joint_positions[driver._root_q_slice], estimated_root.position
    )
    np.testing.assert_allclose(
        published.joint_velocities[driver._root_v_slice],
        estimated_root.velocity,
    )

    driver.on_close()
    assert actuator_system.stop_calls == 1


def test_mobile_driver_uses_dead_reckoning():
    actuator_system = StubActuatorSystem()
    with (
        patch("humanoid.nodes.robot.driver.Subscriber"),
        patch("humanoid.nodes.robot.driver.Publisher"),
    ):
        driver = RobotDriverNode(
            robot_config=ELROBOT_MOBILE_CONFIG,
            actuator_system=actuator_system,
        )

    assert isinstance(driver.root_state_estimator, WheelDeadReckoningRootStateEstimator)

    driver.on_close()


def test_mobile_driver_ignores_commanded_root_velocity():
    root_state_estimator = StubRootStateEstimator(
        RootState(
            position=np.array([0.0, 0.0, 1.0, 0.0]),
            velocity=np.zeros(3),
        )
    )
    actuator_system = StubActuatorSystem()
    with (
        patch("humanoid.nodes.robot.driver.Subscriber"),
        patch("humanoid.nodes.robot.driver.Publisher"),
    ):
        driver = RobotDriverNode(
            robot_config=ELROBOT_MOBILE_CONFIG,
            actuator_system=actuator_system,
            root_state_estimator=root_state_estimator,
        )

    assert driver._root_v_slice is not None
    command_velocities = np.zeros(driver.robot.model.nv)
    command_velocities[driver._root_v_slice] = 1_000.0
    driver.subscriber.receive = Mock(  # ty: ignore[invalid-assignment]
        return_value=RobotJointCommand(
            timestamp=0.0,
            joint_positions=ELROBOT_MOBILE_CONFIG.homing_presets[HomingPreset.HOME].copy(),
            joint_velocities=command_velocities,
        )
    )

    driver.receive()

    assert root_state_estimator.updates == []

    actuator_system.states = {
        joint_name: ActuatorState(position=0.0, velocity=0.0)
        for joint_name in driver.actuator_joint_names
    }
    driver.publish()

    _, measured_velocity = root_state_estimator.updates[-1]
    np.testing.assert_allclose(measured_velocity[driver._root_v_slice], np.zeros(3))


def test_publish_rejects_incomplete_actuator_feedback(robot_driver):
    _actuator_system(robot_driver).states = {"joint_0": ActuatorState(position=0.0, velocity=0.0)}

    with pytest.raises(RuntimeError, match="Incomplete actuator feedback"):
        robot_driver.publish()


def test_close_stops_hardware_before_closing_middleware(robot_driver):
    events = []
    actuator_system = _actuator_system(robot_driver)
    robot_driver.subscriber.close = Mock(side_effect=lambda: events.append("subscriber"))

    with (
        patch.object(actuator_system, "stop", side_effect=lambda: events.append("stop")),
        patch.object(
            actuator_system,
            "disconnect",
            side_effect=lambda: events.append("disconnect"),
        ),
    ):
        robot_driver.on_close()

    assert events == ["stop", "subscriber", "disconnect"]


def test_close_disconnects_hardware_when_subscriber_close_fails(robot_driver):
    events = []
    actuator_system = _actuator_system(robot_driver)

    def fail_subscriber_close():
        events.append("subscriber")
        raise RuntimeError("subscriber close failed")

    robot_driver.subscriber.close = Mock(side_effect=fail_subscriber_close)

    with (
        patch.object(actuator_system, "stop", side_effect=lambda: events.append("stop")),
        patch.object(
            actuator_system,
            "disconnect",
            side_effect=lambda: events.append("disconnect"),
        ),
        pytest.raises(RuntimeError, match="subscriber close failed"),
    ):
        robot_driver.on_close()

    assert events == ["stop", "subscriber", "disconnect"]
