from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from humanoid.nodes.robot_driver import RobotDriver
from humanoid.types.robot import RobotConfig, RobotJointCommand
from humanoid.types.servo import ServoControlMode


def _make_driver(
    robot_config: RobotConfig, velocity_limit: np.ndarray | None = None
) -> RobotDriver:
    """Build a RobotDriver with mocked LCM, motors, and Robot model."""
    n_joints = len(robot_config.joint_idx_to_servo_id)
    if velocity_limit is None:
        velocity_limit = np.full(n_joints, 5.0)

    with (
        patch("humanoid.nodes.robot_driver.Subscriber"),
        patch("humanoid.nodes.robot_driver.Publisher"),
        patch("humanoid.nodes.robot_driver.Robot") as mock_robot_cls,
        patch("humanoid.nodes.robot_driver.SimulatedMotorController") as mock_controller,
    ):
        mock_robot = Mock()
        mock_robot.config = robot_config
        mock_robot.model.lowerPositionLimit = np.full(n_joints, -3.0)
        mock_robot.model.upperPositionLimit = np.full(n_joints, 3.0)
        mock_robot.model.velocityLimit = velocity_limit
        mock_robot.joint_idx_to_position_idx.side_effect = lambda i: i
        mock_robot.joint_idx_to_velocity_idx.side_effect = lambda i: i
        # Fixed-base mock — no planar root joint to echo.
        mock_robot.get_root_q_slice.return_value = None
        mock_robot_cls.return_value = mock_robot

        mock_controller_instance = MagicMock()
        mock_controller.return_value = mock_controller_instance

        driver = RobotDriver(robot_config=robot_config)
        driver.controller = mock_controller_instance

        return driver


@pytest.fixture
def mock_robot_driver():
    """RobotDriver with 7 position-controlled servos and panda-like position limits."""
    robot_config = RobotConfig(
        name="panda",
        tool_frame="panda_hand",
        home_position=np.zeros(7),
        rest_position=np.ones(7),
        joint_idx_to_servo_id={i: i + 1 for i in range(7)},
        servo_control_modes=dict.fromkeys(range(1, 8), ServoControlMode.POSITION),
    )
    driver = _make_driver(robot_config)
    # Override default position limits with panda-specific limits
    driver.joint_lower_limits = np.array(
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    )
    driver.joint_upper_limits = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    yield driver


def test_position_clipping_within_limits(mock_robot_driver):
    """Test that positions within limits are not modified."""
    valid_positions = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.0, 0.0])
    command = RobotJointCommand(timestamp=0.0, joint_positions=valid_positions)

    mock_robot_driver.subscriber.receive = Mock(return_value=command)
    mock_robot_driver.receive()

    mock_robot_driver.controller.write_position.assert_called_once()
    written_positions = mock_robot_driver.controller.write_position.call_args[0][0]

    for joint_idx, position in enumerate(valid_positions):
        servo_id = mock_robot_driver.joint_idx_to_servo_id[joint_idx]
        assert written_positions[servo_id] == pytest.approx(position)


def test_position_clipping_above_upper_limits(mock_robot_driver):
    """Test that positions above upper limits are clipped."""
    excessive_positions = np.array([3.0, 2.0, 3.0, 0.0, 3.0, 4.0, 3.0])
    command = RobotJointCommand(timestamp=0.0, joint_positions=excessive_positions)

    mock_robot_driver.subscriber.receive = Mock(return_value=command)
    mock_robot_driver.receive()

    mock_robot_driver.controller.write_position.assert_called_once()
    written_positions = mock_robot_driver.controller.write_position.call_args[0][0]

    for joint_idx in range(len(excessive_positions)):
        servo_id = mock_robot_driver.joint_idx_to_servo_id[joint_idx]
        assert written_positions[servo_id] <= mock_robot_driver.joint_upper_limits[joint_idx]


def test_position_clipping_below_lower_limits(mock_robot_driver):
    """Test that positions below lower limits are clipped."""
    low_positions = np.array([-3.0, -2.0, -3.0, -4.0, -3.0, -1.0, -3.0])
    command = RobotJointCommand(timestamp=0.0, joint_positions=low_positions)

    mock_robot_driver.subscriber.receive = Mock(return_value=command)
    mock_robot_driver.receive()

    mock_robot_driver.controller.write_position.assert_called_once()
    written_positions = mock_robot_driver.controller.write_position.call_args[0][0]

    for joint_idx in range(len(low_positions)):
        servo_id = mock_robot_driver.joint_idx_to_servo_id[joint_idx]
        assert written_positions[servo_id] >= mock_robot_driver.joint_lower_limits[joint_idx]


def test_position_clipping_mixed_violations(mock_robot_driver):
    """Test that mixed violations (some above, some below) are all clipped correctly."""
    mixed_positions = np.array([-3.0, 2.0, 0.0, -4.0, 3.0, -1.0, 0.5])
    command = RobotJointCommand(timestamp=0.0, joint_positions=mixed_positions)

    mock_robot_driver.subscriber.receive = Mock(return_value=command)
    mock_robot_driver.receive()

    mock_robot_driver.controller.write_position.assert_called_once()
    written_positions = mock_robot_driver.controller.write_position.call_args[0][0]

    for joint_idx in range(len(mixed_positions)):
        servo_id = mock_robot_driver.joint_idx_to_servo_id[joint_idx]
        lower = mock_robot_driver.joint_lower_limits[joint_idx]
        upper = mock_robot_driver.joint_upper_limits[joint_idx]
        assert lower <= written_positions[servo_id] <= upper


def test_position_only_servos_skip_velocity(mock_robot_driver):
    """A purely position-controlled robot should write zero velocity entries."""
    positions = np.zeros(7)
    velocities = np.full(7, 0.5)
    command = RobotJointCommand(
        timestamp=0.0, joint_positions=positions, joint_velocities=velocities
    )

    mock_robot_driver.subscriber.receive = Mock(return_value=command)
    mock_robot_driver.receive()

    mock_robot_driver.controller.write_velocity.assert_called_once()
    written_velocities = mock_robot_driver.controller.write_velocity.call_args[0][0]
    assert written_velocities == {}


def _make_mixed_driver():
    """Driver with 3 velocity-controlled (base) joints + 8 position-controlled (arm) joints."""
    joint_idx_to_servo_id = {
        0: 250,
        1: 251,
        2: 252,
        3: 1,
        4: 2,
        5: 3,
        6: 4,
        7: 5,
        8: 6,
        9: 7,
        10: 8,
    }
    servo_control_modes = {
        **dict.fromkeys([250, 251, 252], ServoControlMode.VELOCITY),
        **dict.fromkeys(range(1, 9), ServoControlMode.POSITION),
    }
    config = RobotConfig(
        name="mixed",
        tool_frame="tool",
        base_frame="base",
        home_position=np.zeros(11),
        rest_position=np.zeros(11),
        joint_idx_to_servo_id=joint_idx_to_servo_id,
        servo_control_modes=servo_control_modes,
    )
    return _make_driver(config, velocity_limit=np.full(11, 4.0))


def test_mixed_modes_route_position_and_velocity_separately():
    """
    Position-controlled servos receive only positions;
    velocity-controlled servos receive only velocities.
    """
    driver = _make_mixed_driver()

    positions = np.linspace(-0.5, 0.5, 11)
    velocities = np.linspace(-1.0, 1.0, 11)
    command = RobotJointCommand(
        timestamp=0.0, joint_positions=positions, joint_velocities=velocities
    )

    driver.subscriber.receive = Mock(return_value=command)
    driver.receive()

    written_positions = driver.controller.write_position.call_args[0][0]
    written_velocities = driver.controller.write_velocity.call_args[0][0]

    # Velocity-controlled base servos must NOT appear in position dict.
    for servo_id in (250, 251, 252):
        assert servo_id not in written_positions
        assert servo_id in written_velocities

    # Position-controlled arm servos must NOT appear in velocity dict.
    for servo_id in range(1, 9):
        assert servo_id in written_positions
        assert servo_id not in written_velocities

    # Spot check a value made it through with the right routing.
    arm_joint_idx = 5  # maps to servo 3 (position-controlled)
    assert written_positions[3] == pytest.approx(positions[arm_joint_idx])

    base_joint_idx = 1  # maps to servo 251 (velocity-controlled)
    assert written_velocities[251] == pytest.approx(velocities[base_joint_idx])


def test_velocity_clamping():
    """Velocities are clamped to +/- velocityLimit."""
    driver = _make_mixed_driver()

    positions = np.zeros(11)
    # Large velocities should be clamped to +/- 4.0
    velocities = np.array([10.0, -10.0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0])
    command = RobotJointCommand(
        timestamp=0.0, joint_positions=positions, joint_velocities=velocities
    )

    driver.subscriber.receive = Mock(return_value=command)
    driver.receive()

    written_velocities = driver.controller.write_velocity.call_args[0][0]
    assert written_velocities[250] == pytest.approx(4.0)
    assert written_velocities[251] == pytest.approx(-4.0)
    assert written_velocities[252] == pytest.approx(4.0)


def test_no_velocities_in_command_skips_velocity_writes():
    """When command.joint_velocities is None, no velocity entries are produced."""
    driver = _make_mixed_driver()

    command = RobotJointCommand(timestamp=0.0, joint_positions=np.zeros(11))

    driver.subscriber.receive = Mock(return_value=command)
    driver.receive()

    driver.controller.write_velocity.assert_called_once_with({})


def test_no_command_does_nothing(mock_robot_driver):
    """If no command is queued, neither write should be invoked."""
    mock_robot_driver.subscriber.receive = Mock(return_value=None)
    mock_robot_driver.receive()

    mock_robot_driver.controller.write_position.assert_not_called()
    mock_robot_driver.controller.write_velocity.assert_not_called()
