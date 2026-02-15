from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from humanoid.nodes.robot_driver import RobotDriver
from humanoid.types.robot import RobotConfig, RobotJointCommand


@pytest.fixture
def mock_robot_driver():
    """Create a RobotDriver with mocked dependencies."""
    # Mock the robot config
    robot_config = RobotConfig(
        name="panda",
        end_effector_frame="panda_hand",
        home_position=np.zeros(7),
        rest_position=np.ones(7),
        joint_idx_to_servo_id={i: i + 1 for i in range(7)},
    )

    # Mock all external dependencies
    with (
        patch("humanoid.nodes.robot_driver.Subscriber"),
        patch("humanoid.nodes.robot_driver.Publisher"),
        patch("humanoid.nodes.robot_driver.Robot") as mock_robot_cls,
        patch("humanoid.nodes.robot_driver.SimulatedMotorController") as mock_controller,
    ):
        # Setup mock robot with joint limits
        mock_robot = Mock()
        mock_robot.model.lowerPositionLimit = np.array(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        )
        mock_robot.model.upperPositionLimit = np.array(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        )
        mock_robot_cls.from_name.return_value = mock_robot

        # Setup mock controller
        mock_controller_instance = MagicMock()
        mock_controller.return_value = mock_controller_instance

        driver = RobotDriver(robot_config=robot_config)
        driver.controller = mock_controller_instance

        yield driver


def test_position_clipping_within_limits(mock_robot_driver):
    """Test that positions within limits are not modified."""
    # Create command with positions within limits
    valid_positions = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.0, 0.0])
    command = RobotJointCommand(timestamp=0.0, joint_positions=valid_positions)

    # Mock subscriber to return the command
    mock_robot_driver.subscriber.receive = Mock(return_value=command)

    # Execute receive
    mock_robot_driver.receive()

    # Verify controller was called with unclamped positions
    mock_robot_driver.controller.write_position.assert_called_once()
    written_positions = mock_robot_driver.controller.write_position.call_args[0][0]

    # Check that positions match the original command (converted to servo IDs)
    for joint_idx, position in enumerate(valid_positions):
        servo_id = mock_robot_driver.joint_idx_to_servo_id[joint_idx]
        assert written_positions[servo_id] == pytest.approx(position)


def test_position_clipping_above_upper_limits(mock_robot_driver):
    """Test that positions above upper limits are clipped."""
    # Create command with positions exceeding upper limits
    excessive_positions = np.array([3.0, 2.0, 3.0, 0.0, 3.0, 4.0, 3.0])
    command = RobotJointCommand(timestamp=0.0, joint_positions=excessive_positions)

    # Mock subscriber to return the command
    mock_robot_driver.subscriber.receive = Mock(return_value=command)

    # Execute receive
    mock_robot_driver.receive()

    # Verify controller was called with clamped positions
    mock_robot_driver.controller.write_position.assert_called_once()
    written_positions = mock_robot_driver.controller.write_position.call_args[0][0]

    # Check that all positions are within limits
    for joint_idx in range(len(excessive_positions)):
        servo_id = mock_robot_driver.joint_idx_to_servo_id[joint_idx]
        assert written_positions[servo_id] <= mock_robot_driver.joint_upper_limits[joint_idx]


def test_position_clipping_below_lower_limits(mock_robot_driver):
    """Test that positions below lower limits are clipped."""
    # Create command with positions below lower limits
    low_positions = np.array([-3.0, -2.0, -3.0, -4.0, -3.0, -1.0, -3.0])
    command = RobotJointCommand(timestamp=0.0, joint_positions=low_positions)

    # Mock subscriber to return the command
    mock_robot_driver.subscriber.receive = Mock(return_value=command)

    # Execute receive
    mock_robot_driver.receive()

    # Verify controller was called with clamped positions
    mock_robot_driver.controller.write_position.assert_called_once()
    written_positions = mock_robot_driver.controller.write_position.call_args[0][0]

    # Check that all positions are within limits
    for joint_idx in range(len(low_positions)):
        servo_id = mock_robot_driver.joint_idx_to_servo_id[joint_idx]
        assert written_positions[servo_id] >= mock_robot_driver.joint_lower_limits[joint_idx]


def test_position_clipping_mixed_violations(mock_robot_driver):
    """Test that mixed violations (some above, some below) are all clipped correctly."""
    # Create command with mixed violations
    mixed_positions = np.array([-3.0, 2.0, 0.0, -4.0, 3.0, -1.0, 0.5])
    command = RobotJointCommand(timestamp=0.0, joint_positions=mixed_positions)

    # Mock subscriber to return the command
    mock_robot_driver.subscriber.receive = Mock(return_value=command)

    # Execute receive
    mock_robot_driver.receive()

    # Verify controller was called with clamped positions
    mock_robot_driver.controller.write_position.assert_called_once()
    written_positions = mock_robot_driver.controller.write_position.call_args[0][0]

    # Check that all positions are within limits
    for joint_idx in range(len(mixed_positions)):
        servo_id = mock_robot_driver.joint_idx_to_servo_id[joint_idx]
        lower = mock_robot_driver.joint_lower_limits[joint_idx]
        upper = mock_robot_driver.joint_upper_limits[joint_idx]
        assert lower <= written_positions[servo_id] <= upper
