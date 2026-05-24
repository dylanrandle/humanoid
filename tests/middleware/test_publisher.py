from unittest.mock import MagicMock, patch

import numpy as np
import pinocchio as pin
import pytest

from humanoid.constants import Topic
from humanoid.middleware.publisher import Publisher
from humanoid.types.lcm import (
    robot_joint_command_t,
    robot_state_t,
)
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)


@pytest.fixture
def mock_lcm():
    """Patch lcm.LCM so no actual broker is needed."""
    mock_instance = MagicMock()
    with patch("humanoid.middleware.publisher.lcm.LCM", return_value=mock_instance):
        yield mock_instance


def _make_joint_command():
    return RobotJointCommand(
        timestamp=1.0,
        joint_positions=np.array([0.1, 0.2, 0.3]),
        joint_velocities=np.array([0.4, 0.5, 0.6]),
    )


def _make_state():
    return RobotState(
        timestamp=2.0,
        joint_positions=np.array([0.1, 0.2]),
        joint_velocities=np.array([0.3, 0.4]),
        motor_temperatures=np.array([30.0, 31.0]),
    )


def _make_tool_command():
    return RobotToolCommand(timestamp=3.0, pose=pin.SE3.Identity())


def _make_base_command():
    return RobotBaseCommand(timestamp=4.0, pose=pin.SE3.Identity())


class TestPublisher:
    def test_publish_joint_command_uses_joint_topic(self, mock_lcm):
        publisher = Publisher()
        cmd = _make_joint_command()

        publisher.publish(cmd)

        mock_lcm.publish.assert_called_once()
        channel, data_bytes = mock_lcm.publish.call_args[0]
        assert channel == Topic.ROBOT_JOINT_COMMAND.value
        assert isinstance(data_bytes, bytes)

        # The published bytes should be decodable back to the original.
        recovered = LCMConverter.robot_joint_command_from_lcm(
            robot_joint_command_t.decode(data_bytes)
        )
        np.testing.assert_allclose(recovered.joint_positions, cmd.joint_positions)
        np.testing.assert_allclose(recovered.joint_velocities, cmd.joint_velocities)  # ty:ignore[no-matching-overload]

    def test_publish_state_uses_state_topic(self, mock_lcm):
        publisher = Publisher()
        state = _make_state()

        publisher.publish(state)

        channel, data_bytes = mock_lcm.publish.call_args[0]
        assert channel == Topic.ROBOT_STATE.value
        recovered = LCMConverter.robot_state_from_lcm(robot_state_t.decode(data_bytes))
        np.testing.assert_allclose(recovered.motor_temperatures, state.motor_temperatures)

    def test_publish_tool_command_uses_tool_topic(self, mock_lcm):
        publisher = Publisher()
        publisher.publish(_make_tool_command())

        channel, _ = mock_lcm.publish.call_args[0]
        assert channel == Topic.ROBOT_TOOL_COMMAND.value

    def test_publish_base_command_uses_base_topic(self, mock_lcm):
        publisher = Publisher()
        publisher.publish(_make_base_command())

        channel, _ = mock_lcm.publish.call_args[0]
        assert channel == Topic.ROBOT_BASE_COMMAND.value

    def test_publish_unsupported_type_raises(self, mock_lcm):
        publisher = Publisher()
        with pytest.raises(TypeError, match="Unsupported data type"):
            publisher.publish("not a valid type")  # type: ignore[arg-type]
        mock_lcm.publish.assert_not_called()
