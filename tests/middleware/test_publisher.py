from unittest.mock import MagicMock, patch

import numpy as np
import pinocchio as pin
import pytest

from humanoid.constants import Topic
from humanoid.middleware.publisher import Publisher
from humanoid.types.lcm import (
    logging_status_t,
    orchestrator_mode_t,
    robot_joint_command_t,
    robot_state_t,
)
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.logging import LoggingState, LoggingStatus
from humanoid.types.orchestrator import Mode, OrchestratorMode
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
        actuator_temperatures=np.array([30.0, 31.0]),
    )


def _make_tool_command():
    return RobotToolCommand(timestamp=3.0, pose=pin.SE3.Identity())


def _make_base_command():
    return RobotBaseCommand(timestamp=4.0, pose=pin.SE3.Identity())


class TestPublisher:
    def test_publishes_to_explicit_topic(self, mock_lcm):
        publisher = Publisher()
        cmd = _make_joint_command()

        publisher.publish(cmd, topic=Topic.ROBOT_JOINT_COMMAND)

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

    def test_publishes_same_type_to_different_topics(self, mock_lcm):
        """A joint command can be published to multiple distinct topics."""
        publisher = Publisher()
        cmd = _make_joint_command()

        publisher.publish(cmd, topic=Topic.CONTROLLER_JOINT_COMMAND)
        publisher.publish(cmd, topic=Topic.HOMING_JOINT_COMMAND)

        channels = [call.args[0] for call in mock_lcm.publish.call_args_list]
        assert channels == [
            Topic.CONTROLLER_JOINT_COMMAND.value,
            Topic.HOMING_JOINT_COMMAND.value,
        ]

    def test_publish_state(self, mock_lcm):
        publisher = Publisher()
        state = _make_state()

        publisher.publish(state, topic=Topic.ROBOT_STATE)

        channel, data_bytes = mock_lcm.publish.call_args[0]
        assert channel == Topic.ROBOT_STATE.value
        recovered = LCMConverter.robot_state_from_lcm(robot_state_t.decode(data_bytes))
        np.testing.assert_allclose(recovered.actuator_temperatures, state.actuator_temperatures)

    def test_publish_tool_command(self, mock_lcm):
        publisher = Publisher()
        publisher.publish(_make_tool_command(), topic=Topic.OCULUS_TOOL_COMMAND)

        channel, _ = mock_lcm.publish.call_args[0]
        assert channel == Topic.OCULUS_TOOL_COMMAND.value

    def test_publish_base_command(self, mock_lcm):
        publisher = Publisher()
        publisher.publish(_make_base_command(), topic=Topic.KEYBOARD_BASE_COMMAND)

        channel, _ = mock_lcm.publish.call_args[0]
        assert channel == Topic.KEYBOARD_BASE_COMMAND.value

    def test_publish_orchestrator_mode(self, mock_lcm):
        publisher = Publisher()
        mode = OrchestratorMode(timestamp=5.0, mode=Mode.OCULUS)

        publisher.publish(mode, topic=Topic.ORCHESTRATOR_MODE)

        channel, data_bytes = mock_lcm.publish.call_args[0]
        assert channel == Topic.ORCHESTRATOR_MODE.value
        recovered = LCMConverter.orchestrator_mode_from_lcm(orchestrator_mode_t.decode(data_bytes))
        assert recovered.mode is Mode.OCULUS

    def test_publish_logging_status(self, mock_lcm):
        publisher = Publisher()
        status = LoggingStatus(
            timestamp=6.0,
            state=LoggingState.FAILED,
            file_name="logs/lcmlog",
            error="logger exited",
        )

        publisher.publish(status, topic=Topic.LOGGING_STATUS)

        channel, data_bytes = mock_lcm.publish.call_args[0]
        assert channel == Topic.LOGGING_STATUS.value
        recovered = LCMConverter.logging_status_from_lcm(logging_status_t.decode(data_bytes))
        assert recovered == status

    def test_publish_topic_type_mismatch_raises(self, mock_lcm):
        publisher = Publisher()
        with pytest.raises(TypeError, match="expects"):
            publisher.publish(_make_joint_command(), topic=Topic.ROBOT_TOOL_COMMAND)
        mock_lcm.publish.assert_not_called()

    def test_publish_unsupported_type_raises(self, mock_lcm):
        publisher = Publisher()
        with pytest.raises(TypeError):
            publisher.publish("not a valid type", topic=Topic.ROBOT_STATE)  # type: ignore[arg-type]
        mock_lcm.publish.assert_not_called()
