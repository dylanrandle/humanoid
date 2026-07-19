import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pinocchio as pin
import pytest

from humanoid.constants import Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.logging import LoggingState, LoggingStatus
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
    with patch("humanoid.middleware.subscriber.lcm.LCM", return_value=mock_instance):
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


class TestSubscriber:
    def test_subscribes_to_each_topic(self, mock_lcm):
        topics = [Topic.ROBOT_STATE, Topic.ROBOT_JOINT_COMMAND]
        Subscriber(topics=topics)

        subscribed_channels = [call.args[0] for call in mock_lcm.subscribe.call_args_list]
        assert subscribed_channels == [t.value for t in topics]

    def test_sets_queue_capacity_when_provided(self, mock_lcm):
        mock_subscription = MagicMock()
        mock_lcm.subscribe.return_value = mock_subscription

        Subscriber(topics=[Topic.ROBOT_STATE], queue_size=5)

        mock_subscription.set_queue_capacity.assert_called_once_with(5)

    def test_skips_queue_capacity_when_none(self, mock_lcm):
        mock_subscription = MagicMock()
        mock_lcm.subscribe.return_value = mock_subscription

        Subscriber(topics=[Topic.ROBOT_STATE], queue_size=None)

        mock_subscription.set_queue_capacity.assert_not_called()

    def test_receive_with_no_message_returns_none(self, mock_lcm):
        sub = Subscriber(topics=[Topic.ROBOT_STATE])
        result = sub.receive(Topic.ROBOT_STATE, timeout=10)
        assert result is None

    def test_blocking_receive_returns_when_message_arrives(self, mock_lcm):
        sub = Subscriber(topics=[Topic.ROBOT_STATE])
        state = _make_state()
        encoded = LCMConverter.robot_state_to_lcm(state).encode()

        def deliver():
            time.sleep(0.02)
            sub._handle_message(Topic.ROBOT_STATE.value, encoded)

        threading.Thread(target=deliver, daemon=True).start()
        result = sub.receive(Topic.ROBOT_STATE, timeout=500)

        assert isinstance(result, RobotState)
        np.testing.assert_allclose(result.joint_positions, state.joint_positions)

    def test_handle_message_decodes_and_queues_state(self, mock_lcm):
        sub = Subscriber(topics=[Topic.ROBOT_STATE])
        state = _make_state()
        encoded = LCMConverter.robot_state_to_lcm(state).encode()

        sub._handle_message(Topic.ROBOT_STATE.value, encoded)

        result = sub.receive(Topic.ROBOT_STATE, timeout=10)
        assert isinstance(result, RobotState)
        np.testing.assert_allclose(result.joint_positions, state.joint_positions)
        np.testing.assert_allclose(result.actuator_temperatures, state.actuator_temperatures)

    def test_handle_message_decodes_each_supported_type(self, mock_lcm):
        sub = Subscriber(
            topics=[
                Topic.ROBOT_JOINT_COMMAND,
                Topic.ROBOT_STATE,
                Topic.ROBOT_TOOL_COMMAND,
                Topic.ROBOT_BASE_COMMAND,
                Topic.LOGGING_STATUS,
            ]
        )

        cases = [
            (
                Topic.ROBOT_JOINT_COMMAND,
                LCMConverter.robot_joint_command_to_lcm(_make_joint_command()),
                RobotJointCommand,
            ),
            (
                Topic.ROBOT_STATE,
                LCMConverter.robot_state_to_lcm(_make_state()),
                RobotState,
            ),
            (
                Topic.ROBOT_TOOL_COMMAND,
                LCMConverter.robot_tool_command_to_lcm(_make_tool_command()),
                RobotToolCommand,
            ),
            (
                Topic.ROBOT_BASE_COMMAND,
                LCMConverter.robot_base_command_to_lcm(_make_base_command()),
                RobotBaseCommand,
            ),
            (
                Topic.LOGGING_STATUS,
                LCMConverter.logging_status_to_lcm(
                    LoggingStatus(
                        timestamp=5.0,
                        state=LoggingState.RUNNING,
                        file_name="logs/lcmlog",
                    )
                ),
                LoggingStatus,
            ),
        ]

        for topic, lcm_msg, expected_type in cases:
            sub._handle_message(topic.value, lcm_msg.encode())
            result = sub.receive(topic, timeout=10)
            assert isinstance(result, expected_type), f"Expected {expected_type} for {topic}"

    def test_queues_are_isolated_per_topic(self, mock_lcm):
        sub = Subscriber(topics=[Topic.ROBOT_STATE, Topic.ROBOT_JOINT_COMMAND])

        state_bytes = LCMConverter.robot_state_to_lcm(_make_state()).encode()
        sub._handle_message(Topic.ROBOT_STATE.value, state_bytes)

        # Receiving on the other topic returns None and doesn't drain state queue.
        assert sub.receive(Topic.ROBOT_JOINT_COMMAND, timeout=10) is None
        # State message is still available.
        assert isinstance(sub.receive(Topic.ROBOT_STATE, timeout=10), RobotState)

    def test_queue_capacity_drops_oldest(self, mock_lcm):
        sub = Subscriber(topics=[Topic.ROBOT_STATE], queue_size=1)

        first = _make_state()
        second = RobotState(
            timestamp=99.0,
            joint_positions=np.array([9.0, 9.0]),
            joint_velocities=np.array([9.0, 9.0]),
            actuator_temperatures=np.array([99.0, 99.0]),
        )

        sub._handle_message(
            Topic.ROBOT_STATE.value, LCMConverter.robot_state_to_lcm(first).encode()
        )
        sub._handle_message(
            Topic.ROBOT_STATE.value, LCMConverter.robot_state_to_lcm(second).encode()
        )

        result = sub.receive(Topic.ROBOT_STATE, timeout=10)
        assert result is not None
        np.testing.assert_allclose(result.joint_positions, second.joint_positions)
        # Queue should now be empty.
        assert sub.receive(Topic.ROBOT_STATE, timeout=10) is None

    def test_handle_message_swallows_decode_errors(self, mock_lcm):
        sub = Subscriber(topics=[Topic.ROBOT_STATE])

        # Garbage bytes should not raise — just get logged and dropped.
        sub._handle_message(Topic.ROBOT_STATE.value, b"not a valid lcm message")

        assert sub.receive(Topic.ROBOT_STATE, timeout=10) is None

    def test_receive_returns_queued_message_immediately(self, mock_lcm):
        """A pre-queued message is returned without waiting."""
        sub = Subscriber(topics=[Topic.ROBOT_STATE])
        sub._handle_message(
            Topic.ROBOT_STATE.value, LCMConverter.robot_state_to_lcm(_make_state()).encode()
        )

        result = sub.receive(Topic.ROBOT_STATE, timeout=0)

        assert isinstance(result, RobotState)

    def test_close_unsubscribes_all(self, mock_lcm):
        subs = [MagicMock(), MagicMock()]
        mock_lcm.subscribe.side_effect = subs

        sub = Subscriber(topics=[Topic.ROBOT_STATE, Topic.ROBOT_JOINT_COMMAND])
        sub.close()

        assert mock_lcm.unsubscribe.call_count == len(subs)
        unsubscribed = [c.args[0] for c in mock_lcm.unsubscribe.call_args_list]
        assert unsubscribed == subs
        assert sub._subscriptions == []
