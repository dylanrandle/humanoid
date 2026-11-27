import time
from unittest.mock import Mock, call

import pytest

from humanoid.nodes.policy.motion import MotionPolicyNode


@pytest.fixture
def node(monkeypatch):
    monkeypatch.setattr("humanoid.nodes.policy.motion.RealtimeEnvironment", Mock())
    monkeypatch.setattr("humanoid.nodes.policy.motion.Robot", Mock())
    monkeypatch.setattr("humanoid.nodes.policy.motion.MotionPolicy", Mock())
    monkeypatch.setattr("humanoid.nodes.policy.motion.MotionExecutor", Mock())
    pipe = Mock()
    pipe.closed.is_set.return_value = False
    pipe.last_received = time.monotonic()
    monkeypatch.setattr("humanoid.nodes.policy.motion.Pipe", Mock(return_value=pipe))
    monkeypatch.setattr("humanoid.nodes.policy.motion.threading.Thread", Mock())
    return MotionPolicyNode(Mock(), "session")


def test_setup_initializes_environment(node):
    node.setup()
    node.env.reset.assert_called_once()
    node.executor.observe.assert_called_once_with(node.env.observe.return_value)
    node.policy.assert_not_called()


def test_each_step_passes_latest_observation_to_policy_and_publishes_action(node):
    first, second = Mock(), Mock()
    node.env.observe.side_effect = [first, second]
    node.step()
    node.policy.assert_called_with(first)
    node.env.step.assert_called_with(node.policy.return_value)
    node.step()
    node.policy.assert_called_with(second)
    assert node.executor.observe.call_args_list == [call(first), call(second)]
    assert node.env.step.call_args_list == [call(node.policy.return_value)] * len((first, second))


def test_close_resets_policy_and_publishes_hold_before_closing_environment(node):
    node.setup()
    node.on_close()
    node.pipe.close.assert_called_once()
    node._worker.join.assert_called_once()
    node.executor.close.assert_called_once()
    node.policy.reset.assert_called_once()
    node.policy.assert_called_once_with(node.env.observe.return_value)
    node.env.step.assert_called_once_with(node.policy.return_value)
    node.env.close.assert_called_once()


def test_close_releases_environment_when_hold_publication_fails(node):
    node.setup()
    node.env.step.side_effect = RuntimeError("Disconnected")
    with pytest.raises(RuntimeError, match="Disconnected"):
        node.on_close()
    node.env.close.assert_called_once()


def test_close_before_first_observation_releases_environment(node):
    node.on_close()
    node.env.observe.assert_not_called()
    node.env.close.assert_called_once()
