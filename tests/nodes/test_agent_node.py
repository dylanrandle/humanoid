from unittest.mock import Mock, call

import pytest

from humanoid.nodes.policy.agent import AgentPolicyNode


@pytest.fixture
def node(monkeypatch):
    monkeypatch.setattr("humanoid.nodes.policy.agent.RealtimeEnvironment", Mock())
    return AgentPolicyNode(Mock(), rate_hz=50)


def test_setup_initializes_environment(node):
    node.setup()
    node.env.reset.assert_called_once()
    node.policy.assert_not_called()


def test_each_step_passes_latest_observation_to_policy_and_publishes_action(node):
    first, second = Mock(), Mock()
    node.env.observe.side_effect = [first, second]
    node.step()
    node.policy.assert_called_with(first)
    node.env.step.assert_called_with(node.policy.return_value)
    node.step()
    node.policy.assert_called_with(second)
    assert node.env.step.call_args_list == [call(node.policy.return_value)] * len((first, second))


def test_close_resets_policy_and_publishes_hold_before_closing_environment(node):
    node.setup()
    node.on_close()
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
