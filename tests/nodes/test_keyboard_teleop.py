from unittest.mock import MagicMock, patch

from humanoid.nodes.policy.teleop.keyboard import KeyboardTeleopNode


def test_keyboard_node_run_stops_when_policy_quits():
    node = object.__new__(KeyboardTeleopNode)
    node.rate_hz = 100_000.0
    node.policy = MagicMock(running=True)
    node.setup = MagicMock()
    node.on_close = MagicMock()
    node.step = MagicMock(side_effect=lambda: setattr(node.policy, "running", False))

    reporter = MagicMock()
    with patch("humanoid.nodes.base.NodeRateReporter", return_value=reporter):
        node.run()

    node.step.assert_called_once_with()
    node.on_close.assert_called_once_with()
    reporter.start.assert_called_once_with()
    reporter.observe_iteration.assert_called_once_with()
