"""Unit tests for the DataLoggerNode."""

import signal
from unittest.mock import MagicMock, patch

from humanoid.constants import DEFAULT_LCM_URL, Topic
from humanoid.nodes.data_logger import DataLoggerNode
from humanoid.types.orchestrator import EventKind, OrchestratorEvent

# Constants for comparisons to adhere to Python Style Rules (no magic numbers in comparisons)
ZERO_CALLS = 0
ONE_CALL = 1
TWO_CALLS = 2
TEST_RATE_HZ = 20.0


def test_data_logger_node_init():
    """Test DataLoggerNode initialization."""
    with patch("humanoid.nodes.data_logger.Subscriber") as mock_sub_class:
        node = DataLoggerNode(
            log_dir="test_logs",
            channel_regex=".*STATE.*",
            lcm_url="udpm://239.255.76.67:7667",
            rate_hz=TEST_RATE_HZ,
        )

        assert node.log_dir == "test_logs"
        assert node.channel_regex == ".*STATE.*"
        assert node.lcm_url == "udpm://239.255.76.67:7667"
        assert node.rate_hz == TEST_RATE_HZ
        assert node.process is None
        mock_sub_class.assert_called_once_with(topics=[Topic.ORCHESTRATOR_EVENT])


def test_data_logger_node_start_logging():
    """Test receiving START_LOGGING starts the lcm-logger subprocess."""
    with (
        patch("humanoid.nodes.data_logger.Subscriber") as mock_sub_class,
        patch("subprocess.Popen") as mock_popen,
        patch("humanoid.nodes.data_logger.Path.mkdir") as mock_mkdir,
    ):
        mock_sub = MagicMock()
        mock_sub_class.return_value = mock_sub

        node = DataLoggerNode(log_dir="test_logs", channel_regex=".*")

        # Mock queue returning START_LOGGING event, then None
        mock_sub.receive.side_effect = [
            OrchestratorEvent(timestamp=123.45, kind=EventKind.START_LOGGING),
            None,
        ]

        node.step()

        # Check directory creation was called
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Check Popen was called to spawn lcm-logger
        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert "lcm-logger" in cmd
        assert "-i" in cmd
        assert "-l" in cmd
        assert DEFAULT_LCM_URL in cmd
        assert "-c" in cmd
        assert ".*" in cmd
        assert any("test_logs" in arg for arg in cmd)


def test_data_logger_node_stop_logging():
    """Test receiving STOP_LOGGING cleanly terminates the subprocess."""
    with (
        patch("humanoid.nodes.data_logger.Subscriber") as mock_sub_class,
        patch("subprocess.Popen"),
    ):
        mock_sub = MagicMock()
        mock_sub_class.return_value = mock_sub

        node = DataLoggerNode()

        # Simulate running process
        mock_process = MagicMock()
        node.process = mock_process

        # Mock queue returning STOP_LOGGING event, then None
        mock_sub.receive.side_effect = [
            OrchestratorEvent(timestamp=123.45, kind=EventKind.STOP_LOGGING),
            None,
        ]

        node.step()

        # Should cleanly send SIGINT and wait
        mock_process.send_signal.assert_called_once_with(signal.SIGINT)
        mock_process.wait.assert_called_once()
        assert node.process is None


def test_data_logger_node_on_close_cleans_up():
    """Test that closing the node stops logging and closes the subscriber."""
    with (
        patch("humanoid.nodes.data_logger.Subscriber") as mock_sub_class,
        patch("subprocess.Popen"),
    ):
        mock_sub = MagicMock()
        mock_sub_class.return_value = mock_sub

        node = DataLoggerNode()

        mock_process = MagicMock()
        node.process = mock_process

        node.on_close()

        # Should stop logging
        mock_process.send_signal.assert_called_once_with(signal.SIGINT)
        mock_process.wait.assert_called_once()
        assert node.process is None

        # Should close subscriber
        mock_sub.close.assert_called_once()
