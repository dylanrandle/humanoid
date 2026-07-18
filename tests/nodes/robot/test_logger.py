import json
import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from humanoid.config import ROBOT_CONFIGS
from humanoid.constants import DEFAULT_LCM_URL, Topic
from humanoid.nodes.robot.logger import RobotLoggerNode
from humanoid.recording import RECORDING_MANIFEST_FILENAME
from humanoid.types.logging import LoggingState, LoggingStatus
from humanoid.types.orchestrator import EventKind, OrchestratorEvent
from humanoid.types.robot import RobotName

TEST_RATE_HZ = 20.0


def _node(tmp_path, **kwargs) -> RobotLoggerNode:
    return RobotLoggerNode(log_dir=str(tmp_path), **kwargs)


def _published_statuses(publisher: MagicMock) -> list[LoggingStatus]:
    return [call.args[0] for call in publisher.publish.call_args_list]


def test_robot_logger_init(tmp_path):
    with (
        patch("humanoid.nodes.robot.logger.Publisher"),
        patch("humanoid.nodes.robot.logger.Subscriber") as subscriber_class,
    ):
        node = _node(
            tmp_path,
            channel_regex=".*STATE.*",
            lcm_url="udpm://239.255.76.67:7667",
            rate_hz=TEST_RATE_HZ,
        )

    assert node.log_dir == str(tmp_path)
    assert node.channel_regex == ".*STATE.*"
    assert node.lcm_url == "udpm://239.255.76.67:7667"
    assert node.rate_hz == TEST_RATE_HZ
    assert node.process is None
    subscriber_class.assert_called_once_with(
        topics=[Topic.ORCHESTRATOR_EVENT],
        url="udpm://239.255.76.67:7667",
    )


def test_start_logging_creates_bundle_and_uses_exact_filename(tmp_path):
    with (
        patch("humanoid.nodes.robot.logger.Subscriber") as subscriber_class,
        patch("humanoid.nodes.robot.logger.Publisher") as publisher_class,
        patch("humanoid.nodes.robot.logger.subprocess.Popen") as popen,
    ):
        subscriber_class.return_value.receive.side_effect = [
            OrchestratorEvent(timestamp=123.45, kind=EventKind.START_LOGGING),
            None,
        ]
        popen.return_value.poll.return_value = None
        node = _node(
            tmp_path,
            channel_regex=".*",
            robot_config=ROBOT_CONFIGS[RobotName.PANDA],
        )

        node.step()

    statuses = _published_statuses(publisher_class.return_value)
    assert [status.state for status in statuses] == [
        LoggingState.STARTING,
        LoggingState.RUNNING,
    ]
    filename = statuses[-1].file_name
    assert filename is not None
    assert popen.call_args.args[0] == [
        "lcm-logger",
        "-l",
        DEFAULT_LCM_URL,
        "-c",
        ".*",
        filename,
    ]
    assert "-i" not in popen.call_args.args[0]
    manifest_path = Path(filename).parent / RECORDING_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    assert manifest["robot"] == RobotName.PANDA


def test_robot_logger_restarts_after_subprocess_crash(tmp_path):
    with (
        patch("humanoid.nodes.robot.logger.Subscriber"),
        patch("humanoid.nodes.robot.logger.Publisher"),
        patch("humanoid.nodes.robot.logger.subprocess.Popen") as popen,
    ):
        node = _node(tmp_path)
        crashed_process = MagicMock()
        crashed_process.poll.return_value = 1
        replacement_process = MagicMock()
        replacement_process.poll.return_value = None
        popen.return_value = replacement_process
        node.process = crashed_process

        node._start_logging()

    popen.assert_called_once()
    assert node.process is replacement_process


def test_stop_logging_reports_clean_exit(tmp_path):
    with (
        patch("humanoid.nodes.robot.logger.Subscriber"),
        patch("humanoid.nodes.robot.logger.Publisher") as publisher_class,
    ):
        node = _node(tmp_path)
        process = MagicMock()
        process.poll.return_value = None
        process.wait.return_value = 0
        node.process = process
        node.logging_status = LoggingStatus(
            timestamp=1.0,
            state=LoggingState.RUNNING,
            file_name="logs/recording/recording.lcm",
        )

        node._stop_logging()

    process.send_signal.assert_called_once_with(signal.SIGINT)
    assert node.process is None
    statuses = _published_statuses(publisher_class.return_value)
    assert [status.state for status in statuses] == [
        LoggingState.STOPPING,
        LoggingState.STOPPED,
    ]
    assert statuses[-1].file_name == "logs/recording/recording.lcm"


def test_stop_logging_reports_nonzero_exit_as_failure(tmp_path):
    with (
        patch("humanoid.nodes.robot.logger.Subscriber"),
        patch("humanoid.nodes.robot.logger.Publisher") as publisher_class,
    ):
        node = _node(tmp_path)
        process = MagicMock()
        process.wait.return_value = 7
        node.process = process
        node.logging_status = LoggingStatus(
            timestamp=1.0,
            state=LoggingState.RUNNING,
            file_name="logs/recording/recording.lcm",
        )

        node._stop_logging()

    status = _published_statuses(publisher_class.return_value)[-1]
    assert status.state is LoggingState.FAILED
    assert status.file_name == "logs/recording/recording.lcm"
    assert status.error == "lcm-logger exited with code 7 while stopping."


def test_stop_logging_reports_forced_termination_as_failure(tmp_path):
    with (
        patch("humanoid.nodes.robot.logger.Subscriber"),
        patch("humanoid.nodes.robot.logger.Publisher") as publisher_class,
    ):
        node = _node(tmp_path)
        process = MagicMock()
        process.wait.side_effect = [subprocess.TimeoutExpired("lcm-logger", 2.0), -9]
        node.process = process
        node.logging_status = LoggingStatus(
            timestamp=1.0,
            state=LoggingState.RUNNING,
            file_name="logs/recording/recording.lcm",
        )

        node._stop_logging()

    process.kill.assert_called_once_with()
    status = _published_statuses(publisher_class.return_value)[-1]
    assert status.state is LoggingState.FAILED
    assert status.file_name == "logs/recording/recording.lcm"
    assert status.error is not None
    assert "log may be incomplete" in status.error


def test_robot_logger_on_close_cleans_up(tmp_path):
    with (
        patch("humanoid.nodes.robot.logger.Subscriber") as subscriber_class,
        patch("humanoid.nodes.robot.logger.Publisher"),
    ):
        node = _node(tmp_path)
        process = MagicMock()
        process.wait.return_value = 0
        node.process = process

        node.on_close()

    process.send_signal.assert_called_once_with(signal.SIGINT)
    assert node.process is None
    subscriber_class.return_value.close.assert_called_once_with()


def test_robot_logger_reports_missing_binary(tmp_path):
    with (
        patch("humanoid.nodes.robot.logger.Subscriber"),
        patch("humanoid.nodes.robot.logger.Publisher") as publisher_class,
        patch("humanoid.nodes.robot.logger.subprocess.Popen", side_effect=FileNotFoundError),
    ):
        node = _node(tmp_path)
        node._start_logging()

    status = _published_statuses(publisher_class.return_value)[-1]
    assert status.state is LoggingState.FAILED
    assert status.error == "lcm-logger was not found in PATH."
    assert node.process is None


def test_robot_logger_reports_recording_directory_failure(tmp_path):
    with (
        patch("humanoid.nodes.robot.logger.Subscriber"),
        patch("humanoid.nodes.robot.logger.Publisher") as publisher_class,
    ):
        node = _node(tmp_path)
        with patch.object(
            node.recording_catalog,
            "create",
            side_effect=PermissionError("read-only filesystem"),
        ):
            node._start_logging()

    status = _published_statuses(publisher_class.return_value)[-1]
    assert status.state is LoggingState.FAILED
    assert status.error is not None
    assert "read-only filesystem" in status.error
    assert node.process is None


def test_robot_logger_reports_subprocess_crash(tmp_path):
    with (
        patch("humanoid.nodes.robot.logger.Subscriber") as subscriber_class,
        patch("humanoid.nodes.robot.logger.Publisher") as publisher_class,
        patch("humanoid.nodes.robot.logger.subprocess.Popen") as popen,
    ):
        subscriber_class.return_value.receive.return_value = None
        process = popen.return_value
        process.poll.return_value = None
        node = _node(tmp_path)
        node._start_logging()
        process.poll.return_value = 9

        node.step()

    status = _published_statuses(publisher_class.return_value)[-1]
    assert status.state is LoggingState.FAILED
    assert status.error == "lcm-logger exited unexpectedly with code 9."
    assert node.process is None
