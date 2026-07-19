import shutil
import signal
import subprocess
import time
from dataclasses import replace
from unittest.mock import ANY, MagicMock

import lcm
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.constants import DEFAULT_LCM_URL, Topic
from humanoid.orchestrator.constants import (
    LCM_LOGPLAYER_COMMAND,
    REPLAY_CHANNEL_PATTERN,
    REPLAY_STDERR_TAIL_BYTES,
    REPLAY_STOP_TIMEOUT_SECONDS,
)
from humanoid.orchestrator.replay import ReplayManager, ReplayManagerError
from humanoid.recording import RecordingCatalog
from humanoid.types.replay import RecordingBundle, ReplayOutcome
from humanoid.types.robot import RobotName


def _recording(
    tmp_path,
    *,
    robot: RobotName = RobotName.PANDA,
) -> RecordingBundle:
    recording = RecordingCatalog(tmp_path).create(ROBOT_CONFIGS[robot])
    recording.log_path.write_bytes(b"log")
    return recording


def _running_process() -> MagicMock:
    process = MagicMock(spec=subprocess.Popen)
    process.poll.return_value = None
    return process


def test_start_runs_logplayer_on_managed_lcm_url(monkeypatch, tmp_path):
    recording = _recording(tmp_path)
    process = _running_process()
    popen = MagicMock(return_value=process)
    monkeypatch.setattr("humanoid.orchestrator.replay.subprocess.Popen", popen)
    manager = ReplayManager()

    status = manager.start(recording)

    popen.assert_called_once_with(
        [
            LCM_LOGPLAYER_COMMAND,
            "-l",
            DEFAULT_LCM_URL,
            "-e",
            REPLAY_CHANNEL_PATTERN,
            str(recording.log_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=ANY,
    )
    assert popen.call_args.kwargs["stderr"] is not subprocess.DEVNULL
    assert status.running is True
    assert status.file_name == str(recording.log_path)
    assert status.outcome is None


def test_natural_completion_preserves_recording(monkeypatch, tmp_path):
    recording = _recording(tmp_path)
    process = _running_process()
    monkeypatch.setattr(
        "humanoid.orchestrator.replay.subprocess.Popen",
        MagicMock(return_value=process),
    )
    manager = ReplayManager()
    manager.start(recording)
    process.poll.return_value = 0

    status = manager.status()

    assert status.running is False
    assert status.outcome is ReplayOutcome.COMPLETED
    assert status.exit_code == 0
    assert recording.log_path.exists()
    assert manager.stop() == status


def test_stop_interrupts_player_without_reporting_completion(monkeypatch, tmp_path):
    recording = _recording(tmp_path)
    process = _running_process()
    process.wait.return_value = -signal.SIGINT
    monkeypatch.setattr(
        "humanoid.orchestrator.replay.subprocess.Popen",
        MagicMock(return_value=process),
    )
    manager = ReplayManager()
    manager.start(recording)

    status = manager.stop()

    process.send_signal.assert_called_once_with(signal.SIGINT)
    process.wait.assert_called_once_with(timeout=REPLAY_STOP_TIMEOUT_SECONDS)
    assert status.running is False
    assert status.outcome is ReplayOutcome.STOPPED
    assert recording.log_path.exists()


def test_stop_preserves_never_started_and_failed_states(monkeypatch, tmp_path):
    manager = ReplayManager()

    assert manager.stop().outcome is None

    process = _running_process()
    monkeypatch.setattr(
        "humanoid.orchestrator.replay.subprocess.Popen",
        MagicMock(return_value=process),
    )
    manager.start(_recording(tmp_path))
    process.poll.return_value = 2

    failed = manager.status()
    stopped = manager.stop()

    assert failed.outcome is ReplayOutcome.FAILED
    assert stopped == failed


def test_nonzero_exit_includes_bounded_stderr_tail(monkeypatch, tmp_path):
    process = _running_process()
    popen = MagicMock(return_value=process)
    monkeypatch.setattr("humanoid.orchestrator.replay.subprocess.Popen", popen)
    manager = ReplayManager()
    manager.start(_recording(tmp_path))
    stderr = popen.call_args.kwargs["stderr"]
    stderr.write(b"x" * (REPLAY_STDERR_TAIL_BYTES + 20) + b" corrupt log")
    process.poll.return_value = 4

    status = manager.status()

    assert status.outcome is ReplayOutcome.FAILED
    assert status.last_output is not None
    assert status.last_output.endswith("corrupt log")
    assert len(status.last_output.encode()) <= REPLAY_STDERR_TAIL_BYTES


def test_missing_logplayer_returns_clear_error_and_preserves_recording(monkeypatch, tmp_path):
    recording = _recording(tmp_path)
    monkeypatch.setattr(
        "humanoid.orchestrator.replay.subprocess.Popen",
        MagicMock(side_effect=FileNotFoundError),
    )
    manager = ReplayManager()

    with pytest.raises(ReplayManagerError, match="not found in PATH"):
        manager.start(recording)

    assert manager.status().outcome is ReplayOutcome.FAILED
    assert recording.log_path.exists()


def test_stderr_capture_failure_preserves_recording(monkeypatch, tmp_path):
    recording = _recording(tmp_path)
    monkeypatch.setattr(
        "humanoid.orchestrator.replay.tempfile.TemporaryFile",
        MagicMock(side_effect=OSError("temporary storage unavailable")),
    )
    manager = ReplayManager()

    with pytest.raises(ReplayManagerError, match="temporary storage unavailable"):
        manager.start(recording)

    assert manager.status().outcome is ReplayOutcome.FAILED
    assert recording.log_path.exists()


def test_validate_accepts_matching_config(tmp_path):
    recording = _recording(tmp_path)

    ReplayManager.validate(recording, ROBOT_CONFIGS[RobotName.PANDA])


def test_validate_rejects_missing_log(tmp_path):
    recording = RecordingCatalog(tmp_path).create(ROBOT_CONFIGS[RobotName.PANDA])

    with pytest.raises(ReplayManagerError, match="LCM log file is unavailable"):
        ReplayManager.validate(recording, ROBOT_CONFIGS[RobotName.PANDA])


def test_validate_rejects_recording_for_different_robot(tmp_path):
    recording = _recording(tmp_path, robot=RobotName.ELROBOT)

    with pytest.raises(ReplayManagerError, match="for elrobot, not panda"):
        ReplayManager.validate(recording, ROBOT_CONFIGS[RobotName.PANDA])


def test_validate_rejects_changed_config_for_same_robot(tmp_path):
    recording = _recording(tmp_path)
    changed_config = dict(recording.robot_config)
    changed_config["tool"] = "different"

    with pytest.raises(ReplayManagerError, match="configuration does not match"):
        ReplayManager.validate(
            replace(recording, robot_config=changed_config),
            ROBOT_CONFIGS[RobotName.PANDA],
        )


@pytest.mark.skipif(shutil.which(LCM_LOGPLAYER_COMMAND) is None, reason="lcm-logplayer unavailable")
def test_actual_logplayer_reaches_natural_completion(tmp_path):
    recording = RecordingCatalog(tmp_path).create(ROBOT_CONFIGS[RobotName.PANDA])
    event_log = lcm.EventLog(recording.log_path, "w", overwrite=True)
    event_log.write_event(0, Topic.ORCHESTRATOR_MODE.value, b"ignored")
    event_log.close()
    manager = ReplayManager(lcm_url="memq://")

    manager.start(recording)
    deadline = time.monotonic() + 2.0
    while (status := manager.status()).running and time.monotonic() < deadline:
        time.sleep(0.01)

    manager.close()
    assert status.outcome is ReplayOutcome.COMPLETED
    assert status.exit_code == 0
    assert recording.log_path.exists()
