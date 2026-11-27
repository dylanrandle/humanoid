import json
import signal
import subprocess
import time
from contextlib import closing
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from humanoid.api.services.recording import RecordingService
from humanoid.config.recording import LCM_LOGGER_COMMAND
from humanoid.config.robot.panda import PANDA_CONFIG
from humanoid.constants import DEFAULT_LCM_URL
from humanoid.recording import RECORDING_MANIFEST_FILENAME
from humanoid.types.recording import RecordingOwner, RecordingState


@pytest.fixture
def recorder(tmp_path):
    with patch("humanoid.api.services.recording.subprocess.Popen") as popen:
        popen.return_value.poll.return_value = None
        popen.return_value.wait.return_value = 0
        with closing(RecordingService(tmp_path, robot_config=PANDA_CONFIG)) as service:
            yield service, popen


def test_recording_creates_bundle_and_flushes_exact_process(recorder):
    service, popen = recorder
    started = service.start()
    assert started.state is RecordingState.RUNNING
    assert popen.call_args.args[0] == [
        LCM_LOGGER_COMMAND,
        "-l",
        DEFAULT_LCM_URL,
        "-c",
        ".*",
        started.file_name,
    ]
    manifest = json.loads(
        Path(started.file_name).with_name(RECORDING_MANIFEST_FILENAME).read_text()
    )
    assert manifest["robot"] == PANDA_CONFIG.name
    stopped = service.stop()
    assert stopped.state is RecordingState.STOPPED
    assert stopped.file_name == started.file_name
    popen.return_value.send_signal.assert_called_once_with(signal.SIGINT)
    assert service.process is None


def test_episode_ownership_and_retries_cannot_replace_recording(recorder):
    service, popen = recorder
    owner = RecordingOwner("runtime", "episode")
    started = service.start(owner=owner)
    assert service.start(owner=owner) == started
    for other in (
        None,
        RecordingOwner("runtime", "other"),
        RecordingOwner("old-runtime", "episode"),
    ):
        with pytest.raises(RuntimeError):
            service.start(owner=other)
        with pytest.raises(RuntimeError, match="does not own"):
            service.stop(owner=other)
    stopped = service.stop(owner=owner)
    assert service.stop(owner=owner) == stopped
    with pytest.raises(RuntimeError, match="already finished"):
        service.start(owner=owner)
    popen.assert_called_once()


@pytest.mark.parametrize("exit_code", [7, -9])
def test_nonzero_or_forced_exit_preserves_failure(recorder, exit_code):
    service, popen = recorder
    started = service.start()
    if exit_code == -signal.SIGKILL:
        popen.return_value.wait.side_effect = [subprocess.TimeoutExpired("recorder", 2), -9]
    else:
        popen.return_value.wait.return_value = exit_code
    stopped = service.stop()
    assert stopped.state is RecordingState.FAILED
    assert stopped.file_name == started.file_name
    assert stopped.error
    assert service.stop() == stopped


def test_crash_is_detected_without_status_polling(recorder):
    service, popen = recorder
    service.start()
    popen.return_value.poll.return_value = 9
    deadline = time.monotonic() + 2
    while service.recording_status.state is RecordingState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.01)
    assert service.recording_status.state is RecordingState.FAILED
    assert service.process is None


@pytest.mark.parametrize("failure", [FileNotFoundError(), PermissionError("denied")])
def test_start_failure_is_reported(recorder, failure):
    service, popen = recorder
    popen.side_effect = failure
    status = service.start()
    assert status.state is RecordingState.FAILED
    assert status.error
    assert service.process is None


def test_failed_directory_creation_does_not_launch_recorder(recorder):
    service, popen = recorder
    service.recording_catalog.create = MagicMock(side_effect=PermissionError("read-only"))
    assert service.start().state is RecordingState.FAILED
    popen.assert_not_called()


def test_backend_finalization_closes_episode_owned_recording(recorder):
    service, popen = recorder
    service.start(owner=RecordingOwner("runtime", "episode"))
    assert service.finalize().state is RecordingState.STOPPED
    popen.return_value.send_signal.assert_called_once_with(signal.SIGINT)


def test_runtime_loss_finalizes_owned_recording_and_stop_retry_is_idempotent(recorder):
    service, popen = recorder
    owner = RecordingOwner("old-runtime", "episode")
    service.start(owner=owner)
    service.reconcile_runtime("old-runtime")
    popen.return_value.send_signal.assert_not_called()
    service.reconcile_runtime("new-runtime")
    assert service.snapshot().state is RecordingState.STOPPED
    popen.return_value.send_signal.assert_called_once_with(signal.SIGINT)
    assert service.stop(owner=owner).state is RecordingState.STOPPED
    service.start(owner=RecordingOwner("new-runtime", "other"))
    assert service.stop(owner=owner).state is RecordingState.STOPPED
    assert service.snapshot().state is RecordingState.RUNNING
