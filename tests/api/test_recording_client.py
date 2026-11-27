import threading
import time
from contextlib import closing
from unittest.mock import Mock

from humanoid.api.clients.recording import RecordingClient
from humanoid.types.recording import RecordingState, RecordingStatus


def wait_until(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_toggle_is_nonblocking_and_failed_toggle_does_not_change_confirmed_state(monkeypatch):
    stopped = RecordingStatus(0, RecordingState.STOPPED)
    api = Mock()
    api.recording_status.return_value = stopped
    entered, release = threading.Event(), threading.Event()

    def failed_toggle():
        entered.set()
        assert release.wait(2)
        raise RuntimeError("Recorder unavailable")

    api.toggle_recording.side_effect = failed_toggle
    monkeypatch.setattr("humanoid.api.clients.recording.RobotClient", lambda: api)
    monkeypatch.setattr("humanoid.api.clients.recording.RECORDING_FEEDBACK_POLL_SECONDS", 0.01)
    with closing(RecordingClient()) as client:
        wait_until(lambda: client.snapshot() == stopped)
        assert client.request()
        assert entered.wait(2)
        try:
            assert client.snapshot() == stopped
            assert not client.request(), "do not accumulate toggles while awaiting a response"
        finally:
            release.set()
        wait_until(lambda: api.recording_status.call_count > 1)
        assert client.snapshot() == stopped
        api.toggle_recording.assert_called_once()


def test_feedback_follows_dashboard_and_reports_unknown_when_disconnected(monkeypatch):
    api = Mock()
    api.recording_status.return_value = RecordingStatus(0, RecordingState.STOPPED)
    monkeypatch.setattr("humanoid.api.clients.recording.RobotClient", lambda: api)
    monkeypatch.setattr("humanoid.api.clients.recording.RECORDING_FEEDBACK_POLL_SECONDS", 0.01)
    with closing(RecordingClient()) as client:
        wait_until(lambda: client.snapshot() is not None)
        running = RecordingStatus(1, RecordingState.RUNNING)
        api.recording_status.return_value = running
        wait_until(lambda: client.snapshot() == running)
        api.recording_status.side_effect = RuntimeError("Disconnected")
        wait_until(lambda: client.snapshot() is None)
        api.recording_status.side_effect = None
        stopped = RecordingStatus(2, RecordingState.STOPPED)
        api.recording_status.return_value = stopped
        wait_until(lambda: client.snapshot() == stopped)
    assert not client.request()
