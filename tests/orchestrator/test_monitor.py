from unittest.mock import MagicMock

import pytest

from humanoid.constants import Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.orchestrator.monitor import LoggingMonitor, OrchestratorMonitor
from humanoid.types.logging import LoggingState, LoggingStatus
from humanoid.types.orchestrator import Mode, OrchestratorMode


def test_reports_latest_orchestrator_mode():
    subscriber = MagicMock(spec=Subscriber)
    subscriber.receive.return_value = OrchestratorMode(timestamp=0.0, mode=Mode.KEYBOARD)
    monitor = OrchestratorMonitor(subscriber=subscriber)

    snapshot = monitor.snapshot()

    assert snapshot.connected is True
    assert snapshot.mode is Mode.KEYBOARD


def test_marks_orchestrator_disconnected_when_messages_become_stale(monkeypatch):
    expected_age_seconds = 3.0
    subscriber = MagicMock(spec=Subscriber)
    subscriber.receive.side_effect = [
        OrchestratorMode(timestamp=0.0, mode=Mode.IDLE),
        None,
    ]
    monotonic = MagicMock(side_effect=[10.0, 10.0, 13.0])
    monkeypatch.setattr("humanoid.orchestrator.monitor.time.monotonic", monotonic)
    monitor = OrchestratorMonitor(subscriber=subscriber, max_age_seconds=2.0)

    connected = monitor.snapshot()
    stale = monitor.snapshot()

    assert connected.connected is True
    assert stale.connected is False
    assert stale.mode is None
    assert stale.age_seconds == expected_age_seconds


def test_reset_discards_queued_mode_and_close_releases_subscriber():
    subscriber = MagicMock(spec=Subscriber)
    subscriber.receive.side_effect = [
        OrchestratorMode(timestamp=0.0, mode=Mode.IDLE),
        None,
    ]
    monitor = OrchestratorMonitor(subscriber=subscriber)

    monitor.reset()
    monitor.close()

    assert subscriber.receive.call_args_list[0].args == (Topic.ORCHESTRATOR_MODE,)
    subscriber.close.assert_called_once_with()


def test_logging_monitor_reports_latest_logger_status():
    subscriber = MagicMock(spec=Subscriber)
    starting = LoggingStatus(timestamp=1.0, state=LoggingState.STARTING)
    running = LoggingStatus(
        timestamp=2.0,
        state=LoggingState.RUNNING,
        file_name="logs/lcmlog_20260101",
    )
    subscriber.receive.side_effect = [starting, running, None]
    monitor = LoggingMonitor(subscriber=subscriber)

    assert monitor.snapshot() == running


def test_logging_monitor_tracks_requests_failures_and_reset(monkeypatch):
    subscriber = MagicMock(spec=Subscriber)
    subscriber.receive.return_value = None
    monkeypatch.setattr("humanoid.orchestrator.monitor.time.time", lambda: 12.0)
    monitor = LoggingMonitor(subscriber=subscriber)

    monitor.start_requested()
    assert monitor.snapshot().state is LoggingState.STARTING

    monitor.stop_requested()
    assert monitor.snapshot().state is LoggingState.STOPPING

    monitor.fail("lcm-logger failed")
    failed = monitor.snapshot()
    assert failed.state is LoggingState.FAILED
    assert failed.error == "lcm-logger failed"

    monitor.reset()
    assert monitor.snapshot() == LoggingStatus(timestamp=12.0, state=LoggingState.STOPPED)


def test_logging_monitor_close_releases_subscriber():
    subscriber = MagicMock(spec=Subscriber)
    monitor = LoggingMonitor(subscriber=subscriber)

    monitor.close()

    subscriber.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("request_method", "expected_action"),
    [
        ("start_requested", "start"),
        ("stop_requested", "stop"),
    ],
)
def test_logging_monitor_expires_unacknowledged_requests(
    monkeypatch,
    request_method,
    expected_action,
):
    subscriber = MagicMock(spec=Subscriber)
    subscriber.receive.return_value = None
    monotonic = MagicMock(side_effect=[10.0, 11.0, 13.0])
    monkeypatch.setattr("humanoid.orchestrator.monitor.time.monotonic", monotonic)
    monitor = LoggingMonitor(subscriber=subscriber, acknowledgement_timeout_seconds=2.0)

    getattr(monitor, request_method)()
    pending = monitor.snapshot()
    expired = monitor.snapshot()

    assert pending.state in {LoggingState.STARTING, LoggingState.STOPPING}
    assert expired.state is LoggingState.FAILED
    assert expired.error == f"Data logging did not acknowledge the {expected_action} request."
