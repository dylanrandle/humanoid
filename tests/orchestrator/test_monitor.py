import queue
import threading
from typing import cast
from unittest.mock import MagicMock

import pytest

from humanoid.constants import Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.orchestrator.monitor.logging import LoggingMonitor
from humanoid.orchestrator.monitor.mode import ModeMonitor
from humanoid.orchestrator.monitor.node import NODE_RATE_SUBSCRIBER_QUEUE_SIZE, NodeRateMonitor
from humanoid.types.logging import LoggingState, LoggingStatus
from humanoid.types.node import NodeRateSample
from humanoid.types.orchestrator import Mode, OrchestratorMode

TEST_TARGET_RATE_HZ = 100.0
TEST_HEALTHY_RATE_HZ = 95.0


class NodeRateSubscriberStub:
    """Block like Subscriber and signal once every initial sample was consumed."""

    def __init__(self, samples: list[NodeRateSample] | None = None):
        self._samples: queue.Queue[NodeRateSample] = queue.Queue()
        for sample in samples or []:
            self._samples.put_nowait(sample)
        self.drained = threading.Event()
        self.closed = False

    def receive(self, topic: Topic, timeout: int | None = None) -> NodeRateSample | None:
        assert topic is Topic.NODE_RATE
        try:
            if timeout is None:
                return self._samples.get()
            return self._samples.get(timeout=timeout / 1000)
        except queue.Empty:
            self.drained.set()
            return None

    def close(self) -> None:
        self.closed = True


def test_reports_latest_orchestrator_mode():
    subscriber = MagicMock(spec=Subscriber)
    subscriber.receive.return_value = OrchestratorMode(timestamp=0.0, mode=Mode.KEYBOARD)
    monitor = ModeMonitor(subscriber=subscriber)

    snapshot = monitor.snapshot()

    assert snapshot.connected is True
    assert snapshot.mode is Mode.KEYBOARD


def test_node_rate_monitor_compares_active_processes_to_target():
    subscriber = NodeRateSubscriberStub(
        [
            NodeRateSample(
                timestamp=1.0,
                node_name="HealthyNode",
                pid=101,
                target_rate_hz=TEST_TARGET_RATE_HZ,
                measured_rate_hz=TEST_HEALTHY_RATE_HZ,
            ),
            NodeRateSample(
                timestamp=1.0,
                node_name="SlowNode",
                pid=102,
                target_rate_hz=TEST_TARGET_RATE_HZ,
                measured_rate_hz=89.0,
            ),
        ]
    )
    monitor = NodeRateMonitor(
        subscriber=cast(Subscriber, subscriber),
        minimum_healthy_ratio=0.9,
        clock=lambda: 10.0,
    )
    assert subscriber.drained.wait(timeout=1.0)

    statuses = monitor.snapshot({"HealthyNode": 101, "SlowNode": 102, "MissingNode": 103})
    monitor.close()

    assert [status.healthy for status in statuses] == [True, False, False]
    assert statuses[0].measured_rate_hz == TEST_HEALTHY_RATE_HZ
    assert statuses[1].target_rate_hz == TEST_TARGET_RATE_HZ
    assert statuses[2].target_rate_hz is None


def test_node_rate_monitor_marks_stale_samples_unhealthy_and_drops_stopped_nodes():
    subscriber = NodeRateSubscriberStub(
        [
            NodeRateSample(
                timestamp=1.0,
                node_name="ExampleNode",
                pid=101,
                target_rate_hz=10.0,
                measured_rate_hz=10.0,
            ),
        ]
    )
    now = 10.0
    monitor = NodeRateMonitor(
        subscriber=cast(Subscriber, subscriber),
        max_age_seconds=2.5,
        clock=lambda: now,
    )
    assert subscriber.drained.wait(timeout=1.0)

    assert monitor.snapshot({"ExampleNode": 101})[0].healthy is True
    now = 13.0
    assert monitor.snapshot({"ExampleNode": 101})[0].healthy is False
    assert monitor.snapshot({}) == []
    monitor.close()


def test_node_rate_monitor_close_releases_subscriber():
    subscriber = NodeRateSubscriberStub()
    monitor = NodeRateMonitor(subscriber=cast(Subscriber, subscriber))

    monitor.close()

    assert subscriber.closed is True


def test_node_rate_monitor_uses_a_bounded_transport_queue(monkeypatch):
    subscriber = NodeRateSubscriberStub()
    subscriber_factory = MagicMock(return_value=subscriber)
    monkeypatch.setattr("humanoid.orchestrator.monitor.node.Subscriber", subscriber_factory)

    monitor = NodeRateMonitor()
    monitor.close()

    subscriber_factory.assert_called_once_with(
        topics=[Topic.NODE_RATE],
        queue_size=NODE_RATE_SUBSCRIBER_QUEUE_SIZE,
    )


def test_marks_orchestrator_disconnected_when_messages_become_stale(monkeypatch):
    expected_age_seconds = 3.0
    subscriber = MagicMock(spec=Subscriber)
    subscriber.receive.side_effect = [
        OrchestratorMode(timestamp=0.0, mode=Mode.IDLE),
        None,
    ]
    monotonic = MagicMock(side_effect=[10.0, 10.0, 13.0])
    monkeypatch.setattr("humanoid.orchestrator.monitor.mode.time.monotonic", monotonic)
    monitor = ModeMonitor(subscriber=subscriber, max_age_seconds=2.0)

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
    monitor = ModeMonitor(subscriber=subscriber)

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
    monkeypatch.setattr("humanoid.orchestrator.monitor.logging.time.time", lambda: 12.0)
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
    monkeypatch.setattr("humanoid.orchestrator.monitor.logging.time.monotonic", monotonic)
    monitor = LoggingMonitor(subscriber=subscriber, acknowledgement_timeout_seconds=2.0)

    getattr(monitor, request_method)()
    pending = monitor.snapshot()
    expired = monitor.snapshot()

    assert pending.state in {LoggingState.STARTING, LoggingState.STOPPING}
    assert expired.state is LoggingState.FAILED
    assert expired.error == f"Data logging did not acknowledge the {expected_action} request."
