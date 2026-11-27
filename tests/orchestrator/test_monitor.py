import queue
import threading
from typing import cast
from unittest.mock import MagicMock

from humanoid.constants import Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.orchestrator.monitor.mode import ModeMonitor
from humanoid.telemetry.node import NODE_RATE_SUBSCRIBER_QUEUE_SIZE, NodeRateMonitor
from humanoid.types.node import NodeRateSample
from humanoid.types.orchestrator import Mode, OrchestratorMode

TEST_TARGET_RATE_HZ = 100.0
TEST_HEALTHY_RATE_HZ = 95.0
TEST_CPU_PERCENT = 18.25
TEST_MEMORY_RSS_MB = 127.75


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
                cpu_percent=TEST_CPU_PERCENT,
                memory_rss_mb=TEST_MEMORY_RSS_MB,
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
    assert statuses[0].cpu_percent == round(TEST_CPU_PERCENT, 1)
    assert statuses[0].memory_rss_mb == round(TEST_MEMORY_RSS_MB, 1)
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
    monkeypatch.setattr("humanoid.telemetry.node.Subscriber", subscriber_factory)

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
