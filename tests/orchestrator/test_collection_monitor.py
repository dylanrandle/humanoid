from dataclasses import replace
from unittest.mock import MagicMock

from humanoid.config.collection import DEFAULT_COLLECTION_SETTINGS
from humanoid.orchestrator.monitor.collection import CollectionMonitor
from humanoid.types.collection import (
    CollectionAction,
    CollectionRequest,
    CollectionRunState,
    CollectionRunStatus,
)


def test_unacknowledged_requests_retry_with_same_identity(monkeypatch):
    subscriber = MagicMock()
    subscriber.receive.return_value = None
    clock = MagicMock(return_value=0)
    monkeypatch.setattr("humanoid.orchestrator.monitor.collection.time.monotonic", clock)
    monitor = CollectionMonitor(subscriber)
    monitor.requested(
        CollectionRequest(0, "same-run", CollectionAction.START, DEFAULT_COLLECTION_SETTINGS)
    )
    assert monitor.retry_request() is None
    clock.return_value = 2
    request = monitor.retry_request()
    assert request is not None and request.run_id == "same-run"
    subscriber.receive.side_effect = [
        CollectionRunStatus(2, CollectionRunState.RUNNING, "same-run"),
        None,
    ]
    monitor.snapshot()
    clock.return_value = 4
    assert monitor.retry_request() is None


def test_pending_request_ignores_old_runs_and_stop_remains_pending_until_terminal():
    subscriber = MagicMock()
    subscriber.receive.return_value = None
    monitor = CollectionMonitor(subscriber)
    monitor.requested(
        CollectionRequest(0, "new", CollectionAction.START, DEFAULT_COLLECTION_SETTINGS)
    )
    subscriber.receive.side_effect = [CollectionRunStatus(0, run_id="old"), None]
    assert monitor.snapshot().state is CollectionRunState.STARTING
    running = CollectionRunStatus(1, CollectionRunState.RUNNING, "new", attempts=1)
    subscriber.receive.side_effect = [running, None]
    assert monitor.snapshot().attempts == 1
    monitor.stop_requested()
    subscriber.receive.side_effect = [running, None]
    assert monitor.snapshot().state is CollectionRunState.STOPPING
    subscriber.receive.side_effect = [replace(running, state=CollectionRunState.ABORTED), None]
    assert not monitor.snapshot().running


def test_scene_preparation_remains_busy_until_scene_ready():
    subscriber = MagicMock()
    subscriber.receive.return_value = None
    monitor = CollectionMonitor(subscriber)
    monitor.requested(
        CollectionRequest(0, "scene", CollectionAction.PREPARE, DEFAULT_COLLECTION_SETTINGS)
    )
    assert monitor.snapshot().state is CollectionRunState.PREPARING
    assert monitor.snapshot().running
    subscriber.receive.side_effect = [
        CollectionRunStatus(
            0, CollectionRunState.IDLE, "scene", task="cleanup", phase="scene_ready"
        ),
        None,
    ]
    assert not monitor.snapshot().running
    assert monitor.retry_request() is None


def test_delayed_progress_cannot_reopen_a_completed_run():
    subscriber = MagicMock()
    subscriber.receive.return_value = None
    monitor = CollectionMonitor(subscriber)
    monitor.requested(
        CollectionRequest(1, "run", CollectionAction.START, DEFAULT_COLLECTION_SETTINGS)
    )
    completed = CollectionRunStatus(3, CollectionRunState.SUCCEEDED, "run", successes=1)
    subscriber.receive.side_effect = [completed, None]
    assert monitor.snapshot() == completed
    subscriber.receive.side_effect = [
        replace(completed, timestamp=2, state=CollectionRunState.RUNNING, successes=0),
        None,
    ]
    assert monitor.snapshot() == completed
