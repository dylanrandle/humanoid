"""Track requests and progress without allowing stale broadcasts to unlock controls."""

import time
from dataclasses import replace

from humanoid.constants import Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.types.collection import (
    CollectionAction,
    CollectionRequest,
    CollectionRunState,
    CollectionRunStatus,
)


class CollectionMonitor:
    def __init__(self, subscriber: Subscriber | None = None):
        self._subscriber = subscriber or Subscriber([Topic.COLLECTION_STATUS])
        self._status = CollectionRunStatus(time.time())
        self._pending: CollectionRequest | None = None
        self._last_sent = 0.0
        self._last_received = float("-inf")

    def snapshot(self) -> CollectionRunStatus:
        while (status := self._subscriber.receive(Topic.COLLECTION_STATUS)) is not None:
            if self._status.run_id and status.run_id != self._status.run_id:
                continue
            if status.timestamp < self._last_received:
                continue
            self._last_received = status.timestamp
            if self._status.state is CollectionRunState.STOPPING and status.running:
                status = replace(status, state=CollectionRunState.STOPPING)
            self._status = status
            if not status.running or (
                self._pending is not None
                and self._pending.action is CollectionAction.START
                and status.state is CollectionRunState.RUNNING
            ):
                self._pending = None
        return self._status

    def retry_request(self) -> CollectionRequest | None:
        """Retry idempotent lifecycle requests until the stack acknowledges them."""
        if self._pending is not None and time.monotonic() - self._last_sent >= 1.0:
            self._last_sent = time.monotonic()
            return self._pending
        return None

    def requested(self, request: CollectionRequest) -> None:
        settings = request.settings
        assert settings is not None and request.action is not CollectionAction.STOP
        self.reset()
        self._pending = request
        self._last_sent = time.monotonic()
        preparing = request.action is CollectionAction.PREPARE
        self._status = CollectionRunStatus(
            request.timestamp,
            CollectionRunState.PREPARING if preparing else CollectionRunState.STARTING,
            request.run_id,
            settings.task,
            episodes=0 if preparing else settings.episodes,
            max_attempts=0 if preparing else settings.max_attempts,
            seed=settings.seed,
            message="Loading the selected task scene."
            if preparing
            else "Waiting for the simulation stack to accept collection.",
        )

    def stop_requested(self) -> None:
        self._pending = CollectionRequest(time.time(), self._status.run_id, CollectionAction.STOP)
        self._last_sent = time.monotonic()
        self._status = replace(
            self._status,
            state=CollectionRunState.STOPPING,
            message="Stopping collection and finalizing the recording.",
        )

    def fail(self, message: str) -> None:
        self._pending = None
        self._status = replace(self._status, state=CollectionRunState.FAILED, message=message)

    def reset(self) -> None:
        self._pending = None
        self._last_received = float("-inf")
        while self._subscriber.receive(Topic.COLLECTION_STATUS) is not None:
            pass
        self._status = CollectionRunStatus(time.time())

    def close(self) -> None:
        self._subscriber.close()
