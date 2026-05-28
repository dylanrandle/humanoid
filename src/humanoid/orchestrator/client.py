"""Client for sending events to the orchestrator.

Wraps the publisher so callers don't have to know which topics each event uses
or remember that some events (e.g., REQUEST_HOMING) require a parameter on a
dedicated topic in addition to the event itself.
"""

import time

import numpy as np

from humanoid.constants import Topic
from humanoid.middleware.publisher import Publisher
from humanoid.types.homing import HomingTarget
from humanoid.types.orchestrator import EventKind, OrchestratorEvent


class OrchestratorClient:
    """Publishes events to the orchestrator's event channel."""

    def __init__(self, publisher: Publisher | None = None):
        """If ``publisher`` is omitted a new one is created."""
        self.publisher = publisher or Publisher()

    def request_homing(self, target_position: np.ndarray) -> None:
        """Request a homing transition with the given target.

        Publishes the ``HomingTarget`` *before* the ``OrchestratorEvent`` so the
        homing node is most likely to have the target in hand when the
        orchestrator flips the mode to HOMING.
        """
        now = time.time()
        self.publisher.publish(
            HomingTarget(timestamp=now, target_position=target_position),
            topic=Topic.HOMING_TARGET,
        )
        self.publisher.publish(
            OrchestratorEvent(timestamp=now, kind=EventKind.REQUEST_HOMING),
            topic=Topic.ORCHESTRATOR_EVENT,
        )

    def request_oculus(self) -> None:
        self._send(EventKind.REQUEST_OCULUS)

    def request_keyboard(self) -> None:
        self._send(EventKind.REQUEST_KEYBOARD)

    def request_idle(self) -> None:
        self._send(EventKind.REQUEST_IDLE)

    def complete(self) -> None:
        """Signal that the current policy has finished (homing → return mode)."""
        self._send(EventKind.COMPLETE)

    def start_logging(self) -> None:
        self._send(EventKind.START_LOGGING)

    def stop_logging(self) -> None:
        self._send(EventKind.STOP_LOGGING)

    def _send(self, kind: EventKind) -> None:
        self.publisher.publish(
            OrchestratorEvent(timestamp=time.time(), kind=kind),
            topic=Topic.ORCHESTRATOR_EVENT,
        )
