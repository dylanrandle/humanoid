"""Control-mode tests for the orchestrator.

The orchestrator's step() does both event handling and topic forwarding. These
tests drive events through ``_handle_event`` directly to focus on transition
behavior, and use a full ``step()`` for the request/forward integration tests.
"""

from collections.abc import Iterable
from unittest.mock import Mock, patch

import pytest

from humanoid.constants import Topic
from humanoid.nodes.orchestrator import OrchestratorNode
from humanoid.types.orchestrator import (
    EventKind,
    Mode,
    OrchestratorEvent,
    OrchestratorMode,
)


def _make_node(mode: Mode = Mode.IDLE) -> OrchestratorNode:
    with (
        patch("humanoid.nodes.orchestrator.Publisher"),
        patch("humanoid.nodes.orchestrator.Subscriber"),
    ):
        return OrchestratorNode(mode=mode)


def _events_then_none(events: Iterable[OrchestratorEvent]):
    """Build a subscriber.receive side-effect that returns the given events on
    ORCHESTRATOR_EVENT and None for everything else (including subsequent calls
    after the events are exhausted)."""
    queue = list(events)

    def receive(topic, timeout=0):
        if topic == Topic.ORCHESTRATOR_EVENT and queue:
            return queue.pop(0)
        return None

    return receive


def _event(kind: EventKind) -> OrchestratorEvent:
    return OrchestratorEvent(timestamp=0.0, kind=kind)


@pytest.fixture
def node() -> OrchestratorNode:
    return _make_node()


class TestTransitions:
    def test_request_oculus_from_idle(self, node):
        node._handle_event(_event(EventKind.REQUEST_OCULUS))
        assert node.mode is Mode.OCULUS
        assert node.return_mode is None

    def test_request_keyboard_from_idle(self, node):
        node._handle_event(_event(EventKind.REQUEST_KEYBOARD))
        assert node.mode is Mode.KEYBOARD

    def test_request_idle_from_teleop(self, node):
        node.mode = Mode.OCULUS
        node._handle_event(_event(EventKind.REQUEST_IDLE))
        assert node.mode is Mode.IDLE

    def test_request_homing_from_teleop_saves_return_mode(self, node):
        node.mode = Mode.OCULUS
        node._handle_event(_event(EventKind.REQUEST_HOMING))

        assert node.mode is Mode.HOMING
        assert node.return_mode is Mode.OCULUS

    def test_request_homing_from_idle_leaves_return_mode_empty(self, node):
        node._handle_event(_event(EventKind.REQUEST_HOMING))
        assert node.mode is Mode.HOMING
        # No teleop to return to — COMPLETE should drop back to IDLE.
        assert node.return_mode is None

    def test_complete_from_homing_returns_to_saved_mode(self, node):
        node.mode = Mode.KEYBOARD
        node._handle_event(_event(EventKind.REQUEST_HOMING))
        assert node.mode is Mode.HOMING
        assert node.return_mode is Mode.KEYBOARD

        node._handle_event(_event(EventKind.COMPLETE))
        assert node.mode is Mode.KEYBOARD
        assert node.return_mode is None

    def test_complete_from_homing_with_no_return_falls_to_idle(self, node):
        node.mode = Mode.HOMING
        node.return_mode = None
        node._handle_event(_event(EventKind.COMPLETE))
        assert node.mode is Mode.IDLE

    def test_complete_with_no_saved_return_drops_to_idle(self, node):
        node.mode = Mode.OCULUS
        node.return_mode = None
        node._handle_event(_event(EventKind.COMPLETE))
        assert node.mode is Mode.IDLE

    def test_re_request_during_homing_preserves_return_mode(self, node):
        node.mode = Mode.OCULUS
        node._handle_event(_event(EventKind.REQUEST_HOMING))
        assert node.return_mode is Mode.OCULUS

        # Re-requesting while already homing should NOT overwrite the saved return.
        node._handle_event(_event(EventKind.REQUEST_HOMING))
        assert node.mode is Mode.HOMING
        assert node.return_mode is Mode.OCULUS

    def test_request_teleop_during_homing_clears_return_mode(self, node):
        node.mode = Mode.OCULUS
        node._handle_event(_event(EventKind.REQUEST_HOMING))
        assert node.return_mode is Mode.OCULUS

        node._handle_event(_event(EventKind.REQUEST_KEYBOARD))
        assert node.mode is Mode.KEYBOARD
        assert node.return_mode is None


class TestStepIntegration:
    def test_step_drains_event_queue_and_transitions(self, node):
        events = [_event(EventKind.REQUEST_HOMING)]
        node.subscriber.receive = Mock(side_effect=_events_then_none(events))

        node.step()

        # Mode is now HOMING after processing the event.
        assert node.mode is Mode.HOMING
        # The mode broadcast happened in step; HomingTarget is NOT the
        # orchestrator's job anymore.
        topics = [c.kwargs["topic"] for c in node.publisher.publish.call_args_list]
        assert Topic.ORCHESTRATOR_MODE in topics
        assert Topic.HOMING_TARGET not in topics

    def test_step_always_broadcasts_current_mode(self, node):
        node.subscriber.receive = Mock(return_value=None)
        node.step()
        mode_calls = [
            c
            for c in node.publisher.publish.call_args_list
            if c.kwargs["topic"] is Topic.ORCHESTRATOR_MODE
        ]
        assert len(mode_calls) == 1
        msg = mode_calls[0].args[0]
        assert isinstance(msg, OrchestratorMode)
        assert msg.mode is Mode.IDLE
