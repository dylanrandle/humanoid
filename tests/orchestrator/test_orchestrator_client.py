"""Tests for OrchestratorClient."""

from unittest.mock import MagicMock

import numpy as np

from humanoid.constants import Topic
from humanoid.orchestrator.client import OrchestratorClient
from humanoid.types.homing import HomingTarget
from humanoid.types.orchestrator import EventKind, OrchestratorEvent


def _make_client() -> tuple[OrchestratorClient, MagicMock]:
    publisher = MagicMock()
    return OrchestratorClient(publisher=publisher), publisher


def test_request_homing_publishes_target_and_event():
    client, publisher = _make_client()
    target = np.array([0.1, 0.2, 0.3])

    client.request_homing(target)

    expected_calls = 2
    assert publisher.publish.call_count == expected_calls

    by_topic = {c.kwargs["topic"]: c.args[0] for c in publisher.publish.call_args_list}
    assert Topic.HOMING_TARGET in by_topic
    assert Topic.ORCHESTRATOR_EVENT in by_topic

    homing_target = by_topic[Topic.HOMING_TARGET]
    assert isinstance(homing_target, HomingTarget)
    np.testing.assert_allclose(homing_target.target_position, target)

    event = by_topic[Topic.ORCHESTRATOR_EVENT]
    assert isinstance(event, OrchestratorEvent)
    assert event.kind is EventKind.REQUEST_HOMING


def test_request_homing_publishes_target_before_event():
    """The homing target must be in flight before the event flips the mode,
    so the homing node has a target to act on when it activates."""
    client, publisher = _make_client()
    client.request_homing(np.array([0.0]))

    topics_in_order = [c.kwargs["topic"] for c in publisher.publish.call_args_list]
    assert topics_in_order == [Topic.HOMING_TARGET, Topic.ORCHESTRATOR_EVENT]


def test_request_oculus_publishes_single_event():
    client, publisher = _make_client()
    client.request_oculus()

    publisher.publish.assert_called_once()
    call = publisher.publish.call_args
    assert call.kwargs["topic"] is Topic.ORCHESTRATOR_EVENT
    assert call.args[0].kind is EventKind.REQUEST_OCULUS


def test_request_keyboard_publishes_single_event():
    client, publisher = _make_client()
    client.request_keyboard()

    call = publisher.publish.call_args
    assert call.args[0].kind is EventKind.REQUEST_KEYBOARD


def test_request_idle_publishes_single_event():
    client, publisher = _make_client()
    client.request_idle()

    call = publisher.publish.call_args
    assert call.args[0].kind is EventKind.REQUEST_IDLE


def test_complete_publishes_single_event():
    client, publisher = _make_client()
    client.complete()

    call = publisher.publish.call_args
    assert call.kwargs["topic"] is Topic.ORCHESTRATOR_EVENT
    assert call.args[0].kind is EventKind.COMPLETE


def test_start_logging_publishes_single_event():
    client, publisher = _make_client()
    client.start_logging()

    call = publisher.publish.call_args
    assert call.kwargs["topic"] is Topic.ORCHESTRATOR_EVENT
    assert call.args[0].kind is EventKind.START_LOGGING


def test_stop_logging_publishes_single_event():
    client, publisher = _make_client()
    client.stop_logging()

    call = publisher.publish.call_args
    assert call.kwargs["topic"] is Topic.ORCHESTRATOR_EVENT
    assert call.args[0].kind is EventKind.STOP_LOGGING
