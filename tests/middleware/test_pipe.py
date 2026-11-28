import threading
import time
from contextlib import closing
from multiprocessing import get_context
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ConfigDict

from humanoid.middleware.pipe import Pipe
from humanoid.types.middleware import PipeConfig


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    value: int


@pytest.fixture
def config():
    return PipeConfig(capacity=2, max_bytes=64, poll_seconds=0.01, join_seconds=1)


def test_pipe_validates_typed_messages_in_both_directions_and_closes_workers(config):
    parent, child = get_context("spawn").Pipe()
    started = time.monotonic()
    with (
        closing(Pipe(parent, Message, config)) as left,
        closing(Pipe(child, Message, config)) as right,
    ):
        left.send(Message(value=1))
        right.send(Message(value=2))
        assert right.inbox.get(timeout=1) == Message(value=1)
        assert left.inbox.get(timeout=1) == Message(value=2)
        assert left.last_received >= started and right.last_received >= started
    for pipe in (left, right):
        assert pipe.closed.is_set()
        assert not pipe._reader.is_alive() and not pipe._writer.is_alive()
        with pytest.raises(RuntimeError, match="closed"):
            pipe.send(Message(value=3))


@pytest.mark.parametrize("payload", [b"not json", b'{"value":"wrong type"}', b"x" * 65])
def test_invalid_or_oversized_incoming_message_closes_pipe(config, payload):
    parent, child = get_context("spawn").Pipe()
    with closing(child), closing(Pipe(parent, Message, config)) as pipe:
        child.send_bytes(payload)
        assert pipe.closed.wait(1)
        assert pipe.inbox.empty()


def test_inbox_overflow_closes_pipe_without_dropping_accepted_messages(config):
    parent, child = get_context("spawn").Pipe()
    with closing(child), closing(Pipe(parent, Message, config)) as pipe:
        for value in range(config.capacity + 1):
            child.send_bytes(Message(value=value).model_dump_json().encode())
        assert pipe.closed.wait(1)
        assert [pipe.inbox.get_nowait().value for _ in range(config.capacity)] == [0, 1]


def test_outgoing_size_limit_rejects_message_before_enqueueing(config):
    parent, child = get_context("spawn").Pipe()
    with closing(child), closing(Pipe(parent, Message, config)) as pipe:
        with pytest.raises(ValueError, match="too large"):
            pipe.send(Message(value=10**100))
        assert not pipe.closed.is_set()
        pipe.send(Message(value=1))
        assert child.poll(1)
        assert Message.model_validate_json(child.recv_bytes()) == Message(value=1)


def test_peer_disconnect_closes_pipe(config):
    parent, child = get_context("spawn").Pipe()
    with closing(child), closing(Pipe(parent, Message, config)) as pipe:
        child.close()
        assert pipe.closed.wait(1)


def test_outbox_overflow_closes_pipe_without_blocking_caller(config):

    blocked, release = threading.Event(), threading.Event()
    connection = MagicMock()
    connection.poll.side_effect = lambda timeout: release.wait(timeout) and False

    def blocked_write(_payload):
        blocked.set()
        release.wait(3)

    connection.send_bytes.side_effect = blocked_write
    pipe = Pipe(connection, Message, config)
    message = Message(value=1)
    try:
        pipe.send(message)
        assert blocked.wait(1)
        for _ in range(config.capacity):
            pipe.send(message)
        with pytest.raises(RuntimeError, match="capacity exceeded"):
            pipe.send(message)
        assert pipe.closed.is_set()
    finally:
        release.set()
        pipe.close()
    assert not pipe._reader.is_alive() and not pipe._writer.is_alive()


def test_concurrent_close_releases_the_connection_only_once(config):
    entered, release, duplicate = threading.Event(), threading.Event(), threading.Event()
    connection = MagicMock()
    connection.poll.side_effect = lambda timeout: release.wait(timeout) and False

    def close():
        if entered.is_set():
            duplicate.set()
        entered.set()
        release.wait(timeout=5)

    connection.close.side_effect = close
    pipe = Pipe(connection, Message, config)
    first = threading.Thread(target=pipe.close, daemon=True)
    second = threading.Thread(target=pipe.close, daemon=True)
    first.start()
    try:
        assert entered.wait(timeout=5)
        second.start()
        assert not duplicate.wait(timeout=0.1)
    finally:
        release.set()
        first.join(timeout=5)
        if second.ident is not None:
            second.join(timeout=5)
        pipe.close()
    assert not first.is_alive() and not second.is_alive()
    connection.close.assert_called_once()
