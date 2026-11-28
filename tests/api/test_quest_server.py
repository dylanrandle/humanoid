"""Exercise real listener shutdown and failed startup without using the default port."""

import asyncio
import socket
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientSession, WSMsgType

from humanoid.config.quest import get_quest_config
from humanoid.constants import EnvironmentVariable
from humanoid.hardware.teleop.quest.input import QuestInputState
from humanoid.hardware.teleop.quest.server import QuestServer
from humanoid.types.quest import QuestCameraSources, QuestVideoServices


def _video_services() -> QuestVideoServices:
    return QuestVideoServices(
        cameras=lambda: QuestCameraSources(generation=0, sources={}),
        offer=MagicMock(),
        remove=MagicMock(),
        metrics=dict,
    )


def test_server_closes_active_viewer_and_releases_listener(monkeypatch):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    monkeypatch.setenv(EnvironmentVariable.QUEST_PORT, str(port))
    input_state = QuestInputState(0.3)
    service = QuestServer(get_quest_config(), input_state, _video_services())
    service.start()

    async def connect_and_close():
        async with (
            ClientSession() as client,
            client.ws_connect(f"http://127.0.0.1:{port}/ws") as viewer,
        ):
            assert (await viewer.receive()).type is WSMsgType.BINARY
            assert (await viewer.receive()).type is WSMsgType.TEXT
            _, message = await asyncio.gather(asyncio.to_thread(service.close), viewer.receive())
            assert message.type is WSMsgType.CLOSE

    try:
        asyncio.run(connect_and_close())
        assert not service._thread.is_alive()
        assert input_state.snapshot().frame is None
        with socket.socket() as probe:
            assert probe.connect_ex(("127.0.0.1", port)) != 0
    finally:
        service.close()


def test_server_bind_failure_cleans_up_thread_and_event_loop(monkeypatch):
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        monkeypatch.setenv(EnvironmentVariable.QUEST_PORT, str(occupied.getsockname()[1]))
        service = QuestServer(get_quest_config(), QuestInputState(0.3), _video_services())
        with pytest.raises(RuntimeError, match="Could not start Quest server"):
            service.start()
        assert not service._thread.is_alive()
        assert service._loop is not None and service._loop.is_closed()
        service.close()
