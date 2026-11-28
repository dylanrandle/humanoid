"""Quest input safety, local origin isolation, and actual WebSocket transport."""

import asyncio
import re
import threading
import time
from dataclasses import replace
from functools import partial
from http import HTTPStatus
from unittest.mock import MagicMock

import msgpack
import numpy as np
import pytest
from aiohttp import ClientSession, WSServerHandshakeError
from aiohttp.test_utils import BaseTestServer, TestServer
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from pydantic import ValidationError

from humanoid.api.server import create_app
from humanoid.api.services.streaming import StreamingService
from humanoid.config.quest import get_quest_config
from humanoid.constants import EnvironmentVariable, Port
from humanoid.hardware.teleop.quest.input import QuestInputState
from humanoid.hardware.teleop.quest.server import VUER_CLIENT
from humanoid.hardware.teleop.quest.server import create_app as create_quest_app
from humanoid.telemetry.streaming import StreamTelemetry
from humanoid.types.camera import CameraEncoding
from humanoid.types.quest import (
    ControllerFrame,
    ControllerState,
    QuestCameraSources,
    QuestVideoServices,
)
from humanoid.types.streaming import StreamAnswer
from humanoid.utils.camera_rendering import compose_camera_views
from tests.orchestrator.test_camera_preview import _frame


def _video_services(backend: MagicMock) -> QuestVideoServices:
    return QuestVideoServices(
        cameras=lambda: QuestCameraSources(
            generation=backend.camera_generation,
            sources={
                name: partial(camera.read_frame, "rgb")
                for name, camera in backend.camera_frame_services.items()
            },
        ),
        offer=backend.streaming_service.offer,
        remove=backend.streaming_service.remove,
        metrics=backend.stream_telemetry.snapshot,
    )


def controller_frame(*, grip: bool = False) -> ControllerFrame:
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    state = ControllerState(
        squeeze=grip, aButton=False, bButton=False, thumbstickValue=(0.25, -0.5)
    )
    return ControllerFrame(
        left=pose.flatten(order="F").tolist(),
        right=pose.flatten(order="F").tolist(),
        leftState=state,
        rightState=state,
    )


def test_mapping_and_release_between_policy_ticks():
    store = QuestInputState(0.3)
    now = time.monotonic() - 0.02
    store.update(controller_frame(grip=True), now)
    assert store.snapshot().frame is None  # A new connection cannot arm while held.
    store.update(controller_frame(), now + 0.001)
    store.update(controller_frame(grip=True), now + 0.002)
    snapshot = store.snapshot().policy_snapshot()
    np.testing.assert_array_equal(snapshot.transforms["r"][:3, 3], [1, 2, 3])
    assert snapshot.buttons["leftJS"] == (0.25, 0.5)
    assert snapshot.buttons["RG"] is True
    assert snapshot.last_disengaged_monotonic == now + 0.001


def test_stale_missing_and_reordered_frames_require_neutral_rearming():
    store = QuestInputState(0.3)
    now = time.monotonic() - 0.02
    store.update(controller_frame(), now)
    store.update(controller_frame(grip=True), now + 0.001)
    store.update(controller_frame(), now - 0.001)
    current = store.snapshot().frame
    assert current is not None and current.engaged
    store.update(controller_frame(grip=True), now - 1)
    assert store.snapshot().frame is None
    store.update(controller_frame(grip=True), now + 0.002)
    assert store.snapshot().frame is None
    store.update(controller_frame(), now + 0.003)
    store.clear()
    store.update(controller_frame(grip=True), now + 0.004)
    assert store.snapshot().frame is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 5.0])
def test_invalid_controller_pose_rejected(bad):
    payload = controller_frame().model_dump()
    payload["right"][0] = bad
    with pytest.raises(ValidationError):
        ControllerFrame.model_validate(payload)


def test_quest_composition_preserves_base_field_and_places_wrist_top_right():
    frame = _frame(CameraEncoding.RGB8)
    intrinsics = replace(frame.intrinsics, width=320, height=240)
    base = replace(
        frame, data=np.full((240, 320, 3), [255, 0, 0], dtype=np.uint8), intrinsics=intrinsics
    )
    wrist = replace(base, data=np.full((240, 320, 3), [0, 255, 0], dtype=np.uint8))
    config = get_quest_config()
    compose = partial(
        compose_camera_views,
        inset_fraction=config.wrist_fraction,
        inset_margin=config.wrist_margin,
        labels=("BASE CAMERA", "WRIST"),
        include_inset=True,
    )
    composite = compose(base, wrist)
    assert composite is not None
    np.testing.assert_array_equal(composite.data[120, 160], [255, 0, 0])
    np.testing.assert_array_equal(composite.data[30, 280], [0, 255, 0])
    np.testing.assert_array_equal(base.data[30, 280], [255, 0, 0])
    missing_wrist = compose(base, None)
    assert missing_wrist is not None
    np.testing.assert_array_equal(missing_wrist.data[60, 280], [0, 0, 0])
    assert compose(None, None) is None


def test_dashboard_public_origin_is_explicit_and_same_origin(monkeypatch):
    monkeypatch.setenv(EnvironmentVariable.PUBLIC_HOST, "triskel.local")
    backend = MagicMock()
    backend.status.return_value = {}
    client = create_app(backend).test_client()
    assert (
        client.get(
            "/api/status", headers={"Host": "triskel.local", "Origin": "https://triskel.local"}
        ).status_code
        == HTTPStatus.OK
    )
    for origin in ("https://evil.local", "http://triskel.local", "https://triskel.local:8443"):
        assert (
            client.get(
                "/api/status", headers={"Host": "triskel.local", "Origin": origin}
            ).status_code
            == HTTPStatus.FORBIDDEN
        )
    assert (
        client.get("/api/status", headers={"Host": "evil.local"}).status_code
        == HTTPStatus.FORBIDDEN
    )


async def _viewer_connected(client: ClientSession, server: BaseTestServer) -> bool:
    async with client.get(server.make_url("/status")) as response:
        assert response.status == HTTPStatus.OK
        return (await response.json())["connected"]


def test_vuer_websocket_clock_guard_ownership_and_disconnect():
    async def run():
        backend = MagicMock()
        backend.quest_input = QuestInputState(0.3)
        backend.camera_frame_services = {}
        backend.camera_generation = 0
        backend.stream_telemetry.snapshot.return_value = {}
        config = get_quest_config()
        app = create_quest_app(config, backend.quest_input, _video_services(backend))
        async with TestServer(app) as server:
            headers = {"Host": f"localhost:{config.port}"}
            async with ClientSession(headers=headers) as client:
                async with client.get(server.make_url("/ws")) as response:
                    assert response.status == HTTPStatus.BAD_REQUEST
                assert not await _viewer_connected(client, server)
                async with client.ws_connect(server.make_url("/ws")) as socket:
                    scene = msgpack.unpackb((await socket.receive()).data, raw=False)
                    assert scene["etype"] == "SET"
                    clock = await socket.receive_json()
                    await socket.send_json(
                        {"kind": "clock", "nonce": clock["nonce"], "client_ms": time.time() * 1000}
                    )
                    with pytest.raises(WSServerHandshakeError) as error:
                        await client.ws_connect(server.make_url("/ws"))
                    assert error.value.status == HTTPStatus.CONFLICT
                    for grip in (False, True):
                        await socket.send_json(
                            {
                                "kind": "frame",
                                "client_ms": time.time() * 1000,
                                "frame_ms": time.time() * 1000,
                            }
                        )
                        await socket.send_bytes(
                            msgpack.packb(
                                {
                                    "etype": "CONTROLLER_MOVE",
                                    "ts": time.time() * 1000,
                                    "value": controller_frame(grip=grip).model_dump(),
                                }
                            )
                        )
                        await asyncio.sleep(0.01)
                    current = backend.quest_input.snapshot().frame
                    assert current is not None and current.engaged
                    await socket.send_bytes(
                        msgpack.packb(
                            {"etype": "CONTROLLER_MOVE", "ts": time.time() * 1000, "value": {}}
                        )
                    )
                    await asyncio.sleep(0.01)
                    assert backend.quest_input.snapshot().frame is None
                await asyncio.sleep(0.01)
                assert not await _viewer_connected(client, server)
                assert backend.quest_input.snapshot().frame is None
                async with client.get(
                    server.make_url("/"), headers={"Origin": "https://evil.local"}
                ) as response:
                    assert response.status == HTTPStatus.FORBIDDEN

    asyncio.run(run())


@pytest.mark.parametrize("interruption", ["disconnect", "runtime_change"])
def test_video_negotiation_interrupted_by_viewer_or_runtime_releases_peer(interruption):
    async def run():
        backend = MagicMock()
        backend.quest_input = QuestInputState(0.3)
        backend.camera_frame_services = {"base": MagicMock()}
        backend.camera_generation = 0
        backend.stream_telemetry.snapshot.return_value = {}
        negotiating, finish = threading.Event(), threading.Event()

        def offer(_offer, _source):
            negotiating.set()
            if not finish.wait(5):
                raise TimeoutError("Test did not release negotiation")
            return StreamAnswer("late-peer", "answer", "answer")

        backend.streaming_service.offer.side_effect = offer
        config = get_quest_config()
        app = create_quest_app(config, backend.quest_input, _video_services(backend))
        async with TestServer(app) as server:
            headers = {"Host": f"localhost:{config.port}"}
            async with (
                ClientSession(headers=headers) as client,
                client.ws_connect(server.make_url("/ws")) as socket,
            ):
                scene = msgpack.unpackb((await socket.receive()).data, raw=False)
                video = next(
                    element
                    for element in scene["data"]["bgChildren"]
                    if element["tag"] == "WebRTCVideoPlane"
                )

                async def negotiate():
                    async with client.post(
                        server.make_url(video["src"]),
                        json={"sdp": "offer", "type": "offer"},
                    ) as response:
                        return response.status

                pending = asyncio.create_task(negotiate())
                try:
                    assert await asyncio.to_thread(negotiating.wait, 2)
                    if interruption == "disconnect":
                        await socket.close()
                        for _ in range(50):
                            if not await _viewer_connected(client, server):
                                break
                            await asyncio.sleep(0.01)
                        assert not await _viewer_connected(client, server)
                    else:
                        backend.camera_generation += 1
                finally:
                    finish.set()
                    status = await pending
                assert status == HTTPStatus.CONFLICT
                backend.streaming_service.remove.assert_called_once_with("late-peer")

    asyncio.run(run())


def test_quest_camera_signaling_delivers_both_views_and_releases_on_disconnect():
    async def run():
        backend = MagicMock()
        backend.quest_input = QuestInputState(0.3)
        backend.camera_generation = 0
        backend.stream_telemetry = StreamTelemetry()
        backend.streaming_service = StreamingService(backend.stream_telemetry)
        backend.camera_frame_services = {"base": MagicMock(), "wrist": MagicMock()}
        for source in backend.camera_frame_services.values():
            source.read_frame.return_value = None
        config = get_quest_config()
        app = create_quest_app(config, backend.quest_input, _video_services(backend))
        peers = []
        try:
            async with TestServer(app) as server:
                headers = {"Host": f"localhost:{config.port}"}
                async with ClientSession(headers=headers) as client:
                    async with client.ws_connect(server.make_url("/ws")) as socket:
                        scene = msgpack.unpackb((await socket.receive()).data, raw=False)
                        assert scene["etype"] == "SET"
                        assert await _viewer_connected(client, server)
                        videos = [
                            element
                            for element in scene["data"]["bgChildren"]
                            if element["tag"] == "WebRTCVideoPlane"
                        ]
                        assert len(videos) == 1  # One composite avoids overlapping media layers.
                        for props in videos:
                            assert props["webRTCOptions"] == {"iceServers": []}
                            peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
                            peers.append(peer)
                            peer.addTransceiver("video", direction="recvonly")
                            tracks = asyncio.Queue()
                            peer.on("track", tracks.put_nowait)
                            await peer.setLocalDescription(await peer.createOffer())
                            async with client.post(
                                server.make_url(props["src"]),
                                json={"sdp": peer.localDescription.sdp, "type": "offer"},
                            ) as response:
                                assert response.status == HTTPStatus.OK
                                answer = await response.json()
                            await peer.setRemoteDescription(RTCSessionDescription(**answer))
                            track = await asyncio.wait_for(tracks.get(), 5)
                            frame = await asyncio.wait_for(track.recv(), 5)
                            # A stopped source visibly reports unavailable, never a frozen image.
                            assert frame.to_ndarray(format="rgb24").max() > 0
                        assert videos[0]["rotation"] == pytest.approx([-np.pi / 6, 0, 0])
                        assert videos[0]["position"] == pytest.approx(
                            [0, -0.75, 1.5 * (1 - np.cos(np.pi / 6))]
                        )
                        assert len(backend.stream_telemetry.snapshot()) == len(videos)
                        async with client.post(
                            server.make_url(props["src"].split("?")[0] + "?generation=old"),
                            json={"sdp": "invalid", "type": "offer"},
                        ) as response:
                            assert response.status == HTTPStatus.CONFLICT
                    for _ in range(50):
                        if not await _viewer_connected(client, server):
                            break
                        await asyncio.sleep(0.02)
                    assert not backend.stream_telemetry.snapshot()
                    assert not await _viewer_connected(client, server)
        finally:
            await asyncio.gather(*(peer.close() for peer in peers))
            backend.streaming_service.close()

    asyncio.run(run())


def test_viewer_preserves_guard_and_serves_original_vuer_scripts():
    async def run():
        config = get_quest_config()
        app = create_quest_app(config, QuestInputState(0.3), _video_services(MagicMock()))
        async with TestServer(app) as server:
            headers = {"Host": f"localhost:{config.port}"}
            async with ClientSession(headers=headers) as client:
                async with client.get(server.make_url("/")) as response:
                    html = await response.text()
                    assert response.status == HTTPStatus.OK
                    assert "/quest-assets/guard.js" in html
                scripts = re.findall(r'(?:src|href)="(/assets/[^" ]+\.js)"', html)
                assert scripts
                for script in scripts:
                    async with client.get(server.make_url(script)) as response:
                        assert response.status == HTTPStatus.OK
                        assert (
                            await response.text() == (VUER_CLIENT / script.lstrip("/")).read_text()
                        )

    asyncio.run(run())


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("localhost:8012", "http://localhost:9876"),
        ("127.0.0.1:8012", "http://127.0.0.1:9876"),
        ("triskel.local:8443", f"https://triskel.local:{Port.DASHBOARD_GATEWAY}"),
    ],
)
def test_connection_page_links_to_dashboard_after_port_is_bound(monkeypatch, host, expected):
    monkeypatch.setenv(EnvironmentVariable.QUEST_PORT, "8012")
    monkeypatch.setenv(EnvironmentVariable.PUBLIC_HOST, "triskel.local")
    monkeypatch.setenv(EnvironmentVariable.API_URL, "http://127.0.0.1:8765")

    async def run():
        app = create_quest_app(
            get_quest_config(), QuestInputState(0.3), _video_services(MagicMock())
        )
        # The CLI learns the actual dashboard port after starting the Quest server.
        monkeypatch.setenv(EnvironmentVariable.API_URL, "http://127.0.0.1:9876")
        async with (
            TestServer(app) as server,
            ClientSession(headers={"Host": host}) as client,
            client.get(server.make_url("/connect")) as response,
        ):
            assert response.status == HTTPStatus.OK
            assert f'id="dashboard" href="{expected}"' in await response.text()

    asyncio.run(run())


def test_local_quest_port_does_not_imply_https(monkeypatch):
    monkeypatch.delenv(EnvironmentVariable.PUBLIC_HOST, raising=False)
    monkeypatch.setenv(EnvironmentVariable.QUEST_PORT, str(Port.QUEST_GATEWAY))
    monkeypatch.setenv(EnvironmentVariable.API_URL, "http://127.0.0.1:9876")

    async def run():
        config = get_quest_config()
        app = create_quest_app(config, QuestInputState(0.3), _video_services(MagicMock()))
        host = f"localhost:{config.port}"
        async with (
            TestServer(app) as server,
            ClientSession(headers={"Host": host, "Origin": f"http://{host}"}) as client,
        ):
            async with client.get(server.make_url("/connect")) as response:
                assert response.status == HTTPStatus.OK
                assert 'id="dashboard" href="http://localhost:9876"' in await response.text()
            async with client.get(
                server.make_url("/status"), headers={"Origin": f"https://{host}"}
            ) as response:
                assert response.status == HTTPStatus.FORBIDDEN

    asyncio.run(run())


@pytest.mark.parametrize(
    "names", [("front", "tool"), ("tool", "front"), ("head",), ("front", "tool", "rear")]
)
def test_viewer_composes_robot_cameras_in_configured_order(names):
    async def run():
        backend = MagicMock()
        backend.quest_input = QuestInputState(0.3)
        backend.camera_generation = 0
        backend.stream_telemetry.snapshot.return_value = {}
        backend.camera_frame_services = {}
        template = _frame(CameraEncoding.RGB8)
        intrinsics = replace(template.intrinsics, width=320, height=240)
        colors = ([255, 0, 0], [0, 255, 0], [0, 0, 255])
        for name, color in zip(names, colors, strict=False):
            reader = MagicMock()
            reader.read_frame.return_value = replace(
                template,
                data=np.full((240, 320, 3), color, dtype=np.uint8),
                intrinsics=intrinsics,
            )
            backend.camera_frame_services[name] = reader

        def offer(payload, source):
            assert payload.camera == names[0]
            frame = source()
            assert frame is not None
            np.testing.assert_array_equal(frame.data[120, 160], colors[0])
            np.testing.assert_array_equal(
                frame.data[30, 280], colors[1] if len(names) > 1 else colors[0]
            )
            return StreamAnswer("configured-cameras", "answer", "answer")

        backend.streaming_service.offer.side_effect = offer
        config = get_quest_config()
        app = create_quest_app(config, backend.quest_input, _video_services(backend))
        async with (
            TestServer(app) as server,
            ClientSession(headers={"Host": f"localhost:{config.port}"}) as client,
            client.ws_connect(server.make_url("/ws")) as socket,
        ):
            scene = msgpack.unpackb((await socket.receive()).data, raw=False)
            videos = [
                element
                for element in scene["data"]["bgChildren"]
                if element["tag"] == "WebRTCVideoPlane"
            ]
            assert len(videos) == 1
            async with client.post(
                server.make_url(videos[0]["src"]), json={"sdp": "offer", "type": "offer"}
            ) as response:
                assert response.status == HTTPStatus.OK
        backend.streaming_service.offer.assert_called_once()
        for name in names[:2]:
            backend.camera_frame_services[name].read_frame.assert_called_once_with("rgb")
        for name in names[2:]:
            backend.camera_frame_services[name].read_frame.assert_not_called()

    asyncio.run(run())
