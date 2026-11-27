import asyncio
import time
from contextlib import closing
from dataclasses import replace

import pytest
from aiortc import (
    RTCConfiguration,
    RTCIceGatherer,
    RTCPeerConnection,
    RTCRtpReceiver,
    RTCSessionDescription,
)

from humanoid.api.services.streaming import StreamingService
from humanoid.config.streaming import get_streaming_config
from humanoid.telemetry.streaming import StreamTelemetry
from humanoid.types.api import ServiceConflict
from humanoid.types.camera import CameraEncoding
from humanoid.types.streaming import StreamOffer
from tests.orchestrator.test_camera_preview import _frame


@pytest.mark.parametrize("codec", ["video/VP8", "video/H264"])
def test_webrtc_negotiates_delivers_frames_and_cleans_up(codec, monkeypatch):
    def reject_default_servers():
        pytest.fail("Local streaming must not use aiortc's default public STUN server")

    monkeypatch.setattr(
        RTCIceGatherer, "getDefaultIceServers", staticmethod(reject_default_servers)
    )
    telemetry = StreamTelemetry()
    with closing(StreamingService(telemetry)) as service:

        async def receive():
            peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
            transceiver = peer.addTransceiver("video", direction="recvonly")
            capabilities = RTCRtpReceiver.getCapabilities("video")
            assert capabilities is not None
            transceiver.setCodecPreferences(
                [item for item in capabilities.codecs if item.mimeType == codec]
            )
            frames = asyncio.Queue()

            @peer.on("track")
            def track_received(track):
                frames.put_nowait(track)

            try:
                await peer.setLocalDescription(await peer.createOffer())
                offer = StreamOffer(
                    camera="base",
                    stream="rgb",
                    depth_far=3,
                    sdp=peer.localDescription.sdp,
                    type="offer",
                )
                answer = await asyncio.to_thread(service.offer, offer, lambda: None)
                await peer.setRemoteDescription(RTCSessionDescription(answer.sdp, answer.type))
                track = await asyncio.wait_for(frames.get(), 5)
                frame = await asyncio.wait_for(track.recv(), 5)
                assert (frame.width, frame.height) == (320, 240)
                metrics = telemetry.snapshot()[answer.session_id]
                assert metrics.frames_sent > 0
                assert metrics.frame_age_seconds is None
                await asyncio.to_thread(service.remove, answer.session_id)
                assert not telemetry.snapshot()
            finally:
                await peer.close()

        asyncio.run(receive())


def test_invalid_negotiation_releases_viewer_slot():
    telemetry = StreamTelemetry()
    config = replace(get_streaming_config(), max_sessions=1)
    with closing(StreamingService(telemetry, config)) as service:
        offer = StreamOffer(camera="base", stream="depth", depth_far=3, sdp="invalid", type="offer")
        for _ in range(2):
            with pytest.raises(ValueError):
                service.offer(offer, lambda: None)
            assert not telemetry.snapshot()


def test_multiple_viewers_receive_rgb_and_depth_and_reset_releases_all_peers():

    telemetry = StreamTelemetry()
    with closing(StreamingService(telemetry)) as service:

        async def receive():
            peers = []
            try:

                async def viewer(stream):
                    peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
                    peers.append(peer)
                    peer.addTransceiver("video", direction="recvonly")
                    tracks = asyncio.Queue()
                    peer.on("track", tracks.put_nowait)
                    await peer.setLocalDescription(await peer.createOffer())
                    offer = StreamOffer(
                        camera="base",
                        stream=stream,
                        depth_far=3,
                        sdp=peer.localDescription.sdp,
                        type="offer",
                    )
                    encoding = CameraEncoding.RGB8 if stream == "rgb" else CameraEncoding.DEPTH16
                    frame = _frame(encoding)
                    started = time.perf_counter()
                    answer = await asyncio.to_thread(
                        service.offer, offer, lambda: replace(frame, timestamp=time.perf_counter())
                    )
                    await peer.setRemoteDescription(RTCSessionDescription(answer.sdp, answer.type))
                    track = await asyncio.wait_for(tracks.get(), 5)
                    decoded = await asyncio.wait_for(track.recv(), 5)
                    assert (decoded.width, decoded.height) == (4, 2)
                    assert decoded.to_ndarray(format="rgb24").max() > 0
                    return time.perf_counter() - started

                timings = await asyncio.gather(viewer("rgb"), viewer("depth"), viewer("rgb"))
                assert len(telemetry.snapshot()) == len(peers)
                assert all(
                    item.frame_age_seconds is not None for item in telemetry.snapshot().values()
                )
                print(
                    "WebRTC offer-to-first-frame seconds:", [round(value, 3) for value in timings]
                )
                await asyncio.to_thread(service.reset)
                assert not telemetry.snapshot()
                assert not service._peers and not service._tracks
            finally:
                await asyncio.gather(*(peer.close() for peer in peers))

        asyncio.run(receive())


def test_unconnected_offer_expires_without_dashboard_polling():

    telemetry = StreamTelemetry()
    config = replace(get_streaming_config(), idle_timeout=0.1)
    with closing(StreamingService(telemetry, config)) as service:

        async def offer_only():
            peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
            try:
                peer.addTransceiver("video", direction="recvonly")
                await peer.setLocalDescription(await peer.createOffer())
                offer = StreamOffer(
                    camera="base",
                    stream="rgb",
                    depth_far=3,
                    sdp=peer.localDescription.sdp,
                    type="offer",
                )
                await asyncio.to_thread(service.offer, offer, lambda: None)
                deadline = time.monotonic() + 3
                while telemetry.snapshot() and time.monotonic() < deadline:
                    await asyncio.sleep(0.02)
                assert not telemetry.snapshot()
            finally:
                await peer.close()

        asyncio.run(offer_only())


def test_reset_during_negotiation_is_a_retriable_conflict(monkeypatch):
    original = RTCPeerConnection.setRemoteDescription
    telemetry = StreamTelemetry()
    with closing(StreamingService(telemetry)) as service:

        async def reset_before_answer(peer, description):
            await original(peer, description)
            await service._reset()

        monkeypatch.setattr(RTCPeerConnection, "setRemoteDescription", reset_before_answer)

        async def negotiate():
            peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
            try:
                peer.addTransceiver("video", direction="recvonly")
                await peer.setLocalDescription(await peer.createOffer())
                offer = StreamOffer(
                    camera="base",
                    stream="rgb",
                    depth_far=3,
                    sdp=peer.localDescription.sdp,
                    type="offer",
                )
                with pytest.raises(ServiceConflict, match="reset during negotiation"):
                    await asyncio.to_thread(service.offer, offer, lambda: None)
                assert not telemetry.snapshot()
                assert not service._peers and not service._tracks
            finally:
                await peer.close()

        asyncio.run(negotiate())
