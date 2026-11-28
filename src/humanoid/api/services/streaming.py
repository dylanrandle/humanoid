"""WebRTC signaling and video delivery hosted by the unified API backend."""

import asyncio
import threading
import time
import uuid
from collections.abc import Callable, Coroutine
from fractions import Fraction
from typing import Any

import numpy as np
from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.exceptions import InvalidStateError
from av import VideoFrame
from PIL import Image, ImageDraw

from humanoid.config.camera import CAMERA_DISPLAY_CONFIG
from humanoid.config.streaming import get_streaming_config
from humanoid.perception.depth import colorize_depth
from humanoid.telemetry.streaming import StreamTelemetry
from humanoid.types.api import ServiceConflict
from humanoid.types.camera import CameraImage
from humanoid.types.streaming import StreamAnswer, StreamingConfig, StreamOffer, VideoSource


class _CameraTrack(VideoStreamTrack):
    def __init__(
        self,
        source: VideoSource,
        offer: StreamOffer,
        config: StreamingConfig,
        report: Callable[[float | None], None],
    ) -> None:
        super().__init__()
        self.source = source
        self.offer = offer
        self.config = config
        self.report = report
        self._started = time.monotonic()
        self._next_frame = self._started

    async def recv(self) -> VideoFrame:
        await asyncio.sleep(max(0, self._next_frame - time.monotonic()))
        self._next_frame = time.monotonic() + 1 / self.config.fps
        frame = self.source()
        age = None if frame is None else max(0.0, time.perf_counter() - frame.timestamp)
        if frame is None:
            pixels = np.zeros((240, 320, 3), dtype=np.uint8)
        elif self.offer.stream == "depth":
            if not isinstance(frame, CameraImage):
                raise ValueError("Depth streaming requires a calibrated camera frame")
            pixels = colorize_depth(
                frame.data.astype(np.float32) * frame.depth_scale,
                CAMERA_DISPLAY_CONFIG.depth_near_meters,
                self.offer.depth_far,
            )
        else:
            pixels = frame.data
        image = Image.fromarray(pixels)
        if frame is None:
            ImageDraw.Draw(image).text((20, 105), "CAMERA UNAVAILABLE", fill="white")
        image.thumbnail((self.config.max_width, self.config.max_width))
        result = VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
        result.pts = int((time.monotonic() - self._started) * 90000)
        result.time_base = Fraction(1, 90000)
        self.report(age)
        return result


class StreamingService:
    def __init__(
        self,
        telemetry: StreamTelemetry,
        config: StreamingConfig | None = None,
    ) -> None:
        self.config = config or get_streaming_config()
        self.telemetry = telemetry
        self._loop = asyncio.new_event_loop()
        self._peers: dict[str, RTCPeerConnection] = {}
        self._tracks: dict[str, _CameraTrack] = {}
        self._expiry: dict[str, asyncio.TimerHandle] = {}
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="webrtc")
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _call[T](self, coroutine: Coroutine[Any, Any, T]) -> T:
        if self._closed:
            coroutine.close()
            raise ServiceConflict("Camera streaming service is closed.")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=self.config.negotiation_timeout)
        except TimeoutError as exc:
            future.cancel()
            raise ServiceConflict("Camera stream negotiation timed out.") from exc

    def offer(self, offer: StreamOffer, source: VideoSource) -> StreamAnswer:
        if offer.depth_far not in CAMERA_DISPLAY_CONFIG.depth_ranges_meters:
            raise ValueError("Unsupported depth preview range.")
        return self._call(self._offer(offer, source))

    async def _offer(
        self,
        offer: StreamOffer,
        source: VideoSource,
    ) -> StreamAnswer:
        if len(self._peers) >= self.config.max_sessions:
            raise ServiceConflict("Camera stream viewer limit reached.")
        session_id = uuid.uuid4().hex
        # Explicitly disable aiortc's default public STUN server; use local candidates only.
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self._peers[session_id] = peer
        self.telemetry.register(session_id, offer.camera, offer.stream)
        track = _CameraTrack(
            source, offer, self.config, lambda age: self.telemetry.frame(session_id, age)
        )
        self._tracks[session_id] = track

        def expire() -> None:
            task = asyncio.create_task(self._remove(session_id))
            self._cleanup_tasks.add(task)
            task.add_done_callback(self._cleanup_tasks.discard)

        self._expiry[session_id] = self._loop.call_later(self.config.idle_timeout, expire)

        @peer.on("connectionstatechange")
        async def state_changed() -> None:
            self.telemetry.connection(session_id, peer.connectionState)
            if peer.connectionState in {"closed", "failed"}:
                await self._remove(session_id)
            elif peer.connectionState == "connected":
                timer = self._expiry.pop(session_id, None)
                if timer is not None:
                    timer.cancel()

        @peer.on("iceconnectionstatechange")
        async def ice_changed() -> None:
            if peer.iceConnectionState in {"closed", "failed"}:
                await self._remove(session_id)

        try:
            await peer.setRemoteDescription(RTCSessionDescription(offer.sdp, offer.type))
            if session_id not in self._peers:
                raise ServiceConflict("Camera stream was reset during negotiation; reconnect.")
            peer.addTrack(track)
            await peer.setLocalDescription(await peer.createAnswer())
            if session_id not in self._peers:
                raise ServiceConflict("Camera stream was reset during negotiation; reconnect.")
            description = peer.localDescription
            assert description is not None
            return StreamAnswer(session_id, description.sdp, description.type)
        except InvalidStateError as exc:
            await self._remove(session_id)
            raise ServiceConflict("Camera stream closed during negotiation; reconnect.") from exc
        except BaseException:
            await self._remove(session_id)
            raise

    def remove(self, session_id: str) -> None:
        self._call(self._remove(session_id))

    async def _remove(self, session_id: str) -> None:
        peer = self._peers.pop(session_id, None)
        track = self._tracks.pop(session_id, None)
        timer = self._expiry.pop(session_id, None)
        if timer is not None:
            timer.cancel()
        if track is not None:
            track.stop()
        self.telemetry.remove(session_id)
        if peer is not None:
            await peer.close()

    async def _reset(self) -> None:
        await asyncio.gather(*(self._remove(session) for session in list(self._peers)))
        if self._cleanup_tasks:
            await asyncio.gather(*self._cleanup_tasks)

    def reset(self) -> None:
        self._call(self._reset())

    def close(self) -> None:
        if self._closed:
            return
        self.reset()
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=self.config.negotiation_timeout)
        self._loop.close()
