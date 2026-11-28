"""Viewer ownership, controller protocol, and video lifecycle on the Quest event loop."""

import asyncio
import contextlib
import math
import secrets
import time
from collections.abc import AsyncIterator

import msgpack
from aiohttp import WSMsgType, web
from vuer.events import Set, Upsert
from vuer.schemas import Element, MotionControllers, Scene, WebRTCVideoPlane

from humanoid.config.camera import CAMERA_DISPLAY_CONFIG
from humanoid.hardware.teleop.quest.input import QuestInputState
from humanoid.types.api import ServiceConflict
from humanoid.types.quest import (
    QuestClockChallenge,
    QuestConfig,
    QuestGuard,
    QuestRoute,
    QuestStatus,
    QuestVideoAnswer,
    QuestVideoOffer,
    QuestVideoServices,
    QuestViewerStatus,
    VuerControllerEvent,
)
from humanoid.types.streaming import DisplayFrame, StreamOffer
from humanoid.utils.camera_rendering import compose_camera_views


class QuestSession:
    def __init__(
        self, config: QuestConfig, input_state: QuestInputState, video: QuestVideoServices
    ) -> None:
        self.config = config
        self.input_state = input_state
        self.video = video
        self._owner: str | None = None
        self._socket: web.WebSocketResponse | None = None
        self._streams: dict[str, str] = {}
        self._generation = -1
        self._video_generation = ""

    @property
    def connected(self) -> bool:
        return self._owner is not None

    def status(self) -> QuestStatus:
        snapshot = self.input_state.snapshot()
        return QuestStatus(
            connected=self.connected,
            tracking=snapshot.frame is not None,
            engaged=snapshot.frame.engaged if snapshot.frame else False,
            cameras=list(self.video.cameras().sources),
        )

    def _videos(self, owner: str, generation: str) -> list[Element]:
        primary = next(iter(self.video.cameras().sources), None)
        if primary is None:
            return []
        pitch = -math.radians(self.config.base_down_degrees)
        distance = 1.5
        # Vuer offsets the panel from [0, 0, -distance] in viewer coordinates.
        position = [0, distance * math.sin(pitch), distance * (1 - math.cos(pitch))]
        return [
            WebRTCVideoPlane(
                key="camera-composite",
                src=QuestRoute.VIDEO_OFFER.format(owner=owner, camera=primary)
                + f"?generation={generation}",
                iceServer=None,
                webRTCOptions={"iceServers": []},
                aspect=4 / 3,
                height=self.config.base_height,
                rotation=[pitch, 0, 0],
                distanceToCamera=distance,
                position=position,
            )
        ]

    async def offer(
        self, owner: str, camera: str, video_generation: str | None, payload: QuestVideoOffer
    ) -> QuestVideoAnswer:
        if owner != self._owner or self._socket is None or self._socket.closed:
            raise ServiceConflict("Quest session closed")
        if video_generation != self._video_generation:
            raise ServiceConflict("Video layout changed")
        cameras = self.video.cameras()
        sources = cameras.sources
        if not sources or camera != next(iter(sources)):
            raise LookupError("Camera unavailable")

        names = list(sources)[:2]
        secondary = names[1] if len(names) > 1 else None

        def source() -> DisplayFrame | None:
            return compose_camera_views(
                sources[camera](),
                sources[secondary]() if secondary is not None else None,
                inset_fraction=self.config.wrist_fraction,
                inset_margin=self.config.wrist_margin,
                labels=(camera.upper(), secondary.upper() if secondary is not None else ""),
                include_inset=secondary is not None,
            )

        offer = StreamOffer(
            camera=camera,
            stream="rgb",
            depth_far=CAMERA_DISPLAY_CONFIG.depth_ranges_meters[0],
            sdp=payload.sdp,
            type=payload.type,
        )
        generation = cameras.generation
        try:
            answer = await asyncio.to_thread(self.video.offer, offer, source)
        except (ValueError, RuntimeError) as exc:
            raise ServiceConflict("Camera stream unavailable; reconnecting") from exc
        if (
            owner != self._owner
            or generation != self.video.cameras().generation
            or video_generation != self._video_generation
            or self._socket is None
            or self._socket.closed
        ):
            await asyncio.to_thread(self.video.remove, answer.session_id)
            raise ServiceConflict("Camera runtime changed")
        old = self._streams.get(camera)
        self._streams[camera] = answer.session_id
        if old:
            await asyncio.to_thread(self.video.remove, old)
        return QuestVideoAnswer(sdp=answer.sdp, type=answer.type)

    async def _clear_streams(self) -> None:
        streams, self._streams = self._streams, {}
        for stream in streams.values():
            await asyncio.to_thread(self.video.remove, stream)

    @contextlib.asynccontextmanager
    async def connection(self, socket: web.WebSocketResponse) -> AsyncIterator[None]:
        """Reserve ownership before the HTTP upgrade, releasing it even if preparation fails."""
        if self.connected:
            raise ServiceConflict("Another Quest session is connected; close its tab first")
        self._owner, self._socket = secrets.token_urlsafe(24), socket
        self.input_state.clear()
        try:
            yield
        finally:
            self.input_state.clear()
            try:
                if socket.prepared:
                    await socket.close()
                await self._clear_streams()
            finally:
                self._owner, self._socket = None, None

    async def run(self, socket: web.WebSocketResponse) -> None:
        """Send the scene and process messages for the reserved viewer connection."""
        if socket is not self._socket or self._owner is None:
            raise ServiceConflict("Quest session closed")
        owner = self._owner
        self._generation = self.video.cameras().generation
        self._video_generation = secrets.token_hex(8)
        scene = Scene(
            bgChildren=[
                MotionControllers(stream=True, left=True, right=True, fps=60),
                *self._videos(owner, self._video_generation),
            ],
            grid=False,
            frameloop="always",
        )
        await socket.send_bytes(msgpack.packb(Set(scene).serialize(), use_bin_type=True))
        await self._receive(socket, owner)

    async def _receive(self, socket: web.WebSocketResponse, owner: str) -> None:
        nonce = secrets.token_hex(16)
        sent = time.monotonic()
        await socket.send_json(QuestClockChallenge(kind="clock", nonce=nonce))
        offset: float | None = None
        guard_capture: float | None = None
        maintenance = asyncio.create_task(self._maintain(socket, owner))
        try:
            async for message in socket:
                try:
                    if message.type is WSMsgType.TEXT:
                        guard = QuestGuard.model_validate_json(message.data)
                        if guard.kind == "clock" and guard.nonce == nonce and offset is None:
                            if time.monotonic() - sent > self.config.input_timeout:
                                break
                            # Lower bound: capture is never dated later than the client clock reply.
                            offset = sent - guard.client_ms / 1000
                        elif guard.kind == "frame" and offset is not None and guard.frame_ms:
                            guard_capture = guard.frame_ms / 1000 + offset
                    elif message.type is WSMsgType.BINARY:
                        payload = msgpack.unpackb(message.data, raw=False)
                        if (
                            not isinstance(payload, dict)
                            or payload.get("etype") != "CONTROLLER_MOVE"
                        ):
                            continue
                        event = VuerControllerEvent.model_validate(payload)
                        if offset is None or guard_capture is None:
                            self.input_state.clear()
                            continue
                        self.input_state.update(
                            event.value, min(event.ts / 1000 + offset, guard_capture)
                        )
                except (ValueError, TypeError, msgpack.UnpackException):
                    self.input_state.clear()
        finally:
            maintenance.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await maintenance

    async def _maintain(self, socket: web.WebSocketResponse, owner: str) -> None:
        retries = 0
        while not socket.closed:
            await asyncio.sleep(1)
            snapshot = self.input_state.snapshot()
            # Periodically remount only failed/reset video peers, retaining healthy streams.
            live = self.video.metrics()
            retries += 1
            cameras = self.video.cameras()
            generation = cameras.generation
            primary = next(iter(cameras.sources), None)
            expected = {primary} if primary is not None else set()
            unhealthy = expected != set(self._streams) or any(
                session not in live
                or live[session].connection_state in {"closed", "failed", "disconnected"}
                for session in self._streams.values()
            )
            if generation != self._generation or (unhealthy and retries % 5 == 0):
                changed = generation != self._generation
                self._generation = generation
                self._video_generation = secrets.token_hex(8)
                await self._clear_streams()
                videos = self._videos(owner, self._video_generation)
                event = (
                    Set(
                        Scene(
                            bgChildren=[
                                MotionControllers(stream=True, left=True, right=True, fps=60),
                                *videos,
                            ],
                            grid=False,
                            frameloop="always",
                        )
                    )
                    if changed
                    else Upsert(videos, to="bgChildren")
                )
                await socket.send_bytes(msgpack.packb(event.serialize(), use_bin_type=True))
            await socket.send_json(
                QuestViewerStatus(
                    kind="status",
                    tracking=snapshot.frame is not None,
                    engaged=snapshot.frame.engaged if snapshot.frame else False,
                    video=len(live),
                )
            )

    async def close(self) -> None:
        if self._socket is not None:
            await self._socket.close()
