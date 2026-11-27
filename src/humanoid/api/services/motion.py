"""Transport-independent API motion lifecycle backed by a private policy pipe."""

import queue
import threading
import time
import uuid
from concurrent.futures import Future, TimeoutError
from dataclasses import asdict
from multiprocessing.connection import Connection

from humanoid.config.motion import (
    MOTION_PIPE_CONFIG,
    MOTION_PROTOCOL_CONFIG,
    get_motion_service_config,
)
from humanoid.middleware.pipe import Pipe
from humanoid.motion.validation import motion_target
from humanoid.types.api import ServiceConflict
from humanoid.types.motion import (
    MotionEnvelope,
    MotionErrorKind,
    MotionMessageKind,
    MotionNotFound,
    MotionRequest,
    MotionServiceConfig,
    MotionStatus,
    MotionSubmission,
)


class MotionService:
    def __init__(self, config: MotionServiceConfig | None = None) -> None:
        self.config = config or get_motion_service_config()
        self._motion_session_id = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._pipe: Pipe[MotionEnvelope] | None = None
        self._worker: threading.Thread | None = None
        self._pending: dict[str, Future[MotionEnvelope]] = {}
        self._deadlines: dict[str, float] = {}
        self._ready = False

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready and self._pipe is not None and not self._pipe.closed.is_set()

    @property
    def motion_session_id(self) -> str:
        with self._lock:
            return self._motion_session_id

    def attach(self, connection: Connection, motion_session_id: str) -> None:
        self.detach()
        with self._lock:
            self._motion_session_id = motion_session_id
            self._deadlines.clear()
            pipe: Pipe[MotionEnvelope] = Pipe(connection, MotionEnvelope, MOTION_PIPE_CONFIG)
            self._pipe = pipe
            self._worker = threading.Thread(
                target=self._receive,
                args=(pipe, motion_session_id),
                daemon=True,
                name="motion-service",
            )
            self._worker.start()

    def require_session(self, motion_session_id: str) -> None:
        if motion_session_id != self.motion_session_id:
            raise ServiceConflict("Control session has expired.")

    def submit(self, request: MotionRequest, motion_session_id: str) -> MotionStatus:
        MotionSubmission.model_validate({**asdict(request), "motion_session_id": motion_session_id})
        if request.expires_in_seconds > self.config.max_start_delay_seconds:
            raise ValueError("Invalid motion start deadline.")
        motion_target(request.operation, request.arguments)
        with self._lock:
            self.require_session(motion_session_id)
            if (
                request.motion_id not in self._deadlines
                and len(self._deadlines) >= self.config.max_motions
            ):
                raise ServiceConflict("Motion history is full; start a new autonomy session.")
            deadline = self._deadlines.setdefault(
                request.motion_id, time.perf_counter() + request.expires_in_seconds
            )
        return self._request(
            MotionEnvelope(
                kind=MotionMessageKind.SUBMIT,
                motion_session_id=motion_session_id,
                correlation_id=uuid.uuid4().hex,
                request=request,
                deadline=deadline,
            )
        )

    def status(self, motion_id: str, motion_session_id: str) -> MotionStatus:
        self.require_session(motion_session_id)
        return self._request(
            MotionEnvelope(
                kind=MotionMessageKind.STATUS,
                motion_session_id=motion_session_id,
                correlation_id=uuid.uuid4().hex,
                motion_id=motion_id,
            )
        )

    def cancel(self, motion_id: str, motion_session_id: str) -> MotionStatus:
        self.require_session(motion_session_id)
        return self._request(
            MotionEnvelope(
                kind=MotionMessageKind.CANCEL,
                motion_session_id=motion_session_id,
                correlation_id=uuid.uuid4().hex,
                motion_id=motion_id,
            )
        )

    def _request(self, message: MotionEnvelope) -> MotionStatus:
        future: Future[MotionEnvelope] = Future()
        with self._lock:
            if self._pipe is None or self._pipe.closed.is_set():
                raise ServiceConflict("Start API before requesting motion.")
            self.require_session(message.motion_session_id)
            if len(self._pending) >= MOTION_PIPE_CONFIG.capacity:
                raise ServiceConflict("Too many pending motion requests; retry later.")
            self._pending[message.correlation_id] = future
            try:
                self._pipe.send(message)
            except RuntimeError as exc:
                self._pending.pop(message.correlation_id)
                raise ServiceConflict(str(exc)) from exc
            except Exception:
                self._pending.pop(message.correlation_id)
                raise
        try:
            reply = future.result(timeout=MOTION_PROTOCOL_CONFIG.request_timeout_seconds)
        except TimeoutError as exc:
            raise ServiceConflict(
                "Motion acknowledgement timed out; query the same motion ID before retrying."
            ) from exc
        finally:
            with self._lock:
                self._pending.pop(message.correlation_id, None)
        if reply.error:
            if reply.error_kind is MotionErrorKind.INVALID:
                raise ValueError(reply.error)
            if reply.error_kind is MotionErrorKind.MISSING:
                raise MotionNotFound(message.motion_id)
            raise ServiceConflict(reply.error)
        if reply.status is None:
            raise ServiceConflict("Executor returned no motion status.")
        return reply.status

    def _receive(self, pipe: Pipe[MotionEnvelope], motion_session_id: str) -> None:
        heartbeat_at = 0.0
        connected = False
        try:
            while not pipe.closed.is_set():
                now = time.monotonic()
                timeout = (
                    MOTION_PROTOCOL_CONFIG.contact_timeout_seconds
                    if connected
                    else MOTION_PROTOCOL_CONFIG.startup_timeout_seconds
                )
                if now - pipe.last_received > timeout:
                    break
                if now >= heartbeat_at and (connected or heartbeat_at == 0):
                    pipe.send(
                        MotionEnvelope(
                            kind=MotionMessageKind.PING,
                            motion_session_id=motion_session_id,
                            correlation_id="",
                        )
                    )
                    heartbeat_at = now + MOTION_PROTOCOL_CONFIG.heartbeat_seconds
                try:
                    reply = pipe.inbox.get(timeout=MOTION_PIPE_CONFIG.poll_seconds)
                except queue.Empty:
                    continue
                if reply.motion_session_id != motion_session_id:
                    break
                connected = True
                with self._lock:
                    self._ready = reply.ready
                    future = self._pending.get(reply.correlation_id)
                    if future is not None and not future.done():
                        future.set_result(reply)
        except (RuntimeError, ValueError):
            pass
        finally:
            pipe.close()
            with self._lock:
                if self._pipe is pipe:
                    self._motion_session_id = uuid.uuid4().hex
                    self._deadlines.clear()
                    for future in self._pending.values():
                        if not future.done():
                            future.set_exception(ServiceConflict("Motion executor disconnected."))

    def detach(self) -> None:
        with self._lock:
            pipe, worker = self._pipe, self._worker
        if pipe is not None:
            pipe.close()
        if worker is not None:
            worker.join(timeout=MOTION_PIPE_CONFIG.join_seconds)
        with self._lock:
            self._pipe = None
            self._ready = False
            self._worker = None
            self._motion_session_id = uuid.uuid4().hex
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ServiceConflict("Motion session ended."))
            self._pending.clear()
            self._deadlines.clear()

    def close(self) -> None:
        self.detach()
