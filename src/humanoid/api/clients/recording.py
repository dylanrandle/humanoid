"""Nonblocking recording toggle and authoritative feedback for input nodes."""

import threading

from humanoid.api.clients.robot import RobotClient
from humanoid.config.recording import RECORDING_FEEDBACK_POLL_SECONDS
from humanoid.logger import get_logger
from humanoid.types.recording import RecordingStatus

logger = get_logger(__name__)


class RecordingClient:
    def __init__(self) -> None:
        self._toggle = threading.Event()
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._pending = False
        self._status: RecordingStatus | None = None
        self._worker = threading.Thread(target=self._run, daemon=True, name="recording-client")
        self._worker.start()

    def request(self) -> bool:
        with self._lock:
            if self._closed.is_set() or self._pending:
                return False
            self._pending = True
            self._toggle.set()
            return True

    def snapshot(self) -> RecordingStatus | None:
        with self._lock:
            return self._status

    def _run(self) -> None:
        client = RobotClient()
        while not self._closed.is_set():
            toggle = self._toggle.is_set()
            try:
                status = client.toggle_recording() if toggle else client.recording_status()
            except (ValueError, RuntimeError, KeyError):
                # A failed/ambiguous toggle is never retried automatically. Poll
                # the authoritative state, including changes from the dashboard.
                status = None
                if toggle:
                    logger.exception("API recording toggle failed")
            with self._lock:
                self._status = status
                if toggle:
                    self._pending = False
                    self._toggle.clear()
            self._toggle.wait(RECORDING_FEEDBACK_POLL_SECONDS)

    def close(self) -> None:
        self._closed.set()
        self._toggle.set()
        self._worker.join(timeout=4)
