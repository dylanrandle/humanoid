"""Interruptible lifecycle wrapper for the third-party Meta Quest reader."""

from __future__ import annotations

import socket
import threading
from contextlib import suppress
from typing import Any


class QuestReaderSession:
    """Own one reader and its blocking ADB logcat worker."""

    def __init__(self, reader: Any) -> None:
        self._reader = reader
        self._connection_lock = threading.Lock()
        self._connection: Any | None = None
        self._worker_error: Exception | None = None
        self._closed = False
        self._reader.running = True
        self._thread = threading.Thread(
            target=self._run,
            name="triskel-quest-logcat",
            daemon=True,
        )
        self._thread.start()

    def get_transformations_and_buttons(self) -> tuple[Any, Any]:
        if self._worker_error is not None:
            raise RuntimeError(
                "Meta Quest logcat worker stopped unexpectedly"
            ) from self._worker_error
        return self._reader.get_transformations_and_buttons()

    def close(self, timeout: float) -> bool:
        """Interrupt logcat and wait up to ``timeout`` seconds for its worker."""

        self._closed = True
        self._reader.running = False
        with self._connection_lock:
            connection = self._connection
        if connection is not None:
            self._interrupt(connection)
        self._thread.join(timeout=max(0.0, timeout))
        return not self._thread.is_alive()

    def _run(self) -> None:
        try:
            self._reader.device.shell("logcat -T 0", self._consume)
        except Exception as exc:
            if not self._closed:
                self._worker_error = exc

    def _consume(self, connection: Any) -> None:
        with self._connection_lock:
            self._connection = connection
            closed = self._closed
        if closed:
            self._interrupt(connection)
            return
        try:
            self._reader.read_logcat_by_line(connection)
        finally:
            with self._connection_lock:
                if self._connection is connection:
                    self._connection = None

    @staticmethod
    def _interrupt(connection: Any) -> None:
        connection_socket = getattr(connection, "socket", None)
        if connection_socket is not None:
            with suppress(OSError):
                connection_socket.shutdown(socket.SHUT_RDWR)
        with suppress(OSError):
            connection.close()
