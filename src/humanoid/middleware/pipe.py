"""Bounded, typed JSON pipe I/O with background reader and writer workers."""

import queue
import threading
import time
from multiprocessing.connection import Connection

from pydantic import BaseModel

from humanoid.types.middleware import PipeConfig


class Pipe[T: BaseModel]:
    def __init__(self, connection: Connection, message_type: type[T], config: PipeConfig) -> None:
        self.connection = connection
        self.message_type = message_type
        self.config = config
        self.inbox: queue.Queue[T] = queue.Queue(config.capacity)
        self._outbox: queue.Queue[bytes] = queue.Queue(config.capacity)
        self.closed = threading.Event()
        self._close_lock = threading.Lock()
        self._connection_closed = False
        self.last_received = time.monotonic()
        self._reader = threading.Thread(target=self._read, daemon=True, name="pipe-read")
        self._writer = threading.Thread(target=self._write, daemon=True, name="pipe-write")
        self._reader.start()
        self._writer.start()

    def send(self, message: T) -> None:
        if self.closed.is_set():
            raise RuntimeError("Pipe connection is closed.")
        payload = message.model_dump_json().encode()
        if len(payload) > self.config.max_bytes:
            raise ValueError("Pipe message is too large.")
        try:
            self._outbox.put_nowait(payload)
        except queue.Full as exc:
            self.closed.set()
            raise RuntimeError("Pipe capacity exceeded; connection interrupted.") from exc

    def _read(self) -> None:
        try:
            while not self.closed.is_set():
                if not self.connection.poll(self.config.poll_seconds):
                    continue
                message = self.message_type.model_validate_json(
                    self.connection.recv_bytes(self.config.max_bytes)
                )
                self.inbox.put_nowait(message)
                self.last_received = time.monotonic()
        except (EOFError, OSError, ValueError, queue.Full):
            self.closed.set()

    def _write(self) -> None:
        try:
            while not self.closed.is_set():
                try:
                    payload = self._outbox.get(timeout=self.config.poll_seconds)
                except queue.Empty:
                    continue
                self.connection.send_bytes(payload)
        except (EOFError, OSError, ValueError):
            self.closed.set()

    def close(self) -> None:
        self.closed.set()
        # The service worker and its owner can close concurrently. Connection.close
        # is not atomic: a second close can hit a descriptor already reused by LCM.
        with self._close_lock:
            if not self._connection_closed:
                self.connection.close()
                self._connection_closed = True
        for worker in (self._reader, self._writer):
            if worker is not threading.current_thread():
                worker.join(timeout=self.config.join_seconds)
