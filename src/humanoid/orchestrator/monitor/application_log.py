"""Multiprocess-safe application logging for the operator console."""

import logging
import os
import sys
import threading
from collections import deque
from logging.handlers import QueueListener, RotatingFileHandler
from multiprocessing import get_context
from multiprocessing.queues import Queue
from pathlib import Path

from humanoid.config.application_logging import (
    APPLICATION_LOG_BACKUP_COUNT,
    APPLICATION_LOG_INITIAL_TAIL_BYTES,
    APPLICATION_LOG_MAX_BYTES,
    APPLICATION_LOG_RETAINED_ENTRIES,
)
from humanoid.config.process import PROCESS_START_METHOD
from humanoid.logger import create_log_formatter
from humanoid.types.logging import ApplicationLogEntry, ApplicationLogSnapshot


class _DashboardLogFilter(logging.Filter):
    """Keep status polling requests from crowding out runtime messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith("werkzeug") or record.levelno >= logging.WARNING


class _ApplicationLogBuffer(logging.Handler):
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity
        self._cursor = 0
        self._entries: deque[ApplicationLogEntry] = deque(maxlen=capacity)
        self._entries_lock = threading.RLock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return
        with self._entries_lock:
            self._append(message)

    def preload(self, messages: list[str]) -> None:
        with self._entries_lock:
            for message in messages[-self.capacity :]:
                self._append(message)

    def snapshot(self, after: int | None = None) -> ApplicationLogSnapshot:
        with self._entries_lock:
            entries = list(self._entries)
            oldest_cursor = entries[0].cursor if entries else self._cursor + 1
            reset = after is not None and (after > self._cursor or after < oldest_cursor - 1)
            if after is None or reset:
                selected = entries
            else:
                selected = [entry for entry in entries if entry.cursor > after]
            return ApplicationLogSnapshot(
                cursor=self._cursor,
                entries=selected,
                reset=reset,
                capacity=self.capacity,
            )

    def _append(self, message: str) -> None:
        self._cursor += 1
        self._entries.append(ApplicationLogEntry(cursor=self._cursor, message=message))


class ApplicationLogMonitor:
    """Collect parent and child logs through one rotating writer and memory buffer."""

    def __init__(
        self,
        path: str | Path,
        *,
        retained_entries: int = APPLICATION_LOG_RETAINED_ENTRIES,
        max_file_bytes: int = APPLICATION_LOG_MAX_BYTES,
        backup_count: int = APPLICATION_LOG_BACKUP_COUNT,
        initial_tail_bytes: int = APPLICATION_LOG_INITIAL_TAIL_BYTES,
    ):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _trim_existing_log(self.path, max_file_bytes)

        formatter = create_log_formatter()
        dashboard_filter = _DashboardLogFilter()
        self._buffer = _ApplicationLogBuffer(retained_entries)
        self._buffer.setFormatter(formatter)
        self._buffer.addFilter(dashboard_filter)
        self._buffer.preload(_read_log_tail(self.path, initial_tail_bytes, retained_entries))

        self._file_handler = RotatingFileHandler(
            self.path,
            maxBytes=max_file_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        self._file_handler.setFormatter(formatter)
        self._file_handler.addFilter(dashboard_filter)

        root_logger = logging.getLogger()
        level = root_logger.level
        self._buffer.setLevel(level)
        self._file_handler.setLevel(level)
        root_logger.addHandler(self._buffer)
        root_logger.addHandler(self._file_handler)

        self._child_console_handler = logging.StreamHandler(sys.stdout)
        self._child_console_handler.setLevel(level)
        self._child_console_handler.setFormatter(formatter)

        self.queue: Queue[logging.LogRecord] = get_context(PROCESS_START_METHOD).Queue()
        self._listener = QueueListener(
            self.queue,
            self._child_console_handler,
            self._file_handler,
            self._buffer,
            respect_handler_level=True,
        )
        self._listener.start()
        self._closed = False

    def snapshot(self, after: int | None = None) -> ApplicationLogSnapshot:
        return self._buffer.snapshot(after)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._listener.stop()
        self.queue.close()
        self.queue.join_thread()

        root_logger = logging.getLogger()
        root_logger.removeHandler(self._buffer)
        root_logger.removeHandler(self._file_handler)
        self._buffer.close()
        self._file_handler.close()
        self._child_console_handler.close()


def _read_log_tail(path: Path, max_bytes: int, max_lines: int) -> list[str]:
    try:
        with path.open("rb") as log_file:
            size = log_file.seek(0, os.SEEK_END)
            start = max(0, size - max_bytes)
            log_file.seek(start)
            content = log_file.read()
    except OSError:
        return []

    lines = content.decode(errors="replace").splitlines()
    if start > 0 and lines:
        lines[0] = f"…{lines[0]}"
    return lines[-max_lines:]


def _trim_existing_log(path: Path, max_bytes: int) -> None:
    try:
        size = path.stat().st_size
        if size <= max_bytes:
            return
        with path.open("rb") as log_file:
            log_file.seek(size - max_bytes)
            content = log_file.read()
        newline = content.find(b"\n")
        if newline >= 0:
            content = content[newline + 1 :]
        path.write_bytes(content)
    except OSError:
        return
