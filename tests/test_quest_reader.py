"""Lifecycle tests for the interruptible Meta Quest reader session."""

import runpy
import socket
import threading
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
QUEST_READER_PATH = (
    REPO_ROOT / "ros_ws" / "src" / "triskel_operator" / "triskel_operator" / "quest_reader.py"
)
QuestReaderSession = runpy.run_path(str(QUEST_READER_PATH))["QuestReaderSession"]


class _Connection:
    def __init__(self) -> None:
        self.socket, self._peer = socket.socketpair()

    def close(self) -> None:
        self.socket.close()
        self._peer.close()


class _Device:
    def __init__(self, reader: "_BlockingReader") -> None:
        self._reader = reader

    def shell(self, command: str, handler: Any) -> None:
        assert command == "logcat -T 0"
        handler(self._reader.connection)


class _BlockingReader:
    def __init__(self) -> None:
        self.running = False
        self.entered_logcat = threading.Event()
        self.connection = _Connection()
        self.device = _Device(self)

    def read_logcat_by_line(self, connection: _Connection) -> None:
        self.entered_logcat.set()
        while self.running:
            try:
                if not connection.socket.recv(1):
                    return
            except OSError:
                return

    @staticmethod
    def get_transformations_and_buttons() -> tuple[dict, dict]:
        return {}, {}


def test_close_interrupts_a_blocked_logcat_reader():
    reader = _BlockingReader()
    session = QuestReaderSession(reader)
    assert reader.entered_logcat.wait(timeout=1.0)

    assert session.close(timeout=1.0) is True
    assert reader.running is False


def test_close_is_idempotent_after_worker_exits():
    reader = _BlockingReader()
    session = QuestReaderSession(reader)
    assert reader.entered_logcat.wait(timeout=1.0)

    assert session.close(timeout=1.0) is True
    assert session.close(timeout=0.0) is True
