"""Latest-frame storage with neutral rearming after any tracking interruption."""

import threading
import time

from humanoid.types.quest import ControllerFrame, QuestSnapshot


class QuestInputState:
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self._lock = threading.RLock()
        self._snapshot = QuestSnapshot()
        self._armed = False

    def clear(self) -> None:
        with self._lock:
            self._snapshot = QuestSnapshot(last_disengaged_monotonic=time.monotonic())
            self._armed = False

    def update(self, frame: ControllerFrame, captured: float) -> None:
        with self._lock:
            now = time.monotonic()
            previous = self._snapshot.captured_monotonic
            if not 0 <= now - captured <= self.timeout:
                self.clear()
                return
            if previous is not None and captured <= previous:
                return
            if previous is None or now - previous > self.timeout:
                self._armed = False
            if not frame.engaged:
                self._armed = True
                self._snapshot.last_disengaged_monotonic = captured
            if not self._armed:
                self.clear()
                return
            self._snapshot = QuestSnapshot(
                frame=frame,
                captured_monotonic=captured,
                last_disengaged_monotonic=self._snapshot.last_disengaged_monotonic,
            )

    def snapshot(self) -> QuestSnapshot:
        with self._lock:
            captured = self._snapshot.captured_monotonic
            if captured is not None and time.monotonic() - captured > self.timeout:
                self.clear()
            return self._snapshot.model_copy(deep=True)
