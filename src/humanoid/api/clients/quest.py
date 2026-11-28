"""Nonblocking local client for the backend's validated Quest input."""

import threading
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from humanoid.config.api import get_api_endpoint
from humanoid.config.quest import get_quest_config
from humanoid.types.api import ApiRoute
from humanoid.types.quest import QuestSnapshot
from humanoid.types.teleop import QuestInputSnapshot


class QuestClient:
    def __init__(self) -> None:
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._snapshot = QuestSnapshot()
        self._thread = threading.Thread(target=self._run, daemon=True, name="quest-input")
        self._thread.start()

    def _run(self) -> None:
        config = get_quest_config()
        url = get_api_endpoint().url.rstrip("/") + ApiRoute.QUEST_INPUT
        opener = build_opener(ProxyHandler({}))
        while not self._closed.is_set():
            try:
                with opener.open(url, timeout=config.request_timeout) as response:
                    snapshot = QuestSnapshot.model_validate_json(response.read(32768))
            except (OSError, URLError, ValueError):
                snapshot = QuestSnapshot()
            with self._lock:
                self._snapshot = snapshot
            self._closed.wait(config.poll_seconds)

    def get_snapshot(self) -> QuestInputSnapshot:
        with self._lock:
            return self._snapshot.policy_snapshot()

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=1)
