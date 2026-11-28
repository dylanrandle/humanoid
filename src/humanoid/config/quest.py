"""Local Quest hosting and input defaults, shared by both hosts."""

import os
import re
from pathlib import Path

from humanoid.constants import EnvironmentVariable, Port
from humanoid.types.quest import QuestConfig
from humanoid.types.teleop import MAX_NETWORK_PORT

QUEST_ASSETS_ROOT = (Path(__file__).parents[1] / "ui" / "quest").resolve()


def get_quest_config(public_host: str | None = None) -> QuestConfig:
    host = (
        public_host if public_host is not None else os.getenv(EnvironmentVariable.PUBLIC_HOST, "")
    ).strip().lower() or None
    if host is not None and not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.local", host):
        raise ValueError(f"{EnvironmentVariable.PUBLIC_HOST} must be a hostname ending in .local")
    port = int(os.getenv(EnvironmentVariable.QUEST_PORT, str(Port.QUEST_SERVER)))
    if not 1 <= port <= MAX_NETWORK_PORT:
        raise ValueError("Quest port must be between 1 and 65535")
    return QuestConfig(
        public_host=host,
        port=port,
        input_timeout=0.3,
        poll_seconds=1 / 60,
        request_timeout=0.2,
        base_height=0.45,
        base_down_degrees=30,
        wrist_fraction=0.3,
        wrist_margin=0.03,
    )
