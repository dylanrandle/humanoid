"""Types for managed node processes."""

import os
from dataclasses import dataclass
from enum import StrEnum


class Runtime(StrEnum):
    SIM = "sim"
    REAL = "real"

    @classmethod
    def from_environment(cls) -> "Runtime":
        """Parse a runtime, using the configured default when unset."""
        # Constants imports this enum, so resolve its environment settings lazily.
        from humanoid.constants import (  # noqa: PLC0415
            DEFAULT_HUMANOID_RUNTIME,
            RUNTIME_ENVIRONMENT_VARIABLE,
        )

        value = os.getenv(RUNTIME_ENVIRONMENT_VARIABLE)
        if value is None or not value.strip():
            return cls(DEFAULT_HUMANOID_RUNTIME)
        normalized = value.lower().strip()
        return cls(normalized)


class ProcessName(StrEnum):
    STACK = "stack"
    REPLAY = "replay"
    KEYBOARD = "keyboard"
    OCULUS = "oculus"


class ProcessAction(StrEnum):
    START = "start"
    STOP = "stop"


@dataclass(frozen=True)
class ProcessStatus:
    running: bool
    pid: int | None
    exit_code: int | None
    runtime: Runtime | None
    uptime_seconds: float | None
    last_output: str | None
