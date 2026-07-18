"""Shared operator-console constants."""

from enum import StrEnum
from pathlib import Path

STATIC_ROOT = Path(__file__).with_name("static").resolve()
MAX_REQUEST_BYTES = 16 * 1024
HOST = "127.0.0.1"
BROWSER_OPEN_DELAY_SECONDS = 0.25
ALLOWED_HOSTS = frozenset({HOST, "localhost"})
JSON_CONTENT_TYPE = "application/json"
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
    "script-src 'self'; style-src 'self'; frame-ancestors 'none'"
)


class ApiRoute(StrEnum):
    ROOT = "/api"
    STATUS = f"{ROOT}/status"
    RUNTIME = f"{ROOT}/runtime"
    ROBOT = f"{ROOT}/robot"
    PROCESSES = f"{ROOT}/processes/<name>/<action>"
    LOGGING = f"{ROOT}/logging/<action>"
    REPLAY = f"{ROOT}/replay/<action>"
    ORCHESTRATOR = f"{ROOT}/orchestrator"


class PayloadKey(StrEnum):
    RUNTIME = "runtime"
    ROBOT = "robot"
    MODE = "mode"
    PARAMETERS = "parameters"
    EXPECTED_RUNTIME = "expected_runtime"
    EXPECTED_ROBOT = "expected_robot"
    REAL_HARDWARE_ACKNOWLEDGED = "real_hardware_acknowledged"
    RECORDING = "recording"
