"""Unified API endpoint, server, and dashboard configuration."""

import os
from pathlib import Path

from humanoid.constants import EnvironmentVariable
from humanoid.types.api import ApiEndpoint

API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8765
DEFAULT_API_ENDPOINT = ApiEndpoint(f"http://{API_HOST}:{DEFAULT_API_PORT}")
STATIC_ROOT = (Path(__file__).parents[1] / "ui" / "static").resolve()
MAX_REQUEST_BYTES = 128 * 1024
BROWSER_OPEN_DELAY_SECONDS = 0.25
ALLOWED_HOSTS = frozenset({API_HOST, "localhost"})
JSON_CONTENT_TYPE = "application/json"
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; connect-src 'self'; img-src 'self' data:; media-src 'self' blob:; "
    "script-src 'self'; style-src 'self'; frame-ancestors 'none'"
)


def get_api_endpoint() -> ApiEndpoint:
    return ApiEndpoint(os.getenv(EnvironmentVariable.API_URL, DEFAULT_API_ENDPOINT.url))
