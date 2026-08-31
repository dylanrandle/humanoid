"""Dependency-free HTTP edge for the local ROS operator node."""

from __future__ import annotations

import json
import mimetypes
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MAX_REQUEST_BYTES = 16 * 1024
STATIC_FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.js": "app.js",
    "/styles.css": "styles.css",
}


class ApiError(Exception):
    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status


def make_server(
    host: str,
    port: int,
    static_root: Path,
    api: Callable[[str, dict[str, Any]], object],
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/api/status":
                self._api_response(path, {})
                return
            relative = STATIC_FILES.get(path)
            if relative is None:
                self._json({"ok": False, "error": "Not found."}, HTTPStatus.NOT_FOUND)
                return
            file_path = static_root / relative
            try:
                body = file_path.read_bytes()
            except OSError:
                self._json(
                    {"ok": False, "error": "Dashboard assets are unavailable."},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type", mimetypes.guess_type(file_path.name)[0] or "text/plain"
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; connect-src 'self'; script-src 'self'; "
                "style-src 'self'; img-src 'self' data:; frame-src http:; "
                "frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            if not path.startswith("/api/"):
                self._json({"ok": False, "error": "Not found."}, HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._json(
                    {"ok": False, "error": "A small JSON request body is required."},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            if self.headers.get_content_type() != "application/json":
                self._json(
                    {"ok": False, "error": "Content-Type must be application/json."},
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                )
                return
            try:
                payload = json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._json(
                    {"ok": False, "error": "Request body must be valid JSON."},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            if not isinstance(payload, dict):
                self._json(
                    {"ok": False, "error": "Request body must be a JSON object."},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            self._api_response(path, payload)

        def _api_response(self, path: str, payload: dict[str, Any]) -> None:
            try:
                result = api(path, payload)
                self._json({"ok": True, "status": result}, HTTPStatus.OK)
            except ApiError as exc:
                self._json({"ok": False, "error": str(exc)}, exc.status)
            except Exception:
                self._json(
                    {"ok": False, "error": "The ROS control request failed unexpectedly."},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def _json(self, value: object, status: HTTPStatus) -> None:
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

    return ThreadingHTTPServer((host, port), Handler)
