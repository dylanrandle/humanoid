"""Independent HTTP listener; robot actions run only on the policy thread."""

import json
import threading
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from humanoid.agent.serialization import encode_frames, encode_state
from humanoid.agent.service import AgentControlService
from humanoid.config.agent import AGENT_HOST
from humanoid.types.agent import (
    AgentRoute,
    CommandRequest,
    ControlOperation,
)
from humanoid.types.camera import DepthView
from humanoid.types.http import HttpKey


def _handler(service: AgentControlService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, format: str, *args) -> None:
            pass  # Keep command payloads out of request logs.

        def _serve(self, *, post: bool) -> None:
            try:
                host = self.headers.get("Host", "")
                origin = self.headers.get("Origin")
                if (
                    host.partition(":")[0].lower() not in {AGENT_HOST, "localhost"}
                    or origin not in (None, f"http://{host}")
                    or self.headers.get("Sec-Fetch-Site") in {"cross-site", "same-site"}
                ):
                    self._reply(HTTPStatus.FORBIDDEN, {"error": "A local request is required."})
                    return
                self._dispatch(post=post)
            except KeyError:
                self._reply(HTTPStatus.NOT_FOUND, {"error": "Unknown command ID."})
            except (ValueError, TypeError, OverflowError) as error:
                self._reply(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except RuntimeError as error:
                self._reply(HTTPStatus.CONFLICT, {"error": str(error)})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                return

        def _dispatch(self, *, post: bool) -> None:
            url = urlsplit(self.path)
            path = url.path
            if not post:
                self._get(path, url.query)
                return
            payload = self._body()
            if path == AgentRoute.COMMANDS:
                required = {
                    "session_id",
                    "command_id",
                    "operation",
                    "arguments",
                    "expires_in_seconds",
                }
                if (
                    set(payload) != required
                    or not isinstance(payload["command_id"], str)
                    or not isinstance(payload["arguments"], dict)
                ):
                    raise ValueError(
                        "Expected session_id, command_id, operation, arguments, "
                        "and expires_in_seconds."
                    )
                if payload["session_id"] != service.session_id:
                    raise RuntimeError("Control session has expired.")
                result = service.submit(
                    CommandRequest(
                        payload["command_id"],
                        ControlOperation(payload["operation"]),
                        payload["arguments"],
                        payload["expires_in_seconds"],
                    )
                )
                self._reply(HTTPStatus.ACCEPTED, asdict(result))
            elif (
                not payload and (command_id := AgentRoute.CANCEL.match_command_id(path)) is not None
            ):
                self._reply(HTTPStatus.OK, asdict(service.cancel(command_id)))
            else:
                self._reply(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})

        def _get(self, path: str, query_string: str) -> None:
            if path == AgentRoute.STATE:
                observation = service.observation()
                self._reply(
                    HTTPStatus.OK,
                    {
                        "session_id": service.session_id,
                        "state": encode_state(observation.robot_state),
                        "mode": observation.mode,
                    },
                )
            elif path == AgentRoute.CAPTURE:
                captures = service.capture()
                self._reply(
                    HTTPStatus.OK,
                    {
                        "session_id": service.session_id,
                        "views": {
                            name: {
                                "state": encode_state(obs.robot_state),
                                "mode": obs.mode,
                                "camera": encode_frames(frames),
                            }
                            for name, (obs, frames) in captures.items()
                        },
                    },
                )
            elif path == AgentRoute.CAMERA:
                query = parse_qs(
                    query_string, keep_blank_values=True, strict_parsing=True, max_num_fields=1
                )
                if query.keys() - {HttpKey.DEPTH_VIEW}:
                    raise ValueError(f"Expected a {HttpKey.DEPTH_VIEW} camera selector.")
                view = DepthView(query.get(HttpKey.DEPTH_VIEW, [DepthView.ALIGNED])[0])
                frames = service.camera_frames(depth_view=view)
                self._reply(HTTPStatus.OK, encode_frames(frames, depth_view=view))
            elif (command_id := AgentRoute.COMMAND.match_command_id(path)) is not None:
                self._reply(
                    HTTPStatus.OK,
                    asdict(service.status(command_id)),
                )
            else:
                self._reply(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})

        def _body(self) -> dict:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Expected application/json.")
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= service.config.max_request_bytes:
                raise ValueError("Invalid request size.")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            return payload

        def _reply(self, status: HTTPStatus, payload: dict) -> None:
            encoded = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            self._serve(post=False)

        def do_POST(self) -> None:
            self._serve(post=True)

    return Handler


class AgentControlServer:
    def __init__(self, service: AgentControlService) -> None:
        self.service = service
        self.server = ThreadingHTTPServer((AGENT_HOST, service.config.port), _handler(service))
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.service.close()
        if self.thread.is_alive():
            self.server.shutdown()
            self.thread.join(timeout=2)
        self.server.server_close()
