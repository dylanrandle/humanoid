"""Independent HTTP listener; robot actions run only on the policy thread."""

import hmac
import json
import threading
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from humanoid.config import ROBOT_CONFIG
from humanoid.config.remote_control import get_remote_control_policy_config
from humanoid.nodes.policy.remote_control import RemoteControlNode
from humanoid.policy.remote_control import RemoteControlPolicy
from humanoid.remote_control.serialization import encode_frames, encode_state
from humanoid.remote_control.service import RemoteControlService
from humanoid.robots.base import Robot
from humanoid.types.remote_api import CommandRequest, ControlOperation, RemoteControlServerConfig
from humanoid.types.remote_control import RemoteControlPolicyConfig
from humanoid.types.robot import RobotConfig


def _handler(service: RemoteControlService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, format: str, *args) -> None:
            pass  # Do not log credentials or command payloads.

        def _serve(self, *, post: bool) -> None:
            try:
                authorization = self.headers.get("Authorization", "")
                if not hmac.compare_digest(authorization, f"Bearer {service.config.token}"):
                    self._reply(HTTPStatus.UNAUTHORIZED, {"error": "Invalid control credential."})
                    return
                self._dispatch(post=post)
            except KeyError:
                self._reply(HTTPStatus.NOT_FOUND, {"error": "Unknown command ID."})
            except (ValueError, TypeError) as error:
                self._reply(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except RuntimeError as error:
                self._reply(HTTPStatus.CONFLICT, {"error": str(error)})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                return

        def _dispatch(self, *, post: bool) -> None:
            path = urlsplit(self.path).path
            if not post:
                if path == "/v1/state":
                    observation = service.observation()
                    self._reply(
                        HTTPStatus.OK,
                        {
                            "session_id": service.session_id,
                            "state": encode_state(observation.robot_state),
                            "mode": observation.mode,
                        },
                    )
                elif path == "/v1/camera":
                    frames = service.observation(camera=True).camera_frames
                    assert frames is not None
                    self._reply(HTTPStatus.OK, encode_frames(frames))
                elif path.startswith("/v1/commands/"):
                    self._reply(
                        HTTPStatus.OK,
                        asdict(service.status(path.removeprefix("/v1/commands/"))),
                    )
                else:
                    self._reply(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})
                return
            payload = self._body()
            if path == "/v1/commands":
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
                        "Expected command_id, operation, arguments, and expires_in_seconds."
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
            elif path.startswith("/v1/commands/") and path.endswith("/cancel") and not payload:
                command_id = path.removeprefix("/v1/commands/").removesuffix("/cancel")
                self._reply(HTTPStatus.OK, asdict(service.cancel(command_id)))
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


class RemoteControlServer:
    def __init__(self, service: RemoteControlService) -> None:
        if not service.config.token:
            raise ValueError("A control API token is required.")
        self.service = service
        self.server = ThreadingHTTPServer(
            (service.config.host, service.config.port), _handler(service)
        )
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


class RemoteControlServerNode(RemoteControlNode):
    """Managed control process usable without RAX or the dashboard."""

    def __init__(
        self,
        server_config: RemoteControlServerConfig,
        robot_config: RobotConfig = ROBOT_CONFIG,
        config: RemoteControlPolicyConfig | None = None,
    ) -> None:
        config = config or get_remote_control_policy_config()
        policy = RemoteControlPolicy(Robot(robot_config), config=config)
        super().__init__(policy, rate_hz=1.0 / config.dt)
        self.service = RemoteControlService(policy, server_config)
        self.server = RemoteControlServer(self.service)

    def setup(self) -> None:
        super().setup()
        self.service.observe(self.env.observe())
        self.server.start()

    def step(self) -> None:
        observation = self.env.observe()
        self.service.observe(observation)
        self.env.step(self.policy(observation))

    def on_close(self) -> None:
        try:
            self.server.close()
        finally:
            super().on_close()
