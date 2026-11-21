"""HTTP client for agents and scripts; waiting never runs on the policy loop."""

import contextlib
import json
import threading
import time
import uuid
from collections.abc import Mapping
from typing import override
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from humanoid.agent.serialization import decode_frames, decode_state
from humanoid.config.agent import DEFAULT_AGENT_ENDPOINT
from humanoid.types.agent import (
    AgentControlEndpoint,
    AgentRoute,
    CommandState,
    ControlOperation,
    MotionResult,
)
from humanoid.types.camera import CameraFrames
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import RobotState


class _NoRedirect(HTTPRedirectHandler):
    @override
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Control API redirects are not allowed.")


class AgentControlClient:
    def __init__(self, endpoint: AgentControlEndpoint = DEFAULT_AGENT_ENDPOINT) -> None:
        self.endpoint = endpoint
        self._session_id: str | None = None
        self._cancelled = threading.Event()
        self._command_lock = threading.Lock()
        self._current_id: str | None = None

    def _request(self, path: str, payload: dict | None = None) -> dict:
        request = Request(
            self.endpoint.url.rstrip("/") + path,
            data=None if payload is None else json.dumps(payload, allow_nan=False).encode(),
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            opener = build_opener(_NoRedirect, ProxyHandler({}))
            with opener.open(request, timeout=3) as response:
                return json.load(response)
        except HTTPError as error:
            detail = json.loads(error.read()).get("error", "Control request failed.")
            raise RuntimeError(detail) from None
        except (URLError, ConnectionError, TimeoutError) as error:
            raise RuntimeError("Robot control service is unreachable.") from error

    def wait_until_ready(self, timeout: float = 30.0) -> None:
        deadline = time.perf_counter() + timeout
        while not self._cancelled.is_set():
            try:
                self.state()
                return
            except RuntimeError:
                if time.perf_counter() >= deadline:
                    raise
                self._cancelled.wait(0.05)
        raise RuntimeError("Agent control client was stopped.")

    def state(self) -> RobotState:
        return self.observation().robot_state

    @property
    def session_id(self) -> str:
        """The control service identity pinned by the first state read."""
        if self._session_id is None:
            self.state()
        assert self._session_id is not None
        return self._session_id

    def observation(self) -> Observation:
        """Read the measured state and operator mode from one API response."""
        started = time.perf_counter()
        payload = self._request(AgentRoute.STATE)
        if self._session_id is None:
            self._session_id = payload["session_id"]
        elif self._session_id != payload["session_id"]:
            raise RuntimeError("Robot control service restarted; create a new client.")
        return Observation(
            robot_state=decode_state(payload["state"], started),
            mode=Mode(payload["mode"]) if payload["mode"] is not None else None,
        )

    def observe(self) -> CameraFrames:
        started = time.perf_counter()
        return decode_frames(self._request(AgentRoute.CAMERA), started)

    def execute(
        self, operation: ControlOperation, arguments: Mapping[str, object], *, timeout: float = 5.0
    ) -> MotionResult:
        if not self._command_lock.acquire(blocking=False):
            raise RuntimeError("A command is already running on this client.")
        try:
            if self._cancelled.is_set():
                raise RuntimeError("Agent control client was stopped.")
            if self._session_id is None:
                self.state()
            command_id = uuid.uuid4().hex
            self._current_id = command_id
            deadline = time.perf_counter() + timeout
            status = self._request(
                AgentRoute.COMMANDS,
                {
                    "session_id": self._session_id,
                    "command_id": command_id,
                    "operation": operation,
                    "arguments": dict(arguments),
                    "expires_in_seconds": 1.0,
                },
            )
            while not self._cancelled.is_set():
                state = CommandState(status["state"])
                if state is CommandState.COMPLETED:
                    return status["result"]
                if state not in (CommandState.QUEUED, CommandState.RUNNING):
                    raise RuntimeError(status["error"] or f"Command {state}.")
                if time.perf_counter() >= deadline:
                    raise RuntimeError("Timed out waiting for the motion command to finish.")
                self._cancelled.wait(0.02)
                status = self._request(AgentRoute.COMMAND.format(command_id=command_id))
            raise RuntimeError("Agent control client was stopped.")
        except BaseException:
            self.hold()
            raise
        finally:
            self._current_id = None
            self._command_lock.release()

    def hold(self) -> None:
        command_id = self._current_id
        if command_id is not None:
            # The policy duration and controller leases also bound motion.
            with contextlib.suppress(RuntimeError):
                self._request(AgentRoute.CANCEL.format(command_id=command_id), {})

    def cancel(self) -> None:
        self._cancelled.set()
        self.hold()

    def close(self) -> None:
        self.cancel()
