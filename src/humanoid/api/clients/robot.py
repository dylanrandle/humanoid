"""HTTP client for agents and scripts; waiting never runs on the policy loop."""

import contextlib
import json
import threading
import time
import uuid
from collections.abc import Mapping
from http import HTTPStatus
from typing import override
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from pydantic import TypeAdapter

from humanoid.api.serialization import decode_frames, decode_state
from humanoid.config.api import get_api_endpoint
from humanoid.config.motion import DEFAULT_MOTION_TIMEOUT_SECONDS
from humanoid.types.api import ApiEndpoint, ApiKey, ApiRoute
from humanoid.types.camera import CameraView, DepthView
from humanoid.types.motion import ControlOperation, MotionResult, MotionState
from humanoid.types.observation import CameraCaptures, Observation
from humanoid.types.orchestrator import Mode
from humanoid.types.recording import RecordingStatus
from humanoid.types.robot import RobotState


class _NoRedirect(HTTPRedirectHandler):
    @override
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Control API redirects are not allowed.")


class RobotClient:
    def __init__(self, endpoint: ApiEndpoint | None = None) -> None:
        self.endpoint = endpoint if endpoint is not None else get_api_endpoint()
        self._runtime_id: str | None = None
        self._motion_session_id: str | None = None
        self._observed_motion_session_id: str | None = None
        self._cancelled = threading.Event()
        self._command_lock = threading.Lock()
        self._current_id: str | None = None

    def _request(self, path: str, payload: dict | None = None) -> dict:
        if path.startswith(ApiRoute.MOTIONS + "/"):
            if payload is None:
                path += "?" + urlencode({"motion_session_id": self.motion_session_id})
            else:
                payload = {**payload, "motion_session_id": self.motion_session_id}
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
            if path == ApiRoute.MOTIONS and error.code == HTTPStatus.BAD_REQUEST:
                raise ValueError(detail) from None
            raise RuntimeError(detail) from None
        except (URLError, ConnectionError, TimeoutError) as error:
            raise RuntimeError("Robot control service is unreachable.") from error

    def recording_status(self) -> RecordingStatus:
        return TypeAdapter(RecordingStatus).validate_python(
            self._request(ApiRoute.RECORDINGS)["status"]
        )

    def toggle_recording(self) -> RecordingStatus:
        return TypeAdapter(RecordingStatus).validate_python(
            self._request(ApiRoute.RECORDING_TOGGLE, {})["status"]
        )

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

    def motion_ready(self) -> bool:
        """Return the executor's latest readiness report; admission is rechecked on submit."""
        started = time.perf_counter()
        payload = self._request(ApiRoute.STATE)
        self._decode_observation(payload, started)
        return payload.get("motion_ready") is True

    def state(self) -> RobotState:
        return self.observation().robot_state

    @property
    def runtime_id(self) -> str:
        """Robot runtime identity, independent of motion executor restarts."""
        if self._runtime_id is None:
            self.state()
        assert self._runtime_id is not None
        return self._runtime_id

    @property
    def motion_session_id(self) -> str:
        """Pin motion authority separately; never silently adopt a new executor."""
        if self._observed_motion_session_id is None:
            self.state()
        if self._motion_session_id is None:
            self._motion_session_id = self._observed_motion_session_id
        elif self._motion_session_id != self._observed_motion_session_id:
            raise RuntimeError("Motion executor restarted; create a new client for motion.")
        assert self._motion_session_id is not None
        return self._motion_session_id

    def observation(self) -> Observation:
        """Read the measured state and operator mode from one API response."""
        started = time.perf_counter()
        payload = self._request(ApiRoute.STATE)
        return self._decode_observation(payload, started)

    def _decode_observation(self, payload: dict, started: float) -> Observation:
        if self._runtime_id is None:
            self._runtime_id = payload["runtime_id"]
        elif self._runtime_id != payload["runtime_id"]:
            raise RuntimeError("Robot runtime restarted; create a new client.")
        if "motion_session_id" in payload:
            self._observed_motion_session_id = payload["motion_session_id"]
        return Observation(
            robot_state=decode_state(payload["state"], started),
            mode=Mode(payload["mode"]) if payload["mode"] is not None else None,
        )

    def capture(self) -> CameraCaptures:
        started = time.perf_counter()
        payload = self._request(ApiRoute.CAPTURE)
        if any(
            record["camera"].get(ApiKey.DEPTH_VIEW) != DepthView.BOTH
            for record in payload["views"].values()
        ):
            raise ValueError("Camera capture requires both native and registered depth views.")
        return {
            name: (
                self._decode_observation({**record, "runtime_id": payload["runtime_id"]}, started),
                decode_frames(record["camera"], started),
            )
            for name, record in payload["views"].items()
        }

    def observe(self, *, depth_view: DepthView | str = DepthView.ALIGNED) -> CameraView:
        """Get RGB and measured depth in the selected camera frame."""
        view = DepthView(depth_view)
        started = time.perf_counter()
        path: str = ApiRoute.CAMERA
        if view is not DepthView.ALIGNED:
            path += "?" + urlencode({ApiKey.DEPTH_VIEW: view.value})
        payload = self._request(path)
        if payload.get(ApiKey.DEPTH_VIEW) != view.value:
            raise ValueError(f"Expected camera {ApiKey.DEPTH_VIEW}={view.value!r}.")
        return decode_frames(payload, started)

    def execute(
        self,
        operation: ControlOperation,
        arguments: Mapping[str, object],
        *,
        timeout: float = DEFAULT_MOTION_TIMEOUT_SECONDS,
    ) -> MotionResult:
        if not self._command_lock.acquire(blocking=False):
            raise RuntimeError("A command is already running on this client.")
        try:
            if self._cancelled.is_set():
                raise RuntimeError("Agent control client was stopped.")
            motion_session_id = self.motion_session_id
            motion_id = uuid.uuid4().hex
            self._current_id = motion_id
            deadline = time.perf_counter() + timeout
            status = self._request(
                ApiRoute.MOTIONS,
                {
                    "motion_session_id": motion_session_id,
                    "motion_id": motion_id,
                    "operation": operation,
                    "arguments": dict(arguments),
                    "expires_in_seconds": 1.0,
                },
            )
            while not self._cancelled.is_set():
                state = MotionState(status["state"])
                if state is MotionState.COMPLETED:
                    return status["result"]
                if state is MotionState.REJECTED:
                    raise ValueError(status["error"] or "Command rejected.")
                if state not in (MotionState.QUEUED, MotionState.RUNNING):
                    raise RuntimeError(status["error"] or f"Command {state}.")
                if time.perf_counter() >= deadline:
                    raise RuntimeError("Timed out waiting for the motion command to finish.")
                self._cancelled.wait(0.02)
                status = self._request(ApiRoute.MOTION.replace("<motion_id>", motion_id))
            raise RuntimeError("Agent control client was stopped.")
        except BaseException:
            self.hold()
            raise
        finally:
            self._current_id = None
            self._command_lock.release()

    def release_autonomy(self) -> None:
        """Release API and wait for feedback confirming it is no longer active."""
        self.hold()
        session = self.motion_session_id
        deadline = time.monotonic() + 3
        while True:
            self._request(ApiRoute.RELEASE_AUTONOMY, {"motion_session_id": session})
            try:
                mode = self.observation().mode
            except RuntimeError as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError("API release requested but unconfirmed.") from exc
                mode = None
            if mode is not None and mode is not Mode.AUTONOMY:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for API to be released.")
            time.sleep(0.05)

    def hold(self) -> None:
        motion_id = self._current_id
        if motion_id is not None:
            # The policy duration and controller leases also bound motion.
            with contextlib.suppress(RuntimeError):
                self._request(ApiRoute.MOTION_CANCEL.replace("<motion_id>", motion_id), {})

    def cancel(self) -> None:
        self._cancelled.set()
        self.hold()

    def close(self) -> None:
        self.cancel()
