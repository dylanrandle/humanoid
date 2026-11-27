"""Control one hardware episode's LCM recording through the unified API."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from http.client import HTTPException
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from pydantic import TypeAdapter

from humanoid.types.api import ApiRoute
from humanoid.types.process import ProcessAction
from humanoid.types.recording import RecordingStatus

_RECORDING_STATUS = TypeAdapter(RecordingStatus)


@contextmanager
def record_lcm_episode(api_url: str, runtime_id: str) -> Iterator[str]:
    payload = json.dumps({"episode_id": uuid4().hex, "runtime_id": runtime_id}).encode()

    def request(action: ProcessAction) -> RecordingStatus:
        path = ApiRoute.RECORDING_ACTION.replace("<action>", action.value)
        req = Request(api_url.rstrip("/") + path, payload, {"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=10) as response:
                return _RECORDING_STATUS.validate_python(json.load(response)["status"])
        except HTTPError as exc:
            raise RuntimeError(
                json.load(exc).get("error", f"LCM recording {action} failed.")
            ) from exc

    try:
        started = request(ProcessAction.START)
    except (OSError, HTTPException, ValueError, KeyError, KeyboardInterrupt) as exc:
        # The backend may have started recording before the connection was lost.
        try:
            request(ProcessAction.STOP)
        except (OSError, HTTPException, RuntimeError, ValueError, KeyError) as stop_error:
            exc.add_note(f"Could not confirm recording cleanup: {stop_error}")
        raise
    try:
        if not started.file_name:
            raise RuntimeError("The logger did not report an LCM recording path.")
        yield started.file_name
    finally:
        try:
            request(ProcessAction.STOP)
        except (OSError, HTTPException):
            # Retrying the same episode ID also handles a lost stop acknowledgement.
            request(ProcessAction.STOP)
