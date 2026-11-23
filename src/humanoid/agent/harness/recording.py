"""Control one hardware episode's LCM recording through the operator console."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from http.client import HTTPException
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from pydantic import TypeAdapter

from humanoid.types.logging import LoggingStatus
from humanoid.types.process import ProcessAction
from humanoid.ui.constants import ApiRoute

_LOGGING_STATUS = TypeAdapter(LoggingStatus)


@contextmanager
def record_lcm_episode(console_url: str, session_id: str) -> Iterator[str]:
    payload = json.dumps({"episode_id": uuid4().hex, "session_id": session_id}).encode()

    def request(action: ProcessAction) -> LoggingStatus:
        path = ApiRoute.LOGGING.replace("<action>", action.value)
        req = Request(console_url.rstrip("/") + path, payload, {"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=10) as response:
                return _LOGGING_STATUS.validate_python(json.load(response)["status"])
        except HTTPError as exc:
            raise RuntimeError(
                json.load(exc).get("error", f"LCM logging {action} failed.")
            ) from exc

    try:
        started = request(ProcessAction.START)
    except (OSError, HTTPException, ValueError, KeyError, KeyboardInterrupt) as exc:
        # The console may have started recording before the connection was lost.
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
