"""LCM recording status and control routes."""

from http import HTTPStatus

from flask import Flask

from humanoid.api.application import BackendApplication
from humanoid.api.routes.common import json_payload, success
from humanoid.api.validation import parse_process_action
from humanoid.types.api import ApiError, ApiRoute


def register_recording_routes(app: Flask, backend: BackendApplication) -> None:
    @app.get(ApiRoute.RECORDINGS)
    def recording_status():
        return success(backend.recording_status())

    @app.post(ApiRoute.RECORDING_TOGGLE)
    def toggle_recording():
        if json_payload():
            raise ApiError("Recording toggle does not accept parameters.", HTTPStatus.BAD_REQUEST)
        return success(backend.toggle_recording())

    @app.post(ApiRoute.RECORDING_ACTION)
    def update_recording(action: str):
        payload = json_payload()
        recording_action = parse_process_action(action)
        if not payload:
            return success(backend.set_recording(recording_action))

        episode_id, runtime_id = payload.get("episode_id"), payload.get("runtime_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ApiError("Expected an episode_id string.", HTTPStatus.BAD_REQUEST)
        if not isinstance(runtime_id, str) or not runtime_id:
            raise ApiError("Expected a runtime_id string.", HTTPStatus.BAD_REQUEST)
        status = backend.set_episode_recording(recording_action, episode_id, runtime_id)
        return success(status)
