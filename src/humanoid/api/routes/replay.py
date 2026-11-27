"""LCM recording replay control routes."""

from http import HTTPStatus

from flask import Flask

from humanoid.api.application import BackendApplication
from humanoid.api.routes.common import json_payload, safety_context, success
from humanoid.api.validation import parse_process_action
from humanoid.types.api import ApiError, ApiKey, ApiRoute
from humanoid.types.process import ProcessAction


def register_replay_routes(app: Flask, backend: BackendApplication) -> None:
    @app.post(ApiRoute.REPLAY)
    def update_replay(action: str):
        replay_action = parse_process_action(action)
        payload = json_payload()
        if replay_action is ProcessAction.STOP:
            return success(backend.stop_replay())

        recording_id = payload.get(ApiKey.RECORDING)
        if not isinstance(recording_id, str) or not recording_id:
            raise ApiError("Select a recording to replay.", HTTPStatus.BAD_REQUEST)
        return success(backend.start_replay(recording_id, safety_context(payload)))
