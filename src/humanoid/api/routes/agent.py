"""Agent status, lifecycle, and task execution routes."""

from http import HTTPStatus

from flask import Flask, jsonify

from humanoid.api.application import BackendApplication
from humanoid.api.routes.common import json_payload, safety_context, success
from humanoid.types.api import ApiError, ApiRoute
from humanoid.types.orchestrator import Mode


def register_agent_routes(app: Flask, backend: BackendApplication) -> None:
    @app.get(ApiRoute.AGENT)
    def agent_status():
        return jsonify(backend.agent_task_service.status())

    @app.post(ApiRoute.AGENT_CONNECT)
    def agent_connect():
        json_payload()
        return success(backend.agent_task_service.start())

    @app.post(ApiRoute.AGENT_STOP)
    def agent_stop():
        json_payload()
        return success(backend.stop_agent())

    @app.post(ApiRoute.AGENT_RUN)
    def agent_run():
        payload = json_payload()
        instruction, model = payload.get("instruction"), payload.get("model")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ApiError("Enter a task for the agent.", HTTPStatus.BAD_REQUEST)
        if not isinstance(model, str) or not model.strip():
            raise ApiError("Select a model.", HTTPStatus.BAD_REQUEST)
        try:
            mode = Mode(payload.get("mode"))
        except ValueError as error:
            raise ApiError("Invalid task mode.", HTTPStatus.BAD_REQUEST) from error
        return success(backend.run_agent(instruction.strip(), model, mode, safety_context(payload)))
