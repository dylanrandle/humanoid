"""Backend status, runtime configuration, process, and mode routes."""

from flask import Flask, jsonify

from humanoid.api.application import BackendApplication
from humanoid.api.routes.common import json_payload, safety_context, success
from humanoid.api.validation import (
    parse_orchestrator_request,
    parse_process_action,
    parse_process_name,
    parse_robot_name,
    parse_runtime,
    parse_scene,
)
from humanoid.types.api import ApiKey, ApiRoute
from humanoid.types.process import ProcessAction


def register_runtime_routes(app: Flask, backend: BackendApplication) -> None:
    @app.get(ApiRoute.STATUS)
    def status():
        return jsonify(backend.status())

    @app.post(ApiRoute.RUNTIME)
    def set_runtime():
        payload = json_payload()
        runtime = parse_runtime(str(payload.get(ApiKey.RUNTIME, "")))
        safety = safety_context(payload)
        return success(backend.set_runtime(runtime, safety))

    @app.post(ApiRoute.SCENE)
    def set_scene():
        payload = json_payload()
        scene = parse_scene(str(payload.get(ApiKey.SCENE, "")))
        return success(backend.set_scene(scene, safety_context(payload)))

    @app.post(ApiRoute.ROBOT)
    def set_robot():
        payload = json_payload()
        robot = parse_robot_name(str(payload.get(ApiKey.ROBOT, "")))
        safety = safety_context(payload)
        return success(backend.set_robot(robot, safety))

    @app.post(ApiRoute.PROCESSES)
    def update_process(name: str, action: str):
        payload = json_payload()
        process_name = parse_process_name(name)
        process_action = parse_process_action(action)
        if process_action is ProcessAction.START:
            status = backend.start_process(process_name, safety_context(payload))
        else:
            status = backend.stop_process(process_name)
        return success(status)

    @app.post(ApiRoute.ORCHESTRATOR)
    def request_orchestrator_mode():
        payload = json_payload()
        orchestrator_request = parse_orchestrator_request(
            str(payload.get(ApiKey.MODE, "")),
            payload.get(ApiKey.PARAMETERS),
        )
        return success(backend.request_mode(orchestrator_request))
