"""Log retrieval routes."""

from flask import Flask, jsonify, request

from humanoid.api.application import BackendApplication
from humanoid.api.validation import parse_log_cursor
from humanoid.types.api import ApiKey, ApiRoute


def register_logs_routes(app: Flask, backend: BackendApplication) -> None:
    @app.get(ApiRoute.LOGS)
    def logs():
        cursor = parse_log_cursor(request.args.get(ApiKey.AFTER))
        return jsonify(backend.logs(cursor))
