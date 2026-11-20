"""Flask server for the local humanoid operator console."""

import argparse
import os
import signal
import threading
import webbrowser
from http import HTTPStatus
from typing import Any
from urllib.parse import urlsplit

from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import BadRequest, HTTPException
from werkzeug.serving import make_server

from humanoid.config.camera import CAMERA_PREVIEW_CONFIG
from humanoid.config.teleop import get_oculus_teleop_policy_config
from humanoid.constants import OCULUS_IP_ENVIRONMENT_VARIABLE
from humanoid.logger import get_logger, setup_logging
from humanoid.orchestrator.service import OrchestratorService
from humanoid.types.orchestrator import OrchestratorError
from humanoid.types.process import ProcessAction
from humanoid.ui.constants import (
    ALLOWED_HOSTS,
    BROWSER_OPEN_DELAY_SECONDS,
    CONTENT_SECURITY_POLICY,
    HOST,
    JSON_CONTENT_TYPE,
    MAX_REQUEST_BYTES,
    STATIC_ROOT,
    ApiRoute,
    PayloadKey,
)
from humanoid.ui.errors import ApiError
from humanoid.ui.validation import (
    parse_application_log_cursor,
    parse_orchestrator_request,
    parse_process_action,
    parse_process_name,
    parse_robot_name,
    parse_runtime,
    parse_safety_context,
)

logger = get_logger(__name__)


def create_app(  # noqa: PLR0915 - route definitions form the application factory
    orchestrator_service: OrchestratorService | None = None,
) -> Flask:
    """Create the operator-console application."""
    if orchestrator_service is None:
        orchestrator_service = OrchestratorService()
    app = Flask(__name__, static_folder=str(STATIC_ROOT), static_url_path="")
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
    app.extensions["orchestrator_service"] = orchestrator_service

    @app.before_request
    def protect_api_requests() -> None:
        if _is_api_request():
            _check_request_origin()

    @app.after_request
    def set_response_headers(response: Response) -> Response:
        response.headers["X-Content-Type-Options"] = "nosniff"
        if _is_api_request():
            response.headers["Cache-Control"] = "no-store"
        else:
            response.headers["Cache-Control"] = "no-cache"
            response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        return response

    @app.errorhandler(ApiError)
    def handle_api_error(error: ApiError):
        return jsonify(ok=False, error=str(error)), error.status

    @app.errorhandler(OrchestratorError)
    def handle_orchestrator_error(error: OrchestratorError):
        return jsonify(ok=False, error=str(error)), HTTPStatus.CONFLICT

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        if not _is_api_request():
            return error
        return jsonify(ok=False, error=error.description), error.code

    @app.errorhandler(Exception)
    def handle_unexpected_error(_error: Exception):
        logger.exception("Unhandled operator-console request error")
        return (
            jsonify(ok=False, error="The control request failed unexpectedly."),
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get(ApiRoute.STATUS)
    def status():
        return jsonify(orchestrator_service.status())

    @app.get(ApiRoute.APPLICATION_LOGS)
    def application_logs():
        cursor = parse_application_log_cursor(request.args.get("after"))
        return jsonify(orchestrator_service.application_logs(cursor))

    @app.get(ApiRoute.HARDWARE_OBSERVATION)
    def hardware_observation():
        return jsonify(orchestrator_service.hardware_observation())

    @app.get(ApiRoute.CAMERA_PREVIEWS)
    def camera_previews():
        try:
            depth_far = float(
                request.args.get("depth_far", CAMERA_PREVIEW_CONFIG.default_depth_far_meters)
            )
        except ValueError as exc:
            raise ApiError("Invalid depth preview range.", HTTPStatus.BAD_REQUEST) from exc
        if depth_far not in CAMERA_PREVIEW_CONFIG.depth_ranges_meters:
            raise ApiError("Invalid depth preview range.", HTTPStatus.BAD_REQUEST)
        return jsonify(orchestrator_service.camera_previews(depth_far))

    @app.post(ApiRoute.RUNTIME)
    def set_runtime():
        payload = _json_payload()
        runtime = parse_runtime(str(payload.get(PayloadKey.RUNTIME, "")))
        safety = _safety_context(payload)
        return _success(orchestrator_service.set_runtime(runtime, safety))

    @app.post(ApiRoute.ROBOT)
    def set_robot():
        payload = _json_payload()
        robot = parse_robot_name(str(payload.get(PayloadKey.ROBOT, "")))
        safety = _safety_context(payload)
        return _success(orchestrator_service.set_robot(robot, safety))

    @app.post(ApiRoute.PROCESSES)
    def update_process(name: str, action: str):
        payload = _json_payload()
        process_name = parse_process_name(name)
        process_action = parse_process_action(action)
        if process_action is ProcessAction.START:
            status = orchestrator_service.start_process(process_name, _safety_context(payload))
        else:
            status = orchestrator_service.stop_process(process_name)
        return _success(status)

    @app.post(ApiRoute.LOGGING)
    def update_logging(action: str):
        _json_payload()
        logging_action = parse_process_action(action)
        return _success(orchestrator_service.set_logging(logging_action))

    @app.post(ApiRoute.REPLAY)
    def update_replay(action: str):
        return _update_replay(orchestrator_service, action)

    @app.post(ApiRoute.ORCHESTRATOR)
    def request_orchestrator_mode():
        payload = _json_payload()
        orchestrator_request = parse_orchestrator_request(
            str(payload.get(PayloadKey.MODE, "")),
            payload.get(PayloadKey.PARAMETERS),
        )
        return _success(orchestrator_service.request_mode(orchestrator_request))

    @app.get(ApiRoute.AUTONOMY)
    def autonomy_status():
        return jsonify(orchestrator_service.autonomy.status())

    @app.post(ApiRoute.AUTONOMY_SELECT)
    def select_autonomy_task():
        payload = _json_payload()
        task = payload.get("task")
        if not isinstance(task, str):
            raise ApiError("Select an autonomy task.", HTTPStatus.BAD_REQUEST)
        return _success(orchestrator_service.autonomy.select(task, _safety_context(payload)))

    @app.post(ApiRoute.AUTONOMY_START)
    def start_autonomy():
        return _success(orchestrator_service.autonomy.start(_safety_context(_json_payload())))

    @app.post(ApiRoute.AUTONOMY_STOP)
    def stop_autonomy():
        _json_payload()
        return _success(orchestrator_service.autonomy.stop())

    @app.post(ApiRoute.AUTONOMY_RPC)
    def autonomy_rpc():
        payload = _json_payload()
        try:
            result = orchestrator_service.autonomy.rpc(
                request.headers.get("Authorization", ""), payload
            )
            return jsonify(result=result)
        except PermissionError:
            raise ApiError(
                "Invalid autonomy supervisor credential.", HTTPStatus.FORBIDDEN
            ) from None
        except (ValueError, TypeError, KeyError, RuntimeError) as error:
            return jsonify(error=str(error))

    @app.route(f"{ApiRoute.ROOT}/<path:_path>", methods=["GET", "POST"])
    def unknown_api(_path: str):
        raise ApiError("Unknown endpoint.", HTTPStatus.NOT_FOUND)

    return app


def _success(status: object):
    return jsonify(ok=True, status=status)


def _safety_context(payload: dict[str, Any]):
    return parse_safety_context(
        payload.get(PayloadKey.EXPECTED_RUNTIME),
        payload.get(PayloadKey.EXPECTED_ROBOT),
        payload.get(PayloadKey.REAL_HARDWARE_ACKNOWLEDGED),
    )


def _is_api_request() -> bool:
    return request.path == ApiRoute.ROOT or request.path.startswith(f"{ApiRoute.ROOT}/")


def _check_request_origin() -> None:
    host = request.headers.get("Host")
    try:
        hostname = urlsplit(f"//{host}").hostname if host else None
    except ValueError:
        hostname = None
    if hostname is None or hostname.lower() not in ALLOWED_HOSTS:
        raise ApiError("Invalid request host.", HTTPStatus.FORBIDDEN)

    origin = request.headers.get("Origin")
    if origin and urlsplit(origin).netloc != host:
        raise ApiError("Cross-origin control requests are not allowed.", HTTPStatus.FORBIDDEN)


def _json_payload() -> dict[str, Any]:
    if request.mimetype != JSON_CONTENT_TYPE:
        raise ApiError("Content-Type must be application/json.", HTTPStatus.UNSUPPORTED_MEDIA_TYPE)

    content_length = request.content_length
    if content_length is not None and content_length > MAX_REQUEST_BYTES:
        raise ApiError("Request body is too large.", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
    if not content_length:
        raise ApiError("A JSON request body is required.", HTTPStatus.BAD_REQUEST)

    try:
        payload = request.get_json()
    except (BadRequest, UnicodeDecodeError) as exc:
        raise ApiError("Request body must be valid JSON.", HTTPStatus.BAD_REQUEST) from exc
    if not isinstance(payload, dict):
        raise ApiError("Request body must be a JSON object.", HTTPStatus.BAD_REQUEST)
    return payload


def _update_replay(orchestrator_service: OrchestratorService, action: str):
    replay_action = parse_process_action(action)
    payload = _json_payload()
    if replay_action is ProcessAction.STOP:
        return _success(orchestrator_service.stop_replay())

    recording_id = payload.get(PayloadKey.RECORDING)
    if not isinstance(recording_id, str) or not recording_id:
        raise ApiError("Select a recording to replay.", HTTPStatus.BAD_REQUEST)
    return _success(orchestrator_service.start_replay(recording_id, _safety_context(payload)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local humanoid operator console")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765)")
    parser.add_argument(
        "--no-open", action="store_true", help="Do not open a browser automatically"
    )
    parser.add_argument(
        "--oculus-ip",
        metavar="IP",
        help=(
            "Quest IPv4 address for wireless ADB "
            f"(also configurable with {OCULUS_IP_ENVIRONMENT_VARIABLE})"
        ),
    )
    args = parser.parse_args()

    if args.oculus_ip is not None:
        os.environ[OCULUS_IP_ENVIRONMENT_VARIABLE] = args.oculus_ip
    try:
        oculus_config = get_oculus_teleop_policy_config()
    except ValueError as exc:
        parser.error(str(exc))

    setup_logging()
    if oculus_config.ip_address is not None:
        logger.info(
            "Oculus wireless ADB configured for %s:%d",
            oculus_config.ip_address,
            oculus_config.port,
        )
    orchestrator_service = OrchestratorService()
    app = create_app(orchestrator_service)
    try:
        server = make_server(HOST, args.port, app, threaded=True)
    except Exception:
        orchestrator_service.close()
        raise
    url = f"http://{HOST}:{server.server_port}"
    logger.info("Humanoid operator console available at %s", url)

    def signal_handler(_signal_number: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, signal_handler)

    if not args.no_open:
        opener = threading.Timer(BROWSER_OPEN_DELAY_SECONDS, webbrowser.open, args=(url,))
        opener.daemon = True
        opener.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping operator console")
    finally:
        server.server_close()
        orchestrator_service.close()


if __name__ == "__main__":
    main()
