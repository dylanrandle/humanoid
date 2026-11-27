"""Flask server for the local humanoid operator console."""

import argparse
import os
import signal
import threading
import webbrowser
from http import HTTPStatus
from urllib.parse import urlsplit

from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import HTTPException
from werkzeug.serving import make_server

from humanoid.api.application import BackendApplication
from humanoid.api.routes.agent import register_agent_routes
from humanoid.api.routes.logs import register_logs_routes
from humanoid.api.routes.recording import register_recording_routes
from humanoid.api.routes.replay import register_replay_routes
from humanoid.api.routes.robot import register_robot_routes
from humanoid.api.routes.runtime import register_runtime_routes
from humanoid.api.routes.streaming import register_streaming_routes
from humanoid.config.api import (
    ALLOWED_HOSTS,
    API_HOST,
    BROWSER_OPEN_DELAY_SECONDS,
    CONTENT_SECURITY_POLICY,
    DEFAULT_API_PORT,
    MAX_REQUEST_BYTES,
    STATIC_ROOT,
)
from humanoid.config.teleop import get_oculus_teleop_policy_config
from humanoid.constants import EnvironmentVariable
from humanoid.logger import get_logger, setup_logging
from humanoid.types.api import ApiError, ApiRoute, BackendError
from humanoid.types.motion import MotionNotFound

logger = get_logger(__name__)


def create_app(backend: BackendApplication | None = None) -> Flask:
    """Create the operator-console application."""
    if backend is None:
        backend = BackendApplication()
    backend.start()
    app = Flask(__name__, static_folder=str(STATIC_ROOT), static_url_path="")
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
    app.extensions["backend"] = backend

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

    @app.errorhandler(BackendError)
    def handle_orchestrator_error(error: BackendError):
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

    @app.errorhandler(ValueError)
    def handle_invalid_request(error: ValueError):
        return jsonify(ok=False, error=str(error)), HTTPStatus.BAD_REQUEST

    @app.errorhandler(MotionNotFound)
    def handle_missing(_error: MotionNotFound):
        return jsonify(ok=False, error="Unknown motion ID."), HTTPStatus.NOT_FOUND

    register_agent_routes(app, backend)
    register_logs_routes(app, backend)
    register_recording_routes(app, backend)
    register_replay_routes(app, backend)
    register_runtime_routes(app, backend)
    register_robot_routes(app, backend)
    register_streaming_routes(app, backend)

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.route(f"{ApiRoute.ROOT}/<path:_path>", methods=["GET", "POST"])
    def unknown_api(_path: str):
        raise ApiError("Unknown endpoint.", HTTPStatus.NOT_FOUND)

    return app


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

    if request.headers.get("Sec-Fetch-Site") in {"cross-site", "same-site"}:
        raise ApiError("A local request is required.", HTTPStatus.FORBIDDEN)
    origin = request.headers.get("Origin")
    if origin and urlsplit(origin).netloc != host:
        raise ApiError("Cross-origin control requests are not allowed.", HTTPStatus.FORBIDDEN)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local humanoid operator console")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_API_PORT, help="Port to bind (default: %(default)s)"
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Do not open a browser automatically"
    )
    parser.add_argument(
        "--oculus-ip",
        metavar="IP",
        help=(
            "Quest IPv4 address for wireless ADB "
            f"(also configurable with {EnvironmentVariable.OCULUS_IP})"
        ),
    )
    args = parser.parse_args()

    if args.oculus_ip is not None:
        os.environ[EnvironmentVariable.OCULUS_IP] = args.oculus_ip
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
    backend = BackendApplication()
    app = create_app(backend)
    try:
        server = make_server(API_HOST, args.port, app, threaded=True)
    except Exception:
        backend.close()
        raise
    url = f"http://{API_HOST}:{server.server_port}"
    os.environ[EnvironmentVariable.API_URL] = url
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
        backend.close()


if __name__ == "__main__":
    main()
