from dataclasses import dataclass
from http import HTTPStatus
from unittest.mock import MagicMock

import pytest
from flask.testing import FlaskClient

from humanoid.orchestrator.service import OrchestratorService
from humanoid.types.homing import HomingPreset
from humanoid.types.logging import LoggingState, LoggingStatus
from humanoid.types.node import NodeRateStatus
from humanoid.types.orchestrator import (
    Mode,
    OrchestratorError,
    OrchestratorParameter,
    OrchestratorRequest,
    OrchestratorState,
    OrchestratorStatus,
    SafetyContext,
)
from humanoid.types.process import ProcessAction, ProcessName, ProcessStatus, Runtime
from humanoid.types.replay import RecordingSummary, ReplayStatus
from humanoid.types.robot import RobotName
from humanoid.types.simulation import MujocoScene
from humanoid.ui.constants import ApiRoute, PayloadKey
from humanoid.ui.server import create_app


@dataclass(frozen=True)
class RouteCase:
    path: str
    payload: dict[str, object]
    method_name: str
    arguments: tuple[object, ...]
    result: dict[str, str]


@pytest.fixture
def server_client() -> tuple[MagicMock, FlaskClient]:
    service = MagicMock(spec=OrchestratorService)
    service.status.return_value = {"runtime": Runtime.SIM}
    app = create_app(service)
    app.config["TESTING"] = True
    return service, app.test_client()


def _safety_payload(*, acknowledged: bool = False) -> dict[str, object]:
    return {
        PayloadKey.EXPECTED_RUNTIME.value: Runtime.SIM,
        PayloadKey.EXPECTED_ROBOT.value: RobotName.ELROBOT_MOBILE,
        PayloadKey.EXPECTED_SCENE.value: MujocoScene.EMPTY,
        PayloadKey.REAL_HARDWARE_ACKNOWLEDGED.value: acknowledged,
    }


def _safety_context(*, acknowledged: bool = False) -> SafetyContext:
    return SafetyContext(
        expected_runtime=Runtime.SIM,
        expected_robot=RobotName.ELROBOT_MOBILE,
        expected_scene=MujocoScene.EMPTY,
        real_hardware_acknowledged=acknowledged,
    )


def test_serves_split_ui_assets(server_client):
    _, client = server_client

    response = client.get("/")
    assert response.status_code == HTTPStatus.OK
    assert b"Humanoid Control" in response.data
    assert b'id="robot-select"' in response.data
    assert b'id="scene-select"' in response.data
    assert b'data-orchestrator-mode="homing"' in response.data
    assert b'data-orchestrator-preset="home"' in response.data
    assert b'data-logging-action="start"' in response.data
    assert b'data-logging-action="stop"' in response.data
    assert b'id="replay-recording"' in response.data
    assert b'id="replay-action"' in response.data
    assert b'id="node-rate-list"' in response.data
    assert b"Home and Rest stay highlighted" not in response.data
    assert response.headers["Cache-Control"] == "no-cache"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]

    response = client.get("/js/app.js")
    assert response.status_code == HTTPStatus.OK
    assert response.content_type.startswith(("text/javascript", "application/javascript"))
    assert b'from "./api.js"' in response.data

    response = client.get("/css/base.css")
    assert response.status_code == HTTPStatus.OK
    assert response.content_type.startswith("text/css")
    assert b":root" in response.data

    response = client.get("/css/data.css")
    assert response.status_code == HTTPStatus.OK
    assert b".data-panel" in response.data
    assert b"overflow-wrap: anywhere" in response.data

    response = client.get("/css/health.css")
    assert response.status_code == HTTPStatus.OK
    assert b".node-rate-row.healthy" in response.data


def test_routes_status(server_client):
    service, client = server_client

    response = client.get(ApiRoute.STATUS)

    assert response.status_code == HTTPStatus.OK
    assert response.json == {"runtime": "sim"}
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    service.status.assert_called_once_with()


def test_serializes_orchestrator_status_dataclass(server_client):
    service, client = server_client
    service.status.return_value = OrchestratorStatus(
        runtime=Runtime.SIM,
        robot=RobotName.PANDA,
        robots=[RobotName.PANDA],
        scene=MujocoScene.FLOOR_AND_CUBE,
        scenes=list(MujocoScene),
        processes={
            ProcessName.STACK: ProcessStatus(
                running=True,
                pid=123,
                exit_code=None,
                runtime=Runtime.SIM,
                uptime_seconds=1.2,
                last_output=None,
            ),
            ProcessName.REPLAY: ProcessStatus(
                running=False,
                pid=None,
                exit_code=None,
                runtime=None,
                uptime_seconds=None,
                last_output=None,
            ),
        },
        node_rates=[
            NodeRateStatus(
                node_name="RobotControllerNode",
                pid=456,
                target_rate_hz=500.0,
                measured_rate_hz=497.2,
                healthy=True,
                age_seconds=0.2,
            )
        ],
        logging=LoggingStatus(
            timestamp=1.0,
            state=LoggingState.RUNNING,
            file_name="logs/lcmlog_20260101",
        ),
        recordings=[
            RecordingSummary(
                id="recording_20260101_120000",
                robot=RobotName.PANDA,
                created_at="2026-01-01T12:00:00+00:00",
            )
        ],
        replay=ReplayStatus(
            running=True,
            file_name="lcmlog_20260101",
            outcome=None,
            exit_code=None,
            last_output=None,
        ),
        orchestrator=OrchestratorState(
            mode=Mode.HOMING,
            connected=True,
            age_seconds=0.1,
            parameters={"preset": HomingPreset.HOME},
        ),
    )

    response = client.get(ApiRoute.STATUS)

    assert response.status_code == HTTPStatus.OK
    assert response.json == {
        "runtime": "sim",
        "robot": "panda",
        "robots": ["panda"],
        "scene": "floor-and-cube",
        "scenes": ["empty", "floor-and-cube"],
        "processes": {
            "stack": {
                "running": True,
                "pid": 123,
                "exit_code": None,
                "runtime": "sim",
                "uptime_seconds": 1.2,
                "last_output": None,
            },
            "replay": {
                "running": False,
                "pid": None,
                "exit_code": None,
                "runtime": None,
                "uptime_seconds": None,
                "last_output": None,
            },
        },
        "node_rates": [
            {
                "node_name": "RobotControllerNode",
                "pid": 456,
                "target_rate_hz": 500.0,
                "measured_rate_hz": 497.2,
                "healthy": True,
                "age_seconds": 0.2,
            }
        ],
        "logging": {
            "timestamp": 1.0,
            "state": "running",
            "file_name": "logs/lcmlog_20260101",
            "error": None,
        },
        "recordings": [
            {
                "id": "recording_20260101_120000",
                "robot": "panda",
                "created_at": "2026-01-01T12:00:00+00:00",
            }
        ],
        "replay": {
            "running": True,
            "file_name": "lcmlog_20260101",
            "outcome": None,
            "exit_code": None,
            "last_output": None,
        },
        "orchestrator": {
            "mode": "homing",
            "connected": True,
            "age_seconds": 0.1,
            "parameters": {"preset": "home"},
        },
    }


@pytest.mark.parametrize(
    "case",
    [
        RouteCase(
            path=ApiRoute.RUNTIME,
            payload={PayloadKey.RUNTIME: "real", **_safety_payload(acknowledged=True)},
            method_name="set_runtime",
            arguments=(Runtime.REAL, _safety_context(acknowledged=True)),
            result={"runtime": "real"},
        ),
        RouteCase(
            path=ApiRoute.ROBOT,
            payload={PayloadKey.ROBOT: "panda", **_safety_payload()},
            method_name="set_robot",
            arguments=(RobotName.PANDA, _safety_context()),
            result={"robot": "panda"},
        ),
        RouteCase(
            path=ApiRoute.SCENE,
            payload={PayloadKey.SCENE: "floor-and-cube", **_safety_payload()},
            method_name="set_scene",
            arguments=(MujocoScene.FLOOR_AND_CUBE, _safety_context()),
            result={"scene": "floor-and-cube"},
        ),
        RouteCase(
            path="/api/processes/stack/start",
            payload=_safety_payload(),
            method_name="start_process",
            arguments=(ProcessName.STACK, _safety_context()),
            result={"stack": "running"},
        ),
        RouteCase(
            path="/api/processes/keyboard/stop",
            payload={},
            method_name="stop_process",
            arguments=(ProcessName.KEYBOARD,),
            result={"keyboard": "stopped"},
        ),
        RouteCase(
            path="/api/logging/start",
            payload={},
            method_name="set_logging",
            arguments=(ProcessAction.START,),
            result={"logging": "started"},
        ),
        RouteCase(
            path="/api/replay/start",
            payload={PayloadKey.RECORDING: "recording_1", **_safety_payload()},
            method_name="start_replay",
            arguments=("recording_1", _safety_context()),
            result={"replay": "running"},
        ),
        RouteCase(
            path="/api/replay/stop",
            payload={},
            method_name="stop_replay",
            arguments=(),
            result={"replay": "stopped"},
        ),
        RouteCase(
            path=ApiRoute.ORCHESTRATOR,
            payload={PayloadKey.MODE: "homing", PayloadKey.PARAMETERS: {"preset": "home"}},
            method_name="request_mode",
            arguments=(
                OrchestratorRequest(
                    mode=Mode.HOMING,
                    parameters={OrchestratorParameter.PRESET: HomingPreset.HOME},
                ),
            ),
            result={"orchestrator": "home"},
        ),
    ],
)
def test_routes_typed_control_actions(
    server_client,
    case,
):
    service, client = server_client
    method = getattr(service, case.method_name)
    method.return_value = case.result

    response = client.post(case.path, json=case.payload)

    assert response.status_code == HTTPStatus.OK
    assert response.json == {"ok": True, "status": case.result}
    method.assert_called_once_with(*case.arguments)


@pytest.mark.parametrize(
    ("path", "data", "content_type", "expected_status"),
    [
        (ApiRoute.ORCHESTRATOR, "{", "application/json", HTTPStatus.BAD_REQUEST),
        (ApiRoute.ORCHESTRATOR, "{}", "text/plain", HTTPStatus.UNSUPPORTED_MEDIA_TYPE),
        (ApiRoute.ORCHESTRATOR, "[]", "application/json", HTTPStatus.BAD_REQUEST),
        (ApiRoute.ORCHESTRATOR, "", "application/json", HTTPStatus.BAD_REQUEST),
        ("/api/runtime", '{"runtime":"hardware"}', "application/json", HTTPStatus.BAD_REQUEST),
        ("/api/robot", '{"robot":"unknown"}', "application/json", HTTPStatus.BAD_REQUEST),
        ("/api/scene", '{"scene":"warehouse"}', "application/json", HTTPStatus.BAD_REQUEST),
        (ApiRoute.RUNTIME, '{"runtime":"sim"}', "application/json", HTTPStatus.BAD_REQUEST),
        (
            "/api/processes/stack/start",
            "{}",
            "application/json",
            HTTPStatus.BAD_REQUEST,
        ),
        (
            ApiRoute.ORCHESTRATOR,
            '{"mode":"dance"}',
            "application/json",
            HTTPStatus.BAD_REQUEST,
        ),
        (
            ApiRoute.ORCHESTRATOR,
            '{"mode":"homing"}',
            "application/json",
            HTTPStatus.BAD_REQUEST,
        ),
        (
            ApiRoute.ORCHESTRATOR,
            '{"mode":"idle","parameters":{"preset":"home"}}',
            "application/json",
            HTTPStatus.BAD_REQUEST,
        ),
        (
            ApiRoute.ORCHESTRATOR,
            '{"mode":"homing","parameters":{"preset":"dance"}}',
            "application/json",
            HTTPStatus.BAD_REQUEST,
        ),
        ("/api/processes/unknown/start", "{}", "application/json", HTTPStatus.NOT_FOUND),
        ("/api/processes/stack/restart", "{}", "application/json", HTTPStatus.NOT_FOUND),
        ("/api/processes/replay/start", "{}", "application/json", HTTPStatus.NOT_FOUND),
        ("/api/logging/restart", "{}", "application/json", HTTPStatus.NOT_FOUND),
        ("/api/replay/restart", "{}", "application/json", HTTPStatus.NOT_FOUND),
        ("/api/replay/start", "{}", "application/json", HTTPStatus.BAD_REQUEST),
        ("/api/unknown", "{}", "application/json", HTTPStatus.NOT_FOUND),
    ],
)
def test_rejects_invalid_api_requests(
    server_client,
    path,
    data,
    content_type,
    expected_status,
):
    _, client = server_client

    response = client.post(path, data=data, content_type=content_type)

    assert response.status_code == expected_status
    assert response.json["ok"] is False


def test_rejects_oversized_requests(server_client):
    _, client = server_client

    response = client.post(
        ApiRoute.ORCHESTRATOR,
        data=b"x" * (16 * 1024 + 1),
        content_type="application/json",
    )

    assert response.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    assert response.json == {"ok": False, "error": "Request body is too large."}


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://example.com"},
        {"Host": "example.com"},
        {"Host": "[invalid"},
    ],
)
def test_rejects_non_local_requests(server_client, headers):
    service, client = server_client

    response = client.post(
        ApiRoute.RUNTIME,
        json={PayloadKey.RUNTIME: "sim"},
        headers=headers,
    )

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json["ok"] is False
    service.set_runtime.assert_not_called()


def test_service_error_returns_conflict(server_client):
    service, client = server_client
    service.start_process.side_effect = OrchestratorError("stack unavailable")

    response = client.post("/api/processes/stack/start", json=_safety_payload())

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json == {"ok": False, "error": "stack unavailable"}


def test_external_stack_status_returns_conflict(server_client):
    service, client = server_client
    service.status.side_effect = OrchestratorError(
        "Another stack is already broadcasting. Stop it before using this console."
    )

    response = client.get(ApiRoute.STATUS)

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json == {
        "ok": False,
        "error": "Another stack is already broadcasting. Stop it before using this console.",
    }


def test_status_failure_returns_json_error(server_client):
    service, client = server_client
    service.status.side_effect = RuntimeError("boom")

    response = client.get(ApiRoute.STATUS)

    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert response.json == {"ok": False, "error": "The control request failed unexpectedly."}


def test_static_path_cannot_escape_ui_directory(server_client):
    _, client = server_client

    response = client.get("/../pyproject.toml")

    assert response.status_code == HTTPStatus.NOT_FOUND
