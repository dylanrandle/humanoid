from dataclasses import dataclass
from http import HTTPStatus
from unittest.mock import MagicMock

import pytest
from flask.testing import FlaskClient

from humanoid.api.application import BackendApplication
from humanoid.api.server import create_app
from humanoid.config.api import MAX_REQUEST_BYTES
from humanoid.types.actuator import ActuatorHealth, ActuatorHealthStatus
from humanoid.types.api import ApiKey, ApiRoute, BackendError, BackendStatus, SafetyContext
from humanoid.types.homing import HomingPreset
from humanoid.types.host import HostResourceStatus
from humanoid.types.logs import LogEntry, LogSnapshot
from humanoid.types.node import NodeRateStatus
from humanoid.types.orchestrator import (
    Mode,
    OrchestratorParameter,
    OrchestratorRequest,
    OrchestratorState,
)
from humanoid.types.process import ProcessAction, ProcessName, ProcessStatus, Runtime
from humanoid.types.recording import RecordingState, RecordingStatus
from humanoid.types.replay import RecordingSummary, ReplayStatus
from humanoid.types.robot import RobotName
from humanoid.types.simulation import SimulationScene
from humanoid.types.telemetry import LCMChannelStatus


@dataclass(frozen=True)
class RouteCase:
    path: str
    payload: dict[str, object]
    method_name: str
    arguments: tuple[object, ...]
    result: dict[str, str]


@pytest.fixture
def server_client() -> tuple[MagicMock, FlaskClient]:
    service = MagicMock(spec=BackendApplication)
    service.status.return_value = {"runtime": Runtime.SIM}
    app = create_app(service)
    app.config["TESTING"] = True
    return service, app.test_client()


def _safety_payload(*, acknowledged: bool = False) -> dict[str, object]:
    return {
        ApiKey.EXPECTED_RUNTIME.value: Runtime.SIM,
        ApiKey.EXPECTED_ROBOT.value: RobotName.TRISKEL,
        ApiKey.REAL_HARDWARE_ACKNOWLEDGED.value: acknowledged,
    }


def _safety_context(*, acknowledged: bool = False) -> SafetyContext:
    return SafetyContext(
        expected_runtime=Runtime.SIM,
        expected_robot=RobotName.TRISKEL,
        real_hardware_acknowledged=acknowledged,
    )


def test_serves_split_ui_assets(server_client):
    _, client = server_client

    response = client.get("/")
    assert response.status_code == HTTPStatus.OK
    assert b"Humanoid Control" in response.data
    assert b'id="robot-select"' in response.data
    assert b'data-orchestrator-mode="homing"' in response.data
    assert b'data-orchestrator-preset="home"' in response.data
    assert b'data-recording-action="start"' in response.data
    assert b'data-recording-action="stop"' in response.data
    assert b'id="replay-recording"' in response.data
    assert b'id="replay-action"' in response.data
    assert b'id="node-rate-list"' in response.data
    assert b'id="actuator-health-list"' in response.data
    assert b'id="actuator-health-error"' in response.data
    assert b'id="log-output"' in response.data
    assert b'role="region"' in response.data
    assert b'role="log"' not in response.data
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
    assert b".actuator-health-row.healthy" in response.data

    response = client.get("/css/logs.css")
    assert response.status_code == HTTPStatus.OK
    assert b".log-output" in response.data
    assert b"overflow-y: scroll" in response.data
    assert b"scrollbar-gutter: stable" in response.data


def test_routes_status(server_client):
    service, client = server_client

    response = client.get(ApiRoute.STATUS)

    assert response.status_code == HTTPStatus.OK
    assert response.json == {"runtime": "sim"}
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    service.status.assert_called_once_with()


def test_routes_incremental_logs(server_client):
    service, client = server_client
    service.logs.return_value = LogSnapshot(
        cursor=4,
        entries=[LogEntry(cursor=4, message="[INFO] Stack started")],
        reset=False,
        capacity=200,
    )

    response = client.get(f"{ApiRoute.LOGS}?after=3")

    assert response.status_code == HTTPStatus.OK
    assert response.json == {
        "cursor": 4,
        "entries": [{"cursor": 4, "message": "[INFO] Stack started"}],
        "reset": False,
        "capacity": 200,
    }
    service.logs.assert_called_once_with(3)


@pytest.mark.parametrize("cursor", ["-1", "invalid"])
def test_rejects_invalid_log_cursor(server_client, cursor):
    service, client = server_client

    response = client.get(f"{ApiRoute.LOGS}?after={cursor}")

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json["ok"] is False
    service.logs.assert_not_called()


def test_serializes_orchestrator_status_dataclass(server_client):
    service, client = server_client
    service.status.return_value = BackendStatus(
        host_resources=HostResourceStatus(25.0, 50.0, 4096, 8192),
        runtime=Runtime.SIM,
        robot=RobotName.PANDA,
        robots=[RobotName.PANDA],
        scene=SimulationScene.EMPTY,
        scenes=list(SimulationScene),
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
        lcm_channels=[LCMChannelStatus("ROBOT/STATE", 100.0, 0.1, True)],
        node_rates=[
            NodeRateStatus(
                node_name="RobotControllerNode",
                pid=456,
                target_rate_hz=500.0,
                measured_rate_hz=497.2,
                healthy=True,
                age_seconds=0.2,
                cpu_percent=21.4,
                memory_rss_mb=86.3,
            )
        ],
        actuator_health=ActuatorHealthStatus(
            connected=True,
            healthy=False,
            age_seconds=0.1,
            actuators=(
                ActuatorHealth(
                    joint_name="panda_joint1",
                    controller="main",
                    actuator_id=1,
                    healthy=False,
                    temperature_celsius=42.0,
                    issue="Overload protection triggered.",
                ),
            ),
            error="Incomplete actuator feedback.",
        ),
        recording=RecordingStatus(
            timestamp=1.0,
            state=RecordingState.RUNNING,
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
        "host_resources": {
            "cpu_percent": 25.0,
            "memory_percent": 50.0,
            "memory_used_bytes": 4096,
            "memory_total_bytes": 8192,
        },
        "robot": "panda",
        "robots": ["panda"],
        "scene": "empty",
        "scenes": ["empty", "toy_cleanup"],
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
        "lcm_channels": [
            {"channel": "ROBOT/STATE", "publish_rate_hz": 100.0, "age_seconds": 0.1, "active": True}
        ],
        "node_rates": [
            {
                "node_name": "RobotControllerNode",
                "pid": 456,
                "target_rate_hz": 500.0,
                "measured_rate_hz": 497.2,
                "healthy": True,
                "age_seconds": 0.2,
                "cpu_percent": 21.4,
                "memory_rss_mb": 86.3,
            }
        ],
        "actuator_health": {
            "connected": True,
            "healthy": False,
            "age_seconds": 0.1,
            "actuators": [
                {
                    "joint_name": "panda_joint1",
                    "controller": "main",
                    "actuator_id": 1,
                    "healthy": False,
                    "temperature_celsius": 42.0,
                    "issue": "Overload protection triggered.",
                }
            ],
            "error": "Incomplete actuator feedback.",
        },
        "recording": {
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
            payload={ApiKey.RUNTIME: "real", **_safety_payload(acknowledged=True)},
            method_name="set_runtime",
            arguments=(Runtime.REAL, _safety_context(acknowledged=True)),
            result={"runtime": "real"},
        ),
        RouteCase(
            path=ApiRoute.SCENE,
            payload={ApiKey.SCENE: "toy_cleanup", **_safety_payload()},
            method_name="set_scene",
            arguments=(SimulationScene.TOY_CLEANUP, _safety_context()),
            result={"scene": "toy_cleanup"},
        ),
        RouteCase(
            path=ApiRoute.ROBOT,
            payload={ApiKey.ROBOT: "panda", **_safety_payload()},
            method_name="set_robot",
            arguments=(RobotName.PANDA, _safety_context()),
            result={"robot": "panda"},
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
            path="/api/recordings/toggle",
            payload={},
            method_name="toggle_recording",
            arguments=(),
            result={"state": "running", "file_name": "recording.lcm"},
        ),
        RouteCase(
            path="/api/recordings/start",
            payload={},
            method_name="set_recording",
            arguments=(ProcessAction.START,),
            result={"recording": "started"},
        ),
        RouteCase(
            path="/api/recordings/start",
            payload={"episode_id": "one", "runtime_id": "agent"},
            method_name="set_episode_recording",
            arguments=(ProcessAction.START, "one", "agent"),
            result={"state": "running", "file_name": "recording.lcm"},
        ),
        RouteCase(
            path="/api/recordings/stop",
            payload={},
            method_name="set_recording",
            arguments=(ProcessAction.STOP,),
            result={"recording": "stopping"},
        ),
        RouteCase(
            path="/api/recordings/stop",
            payload={"episode_id": "one", "runtime_id": "agent"},
            method_name="set_episode_recording",
            arguments=(ProcessAction.STOP, "one", "agent"),
            result={"state": "stopped", "file_name": "recording.lcm"},
        ),
        RouteCase(
            path="/api/replay/start",
            payload={ApiKey.RECORDING: "recording_1", **_safety_payload()},
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
            payload={ApiKey.MODE: "homing", ApiKey.PARAMETERS: {"preset": "home"}},
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
    "payload",
    [
        {"episode_id": "one"},
        {"runtime_id": "agent"},
        {"episode_id": "", "runtime_id": "agent"},
        {"episode_id": "one", "runtime_id": None},
        {"epsiode_id": "one"},
    ],
)
def test_malformed_episode_stop_never_stops_manual_recording(server_client, payload):
    service, client = server_client
    response = client.post("/api/recordings/stop", json=payload)
    assert response.status_code == HTTPStatus.BAD_REQUEST
    service.set_recording.assert_not_called()
    service.set_episode_recording.assert_not_called()


@pytest.mark.parametrize(
    ("path", "data", "content_type", "expected_status"),
    [
        (ApiRoute.ORCHESTRATOR, "{", "application/json", HTTPStatus.BAD_REQUEST),
        (ApiRoute.ORCHESTRATOR, "{}", "text/plain", HTTPStatus.UNSUPPORTED_MEDIA_TYPE),
        (ApiRoute.ORCHESTRATOR, "[]", "application/json", HTTPStatus.BAD_REQUEST),
        (ApiRoute.ORCHESTRATOR, "", "application/json", HTTPStatus.BAD_REQUEST),
        ("/api/runtime", '{"runtime":"hardware"}', "application/json", HTTPStatus.BAD_REQUEST),
        ("/api/robot", '{"robot":"unknown"}', "application/json", HTTPStatus.BAD_REQUEST),
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
        ("/api/recordings/restart", "{}", "application/json", HTTPStatus.NOT_FOUND),
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
        data=b"x" * (MAX_REQUEST_BYTES + 1),
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
        json={ApiKey.RUNTIME: "sim"},
        headers=headers,
    )

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json["ok"] is False
    service.set_runtime.assert_not_called()


def test_service_error_returns_conflict(server_client):
    service, client = server_client
    service.start_process.side_effect = BackendError("stack unavailable")

    response = client.post("/api/processes/stack/start", json=_safety_payload())

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json == {"ok": False, "error": "stack unavailable"}


def test_external_stack_status_returns_conflict(server_client):
    service, client = server_client
    service.status.side_effect = BackendError(
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


def test_run_agent_uses_operator_configuration_and_mode(server_client):
    service, client = server_client
    service.run_agent.return_value = {"phase": "running"}
    payload = {
        **_safety_payload(),
        "instruction": "read the mode",
        "model": "vision",
        "mode": "idle",
    }
    response = client.post(ApiRoute.AGENT_RUN, json=payload)
    assert response.json == {"ok": True, "status": {"phase": "running"}}
    service.run_agent.assert_called_once_with(
        "read the mode", "vision", Mode.IDLE, _safety_context()
    )
    assert (
        client.post(ApiRoute.AGENT_RUN, json={**payload, "instruction": " "}).status_code
        == HTTPStatus.BAD_REQUEST
    )
    service.run_agent.assert_called_once()


def test_recording_status_is_available_without_robot_observations(server_client):
    service, client = server_client
    service.recording_status.return_value = RecordingStatus(
        timestamp=1, state=RecordingState.STOPPED
    )
    response = client.get("/api/recordings")
    assert response.status_code == HTTPStatus.OK
    assert response.json["status"]["state"] == "stopped"
    service.recording_status.assert_called_once_with()


def test_toggle_rejects_episode_payload(server_client):
    service, client = server_client
    response = client.post("/api/recordings/toggle", json={"episode_id": "other"})
    assert response.status_code == HTTPStatus.BAD_REQUEST
    service.toggle_recording.assert_not_called()
