import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from http import HTTPStatus
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from humanoid.agent.client import AgentControlClient
from humanoid.agent.server import AgentControlServer
from humanoid.types.agent import AgentControlEndpoint, AgentRoute, ControlOperation
from humanoid.types.orchestrator import Mode


@pytest.fixture
def connection(service):
    server = AgentControlServer(service)
    server.start()
    host, port = server.server.server_address[:2]
    client = AgentControlClient(AgentControlEndpoint(f"http://{host}:{port}"))
    yield client, service
    client.close()
    server.close()


def payload(client, command_id="move", **arguments):
    state = client._request(AgentRoute.STATE)
    return {
        "session_id": state["session_id"],
        "command_id": command_id,
        "operation": "move_base",
        "arguments": arguments or {"delta": [0.04, 0, 0], "duration": 0.1},
        "expires_in_seconds": 1.0,
    }


def test_http_submit_returns_before_execution_and_cancel_works(connection):
    client, service = connection
    request = payload(client)
    assert client._request(AgentRoute.COMMANDS, request)["state"] == "queued"
    assert client._request(AgentRoute.COMMANDS, request)["state"] == "queued"
    service.policy(service.observation())
    assert client._request(AgentRoute.COMMAND.format(command_id="move"))["state"] == "running"
    assert client._request(AgentRoute.CANCEL.format(command_id="move"), {})["state"] == "cancelled"


@pytest.mark.parametrize(
    "query",
    ["depth_view=unknown", "registered=1", "depth_view=", "depth_view=native&depth_view=aligned"],
)
def test_camera_rejects_unsupported_depth_selectors(connection, query):
    client, _ = connection
    with pytest.raises(HTTPError) as error:
        urlopen(f"{client.endpoint.url}/v1/camera?{query}", timeout=1)
    assert error.value.code == HTTPStatus.BAD_REQUEST


def test_client_waits_while_policy_loop_continues(connection):
    client, service = connection
    observation = service.observation()
    stopped = threading.Event()
    actions = []

    def loop():
        while not stopped.is_set():
            observation.robot_state.timestamp = time.perf_counter()
            service.observe(observation)
            actions.append(service.policy(observation))
            stopped.wait(0.01)

    worker = threading.Thread(target=loop)
    worker.start()
    try:
        outcome = client.execute(
            ControlOperation.MOVE_BASE, {"delta": [0.04, 0, 0], "duration": 0.1}
        )
        assert not outcome["reached"]  # Stationary feedback cannot fabricate a successful move.
        assert outcome["position_error_m"] == pytest.approx(0.04)
        assert any(action.base_pose is not None for action in actions)
    finally:
        stopped.set()
        worker.join(timeout=1)


@pytest.mark.parametrize(
    "headers",
    [{"Host": "example.org"}, {"Origin": "https://example.org"}, {"Sec-Fetch-Site": "cross-site"}],
)
def test_browser_requests_from_other_sites_are_forbidden(connection, headers):
    client, _ = connection
    request = Request(client.endpoint.url + AgentRoute.STATE, headers=headers)
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=1)
    assert error.value.code == HTTPStatus.FORBIDDEN


def test_old_sessions_cannot_submit(connection):
    client, service = connection
    request = payload(client)
    request["session_id"] = "previous-server-instance"
    with pytest.raises(RuntimeError, match="session has expired"):
        client._request(AgentRoute.COMMANDS, request)
    assert not service._commands


def test_idle_sends_release_while_joint_feedback_is_stale(service):
    sample = service.observation()
    request_count = 0

    def release():
        nonlocal request_count
        request_count += 1
        if request_count > 1:
            # Fresh feedback arrives after the operator has taken control.
            state = replace(sample.robot_state, timestamp=time.perf_counter())
            service.observe(replace(sample, robot_state=state, mode=Mode.KEYBOARD))

    with closing(AgentControlServer(service, release)) as server:
        server.start()
        host, port = server.server.server_address[:2]
        with closing(AgentControlClient(AgentControlEndpoint(f"http://{host}:{port}"))) as client:
            assert client.session_id == service.session_id
            service.observe(replace(sample, robot_state=replace(sample.robot_state, timestamp=0)))
            client.idle()
            assert request_count > 1
            assert client.observation().mode is Mode.KEYBOARD


def test_client_cancellation_does_not_wait_for_motion_completion(connection):
    client, service = connection
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(client.execute, ControlOperation.HOLD, {"duration": 2})
        deadline = time.perf_counter() + 2
        while not service._commands and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert service._commands
        client.cancel()
        with pytest.raises(RuntimeError, match="stopped"):
            result.result(timeout=1)
    assert all(record.result.done() for record in service._commands.values())
