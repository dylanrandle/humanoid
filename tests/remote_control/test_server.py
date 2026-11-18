import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from humanoid.remote_control.client import RemoteControlClient
from humanoid.remote_control.server import RemoteControlServer
from humanoid.types.remote_api import ControlOperation, RemoteControlEndpoint


@pytest.fixture
def connection(service):
    server = RemoteControlServer(service)
    server.start()
    host, port = server.server.server_address[:2]
    client = RemoteControlClient(
        RemoteControlEndpoint(f"http://{host}:{port}", service.config.token)
    )
    yield client, service
    client.close()
    server.close()


def payload(client, command_id="move", **arguments):
    state = client._request("/v1/state")
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
    assert client._request("/v1/commands", request)["state"] == "queued"
    assert client._request("/v1/commands", request)["state"] == "queued"
    service.policy(service.observation())
    assert client._request("/v1/commands/move")["state"] == "running"
    assert client._request("/v1/commands/move/cancel", {})["state"] == "cancelled"


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


def test_bad_credentials_and_old_sessions_cannot_submit(connection):
    client, service = connection
    request = payload(client)
    client.endpoint = replace(client.endpoint, token="wrong")
    with pytest.raises(RuntimeError, match="credential"):
        client._request("/v1/commands", request)
    client.endpoint = replace(client.endpoint, token=service.config.token)
    request["session_id"] = "previous-server-instance"
    with pytest.raises(RuntimeError, match="session has expired"):
        client._request("/v1/commands", request)
    assert not service._commands


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
