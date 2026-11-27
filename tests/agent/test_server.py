import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from http import HTTPStatus
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from humanoid.api.clients.robot import RobotClient
from humanoid.types.api import ApiEndpoint, ApiRoute
from humanoid.types.motion import ControlOperation
from humanoid.types.orchestrator import Mode
from tests.api.helpers import ApiTestServer


@pytest.fixture
def connection(service):
    server = ApiTestServer(service)
    server.start()
    host, port = server.server.server_address[:2]
    client = RobotClient(ApiEndpoint(f"http://{host}:{port}"))
    yield client, service, server.backend.observation_service
    client.close()
    server.close()


def payload(client, motion_id="move", **arguments):
    state = client._request(ApiRoute.STATE)
    return {
        "motion_session_id": state["motion_session_id"],
        "motion_id": motion_id,
        "operation": "move_base",
        "arguments": arguments or {"delta": [0.04, 0, 0], "duration": 0.1},
        "expires_in_seconds": 1.0,
    }


def test_http_submit_returns_before_execution_and_cancel_works(connection):
    client, service, _observations = connection
    request = payload(client)
    assert client._request(ApiRoute.MOTIONS, request)["state"] == "queued"
    assert client._request(ApiRoute.MOTIONS, request)["state"] == "queued"
    service.policy(service._observation)
    assert client._request(ApiRoute.MOTION.replace("<motion_id>", "move"))["state"] == "running"
    assert (
        client._request(ApiRoute.MOTION_CANCEL.replace("<motion_id>", "move"), {})["state"]
        == "cancelled"
    )


@pytest.mark.parametrize(
    "query",
    ["depth_view=unknown", "registered=1", "depth_view=", "depth_view=native&depth_view=aligned"],
)
def test_camera_rejects_unsupported_depth_selectors(connection, query):
    client, _, _ = connection
    with pytest.raises(HTTPError) as error:
        urlopen(f"{client.endpoint.url}/api/camera?{query}", timeout=1)
    assert error.value.code == HTTPStatus.BAD_REQUEST


def test_client_waits_while_policy_loop_continues(connection):
    client, service, _observations = connection
    observation = service._observation
    stopped = threading.Event()
    actions = []

    def loop():
        while not stopped.is_set():
            observation.robot_state.timestamp = time.perf_counter()
            service.observe(observation)
            _observations.observe(observation)
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
    client, _, _ = connection
    request = Request(client.endpoint.url + ApiRoute.STATE, headers=headers)
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=1)
    assert error.value.code == HTTPStatus.FORBIDDEN


def test_old_sessions_cannot_submit(connection):
    client, service, _observations = connection
    request = payload(client)
    request["motion_session_id"] = "previous-server-instance"
    with pytest.raises(RuntimeError, match="session has expired"):
        client._request(ApiRoute.MOTIONS, request)
    assert not service._motions


def test_release_autonomy_sends_request_while_joint_feedback_is_stale(observations):
    sample = observations.observation()
    request_count = 0

    def release():
        nonlocal request_count
        request_count += 1
        if request_count > 1:
            # Fresh feedback arrives after the operator has taken control.
            state = replace(sample.robot_state, timestamp=time.perf_counter())
            observations.observe(replace(sample, robot_state=state, mode=Mode.KEYBOARD))

    with closing(ApiTestServer(observations, release)) as server:
        server.start()
        host, port = server.server.server_address[:2]
        with closing(RobotClient(ApiEndpoint(f"http://{host}:{port}"))) as client:
            assert client.motion_session_id == server.backend.motion_service.motion_session_id
            observations.observe(
                replace(sample, robot_state=replace(sample.robot_state, timestamp=0))
            )
            client.release_autonomy()
            assert request_count > 1
            assert client.observation().mode is Mode.KEYBOARD


def test_client_cancellation_does_not_wait_for_motion_completion(connection):
    client, service, _observations = connection
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(client.execute, ControlOperation.HOLD, {"duration": 2})
        deadline = time.perf_counter() + 2
        while not service._motions and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert service._motions
        client.cancel()
        with pytest.raises(RuntimeError, match="stopped"):
            result.result(timeout=1)
    assert all(record.result.done() for record in service._motions.values())


def test_observations_survive_motion_restart_but_old_motion_authority_does_not(observations):
    with closing(ApiTestServer(observations)) as server:
        server.start()
        host, port = server.server.server_address[:2]
        with closing(RobotClient(ApiEndpoint(f"http://{host}:{port}"))) as client:
            runtime_id = client.runtime_id
            motion_id = client.motion_session_id
            server.backend.motion_service.detach()
            assert server.backend.motion_service.motion_session_id != motion_id
            client.observation()
            assert client.runtime_id == runtime_id
            with pytest.raises(RuntimeError, match="Motion executor restarted"):
                _ = client.motion_session_id
            server.backend.runtime_id = "replacement-runtime"
            with pytest.raises(RuntimeError, match="Robot runtime restarted"):
                client.observation()


def test_read_only_client_can_acquire_first_motion_session_after_restart(observations):
    with closing(ApiTestServer(observations)) as server:
        server.start()
        host, port = server.server.server_address[:2]
        with closing(RobotClient(ApiEndpoint(f"http://{host}:{port}"))) as client:
            client.observation()
            server.backend.motion_service.detach()
            client.observation()
            assert client.motion_session_id == server.backend.motion_service.motion_session_id
