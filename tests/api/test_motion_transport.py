import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from multiprocessing import get_context
from unittest.mock import MagicMock

import numpy as np
import pytest

from humanoid.api.services.motion import MotionService
from humanoid.config.motion import MOTION_PROTOCOL_CONFIG
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.constants import private_lcm_url
from humanoid.nodes.policy.motion import MotionPolicyNode
from humanoid.types.homing import HomingPreset
from humanoid.types.motion import ControlOperation, MotionRequest, MotionState
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import RobotState


def _observation():
    positions = TRISKEL_CONFIG.homing_presets[HomingPreset.HOME].copy()
    return Observation(
        robot_state=RobotState(
            timestamp=time.perf_counter(),
            joint_positions=positions,
            joint_velocities=np.zeros(len(positions)),
            actuator_temperatures=np.zeros(len(positions)),
        ),
        mode=Mode.AUTONOMY,
    )


@pytest.fixture
def connected():
    parent, child = get_context("spawn").Pipe()
    with (
        closing(MotionService()) as service,
        closing(
            MotionPolicyNode(child, "session", TRISKEL_CONFIG, lcm_url=private_lcm_url())
        ) as node,
    ):
        service.attach(parent, "session")
        observation = _observation()
        node.executor.observe(observation)
        yield service, node, observation


def test_pipe_acceptance_execution_cancellation_and_duplicate_suppression(connected):
    service, node, observation = connected
    request = MotionRequest("one", ControlOperation.HOLD, {"duration": 1}, 1)
    assert service.submit(request, "session").state is MotionState.QUEUED
    assert service.submit(request, "session").state is MotionState.QUEUED
    node.policy(observation)
    assert service.status("one", "session").state is MotionState.RUNNING
    assert service.cancel("one", "session").state is MotionState.CANCELLED
    assert service.submit(request, "session").state is MotionState.CANCELLED
    with pytest.raises(ValueError, match="already used"):
        service.submit(replace(request, arguments={"duration": 0.2}), "session")


def test_pipe_preserves_start_deadline_and_rejects_stale_sessions(connected):
    service, node, observation = connected
    request = MotionRequest("expires", ControlOperation.HOLD, {"duration": 1}, 0.01)
    service.submit(request, "session")
    time.sleep(0.03)
    service.submit(request, "session")
    observation.robot_state.timestamp = time.perf_counter()
    node.policy(observation)
    assert service.status("expires", "session").state is MotionState.EXPIRED
    with pytest.raises(RuntimeError, match="session has expired"):
        service.submit(request, "previous")


def test_disconnect_invalidates_session_and_does_not_replay_motion(connected):
    service, node, _ = connected
    service.submit(MotionRequest("one", ControlOperation.HOLD, {}, 1), "session")
    node.pipe.close()
    deadline = time.monotonic() + 3
    while service.motion_session_id == "session" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert service.motion_session_id != "session"
    with pytest.raises(RuntimeError):
        service.status("one", "session")


def test_mode_admission_is_checked_in_executor(connected):
    service, node, observation = connected
    node.executor.observe(replace(observation, mode=Mode.IDLE))
    with pytest.raises(RuntimeError, match="policy mode"):
        service.submit(MotionRequest("idle", ControlOperation.HOLD, {}, 1), "session")
    assert not node.executor._motions


def test_concurrent_duplicate_submissions_share_one_executor_record(connected):

    service, node, _ = connected
    request = MotionRequest("duplicate", ControlOperation.HOLD, {}, 1)
    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: service.submit(request, "session"), range(8)))
    assert all(status.state is MotionState.QUEUED for status in statuses)
    assert list(node.executor._motions) == ["duplicate"]


def test_lost_acknowledgement_retry_does_not_extend_deadline(connected, monkeypatch):
    service, node, observation = connected
    real_send = node.pipe.send
    lost = False

    def drop_first_ack(message):
        nonlocal lost
        if message.status is not None and not lost:
            lost = True
            return
        real_send(message)

    monkeypatch.setattr(node.pipe, "send", drop_first_ack)
    monkeypatch.setattr(
        "humanoid.api.services.motion.MOTION_PROTOCOL_CONFIG",
        replace(MOTION_PROTOCOL_CONFIG, request_timeout_seconds=0.05),
    )
    request = MotionRequest("lost", ControlOperation.HOLD, {}, 0.02)
    with pytest.raises(RuntimeError, match="acknowledgement timed out"):
        service.submit(request, "session")
    assert service.submit(request, "session").state is MotionState.QUEUED
    observation.robot_state.timestamp = time.perf_counter()
    node.policy(observation)
    assert service.status("lost", "session").state is MotionState.EXPIRED
    assert len(node.executor._motions) == 1


def test_disconnect_cancels_motion_publishes_hold_and_releases_mode(connected, monkeypatch):

    service, node, observation = connected
    service.submit(MotionRequest("active", ControlOperation.HOLD, {}, 1), "session")
    node.policy(observation)
    publisher = MagicMock()
    monkeypatch.setattr(node.env, "observe", lambda: observation)
    monkeypatch.setattr(node.env, "step", publisher)
    release = MagicMock()
    monkeypatch.setattr("humanoid.nodes.policy.motion.OrchestratorClient", release)
    service.detach()
    node.pipe.closed.wait(2)
    observation.robot_state.timestamp = time.perf_counter()
    node.step()
    assert node.stop_condition()
    assert node.executor.status("active").state is MotionState.CANCELLED
    action = publisher.call_args.args[0]
    assert action.base_pose is not None
    release.return_value.release_autonomy.assert_called_once()


def test_new_executor_session_rejects_old_ids_and_never_replays(connected):
    service, _, _ = connected
    request = MotionRequest("previous", ControlOperation.HOLD, {}, 1)
    service.submit(request, "session")
    parent, child = get_context("spawn").Pipe()
    with closing(
        MotionPolicyNode(child, "replacement", TRISKEL_CONFIG, lcm_url=private_lcm_url())
    ) as next_node:
        next_node.executor.observe(_observation())
        service.attach(parent, "replacement")
        with pytest.raises(RuntimeError, match="session has expired"):
            service.submit(request, "session")
        with pytest.raises(KeyError):
            service.status(request.motion_id, "replacement")
        assert not next_node.executor._motions


def test_readiness_requires_executor_feedback_in_autonomy(connected):
    service, node, observation = connected
    node.executor.observe(replace(observation, mode=Mode.IDLE))
    assert not node.executor.ready
    observation.robot_state.timestamp = time.perf_counter()
    node.executor.observe(observation)
    assert node.executor.ready
    service.submit(MotionRequest("readiness", ControlOperation.HOLD, {}, 1), "session")
    assert service.ready
    service.detach()
    assert not service.ready
