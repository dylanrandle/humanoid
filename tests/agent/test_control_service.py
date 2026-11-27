import time
from dataclasses import replace

import pytest

from humanoid.types.motion import ControlOperation, MotionRequest, MotionState
from humanoid.types.orchestrator import Mode


def request(motion_id="move", **arguments):
    return MotionRequest(
        motion_id,
        ControlOperation.MOVE_BASE,
        arguments or {"delta": [0.04, 0, 0], "duration": 0.1},
        1.0,
    )


def test_duplicates_return_original_result_and_conflicts_do_not_move(service):
    original = request()
    assert service.submit(original).state is MotionState.QUEUED
    assert service.submit(original).state is MotionState.QUEUED
    with pytest.raises(ValueError, match="different command"):
        service.submit(request(delta=[0.02, 0, 0]))
    service.policy(service._observation)
    assert service.status("move").state is MotionState.RUNNING
    with pytest.raises(RuntimeError, match="already running"):
        service.submit(request("another"))
    assert service.cancel("move").state is MotionState.CANCELLED
    assert service.submit(original).state is MotionState.CANCELLED
    assert service.submit(request("next")).state is MotionState.QUEUED
    # Re-cancelling an old command cannot interrupt a newer one.
    service.cancel("move")
    assert service.status("next").state is MotionState.QUEUED


def test_validation_occurs_on_policy_loop(service):
    service.submit(request(delta=[1, 0, 0]))
    service.policy(service._observation)
    assert service.status("move").state is MotionState.REJECTED


def test_deadlines_use_receiving_clock_and_are_not_refreshed_by_retry(service, monkeypatch):
    original = request()
    service.submit(original)
    now = time.perf_counter() + 1.1
    observation = service._observation
    observation.robot_state.timestamp = now
    monkeypatch.setattr("humanoid.policy.motion.time.perf_counter", lambda: now)
    service.submit(original)
    service.policy(observation)
    assert service.status("move").state is MotionState.EXPIRED


def test_history_limit_never_evicts_ids_that_could_replay_motion(service):
    service.config = replace(service.config, max_motions=1)
    service.submit(request())
    service.cancel("move")
    with pytest.raises(RuntimeError, match="history is full"):
        service.submit(request("new"))
    assert service.submit(request()).state is MotionState.CANCELLED


def test_invalid_hold_does_not_interrupt_another_command(service):
    service.submit(request())
    with pytest.raises(ValueError):
        service.submit(MotionRequest("bad", ControlOperation.HOLD, {"extra": 1}, 1.0))
    assert service.status("move").state is MotionState.QUEUED
    service.submit(MotionRequest("hold", ControlOperation.HOLD, {}, 1.0))
    assert service.status("move").state is MotionState.CANCELLED


def test_shutdown_cancels_work_and_refuses_new_motions(service):
    service.submit(request())
    service.close()
    assert service.status("move").state is MotionState.CANCELLED
    with pytest.raises(RuntimeError, match="not ready"):
        service.submit(request("new"))


def test_inactive_mode_never_executes_queued_motion(service):
    observation = service._observation
    service.submit(request())
    observation.mode = Mode.IDLE
    assert service.policy(observation).base_pose is None
    assert service.status("move").state is MotionState.CANCELLED


def test_idle_submission_cannot_start_when_autonomy_activates(service):
    observation = service._observation
    observation.mode = Mode.IDLE
    with pytest.raises(RuntimeError, match="mode"):
        service.submit(request())
    observation.mode = Mode.AUTONOMY
    service.policy(observation)
    with pytest.raises(KeyError):
        service.status("move")
