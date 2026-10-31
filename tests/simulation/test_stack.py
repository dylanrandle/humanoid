"""Exercise the production control path across independently running processes."""

import time

import numpy as np
import pytest

from humanoid.config.collection import DEFAULT_COLLECTION_CONFIG
from humanoid.constants import Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.simulation.client import SimulationClient
from humanoid.simulation.tasks.cleanup.task import CleanupTask
from humanoid.simulation.tasks.empty import EmptySceneTask
from humanoid.types.collection import CollectionCancelled
from humanoid.types.orchestrator import EventKind, Mode, OrchestratorEvent
from humanoid.types.robot import RobotJointCommand, RobotState
from humanoid.types.simulation import SimulationReset, SimulationState


def test_policy_targets_traverse_normal_processes_and_physics_runs_without_client(running_stack):
    runtime = running_stack.client
    task = CleanupTask()
    witness = Subscriber(
        [
            Topic.SYSTEM_TOOL_COMMAND,
            Topic.CONTROLLER_JOINT_COMMAND,
            Topic.ROBOT_JOINT_COMMAND,
            Topic.ROBOT_STATE,
        ],
        url=running_stack.url,
    )
    try:
        runtime.reset(task, 0, DEFAULT_COLLECTION_CONFIG)
        runtime.activate()
        initial = runtime.state().joint_positions.copy()
        target = initial.copy()
        robot = runtime.engine.robot
        target[robot.get_joint_position_indices(robot.get_arm_joint_indices())[0]] += 0.1
        for _ in range(20):
            runtime.step(target)
        for topic in (
            Topic.SYSTEM_TOOL_COMMAND,
            Topic.CONTROLLER_JOINT_COMMAND,
            Topic.ROBOT_JOINT_COMMAND,
        ):
            assert witness.receive(topic, 1000) is not None
        assert not np.allclose(runtime.state().joint_positions, initial)
        assert len({process.pid for process in running_stack.processes}) == 4  # noqa: PLR2004
        before = runtime.snapshot.sim_seconds
        time.sleep(0.25)  # No policy calls, acknowledgements, or explicit physics stepping.
        runtime.poll()
        assert runtime.snapshot.sim_seconds > before + 0.1
        runtime.stop()
        assert runtime.mode is Mode.IDLE
        assert all(process.is_alive() for process in running_stack.processes)
    finally:
        witness.close()


def test_scene_resets_change_seed_and_retries_do_not_reset_again(running_stack):
    runtime = running_stack.client
    task = CleanupTask()
    pids = [process.pid for process in running_stack.processes]
    runtime.reset(task, 17, DEFAULT_COLLECTION_CONFIG)
    first = task.objects(runtime.engine)[0].pose.translation.copy()
    reset_id = runtime.snapshot.reset_id
    before = runtime.snapshot.sim_seconds
    for _ in range(3):
        runtime.publisher.publish(
            SimulationReset(time.perf_counter(), reset_id, task.name, 17), Topic.SIMULATION_RESET
        )
        time.sleep(0.1)
    runtime.poll()
    assert runtime.snapshot.sim_seconds > before + 0.1
    runtime.reset(task, 24, DEFAULT_COLLECTION_CONFIG)
    second = task.objects(runtime.engine)[0].pose.translation
    assert not np.allclose(first[:2], second[:2])
    assert [process.pid for process in running_stack.processes] == pids
    assert runtime.mode is Mode.IDLE
    assert runtime.logging_status.state.value == "stopped"


def test_scene_reset_is_rejected_in_active_mode(running_stack):
    runtime = running_stack.client
    runtime.reset(CleanupTask(), 0, DEFAULT_COLLECTION_CONFIG)
    runtime.activate()
    original = runtime.snapshot.reset_id
    request = SimulationReset(time.perf_counter(), "unsafe", "cleanup", 1)
    deadline = time.monotonic() + 3
    status = None
    while time.monotonic() < deadline:
        runtime.publisher.publish(request, Topic.SIMULATION_RESET)
        status = runtime.subscriber.receive(Topic.SIMULATION_RESET_STATUS, 100)
        if status is not None and status.request_id == request.request_id:
            break
    assert status is not None and "Idle" in status.error
    runtime.poll()
    assert runtime.snapshot.reset_id == original
    runtime.stop()


def test_operator_mode_change_cancels_policy_without_stopping_physics(stack_runtime):
    runtime = stack_runtime(CleanupTask())
    runtime.publisher.publish(
        OrchestratorEvent(time.perf_counter(), EventKind.REQUEST_IDLE), Topic.ORCHESTRATOR_EVENT
    )
    time.sleep(0.1)
    with pytest.raises(CollectionCancelled):
        runtime.step(runtime.state().joint_positions)
    before = runtime.time
    time.sleep(0.1)
    runtime.poll()
    assert runtime.time > before


def test_empty_scene_removes_task_geometry_without_restarting_stack(running_stack):
    runtime = running_stack.client
    pids = [process.pid for process in running_stack.processes]
    runtime.reset(CleanupTask(), 0, DEFAULT_COLLECTION_CONFIG)
    assert runtime.engine.model.ncam > 0
    runtime.reset(EmptySceneTask(runtime.robot_config), 0, DEFAULT_COLLECTION_CONFIG)
    assert runtime.engine.model.ncam == 0
    assert runtime.engine.read_scene_state(0).geoms == ()
    assert runtime.mode is Mode.IDLE
    assert [process.pid for process in running_stack.processes] == pids
    runtime.activate()
    initial = runtime.time
    for _ in range(10):
        runtime.step(runtime.state().joint_positions)
    assert runtime.time > initial
    runtime.stop()


def test_editing_an_expert_seed_cannot_change_cached_feedback():
    client = SimulationClient(lcm_url="memq://")
    try:
        client.snapshot = SimulationState(
            1.0,
            "scene",
            0.1,
            np.zeros(3),
            np.zeros(3),
            RobotState(1.0, np.zeros(3), np.zeros(3), np.zeros(3)),
            RobotJointCommand(0.9, np.zeros(3), np.zeros(3)),
        )
        seed = client.state().joint_positions
        seed[:] = 2.0
        assert client.state().joint_positions == pytest.approx(0)
        assert client.snapshot.state.joint_positions == pytest.approx(0)
    finally:
        client.close()
