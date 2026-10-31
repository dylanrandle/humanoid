"""LCM boundary tests for the MuJoCo simulation node."""

import multiprocessing
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import mujoco
import numpy as np
import pytest

from humanoid.config.robot.so101 import SO101_CONFIG
from humanoid.constants import Topic
from humanoid.middleware.publisher import Publisher
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.robot import simulation
from humanoid.nodes.robot.simulation import MujocoSimulationNode
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.types.logging import LoggingState, LoggingStatus
from humanoid.types.orchestrator import Mode, OrchestratorMode
from humanoid.types.robot import NormalizedRobotJointCommand, RobotJointCommand, RobotState
from humanoid.types.simulation import MujocoSimulationConfig, SimulationReset

SPAWNED_SIMULATION_TIMEOUT_SECONDS = 15.0
EXPECTED_SIMULATION_RATE_HZ = 200.0
EXPECTED_SIMULATION_SUBSTEPS = 1


def _run_spawned_simulation(result_queue) -> None:
    node = MujocoSimulationNode(robot_config=SO101_CONFIG, lcm_url="memq://")
    try:
        for _ in range(10):
            node.step()
        state = node.engine.read_robot_state(timestamp=1.0)
        result_queue.put(state.joint_positions.tolist())
    finally:
        node.on_close()


def _state(timestamp: float = 1.0) -> RobotState:
    return RobotState(
        timestamp=timestamp,
        joint_positions=np.zeros(6),
        joint_velocities=np.zeros(6),
        actuator_temperatures=np.zeros(6),
    )


def _make_node(monkeypatch, *, clock=None, timeout: float = 0.25):
    subscriber = MagicMock(spec=Subscriber)
    subscriber.receive.return_value = None
    publisher = MagicMock(spec=Publisher)
    engine = MagicMock(spec=NativeMujocoEngine)
    engine.physics_timestep = MujocoSimulationConfig().physics_timestep
    engine.robot = MagicMock()
    engine.robot.model.nv = 6
    engine.read_robot_state.return_value = _state()
    engine.apply_joint_command.return_value = NormalizedRobotJointCommand(
        joint_positions=np.zeros(6),
        joint_velocities=np.zeros(6),
    )
    monkeypatch.setattr(
        "humanoid.nodes.robot.simulation.Subscriber",
        MagicMock(return_value=subscriber),
    )
    monkeypatch.setattr(
        "humanoid.nodes.robot.simulation.Publisher",
        MagicMock(return_value=publisher),
    )
    node = MujocoSimulationNode(
        robot_config=SO101_CONFIG,
        engine=engine,
        command_timeout_seconds=timeout,
        clock=clock or MagicMock(return_value=1.0),
    )
    return node, engine, subscriber, publisher


def test_steps_physics_and_publishes_state(monkeypatch):
    clock = MagicMock(side_effect=[1.0, 1.0, 1.01])
    node, engine, subscriber, publisher = _make_node(monkeypatch, clock=clock)
    command = RobotJointCommand(timestamp=1.0, joint_positions=np.zeros(6))
    subscriber.receive.side_effect = lambda topic: (
        command if topic is Topic.ROBOT_JOINT_COMMAND else None
    )

    node.step()

    engine.apply_joint_command.assert_called_once_with(command)
    engine.step.assert_called_once_with(EXPECTED_SIMULATION_SUBSTEPS)
    engine.read_robot_state.assert_called_once_with(timestamp=1.01)
    publisher.publish.assert_called_once_with(
        engine.read_robot_state.return_value,
        topic=Topic.ROBOT_STATE,
    )


def test_default_simulation_runs_at_two_hundred_hz(monkeypatch):
    node, _, _, _ = _make_node(monkeypatch)

    assert node.rate_hz == pytest.approx(EXPECTED_SIMULATION_RATE_HZ)
    assert node.substeps == EXPECTED_SIMULATION_SUBSTEPS


def test_watchdog_stops_velocity_actuators_once(monkeypatch):
    clock = MagicMock(return_value=1.0)
    node, engine, subscriber, _ = _make_node(monkeypatch, clock=clock, timeout=0.25)
    engine.apply_joint_command.return_value = NormalizedRobotJointCommand(
        joint_positions=np.zeros(6),
        joint_velocities=np.ones(6),
    )
    commands = iter([RobotJointCommand(timestamp=1.0, joint_positions=np.zeros(6)), None, None])
    subscriber.receive.side_effect = lambda topic: (
        next(commands) if topic is Topic.ROBOT_JOINT_COMMAND else None
    )
    node.step()
    clock.return_value = 1.3
    node.step()
    clock.return_value = 1.6
    node.step()

    engine.stop_velocity_actuators.assert_called_once_with()


def test_close_stops_simulation_before_closing_transport(monkeypatch):
    node, engine, subscriber, _ = _make_node(monkeypatch)
    operations = MagicMock()
    engine.stop_velocity_actuators.side_effect = lambda: operations("stop")
    subscriber.close.side_effect = lambda: operations("close")

    node.on_close()

    assert operations.call_args_list == [call("stop"), call("close")]


def test_rejects_invalid_timing_or_watchdog(monkeypatch):
    with pytest.raises(ValueError, match="timeout must be positive"):
        _make_node(monkeypatch, timeout=0.0)

    engine = MagicMock(spec=NativeMujocoEngine)
    engine.physics_timestep = 0.003
    with pytest.raises(ValueError, match="must not exceed"):
        MujocoSimulationNode(
            robot_config=SO101_CONFIG,
            simulation_config=MujocoSimulationConfig(publish_rate_hz=500.0),
            engine=engine,
        )


def test_simulation_constructs_and_steps_in_a_spawned_process():
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_run_spawned_simulation, args=(result_queue,))
    process.start()
    process.join(timeout=SPAWNED_SIMULATION_TIMEOUT_SECONDS)
    try:
        assert process.is_alive() is False
        assert process.exitcode == 0
        assert np.isfinite(result_queue.get(timeout=1.0)).all()
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        result_queue.close()
        result_queue.join_thread()


def test_scene_reset_uses_registered_task_and_discards_old_commands(monkeypatch):
    def scene(spec):
        spec.worldbody.add_geom(
            name="selected_goal",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.17, 1, 1],
            size=[0.1, 0.1, 0.1],
            contype=2,
        )

    task = SimpleNamespace(
        robot=SO101_CONFIG.name,
        create_engine=MagicMock(
            side_effect=lambda seed: NativeMujocoEngine(SO101_CONFIG, scene_builder=scene)
        ),
    )
    monkeypatch.setitem(simulation.TASKS, "alternate", lambda **kwargs: task)
    node = MujocoSimulationNode(robot_config=SO101_CONFIG, lcm_url="memq://")
    node.subscriber.close()
    messages = {
        Topic.ORCHESTRATOR_MODE: OrchestratorMode(0, Mode.IDLE),
        Topic.LOGGING_STATUS: LoggingStatus(time.time(), LoggingState.STOPPED),
    }
    node.subscriber = MagicMock()
    node.subscriber.receive.side_effect = messages.get
    node.publisher = MagicMock()
    try:
        request = SimulationReset(time.perf_counter(), "scene", "alternate", 17)
        node._reset(request)
        assert node._reset_status is not None
        assert node._reset_status.error == ""
        assert node.engine.model.geom("selected_goal").pos[0] == pytest.approx(0.17)
        node.engine.step(2)
        advanced = node.engine.data.time
        node._reset(request)
        assert node.engine.data.time == advanced
        task.create_engine.assert_called_once_with(17)
        stale = RobotJointCommand(node._accept_commands_after - 1, np.ones(6))
        messages[Topic.ROBOT_JOINT_COMMAND] = stale
        apply = MagicMock(wraps=node.engine.apply_joint_command)
        monkeypatch.setattr(node.engine, "apply_joint_command", apply)
        node.step()
        apply.assert_not_called()
    finally:
        node.on_close()


def test_scene_timestep_can_differ_without_changing_driver_rate():
    engine = NativeMujocoEngine(SO101_CONFIG, MujocoSimulationConfig(physics_timestep=0.002))
    node = MujocoSimulationNode(robot_config=SO101_CONFIG, engine=engine, lcm_url="memq://")
    try:
        for _ in range(10):
            node.step()
        assert engine.data.time == pytest.approx(10 / EXPECTED_SIMULATION_RATE_HZ)
        assert node.rate_hz == EXPECTED_SIMULATION_RATE_HZ
    finally:
        node.on_close()


def test_empty_driver_does_not_erase_replayed_scene_geometry():
    node = MujocoSimulationNode(robot_config=SO101_CONFIG, lcm_url="memq://")
    node.publisher = MagicMock()
    try:
        for _ in range(node._scene_interval):
            node.step()
        assert all(
            call.kwargs.get("topic") is Topic.ROBOT_STATE
            for call in node.publisher.publish.call_args_list
        )
        node._reset_id = "empty-scene"
        node._applied = RobotJointCommand(0, node.engine.read_robot_state(0).joint_positions)
        for _ in range(node._scene_interval):
            node.step()
        scenes = [
            call.args[0]
            for call in node.publisher.publish.call_args_list
            if len(call.args) > 1 and call.args[1] is Topic.SIMULATION_SCENE
        ]
        assert len(scenes) == 1
        assert scenes[0].geoms == ()
    finally:
        node.on_close()
