"""LCM boundary tests for the MuJoCo simulation node."""

import multiprocessing
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from humanoid.config.robot.so101 import SO101_CONFIG
from humanoid.constants import Topic
from humanoid.middleware.publisher import Publisher
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.robot.simulation import MujocoSimulationNode
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.types.robot import NormalizedRobotJointCommand, RobotJointCommand, RobotState
from humanoid.types.simulation import MujocoSimulationConfig

SPAWNED_SIMULATION_TIMEOUT_SECONDS = 15.0


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
    publisher = MagicMock(spec=Publisher)
    engine = MagicMock(spec=NativeMujocoEngine)
    engine.physics_timestep = 0.001
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
    clock = MagicMock(side_effect=[1.0, 1.01])
    node, engine, subscriber, publisher = _make_node(monkeypatch, clock=clock)
    command = RobotJointCommand(timestamp=1.0, joint_positions=np.zeros(6))
    subscriber.receive.return_value = command

    node.step()

    engine.apply_joint_command.assert_called_once_with(command)
    engine.step.assert_called_once_with(2)
    engine.read_robot_state.assert_called_once_with(timestamp=1.01)
    publisher.publish.assert_called_once_with(
        engine.read_robot_state.return_value,
        topic=Topic.ROBOT_STATE,
    )


def test_watchdog_stops_velocity_actuators_once(monkeypatch):
    clock = MagicMock(side_effect=[1.0, 1.0, 1.3, 1.3, 1.6, 1.6])
    node, engine, subscriber, _ = _make_node(monkeypatch, clock=clock, timeout=0.25)
    engine.apply_joint_command.return_value = NormalizedRobotJointCommand(
        joint_positions=np.zeros(6),
        joint_velocities=np.ones(6),
    )
    subscriber.receive.side_effect = [
        RobotJointCommand(timestamp=1.0, joint_positions=np.zeros(6)),
        None,
        None,
    ]

    node.step()
    node.step()
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
    with pytest.raises(ValueError, match="integer multiple"):
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
