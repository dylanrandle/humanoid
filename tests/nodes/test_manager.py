import os
import signal
import threading
from collections.abc import Callable
from multiprocessing.process import BaseProcess
from unittest.mock import MagicMock, call

import pytest

from humanoid.constants import (
    DEFAULT_HUMANOID_ROBOT,
    DEFAULT_HUMANOID_RUNTIME,
    ROBOT_ENVIRONMENT_VARIABLE,
    RUNTIME_ENVIRONMENT_VARIABLE,
)
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.groups import NODE_GROUPS, PROCESS_STOP_ORDER
from humanoid.nodes.manager import (
    NodeManager,
    NodeManagerError,
)
from humanoid.nodes.orchestrator import OrchestratorNode
from humanoid.nodes.policy.homing import HomingNode
from humanoid.nodes.policy.teleop.keyboard import KeyboardTeleopNode
from humanoid.nodes.policy.teleop.oculus import OculusTeleopNode
from humanoid.nodes.robot.controller import RobotControllerNode
from humanoid.nodes.robot.driver import RobotDriverNode
from humanoid.nodes.robot.logger import RobotLoggerNode
from humanoid.nodes.robot.visualizer import RobotVisualizerNode
from humanoid.types.node import ProcessContext
from humanoid.types.process import ProcessName, Runtime
from humanoid.types.robot import RobotName


def _fake_process(
    name: str = "Node",
    *,
    running: bool = True,
    exit_code: int | None = None,
    pid: int = 321,
) -> MagicMock:
    process = MagicMock(spec=BaseProcess)
    process.name = name
    process.pid = pid
    process.exitcode = exit_code
    process.is_alive.return_value = running
    return process


def _fake_context(processes: list[MagicMock]) -> MagicMock:
    context = MagicMock(spec=ProcessContext)

    def create_process(*, target: Callable[..., None], name: str) -> MagicMock:
        process = _fake_process(name=name, pid=321 + len(processes))
        processes.append(process)
        return process

    context.Process.side_effect = create_process
    return context


def _use_process_context(monkeypatch, context: MagicMock | None = None) -> MagicMock:
    context = context if context is not None else MagicMock(spec=ProcessContext)
    monkeypatch.setattr("humanoid.nodes.manager.get_context", MagicMock(return_value=context))
    return context


def _join_stack_startup(manager: NodeManager) -> None:
    startup_thread = manager._groups[ProcessName.STACK].startup_thread
    assert startup_thread is not None
    startup_thread.join(timeout=1.0)
    assert startup_thread.is_alive() is False


def test_node_groups_reference_imported_node_classes():
    assert NODE_GROUPS[ProcessName.STACK].nodes == (
        RobotDriverNode,
        RobotControllerNode,
        RobotVisualizerNode,
        OrchestratorNode,
        RobotLoggerNode,
    )
    assert NODE_GROUPS[ProcessName.STACK].deferred_nodes == (HomingNode,)
    assert NODE_GROUPS[ProcessName.REPLAY].nodes == (RobotDriverNode, RobotVisualizerNode)
    assert NODE_GROUPS[ProcessName.KEYBOARD].nodes == (KeyboardTeleopNode,)
    assert NODE_GROUPS[ProcessName.OCULUS].nodes == (OculusTeleopNode,)


@pytest.mark.parametrize("runtime", list(Runtime))
def test_replay_group_starts_driver_and_visualizer(monkeypatch, runtime):
    processes: list[MagicMock] = []
    context = _fake_context(processes)
    _use_process_context(monkeypatch, context)
    manager = NodeManager(runtime=runtime)

    manager.start(ProcessName.REPLAY)

    assert [entry.kwargs["target"] for entry in context.Process.call_args_list] == [
        RobotDriverNode.main,
        RobotVisualizerNode.main,
    ]


def test_starts_core_nodes_before_homing_with_selected_configuration(monkeypatch):
    processes: list[MagicMock] = []
    context = _fake_context(processes)
    observed_configurations: list[tuple[str | None, str | None]] = []

    def capture_configuration():
        observed_configurations.append(
            (
                os.getenv(RUNTIME_ENVIRONMENT_VARIABLE),
                os.getenv(ROBOT_ENVIRONMENT_VARIABLE),
            )
        )

    context.Process.side_effect = lambda *, target, name: _process_with_start_callback(
        name,
        capture_configuration,
        processes,
    )
    monkeypatch.setenv(RUNTIME_ENVIRONMENT_VARIABLE, Runtime.SIM)
    monkeypatch.setenv(ROBOT_ENVIRONMENT_VARIABLE, RobotName.ELROBOT)
    _use_process_context(monkeypatch, context)
    manager = NodeManager(
        runtime=Runtime.REAL,
        robot=RobotName.PANDA,
    )
    monkeypatch.setattr(manager, "_wait_for_robot_state", MagicMock(return_value=True))

    status = manager.start(ProcessName.STACK)
    _join_stack_startup(manager)

    expected_nodes = (*NODE_GROUPS[ProcessName.STACK].nodes, HomingNode)
    assert [entry.kwargs["target"] for entry in context.Process.call_args_list] == [
        node.main for node in expected_nodes
    ]
    assert observed_configurations == [(Runtime.REAL, RobotName.PANDA)] * len(expected_nodes)
    assert os.getenv(RUNTIME_ENVIRONMENT_VARIABLE) == Runtime.SIM
    assert os.getenv(ROBOT_ENVIRONMENT_VARIABLE) == RobotName.ELROBOT
    assert status.running is True


def _process_with_start_callback(
    name: str,
    callback: Callable[[], None],
    processes: list[MagicMock],
) -> MagicMock:
    process = _fake_process(name=name, pid=321 + len(processes))
    process.start.side_effect = callback
    processes.append(process)
    return process


@pytest.mark.parametrize(
    ("environment_value", "expected"),
    [
        ("real", Runtime.REAL),
        ("sim", Runtime.SIM),
    ],
)
def test_environment_runtime_uses_known_values(monkeypatch, environment_value, expected):
    monkeypatch.setenv(RUNTIME_ENVIRONMENT_VARIABLE, environment_value)
    _use_process_context(monkeypatch)

    manager = NodeManager()

    assert manager.runtime is expected


@pytest.mark.parametrize("value", ["simulation", "typo"])
def test_environment_runtime_rejects_unknown_values(monkeypatch, value):
    monkeypatch.setenv(RUNTIME_ENVIRONMENT_VARIABLE, value)
    _use_process_context(monkeypatch)

    with pytest.raises(ValueError, match=value):
        NodeManager()


@pytest.mark.parametrize("environment_value", [None, ""])
def test_environment_uses_explicit_defaults(monkeypatch, environment_value):
    if environment_value is None:
        monkeypatch.delenv(RUNTIME_ENVIRONMENT_VARIABLE, raising=False)
        monkeypatch.delenv(ROBOT_ENVIRONMENT_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(RUNTIME_ENVIRONMENT_VARIABLE, environment_value)
        monkeypatch.setenv(ROBOT_ENVIRONMENT_VARIABLE, environment_value)
    _use_process_context(monkeypatch)

    manager = NodeManager()

    assert manager.runtime is DEFAULT_HUMANOID_RUNTIME
    assert manager.robot is DEFAULT_HUMANOID_ROBOT


@pytest.mark.parametrize(
    "environment_value",
    [
        "typo",
        "unknown",
    ],
)
def test_environment_robot_rejects_unknown_values(monkeypatch, environment_value):
    monkeypatch.setenv(ROBOT_ENVIRONMENT_VARIABLE, environment_value)
    _use_process_context(monkeypatch)

    with pytest.raises(ValueError, match=environment_value):
        NodeManager()


@pytest.mark.parametrize("robot", [RobotName.PANDA, RobotName.ELROBOT_MOBILE])
def test_environment_robot_uses_known_values(monkeypatch, robot):
    monkeypatch.setenv(ROBOT_ENVIRONMENT_VARIABLE, robot)
    _use_process_context(monkeypatch)

    assert NodeManager().robot is robot


def test_rejects_configuration_change_and_duplicate_start_while_group_is_running(monkeypatch):
    processes: list[MagicMock] = []
    _use_process_context(monkeypatch, _fake_context(processes))
    manager = NodeManager(runtime=Runtime.SIM)
    manager.start(ProcessName.KEYBOARD)

    with pytest.raises(NodeManagerError, match="Stop the stack and teleop"):
        manager.set_runtime(Runtime.REAL)
    with pytest.raises(NodeManagerError, match="Stop the stack and teleop"):
        manager.set_robot(RobotName.PANDA)
    with pytest.raises(NodeManagerError, match="already running"):
        manager.start(ProcessName.KEYBOARD)


def test_changes_robot_while_stopped(monkeypatch):
    _use_process_context(monkeypatch)
    manager = NodeManager()

    manager.set_robot(RobotName.SO101)

    assert manager.robot is RobotName.SO101


def test_robot_state_timeout_always_closes_subscriber(monkeypatch):
    subscriber = MagicMock(spec=Subscriber)
    subscriber.receive.return_value = None
    monotonic = MagicMock(side_effect=[0.0, 0.0])
    monkeypatch.setattr("humanoid.nodes.manager.Subscriber", MagicMock(return_value=subscriber))
    monkeypatch.setattr("humanoid.nodes.manager.time.monotonic", monotonic)
    _use_process_context(monkeypatch)
    manager = NodeManager(state_timeout_seconds=0.0)

    with pytest.raises(RuntimeError, match="Timed out"):
        manager._wait_for_robot_state(threading.Event())

    subscriber.close.assert_called_once_with()


def test_public_robot_readiness_timeout_is_a_manager_error(monkeypatch):
    _use_process_context(monkeypatch)
    manager = NodeManager()
    monkeypatch.setattr(
        manager,
        "_wait_for_robot_state",
        MagicMock(side_effect=RuntimeError("state unavailable")),
    )

    with pytest.raises(NodeManagerError, match="state unavailable"):
        manager.wait_until_robot_ready()


def test_stack_startup_failure_stops_core_nodes_and_reports_error(monkeypatch):
    processes: list[MagicMock] = []
    context = _fake_context(processes)
    _use_process_context(monkeypatch, context)
    manager = NodeManager()
    monkeypatch.setattr(
        manager,
        "_wait_for_robot_state",
        MagicMock(side_effect=RuntimeError("state unavailable")),
    )

    manager.start(ProcessName.STACK)
    for process in processes:
        process.is_alive.return_value = False
        process.exitcode = 0
    _join_stack_startup(manager)
    status = manager.status()[ProcessName.STACK]

    assert status.running is False
    assert status.exit_code == 1
    assert status.last_output == "Main stack failed to start: state unavailable"


def test_stopping_stack_cancels_deferred_startup(monkeypatch):
    processes: list[MagicMock] = []
    context = _fake_context(processes)
    waiting = threading.Event()
    _use_process_context(monkeypatch, context)
    manager = NodeManager()

    def wait_for_stop(stop_event):
        waiting.set()
        stop_event.wait(timeout=1.0)
        return False

    monkeypatch.setattr(manager, "_wait_for_robot_state", wait_for_stop)
    manager.start(ProcessName.STACK)
    assert waiting.wait(timeout=1.0)
    for process in processes:
        process.join.side_effect = lambda timeout=None, process=process: (
            process.is_alive.configure_mock(return_value=False)
        )

    status = manager.stop(ProcessName.STACK)

    assert len(processes) == len(NODE_GROUPS[ProcessName.STACK].nodes)
    assert status.running is False
    assert status.exit_code == 0


def test_partial_group_failure_stops_siblings_and_reports_failure(monkeypatch):
    expected_exit_code = 7
    processes: list[MagicMock] = []
    context = _fake_context(processes)
    _use_process_context(monkeypatch, context)
    manager = NodeManager()
    monkeypatch.setattr(manager, "_wait_for_robot_state", MagicMock(return_value=True))
    stop_processes = MagicMock()
    monkeypatch.setattr(manager, "_stop_processes", stop_processes)
    manager.start(ProcessName.STACK)
    _join_stack_startup(manager)

    failed_process = processes[0]
    failed_process.is_alive.return_value = False
    failed_process.exitcode = expected_exit_code

    status = manager.status()[ProcessName.STACK]

    assert status.running is False
    assert status.exit_code == expected_exit_code
    assert status.last_output == (
        f"{failed_process.name} exited unexpectedly with code {expected_exit_code}"
    )
    stop_processes.assert_called_once_with(processes)


def test_stop_processes_interrupts_then_escalates_stragglers(monkeypatch):
    process = _fake_process()
    process.is_alive.side_effect = [True, True, True]
    monotonic = MagicMock(side_effect=[0.0, 0.0])
    send_signal = MagicMock()
    monkeypatch.setattr("humanoid.nodes.manager.time.monotonic", monotonic)
    monkeypatch.setattr("humanoid.nodes.manager.os.kill", send_signal)

    NodeManager._stop_processes([process], timeout=5.0)

    if os.name == "posix":
        send_signal.assert_called_once_with(process.pid, signal.SIGINT)
    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.join.call_args_list == [
        call(timeout=5.0),
        call(timeout=2.0),
        call(timeout=2.0),
    ]


def test_close_uses_dependency_safe_group_order(monkeypatch):
    _use_process_context(monkeypatch)
    manager = NodeManager()
    stop = MagicMock()
    monkeypatch.setattr(manager, "stop", stop)

    manager.close()

    assert stop.call_args_list == [call(name) for name in PROCESS_STOP_ORDER]
