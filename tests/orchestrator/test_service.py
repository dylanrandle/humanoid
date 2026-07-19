from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.nodes.manager import NodeManager, NodeManagerError
from humanoid.orchestrator.client import OrchestratorClient
from humanoid.orchestrator.monitor import LoggingMonitor, NodeRateMonitor, OrchestratorMonitor
from humanoid.orchestrator.replay import ReplayManager, ReplayManagerError
from humanoid.orchestrator.service import OrchestratorService
from humanoid.recording import RecordingCatalog, RecordingError
from humanoid.types.homing import HomingPreset
from humanoid.types.logging import LoggingState, LoggingStatus
from humanoid.types.orchestrator import (
    Mode,
    ModeStatus,
    OrchestratorError,
    OrchestratorParameter,
    OrchestratorRequest,
    SafetyContext,
)
from humanoid.types.process import ProcessAction, ProcessName, ProcessStatus, Runtime
from humanoid.types.replay import RecordingBundle, RecordingSummary, ReplayOutcome, ReplayStatus
from humanoid.types.robot import RobotName


def _process_status(running: bool = False) -> ProcessStatus:
    return ProcessStatus(
        running=running,
        pid=123 if running else None,
        exit_code=None,
        runtime=Runtime.SIM if running else None,
        uptime_seconds=1.0 if running else None,
        last_output=None,
    )


def _all_processes(
    *,
    stack: bool = False,
    replay: bool = False,
    keyboard: bool = False,
    oculus: bool = False,
) -> dict[ProcessName, ProcessStatus]:
    return {
        ProcessName.STACK: _process_status(stack),
        ProcessName.REPLAY: _process_status(replay),
        ProcessName.KEYBOARD: _process_status(keyboard),
        ProcessName.OCULUS: _process_status(oculus),
    }


def _make_service(
    *,
    stack: bool = False,
    keyboard: bool = False,
    oculus: bool = False,
    connected: bool = False,
    mode: Mode | None = None,
) -> tuple[OrchestratorService, MagicMock, MagicMock, MagicMock]:
    manager = MagicMock(spec=NodeManager)
    manager.runtime = Runtime.SIM
    manager.robot = RobotName.ELROBOT_MOBILE
    manager.status.return_value = _all_processes(
        stack=stack,
        keyboard=keyboard,
        oculus=oculus,
    )
    manager.active_nodes.return_value = {}
    client = MagicMock(spec=OrchestratorClient)
    monitor = MagicMock(spec=OrchestratorMonitor)
    monitor.snapshot.return_value = ModeStatus(
        mode=mode,
        connected=connected,
        age_seconds=0.0 if connected else None,
    )
    replay_manager = MagicMock(spec=ReplayManager)
    replay_manager.status.return_value = ReplayStatus(
        running=False,
        file_name=None,
        outcome=None,
        exit_code=None,
        last_output=None,
    )
    logging_monitor = MagicMock(spec=LoggingMonitor)
    logging_monitor.snapshot.return_value = LoggingStatus(
        timestamp=0.0,
        state=LoggingState.STOPPED,
    )
    node_rate_monitor = MagicMock(spec=NodeRateMonitor)
    node_rate_monitor.snapshot.return_value = []
    service = OrchestratorService(
        node_manager=manager,
        orchestrator_client=client,
        orchestrator_monitor=monitor,
        logging_monitor=logging_monitor,
        node_rate_monitor=node_rate_monitor,
        replay_manager=replay_manager,
    )
    recording_catalog = MagicMock(spec=RecordingCatalog)
    recording_catalog.list.return_value = []
    service.recording_catalog = recording_catalog
    return service, manager, client, monitor


def _recording(tmp_path, robot: RobotName = RobotName.ELROBOT_MOBILE) -> RecordingBundle:
    recording = RecordingCatalog(tmp_path).create(ROBOT_CONFIGS[robot])
    recording.log_path.write_bytes(b"log")
    return recording


def _homing_request(preset: HomingPreset) -> OrchestratorRequest:
    return OrchestratorRequest(
        mode=Mode.HOMING,
        parameters={OrchestratorParameter.PRESET: preset},
    )


def _safety_context(
    *,
    runtime: Runtime = Runtime.SIM,
    robot: RobotName = RobotName.ELROBOT_MOBILE,
    acknowledged: bool = False,
) -> SafetyContext:
    return SafetyContext(
        expected_runtime=runtime,
        expected_robot=robot,
        real_hardware_acknowledged=acknowledged,
    )


def test_teleop_requires_main_stack():
    service, manager, _, _ = _make_service()

    with pytest.raises(OrchestratorError, match="Start the main stack"):
        service.start_process(ProcessName.OCULUS, _safety_context())

    manager.start.assert_not_called()


def test_teleop_waits_for_orchestrator_readiness():
    service, manager, _, _ = _make_service(stack=True)

    with pytest.raises(OrchestratorError, match="finish starting"):
        service.start_process(ProcessName.KEYBOARD, _safety_context())

    manager.start.assert_not_called()


def test_starting_teleop_uses_ready_stack():
    service, manager, _, _ = _make_service(stack=True, connected=True, mode=Mode.IDLE)

    service.start_process(ProcessName.KEYBOARD, _safety_context())

    manager.start.assert_called_once_with(ProcessName.KEYBOARD)


def test_manager_start_failure_is_returned_as_control_error():
    service, manager, _, _ = _make_service(stack=True, connected=True, mode=Mode.IDLE)
    manager.start.side_effect = NodeManagerError("process failed")

    with pytest.raises(OrchestratorError, match="process failed"):
        service.start_process(ProcessName.KEYBOARD, _safety_context())


def test_start_stack_rejects_external_orchestrator():
    service, manager, _, _ = _make_service(connected=True, mode=Mode.IDLE)

    with pytest.raises(OrchestratorError, match="already broadcasting"):
        service.start_process(ProcessName.STACK, _safety_context())

    manager.start.assert_not_called()


def test_status_rejects_external_orchestrator():
    service, _, _, _ = _make_service(connected=True, mode=Mode.IDLE)

    with pytest.raises(OrchestratorError, match="Stop it before using this console"):
        service.status()


def test_stack_failure_clears_its_recent_mode_before_reporting_status():
    failure_code = 7
    service, manager, _, monitor = _make_service(connected=True, mode=Mode.IDLE)
    processes = _all_processes()
    processes[ProcessName.STACK] = ProcessStatus(
        running=False,
        pid=None,
        exit_code=failure_code,
        runtime=Runtime.SIM,
        uptime_seconds=2.0,
        last_output="RobotDriverNode exited unexpectedly with code 7",
    )
    manager.status.return_value = processes
    monitor.reset.side_effect = lambda: monitor.snapshot.configure_mock(
        return_value=ModeStatus(mode=None, connected=False, age_seconds=None)
    )

    status = service.status()

    assert status.processes[ProcessName.STACK].exit_code == failure_code
    assert status.orchestrator.connected is False
    monitor.reset.assert_called_once_with()
    cast(MagicMock, service.logging_monitor).reset.assert_called_once_with()

    monitor.snapshot.return_value = ModeStatus(mode=Mode.IDLE, connected=True, age_seconds=0.0)
    with pytest.raises(OrchestratorError, match="already broadcasting"):
        service.status()
    monitor.reset.assert_called_once_with()


def test_runtime_change_is_rejected_while_external_stack_is_active():
    service, manager, _, _ = _make_service(connected=True, mode=Mode.IDLE)

    with pytest.raises(OrchestratorError, match="already broadcasting"):
        service.set_runtime(Runtime.REAL, _safety_context(acknowledged=True))

    manager.set_runtime.assert_not_called()


def test_robot_change_is_rejected_while_external_stack_is_active():
    service, manager, _, _ = _make_service(connected=True, mode=Mode.IDLE)

    with pytest.raises(OrchestratorError, match="already broadcasting"):
        service.set_robot(RobotName.PANDA, _safety_context())

    manager.set_robot.assert_not_called()


def test_stop_process_does_not_control_external_stack():
    service, manager, client, _ = _make_service(connected=True, mode=Mode.KEYBOARD)

    with pytest.raises(OrchestratorError, match="already broadcasting"):
        service.stop_process(ProcessName.KEYBOARD)

    manager.stop.assert_not_called()
    client.request_idle.assert_not_called()


def test_robot_change_uses_typed_robot_name():
    service, manager, _, _ = _make_service()

    status = service.set_robot(RobotName.PANDA, _safety_context())

    manager.set_robot.assert_called_once_with(RobotName.PANDA)
    assert status.robots == list(RobotName)


@pytest.mark.parametrize(
    "safety",
    [
        _safety_context(runtime=Runtime.REAL),
        _safety_context(robot=RobotName.PANDA),
    ],
)
def test_configuration_change_rejects_stale_operator_snapshot(safety):
    service, manager, _, _ = _make_service()

    with pytest.raises(OrchestratorError, match="Configuration changed"):
        service.set_robot(RobotName.PANDA, safety)

    manager.set_robot.assert_not_called()


def test_real_runtime_selection_requires_hardware_acknowledgement():
    service, manager, _, _ = _make_service()

    with pytest.raises(OrchestratorError, match="acknowledgement is required"):
        service.set_runtime(Runtime.REAL, _safety_context())

    manager.set_runtime.assert_not_called()


def test_real_stack_start_requires_current_configuration_and_acknowledgement():
    service, manager, _, _ = _make_service()
    manager.runtime = Runtime.REAL

    with pytest.raises(OrchestratorError, match="acknowledgement is required"):
        service.start_process(
            ProcessName.STACK,
            _safety_context(runtime=Runtime.REAL),
        )

    manager.start.assert_not_called()


def test_stack_start_rejects_stale_operator_snapshot():
    service, manager, _, _ = _make_service()
    manager.runtime = Runtime.REAL

    with pytest.raises(OrchestratorError, match="Configuration changed"):
        service.start_process(
            ProcessName.STACK,
            _safety_context(acknowledged=True),
        )

    manager.start.assert_not_called()


def test_stale_client_cannot_start_after_another_client_selects_real_runtime():
    service, manager, _, _ = _make_service()
    manager.set_runtime.side_effect = lambda runtime: setattr(manager, "runtime", runtime)

    service.set_runtime(Runtime.REAL, _safety_context(acknowledged=True))

    with pytest.raises(OrchestratorError, match="Configuration changed"):
        service.start_process(ProcessName.STACK, _safety_context())

    manager.start.assert_not_called()


def test_request_waits_for_orchestrator_readiness():
    service, _, client, _ = _make_service(stack=True)

    with pytest.raises(OrchestratorError, match="finish starting"):
        service.request_mode(_homing_request(HomingPreset.HOME))

    client.request_homing.assert_not_called()


def test_teleop_mode_requires_matching_process():
    service, _, client, _ = _make_service(stack=True, connected=True, mode=Mode.IDLE)

    with pytest.raises(OrchestratorError, match="Start Keyboard teleop"):
        service.request_mode(OrchestratorRequest(mode=Mode.KEYBOARD))

    client.request_keyboard.assert_not_called()


def test_teleop_mode_uses_running_process():
    service, _, client, _ = _make_service(
        stack=True,
        keyboard=True,
        connected=True,
        mode=Mode.IDLE,
    )

    service.request_mode(OrchestratorRequest(mode=Mode.KEYBOARD))

    client.request_keyboard.assert_called_once_with()


@pytest.mark.parametrize(
    ("action", "method_name"),
    [
        (ProcessAction.START, "start_logging"),
        (ProcessAction.STOP, "stop_logging"),
    ],
)
def test_logging_controls_publish_requested_event(action, method_name):
    service, _, client, _ = _make_service(stack=True, connected=True, mode=Mode.IDLE)

    service.set_logging(action)

    getattr(client, method_name).assert_called_once_with()
    logging_monitor = cast(MagicMock, service.logging_monitor)
    expected_monitor_method = (
        "start_requested" if action is ProcessAction.START else "stop_requested"
    )
    getattr(logging_monitor, expected_monitor_method).assert_called_once_with()


def test_logging_controls_require_ready_stack():
    service, _, client, _ = _make_service()

    with pytest.raises(OrchestratorError, match="Start the main stack"):
        service.set_logging(ProcessAction.START)

    client.start_logging.assert_not_called()


def test_logging_request_failure_is_reported_by_monitor():
    service, _, client, _ = _make_service(stack=True, connected=True, mode=Mode.IDLE)
    client.start_logging.side_effect = RuntimeError("publish failed")

    with pytest.raises(RuntimeError, match="publish failed"):
        service.set_logging(ProcessAction.START)

    cast(MagicMock, service.logging_monitor).fail.assert_called_once_with(
        "Could not request data logging: publish failed"
    )


def test_status_exposes_latest_logging_lifecycle():
    service, _, _, _ = _make_service()
    expected = LoggingStatus(
        timestamp=2.0,
        state=LoggingState.RUNNING,
        file_name="logs/lcmlog_20260101",
    )
    cast(MagicMock, service.logging_monitor).snapshot.return_value = expected

    assert service.status().logging == expected


def test_status_exposes_server_managed_recordings():
    service, _, _, _ = _make_service()
    recordings = [
        RecordingSummary(
            id="recording_20260101_120000",
            robot=RobotName.PANDA,
            created_at="2026-01-01T12:00:00+00:00",
        )
    ]
    cast(MagicMock, service.recording_catalog).list.return_value = recordings

    assert service.status().recordings == recordings


def test_replay_starts_simulation_nodes_before_logplayer(tmp_path):
    service, manager, _, _ = _make_service()
    replay_manager = cast(MagicMock, service.replay_manager)
    recording = _recording(tmp_path)
    cast(MagicMock, service.recording_catalog).get.return_value = recording
    events = []

    def start_nodes(name):
        events.append(("nodes", name))
        manager.status.return_value[ProcessName.REPLAY] = _process_status(True)

    def wait_until_ready():
        events.append(("ready", None))

    def start_player(selected_recording):
        events.append(("player", selected_recording))
        replay_manager.status.return_value = ReplayStatus(
            running=True,
            file_name=selected_recording.file_name,
            outcome=None,
            exit_code=None,
            last_output=None,
        )
        return replay_manager.status.return_value

    manager.start.side_effect = start_nodes
    manager.wait_until_robot_ready.side_effect = wait_until_ready
    replay_manager.validate.side_effect = lambda selected_recording, _config: events.append(
        ("validate", selected_recording)
    )
    replay_manager.start.side_effect = start_player

    status = service.start_replay(recording.id, _safety_context())

    assert events == [
        ("validate", recording),
        ("nodes", ProcessName.REPLAY),
        ("ready", None),
        ("player", recording),
    ]
    replay_manager.validate.assert_called_once_with(
        recording,
        ROBOT_CONFIGS[RobotName.ELROBOT_MOBILE],
    )
    assert status.replay.running is True


@pytest.mark.parametrize("method_name", ["start_process", "stop_process"])
def test_replay_cannot_use_generic_process_controls(method_name):
    service, manager, _, _ = _make_service(stack=True, connected=True)
    method = getattr(service, method_name)
    arguments = (
        (ProcessName.REPLAY, _safety_context())
        if method_name == "start_process"
        else (ProcessName.REPLAY,)
    )

    with pytest.raises(OrchestratorError, match="dedicated replay controls"):
        method(*arguments)

    manager.start.assert_not_called()
    manager.stop.assert_not_called()


def test_real_replay_requires_hardware_acknowledgement(tmp_path):
    service, manager, _, _ = _make_service()
    manager.runtime = Runtime.REAL
    recording = _recording(tmp_path)
    cast(MagicMock, service.recording_catalog).get.return_value = recording

    with pytest.raises(OrchestratorError, match="acknowledgement is required"):
        service.start_replay(recording.id, _safety_context(runtime=Runtime.REAL))

    manager.start.assert_not_called()


def test_real_replay_starts_after_hardware_acknowledgement(tmp_path):
    service, manager, _, _ = _make_service()
    manager.runtime = Runtime.REAL
    recording = _recording(tmp_path)
    cast(MagicMock, service.recording_catalog).get.return_value = recording
    manager.start.side_effect = lambda _name: manager.status.return_value.__setitem__(
        ProcessName.REPLAY,
        _process_status(True),
    )
    replay_manager = cast(MagicMock, service.replay_manager)
    replay_manager.start.side_effect = lambda _recording: replay_manager.status.configure_mock(
        return_value=ReplayStatus(
            running=True,
            file_name=recording.file_name,
            outcome=None,
            exit_code=None,
            last_output=None,
        )
    )

    service.start_replay(
        recording.id,
        _safety_context(runtime=Runtime.REAL, acknowledged=True),
    )

    replay_manager.validate.assert_called_once_with(
        recording,
        ROBOT_CONFIGS[RobotName.ELROBOT_MOBILE],
    )
    manager.start.assert_called_once_with(ProcessName.REPLAY)


def test_replay_start_failure_stops_nodes_and_preserves_recording(tmp_path):
    service, manager, _, monitor = _make_service()
    replay_manager = cast(MagicMock, service.replay_manager)
    recording = _recording(tmp_path)
    cast(MagicMock, service.recording_catalog).get.return_value = recording
    manager.wait_until_robot_ready.side_effect = NodeManagerError("Robot state unavailable")

    with pytest.raises(OrchestratorError, match="Robot state unavailable"):
        service.start_replay(recording.id, _safety_context())

    manager.stop.assert_called_once_with(ProcessName.REPLAY)
    replay_manager.start.assert_not_called()
    monitor.reset.assert_called_once_with()
    assert recording.log_path.exists()


def test_replay_validation_failure_prevents_node_start_and_preserves_recording(tmp_path):
    service, manager, _, _ = _make_service()
    replay_manager = cast(MagicMock, service.replay_manager)
    recording = _recording(tmp_path)
    cast(MagicMock, service.recording_catalog).get.return_value = recording
    replay_manager.validate.side_effect = ReplayManagerError(
        "The recorded robot configuration does not match the selected configuration."
    )

    with pytest.raises(OrchestratorError, match="configuration does not match"):
        service.start_replay(recording.id, _safety_context())

    manager.start.assert_not_called()
    assert recording.log_path.exists()


def test_replay_requires_all_managed_processes_to_be_stopped(tmp_path):
    service, manager, _, _ = _make_service(stack=True, connected=True)
    recording = _recording(tmp_path)
    cast(MagicMock, service.recording_catalog).get.return_value = recording

    with pytest.raises(OrchestratorError, match="Stop the stack and teleop"):
        service.start_replay(recording.id, _safety_context())

    manager.start.assert_not_called()


def test_replay_rejects_unknown_recording(tmp_path):
    service, manager, _, _ = _make_service()
    cast(MagicMock, service.recording_catalog).get.side_effect = RecordingError(
        "The selected recording is unavailable."
    )

    with pytest.raises(OrchestratorError, match="selected recording is unavailable"):
        service.start_replay("recording_missing", _safety_context())

    manager.start.assert_not_called()


def test_managed_replay_mode_is_not_reported_as_an_external_stack():
    service, manager, _, _ = _make_service(connected=True, mode=Mode.IDLE)
    replay_manager = cast(MagicMock, service.replay_manager)
    manager.status.return_value[ProcessName.REPLAY] = _process_status(True)
    replay_manager.status.return_value = ReplayStatus(
        running=True,
        file_name="lcmlog",
        outcome=None,
        exit_code=None,
        last_output=None,
    )

    status = service.status()

    assert status.replay.running is True
    assert status.orchestrator.connected is True


def test_completed_player_stops_replay_nodes():
    service, manager, _, monitor = _make_service()
    replay_manager = cast(MagicMock, service.replay_manager)
    manager.status.return_value[ProcessName.REPLAY] = _process_status(True)
    replay_manager.status.return_value = ReplayStatus(
        running=False,
        file_name="lcmlog",
        outcome=ReplayOutcome.COMPLETED,
        exit_code=0,
        last_output=None,
    )
    manager.stop.side_effect = lambda _name: manager.status.return_value.__setitem__(
        ProcessName.REPLAY,
        _process_status(),
    )

    status = service.status()

    manager.stop.assert_called_once_with(ProcessName.REPLAY)
    monitor.reset.assert_called_once_with()
    assert status.replay.exit_code == 0


def test_replay_node_failure_stops_player_and_surfaces_error():
    failure_code = 8
    service, manager, _, monitor = _make_service()
    replay_manager = cast(MagicMock, service.replay_manager)
    manager.status.return_value[ProcessName.REPLAY] = ProcessStatus(
        running=False,
        pid=None,
        exit_code=failure_code,
        runtime=Runtime.SIM,
        uptime_seconds=1.0,
        last_output="RobotDriverNode exited unexpectedly with code 8",
    )
    replay_manager.status.return_value = ReplayStatus(
        running=True,
        file_name="lcmlog",
        outcome=None,
        exit_code=None,
        last_output=None,
    )
    failed = ReplayStatus(
        running=False,
        file_name="lcmlog",
        outcome=ReplayOutcome.FAILED,
        exit_code=1,
        last_output="RobotDriverNode exited unexpectedly with code 8",
    )
    replay_manager.fail.return_value = failed

    status = service.status()

    replay_manager.fail.assert_called_once_with(failed.last_output)
    monitor.reset.assert_called_once_with()
    assert status.replay == failed


def test_stop_replay_stops_player_and_nodes():
    service, manager, _, monitor = _make_service()
    replay_manager = cast(MagicMock, service.replay_manager)

    service.stop_replay()

    replay_manager.stop.assert_called_once_with()
    manager.stop.assert_called_once_with(ProcessName.REPLAY)
    monitor.reset.assert_called_once_with()


def test_home_request_uses_configured_home_target():
    service, manager, client, _ = _make_service(
        stack=True,
        connected=True,
        mode=Mode.IDLE,
    )
    manager.robot = RobotName.PANDA

    status = service.request_mode(_homing_request(HomingPreset.HOME))

    target = client.request_homing.call_args.args[0]
    np.testing.assert_allclose(
        target, ROBOT_CONFIGS[RobotName.PANDA].homing_presets[HomingPreset.HOME]
    )
    assert status.orchestrator.parameters[OrchestratorParameter.PRESET] is HomingPreset.HOME


def test_status_exposes_selected_robot_and_available_options():
    service, manager, _, _ = _make_service()
    manager.robot = RobotName.SO101

    status = service.status()

    assert status.robot is RobotName.SO101
    assert status.robots == list(RobotName)


def test_status_requests_rates_for_active_managed_nodes():
    service, manager, _, _ = _make_service(stack=True)
    active_nodes = {"RobotControllerNode": 123}
    manager.active_nodes.return_value = active_nodes

    status = service.status()

    cast(MagicMock, service.node_rate_monitor).snapshot.assert_called_once_with(active_nodes)
    assert status.node_rates == []


def test_homing_parameters_stay_selected_until_homing_completes():
    service, _, _, monitor = _make_service(stack=True, connected=True, mode=Mode.IDLE)
    monitor.snapshot.side_effect = [
        ModeStatus(mode=Mode.IDLE, connected=True, age_seconds=0.0),
        ModeStatus(mode=Mode.IDLE, connected=True, age_seconds=0.0),
        ModeStatus(mode=Mode.HOMING, connected=True, age_seconds=0.0),
        ModeStatus(mode=Mode.IDLE, connected=True, age_seconds=0.0),
    ]

    requested = service.request_mode(_homing_request(HomingPreset.REST))
    moving = service.status()
    completed = service.status()

    assert requested.orchestrator.parameters[OrchestratorParameter.PRESET] is HomingPreset.REST
    assert moving.orchestrator.parameters[OrchestratorParameter.PRESET] is HomingPreset.REST
    assert completed.orchestrator.parameters == {}
    assert completed.orchestrator.mode is Mode.IDLE


def test_pending_homing_parameters_expire_if_transition_is_not_seen(monkeypatch):
    service, _, _, _ = _make_service(stack=True, connected=True, mode=Mode.IDLE)
    monotonic = MagicMock(side_effect=[0.0, 0.0, 3.0])
    monkeypatch.setattr("humanoid.orchestrator.service.time.monotonic", monotonic)

    requested = service.request_mode(_homing_request(HomingPreset.HOME))
    expired = service.status()

    assert requested.orchestrator.parameters[OrchestratorParameter.PRESET] is HomingPreset.HOME
    assert expired.orchestrator.parameters == {}
    assert expired.orchestrator.mode is Mode.IDLE


def test_stopping_active_teleop_returns_orchestrator_to_idle():
    service, manager, client, monitor = _make_service(
        stack=True,
        keyboard=True,
        connected=True,
        mode=Mode.KEYBOARD,
    )
    monitor.snapshot.side_effect = [
        ModeStatus(mode=Mode.KEYBOARD, connected=True, age_seconds=0.0),
        ModeStatus(mode=Mode.IDLE, connected=True, age_seconds=0.0),
        ModeStatus(mode=Mode.IDLE, connected=True, age_seconds=0.0),
    ]
    manager.stop.side_effect = lambda _name: client.request_idle.assert_called_once_with()

    service.stop_process(ProcessName.KEYBOARD)

    manager.stop.assert_called_once_with(ProcessName.KEYBOARD)
    client.request_idle.assert_called_once_with()


def test_stopping_inactive_teleop_does_not_change_mode():
    service, _, client, _ = _make_service(
        stack=True,
        keyboard=True,
        connected=True,
        mode=Mode.OCULUS,
    )

    service.stop_process(ProcessName.KEYBOARD)

    client.request_idle.assert_not_called()


def test_stopping_stack_idles_and_stops_managed_children():
    service, manager, client, monitor = _make_service(
        stack=True,
        keyboard=True,
        connected=True,
        mode=Mode.KEYBOARD,
    )

    service.stop_process(ProcessName.STACK)

    client.request_idle.assert_called_once_with()
    manager.close.assert_called_once_with()
    monitor.reset.assert_called_once_with()


def test_stopping_stack_control_stops_orphaned_managed_teleop():
    service, manager, client, monitor = _make_service(keyboard=True)

    service.stop_process(ProcessName.STACK)

    client.request_idle.assert_not_called()
    manager.close.assert_called_once_with()
    monitor.reset.assert_called_once_with()


def test_close_releases_mode_monitor_when_stack_status_fails():
    service, manager, _, monitor = _make_service()
    manager.status.side_effect = RuntimeError("status failed")

    service.close()

    manager.close.assert_called_once_with()
    monitor.close.assert_called_once_with()
    cast(MagicMock, service.node_rate_monitor).close.assert_called_once_with()
