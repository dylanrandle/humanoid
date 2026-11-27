import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import cast
from unittest.mock import MagicMock, call, patch

import lcm
import numpy as np
import pytest
from werkzeug.serving import make_server

from humanoid.agent.harness.recording import record_lcm_episode
from humanoid.api.application import BackendApplication
from humanoid.api.server import create_app
from humanoid.api.services.logs import LogService
from humanoid.api.services.recording import RecordingService
from humanoid.api.services.replay import ReplayManager, ReplayManagerError
from humanoid.config import ROBOT_CONFIGS
from humanoid.config.backend import INPUT_PROCESSES
from humanoid.constants import Topic, private_lcm_url
from humanoid.middleware.publisher import Publisher
from humanoid.nodes.manager import NodeManager, NodeManagerError
from humanoid.orchestrator.client import OrchestratorClient
from humanoid.orchestrator.monitor.mode import ModeMonitor
from humanoid.recording import RecordingCatalog, RecordingError
from humanoid.telemetry.actuator import ActuatorHealthMonitor
from humanoid.telemetry.node import NodeRateMonitor
from humanoid.types.actuator import ActuatorHealthStatus
from humanoid.types.api import BackendError, SafetyContext
from humanoid.types.homing import HomingPreset
from humanoid.types.logs import LogEntry, LogSnapshot
from humanoid.types.orchestrator import Mode, ModeStatus, OrchestratorParameter, OrchestratorRequest
from humanoid.types.process import ProcessAction, ProcessName, ProcessStatus, Runtime
from humanoid.types.recording import RecordingOwner, RecordingState, RecordingStatus
from humanoid.types.replay import RecordingBundle, RecordingSummary, ReplayOutcome, ReplayStatus
from humanoid.types.robot import RobotName
from humanoid.types.simulation import SimulationScene


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
        ProcessName.AUTONOMY: _process_status(),
    }


def _make_service(
    *,
    stack: bool = False,
    keyboard: bool = False,
    oculus: bool = False,
    connected: bool = False,
    mode: Mode | None = None,
) -> tuple[BackendApplication, MagicMock, MagicMock, MagicMock]:
    manager = MagicMock(spec=NodeManager)
    manager.runtime = Runtime.SIM
    manager.robot = RobotName.TRISKEL
    manager.scene = SimulationScene.EMPTY
    manager.status.return_value = _all_processes(
        stack=stack,
        keyboard=keyboard,
        oculus=oculus,
    )
    manager.active_nodes.return_value = {}
    client = MagicMock(spec=OrchestratorClient)
    monitor = MagicMock(spec=ModeMonitor)
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
    recording = MagicMock(spec=RecordingService)
    recording.snapshot.return_value = RecordingStatus(
        timestamp=0.0,
        state=RecordingState.STOPPED,
    )
    node_rate_monitor = MagicMock(spec=NodeRateMonitor)
    node_rate_monitor.snapshot.return_value = []
    actuator_health_monitor = MagicMock(spec=ActuatorHealthMonitor)
    actuator_health_monitor.snapshot.return_value = ActuatorHealthStatus(
        connected=False,
        healthy=False,
        age_seconds=None,
        actuators=(),
    )
    log_service = MagicMock(spec=LogService)
    log_service.snapshot.return_value = LogSnapshot(
        cursor=0,
        entries=[],
        reset=False,
        capacity=200,
    )
    log_service.queue = MagicMock()
    with patch.multiple(
        "humanoid.api.application",
        NodeManager=MagicMock(return_value=manager),
        OrchestratorClient=MagicMock(return_value=client),
        ModeMonitor=MagicMock(return_value=monitor),
        RecordingService=MagicMock(return_value=recording),
        NodeRateMonitor=MagicMock(return_value=node_rate_monitor),
        ActuatorHealthMonitor=MagicMock(return_value=actuator_health_monitor),
        ReplayManager=MagicMock(return_value=replay_manager),
        LogService=MagicMock(return_value=log_service),
        Publisher=MagicMock(),
        Subscriber=MagicMock(),
    ):
        service = BackendApplication()
    recording_catalog = MagicMock(spec=RecordingCatalog)
    recording_catalog.list.return_value = []
    service.recording_catalog = recording_catalog
    service.runtime_service.recording_catalog = recording_catalog
    return service, manager, client, monitor


def _recording(tmp_path, robot: RobotName = RobotName.TRISKEL) -> RecordingBundle:
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
    robot: RobotName = RobotName.TRISKEL,
    acknowledged: bool = False,
) -> SafetyContext:
    return SafetyContext(
        expected_runtime=runtime,
        expected_robot=robot,
        real_hardware_acknowledged=acknowledged,
    )


def test_teleop_requires_main_stack():
    service, manager, _, _ = _make_service()

    with pytest.raises(BackendError, match="Start the main stack"):
        service.start_process(ProcessName.OCULUS, _safety_context())

    manager.start.assert_not_called()


def test_teleop_waits_for_orchestrator_readiness():
    service, manager, _, _ = _make_service(stack=True)

    with pytest.raises(BackendError, match="finish starting"):
        service.start_process(ProcessName.KEYBOARD, _safety_context())

    manager.start.assert_not_called()


def test_starting_teleop_uses_ready_stack():
    service, manager, _, _ = _make_service(stack=True, connected=True, mode=Mode.IDLE)

    service.start_process(ProcessName.KEYBOARD, _safety_context())

    manager.start.assert_called_once_with(ProcessName.KEYBOARD)


@pytest.mark.parametrize(("mode", "process"), INPUT_PROCESSES.items())
@pytest.mark.parametrize("idle_acknowledged", [True, False])
def test_restarting_selected_input_requires_idle(mode, process, idle_acknowledged, monkeypatch):
    service, manager, client, _ = _make_service(stack=True, connected=True, mode=mode)

    def wait_for_idle(requested):
        assert requested is Mode.IDLE
        client.request_idle.assert_called_once_with()
        manager.start.assert_not_called()
        return idle_acknowledged

    monkeypatch.setattr(service.mode_service, "wait_for_mode", wait_for_idle)
    if idle_acknowledged:
        service.start_process(process, _safety_context())
        manager.start.assert_called_once_with(process)
    else:
        with pytest.raises(BackendError, match="Select Idle before restarting"):
            service.start_process(process, _safety_context())
        manager.start.assert_not_called()


def test_manager_start_failure_is_returned_as_control_error():
    service, manager, _, _ = _make_service(stack=True, connected=True, mode=Mode.IDLE)
    manager.start.side_effect = NodeManagerError("process failed")

    with pytest.raises(BackendError, match="process failed"):
        service.start_process(ProcessName.KEYBOARD, _safety_context())


def test_start_stack_rejects_external_orchestrator():
    service, manager, _, _ = _make_service(connected=True, mode=Mode.IDLE)

    with pytest.raises(BackendError, match="already broadcasting"):
        service.start_process(ProcessName.STACK, _safety_context())

    manager.start.assert_not_called()


def test_status_rejects_external_orchestrator():
    service, _, _, _ = _make_service(connected=True, mode=Mode.IDLE)

    with pytest.raises(BackendError, match="Stop it before using this console"):
        service.status()


def test_stack_failure_stops_inputs_and_clears_its_recent_mode_before_reporting_status():
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
    for name in INPUT_PROCESSES.values():
        processes[name] = _process_status(True)
    manager.status.side_effect = processes.copy
    manager.stop.side_effect = lambda name: processes.update({name: _process_status()})
    monitor.reset.side_effect = lambda: monitor.snapshot.configure_mock(
        return_value=ModeStatus(mode=None, connected=False, age_seconds=None)
    )

    status = service.status()

    assert manager.stop.call_args_list == [call(name) for name in INPUT_PROCESSES.values()]
    assert all(not status.processes[name].running for name in INPUT_PROCESSES.values())
    assert status.processes[ProcessName.STACK].exit_code == failure_code
    assert status.orchestrator.connected is False
    monitor.reset.assert_called_once_with()
    cast(MagicMock, service.recording_service).finalize.assert_called_once_with()

    monitor.snapshot.return_value = ModeStatus(mode=Mode.IDLE, connected=True, age_seconds=0.0)
    with pytest.raises(BackendError, match="already broadcasting"):
        service.status()
    monitor.reset.assert_called_once_with()


def test_runtime_change_is_rejected_while_external_stack_is_active():
    service, manager, _, _ = _make_service(connected=True, mode=Mode.IDLE)

    with pytest.raises(BackendError, match="already broadcasting"):
        service.set_runtime(Runtime.REAL, _safety_context(acknowledged=True))

    manager.set_runtime.assert_not_called()


def test_robot_change_is_rejected_while_external_stack_is_active():
    service, manager, _, _ = _make_service(connected=True, mode=Mode.IDLE)

    with pytest.raises(BackendError, match="already broadcasting"):
        service.set_robot(RobotName.PANDA, _safety_context())

    manager.set_robot.assert_not_called()


def test_stop_process_does_not_control_external_stack():
    service, manager, client, _ = _make_service(connected=True, mode=Mode.KEYBOARD)

    with pytest.raises(BackendError, match="already broadcasting"):
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

    with pytest.raises(BackendError, match="Configuration changed"):
        service.set_robot(RobotName.PANDA, safety)

    manager.set_robot.assert_not_called()


def test_real_runtime_selection_does_not_require_hardware_acknowledgement():
    service, manager, _, _ = _make_service()

    service.set_runtime(Runtime.REAL, _safety_context())

    manager.set_runtime.assert_called_once_with(Runtime.REAL)


def test_real_stack_start_requires_current_configuration_and_acknowledgement():
    service, manager, _, _ = _make_service()
    manager.runtime = Runtime.REAL

    with pytest.raises(BackendError, match="acknowledgement is required"):
        service.start_process(
            ProcessName.STACK,
            _safety_context(runtime=Runtime.REAL),
        )

    manager.start.assert_not_called()


def test_stack_start_rejects_stale_operator_snapshot():
    service, manager, _, _ = _make_service()
    manager.runtime = Runtime.REAL

    with pytest.raises(BackendError, match="Configuration changed"):
        service.start_process(
            ProcessName.STACK,
            _safety_context(acknowledged=True),
        )

    manager.start.assert_not_called()


def test_stale_client_cannot_start_after_another_client_selects_real_runtime():
    service, manager, _, _ = _make_service()
    manager.set_runtime.side_effect = lambda runtime: setattr(manager, "runtime", runtime)

    service.set_runtime(Runtime.REAL, _safety_context(acknowledged=True))

    with pytest.raises(BackendError, match="Configuration changed"):
        service.start_process(ProcessName.STACK, _safety_context())

    manager.start.assert_not_called()


def test_request_waits_for_orchestrator_readiness():
    service, _, client, _ = _make_service(stack=True)

    with pytest.raises(BackendError, match="finish starting"):
        service.request_mode(_homing_request(HomingPreset.HOME))

    client.request_homing.assert_not_called()


def test_teleop_mode_requires_matching_process():
    service, _, client, _ = _make_service(stack=True, connected=True, mode=Mode.IDLE)

    with pytest.raises(BackendError, match="Start Keyboard teleop"):
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


def test_system_mode_does_not_require_an_input_process():
    service, _, client, _ = _make_service(stack=True, connected=True, mode=Mode.IDLE)
    service.request_mode(OrchestratorRequest(mode=Mode.SYSTEM))
    client.request_system.assert_called_once_with()


def test_agent_start_requires_separate_autonomy_selection():
    service, manager, client, _ = _make_service(stack=True, connected=True, mode=Mode.IDLE)
    request = OrchestratorRequest(mode=Mode.AUTONOMY)
    with pytest.raises(BackendError, match="Start API"):
        service.request_mode(request)
    service.start_process(ProcessName.AUTONOMY, _safety_context())
    manager.start.assert_called_once_with(ProcessName.AUTONOMY)
    client.request_autonomy.assert_not_called()
    manager.status.return_value[ProcessName.AUTONOMY] = _process_status(True)
    service.request_mode(request)
    client.request_autonomy.assert_called_once_with()


@pytest.mark.parametrize("action", [ProcessAction.START, ProcessAction.STOP])
def test_recording_controls_call_backend_service(action):
    service, _, client, _ = _make_service(stack=True, connected=True, mode=Mode.IDLE)
    service.set_recording(action)
    recorder = cast(MagicMock, service.recording_service)
    method = recorder.start if action is ProcessAction.START else recorder.stop
    method.assert_called_once_with()
    client._send.assert_not_called()


def test_recording_controls_require_ready_stack():
    service, _, _, _ = _make_service()
    with pytest.raises(BackendError, match="Start the main stack"):
        service.set_recording(ProcessAction.START)
    cast(MagicMock, service.recording_service).start.assert_not_called()


def test_episode_recording_rejects_stale_runtime_before_requesting_recording():
    service, manager, _, _ = _make_service(stack=True, connected=True)
    manager.runtime = Runtime.REAL
    with pytest.raises(BackendError, match="runtime has expired"):
        service.set_episode_recording(ProcessAction.START, "one", "old-session")
    cast(MagicMock, service.recording_service).start.assert_not_called()
    service.set_episode_recording(ProcessAction.START, "one", service.runtime_id)
    cast(MagicMock, service.recording_service).start.assert_called_once_with(
        owner=RecordingOwner(service.runtime_id, "one")
    )


def test_hardware_episode_records_camera_and_state_channels_even_on_exception(
    tmp_path, monkeypatch
):
    service, manager, _, _ = _make_service(stack=True, connected=True)
    manager.runtime = Runtime.REAL
    url = private_lcm_url()
    publisher = Publisher(url)
    # Multiport discovery needs an active receive loop for this synthetic publisher.
    publisher.lc.subscribe(".*", lambda _channel, _data: None)
    service.orchestrator_client = OrchestratorClient(publisher)
    channels = {
        "CAMERA/BASE/RGB",
        "CAMERA/BASE/DEPTH",
        "CAMERA/WRIST/RGB",
        "CAMERA/WRIST/DEPTH",
        Topic.ROBOT_STATE.value,
    }
    with ExitStack() as resources:
        resources.callback(service.close)
        recorder = resources.enter_context(closing(RecordingService(log_dir=tmp_path, lcm_url=url)))
        service.recording_service = recorder
        service.runtime_service.recording_service = recorder
        server = make_server("127.0.0.1", 0, create_app(service))
        resources.callback(server.server_close)
        pool = resources.enter_context(ThreadPoolExecutor(max_workers=2))
        stop = Event()
        resources.callback(stop.set)
        pool.submit(server.serve_forever, poll_interval=0.01)
        resources.callback(server.shutdown)
        console = f"http://127.0.0.1:{server.server_port}"
        with (
            pytest.raises(ValueError, match="interrupted"),
            record_lcm_episode(console, service.runtime_id) as name,
        ):
            # A late wildcard subscriber can wait for mpudpm's five-second
            # channel-map rebroadcast. Exercise that path with sustained input.
            deadline = time.monotonic() + 6
            while time.monotonic() < deadline:
                for channel in channels:
                    publisher.lc.publish(channel, b"episode sample")
                publisher.lc.handle_timeout(0)
                time.sleep(0.01)
            raise ValueError("episode interrupted")
        assert recorder.process is None
        assert Path(name).with_name("robot.json").is_file()
        with closing(lcm.EventLog(name, "r")) as log:
            assert {event.channel for event in log if event.data == b"episode sample"} == channels


def test_recording_failure_reaches_api_caller():
    service, _, _, _ = _make_service(stack=True, connected=True, mode=Mode.IDLE)
    cast(MagicMock, service.recording_service).start.return_value = RecordingStatus(
        0, RecordingState.FAILED, error="disk full"
    )
    with pytest.raises(BackendError, match="disk full"):
        service.set_recording(ProcessAction.START)


def test_status_exposes_latest_recording_lifecycle():
    service, _, _, _ = _make_service()
    expected = RecordingStatus(
        timestamp=2.0,
        state=RecordingState.RUNNING,
        file_name="logs/lcmlog_20260101",
    )
    cast(MagicMock, service.recording_service).snapshot.return_value = expected

    assert service.status().recording == expected


def test_status_exposes_latest_actuator_health():
    service, _, _, _ = _make_service()
    expected = ActuatorHealthStatus(
        connected=False,
        healthy=False,
        age_seconds=3.0,
        actuators=(),
        error="Robot driver stopped.",
    )
    cast(MagicMock, service.actuator_health_monitor).snapshot.return_value = expected

    assert service.status().actuator_health == expected


def test_logs_exposes_incremental_snapshot():
    service, _, _, _ = _make_service()
    expected = LogSnapshot(
        cursor=8,
        entries=[LogEntry(cursor=8, message="[WARNING] Input is stale")],
        reset=False,
        capacity=200,
    )
    cast(MagicMock, service.log_service).snapshot.return_value = expected

    assert service.logs(after=7) == expected
    cast(MagicMock, service.log_service).snapshot.assert_called_once_with(7)


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
        ROBOT_CONFIGS[RobotName.TRISKEL],
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

    with pytest.raises(BackendError, match="dedicated replay controls"):
        method(*arguments)

    manager.start.assert_not_called()
    manager.stop.assert_not_called()


def test_real_replay_requires_hardware_acknowledgement(tmp_path):
    service, manager, _, _ = _make_service()
    manager.runtime = Runtime.REAL
    recording = _recording(tmp_path)
    cast(MagicMock, service.recording_catalog).get.return_value = recording

    with pytest.raises(BackendError, match="acknowledgement is required"):
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
        ROBOT_CONFIGS[RobotName.TRISKEL],
    )
    manager.start.assert_called_once_with(ProcessName.REPLAY)


def test_replay_start_failure_stops_nodes_and_preserves_recording(tmp_path):
    service, manager, _, monitor = _make_service()
    replay_manager = cast(MagicMock, service.replay_manager)
    recording = _recording(tmp_path)
    cast(MagicMock, service.recording_catalog).get.return_value = recording
    manager.wait_until_robot_ready.side_effect = NodeManagerError("Robot state unavailable")

    with pytest.raises(BackendError, match="Robot state unavailable"):
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

    with pytest.raises(BackendError, match="configuration does not match"):
        service.start_replay(recording.id, _safety_context())

    manager.start.assert_not_called()
    assert recording.log_path.exists()


def test_replay_requires_all_managed_processes_to_be_stopped(tmp_path):
    service, manager, _, _ = _make_service(stack=True, connected=True)
    recording = _recording(tmp_path)
    cast(MagicMock, service.recording_catalog).get.return_value = recording

    with pytest.raises(BackendError, match="Stop the stack and input"):
        service.start_replay(recording.id, _safety_context())

    manager.start.assert_not_called()


def test_replay_rejects_unknown_recording(tmp_path):
    service, manager, _, _ = _make_service()
    cast(MagicMock, service.recording_catalog).get.side_effect = RecordingError(
        "The selected recording is unavailable."
    )

    with pytest.raises(BackendError, match="selected recording is unavailable"):
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
    monkeypatch.setattr("humanoid.api.services.modes.time.monotonic", monotonic)

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
    cast(MagicMock, service.actuator_health_monitor).close.assert_called_once_with()
    cast(MagicMock, service.log_service).close.assert_called_once_with()


def test_camera_monitor_is_lazy_reset_on_configuration_change_and_closed(monkeypatch):
    service, _, _, _ = _make_service()
    base, wrist = MagicMock(), MagicMock()
    factory = MagicMock(side_effect=[base, wrist])
    monkeypatch.setattr("humanoid.api.application.CameraFrameService", factory)
    factory.assert_not_called()
    assert service.camera_frame_services == {"base": base, "wrist": wrist}
    assert service.camera_frame_services == {"base": base, "wrist": wrist}
    assert factory.call_args_list == [
        call(camera_name="base", lcm_url=service.lcm_url),
        call(camera_name="wrist", lcm_url=service.lcm_url),
    ]
    service.set_robot(
        RobotName.PANDA,
        SafetyContext(expected_runtime=Runtime.SIM, expected_robot=RobotName.TRISKEL),
    )
    assert not service._camera_frame_services
    service.close()
    for monitor in (base, wrist):
        monitor.close.assert_called_once()


@pytest.mark.parametrize("robot", [RobotName.PANDA, RobotName.SO101])
def test_robot_without_cameras_has_disconnected_previews(robot):
    service, manager, _, _ = _make_service()
    manager.robot = robot
    preview = service.camera_frame_services
    assert preview == {}
    service.close()


def test_supervision_reaps_stack_failure_without_status_requests():
    service, manager, _, _ = _make_service(stack=True, connected=True)
    processes = manager.status.return_value
    service.start()
    try:
        processes[ProcessName.STACK] = ProcessStatus(
            running=False,
            pid=None,
            exit_code=1,
            runtime=Runtime.SIM,
            uptime_seconds=None,
            last_output="stack crashed",
        )
        deadline = time.monotonic() + 2
        while (
            not cast(MagicMock, service.recording_service).finalize.called
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        cast(MagicMock, service.recording_service).finalize.assert_called()
        assert service._supervisor is not None and service._supervisor.is_alive()
    finally:
        service.close()


def test_read_only_agent_task_does_not_require_motion_process(monkeypatch):
    service, _, _, _ = _make_service(stack=True, connected=True, mode=Mode.IDLE)
    observations = MagicMock()
    observations.observation.return_value.mode = Mode.IDLE
    service._observation_service = observations
    console = MagicMock()
    monkeypatch.setattr(service.agent_task_service, "console", console)
    try:
        service.run_agent("Describe the scene", "test-model", Mode.IDLE, _safety_context())
        console.start.assert_called_once()
        execution = console.start.call_args.args[0]
        assert execution.task.instruction == "Describe the scene"
        assert execution.model == "test-model"
        assert execution.mode is Mode.IDLE
        assert execution.runtime_id == service.runtime_id
        assert execution.motion_session_id is None
        service.motion_service.detach()
        service.stop_process(ProcessName.AUTONOMY)
        console.stop.assert_not_called()
        assert service.runtime_id == execution.runtime_id
    finally:
        service.close()


def test_motion_restart_preserves_runtime_and_episode_recording(tmp_path):
    service, manager, _, _ = _make_service(stack=True, connected=True)
    manager.runtime = Runtime.REAL
    with patch("humanoid.api.services.recording.subprocess.Popen") as popen:
        popen.return_value.poll.return_value = None
        popen.return_value.wait.return_value = 0
        with closing(RecordingService(tmp_path)) as recording:
            service.recording_service = recording
            original_runtime = service.runtime_id
            service.set_episode_recording(ProcessAction.START, "episode", original_runtime)
            service.motion_service.detach()
            service.status()
            assert service.runtime_id == original_runtime
            assert recording.snapshot().state is RecordingState.RUNNING
            service._reset_runtime()
            assert service.runtime_id != original_runtime
            assert recording.snapshot().state is RecordingState.STOPPED
            assert (
                service.set_episode_recording(ProcessAction.STOP, "episode", original_runtime).state
                is RecordingState.STOPPED
            )


def test_toggle_recording_uses_authoritative_state_and_preserves_episode_owner(tmp_path):
    service, _, _, _ = _make_service(stack=True, connected=True)
    with patch("humanoid.api.services.recording.subprocess.Popen") as popen:
        popen.return_value.poll.return_value = None
        popen.return_value.wait.return_value = 0
        with closing(RecordingService(tmp_path)) as recording:
            service.recording_service = recording
            assert service.toggle_recording().state is RecordingState.RUNNING
            assert service.toggle_recording().state is RecordingState.STOPPED
            recording.start(owner=RecordingOwner(service.runtime_id, "episode"))
            with pytest.raises(RuntimeError, match="does not own"):
                service.toggle_recording()
            assert recording.snapshot().state is RecordingState.RUNNING


def test_public_observation_service_creation_replaces_and_closes_owned_service(monkeypatch):
    service, manager, _, _ = _make_service()
    default_config = ROBOT_CONFIGS[manager.robot]
    custom_config = replace(default_config, cameras={"chassis": default_config.cameras["base"]})
    original, replacement = MagicMock(), MagicMock()
    factory = MagicMock(side_effect=[original, replacement])
    monkeypatch.setattr("humanoid.api.application.ObservationService", factory)
    try:
        assert service.observation_service is original
        assert service.create_observation_service(custom_config) is replacement
        original.close.assert_called_once_with()
        assert service.observation_service is replacement
        assert factory.call_args_list == [
            call(default_config, lcm_url=service.lcm_url),
            call(custom_config, lcm_url=service.lcm_url),
        ]
    finally:
        service.close()
    replacement.close.assert_called_once_with()
