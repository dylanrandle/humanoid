"""Application service coordinating processes and orchestrator requests."""

import contextlib
import threading
import time
from collections.abc import Callable

from humanoid.config import ROBOT_CONFIGS
from humanoid.logger import get_logger
from humanoid.nodes.groups import process_display_name
from humanoid.nodes.manager import NodeManager, NodeManagerError
from humanoid.orchestrator.client import OrchestratorClient
from humanoid.orchestrator.constants import (
    CONTROLLED_PROCESS_NAMES,
    EXTERNAL_STACK_ERROR,
    HOMING_TARGETS,
    MODE_TRANSITION_POLL_INTERVAL_SECONDS,
    MODE_TRANSITION_TIMEOUT_SECONDS,
    PARAMETERIZED_REQUEST_TIMEOUT_SECONDS,
    PROCESS_MODES,
    REAL_HARDWARE_ACKNOWLEDGEMENT_ERROR,
    REPLAY_ACTIVE_ERROR,
    REPLAY_ACTIVE_PROCESSES_ERROR,
    REPLAY_DEDICATED_CONTROL_ERROR,
    STALE_CONFIGURATION_ERROR,
    TELEOP_PROCESSES,
)
from humanoid.orchestrator.monitor import LoggingMonitor, OrchestratorMonitor
from humanoid.orchestrator.replay import ReplayManager, ReplayManagerError
from humanoid.recording import RecordingCatalog, RecordingError
from humanoid.types.homing import HomingPreset
from humanoid.types.orchestrator import (
    Mode,
    OrchestratorError,
    OrchestratorParameter,
    OrchestratorRequest,
    OrchestratorState,
    OrchestratorStatus,
    SafetyContext,
)
from humanoid.types.process import ProcessAction, ProcessName, ProcessStatus, Runtime
from humanoid.types.replay import ReplayOutcome, ReplayStatus
from humanoid.types.robot import RobotName

logger = get_logger(__name__)


class OrchestratorService:
    """Coordinates process lifecycle and orchestrator requests."""

    def __init__(
        self,
        node_manager: NodeManager | None = None,
        orchestrator_client: OrchestratorClient | None = None,
        orchestrator_monitor: OrchestratorMonitor | None = None,
        logging_monitor: LoggingMonitor | None = None,
        replay_manager: ReplayManager | None = None,
    ):
        self.node_manager = node_manager if node_manager is not None else NodeManager()
        self.orchestrator_client = (
            orchestrator_client if orchestrator_client is not None else OrchestratorClient()
        )
        self.orchestrator_monitor = (
            orchestrator_monitor if orchestrator_monitor is not None else OrchestratorMonitor()
        )
        self.logging_monitor = logging_monitor if logging_monitor is not None else LoggingMonitor()
        self.replay_manager = replay_manager if replay_manager is not None else ReplayManager()
        self.recording_catalog = RecordingCatalog()
        self._lock = threading.RLock()
        self._parameterized_request: OrchestratorRequest | None = None
        self._parameterized_request_at: float | None = None
        self._parameterized_mode_seen = False
        self._handled_stack_failure: ProcessStatus | None = None

    def status(self) -> OrchestratorStatus:
        with self._lock:
            processes = self.node_manager.status()
            replay, processes_changed = self._reconcile_replay(processes)
            if processes_changed:
                processes = self.node_manager.status()
            stack = processes[ProcessName.STACK]
            stack_failure = stack if stack.exit_code not in {None, 0} else None
            if stack_failure != self._handled_stack_failure:
                self._handled_stack_failure = stack_failure
                if stack_failure is not None:
                    self.orchestrator_monitor.reset()
                    self.logging_monitor.reset()
                    self._clear_parameterized_request()
            orchestrator = self._orchestrator_state()
            stack_running = stack.running
            replay_running = processes[ProcessName.REPLAY].running or replay.running
            if orchestrator.connected and not (stack_running or replay_running):
                raise OrchestratorError(EXTERNAL_STACK_ERROR)
            return OrchestratorStatus(
                runtime=self.node_manager.runtime,
                robot=self.node_manager.robot,
                robots=list(RobotName),
                processes=processes,
                logging=self.logging_monitor.snapshot(),
                recordings=self.recording_catalog.list(),
                replay=replay,
                orchestrator=orchestrator,
            )

    def set_runtime(self, runtime: Runtime, safety: SafetyContext) -> OrchestratorStatus:
        return self._set_configuration(
            lambda: self.node_manager.set_runtime(runtime),
            safety,
            requires_real_acknowledgement=runtime is Runtime.REAL,
        )

    def set_robot(self, robot: RobotName, safety: SafetyContext) -> OrchestratorStatus:
        return self._set_configuration(lambda: self.node_manager.set_robot(robot), safety)

    def _set_configuration(
        self,
        action: Callable[[], None],
        safety: SafetyContext,
        *,
        requires_real_acknowledgement: bool = False,
    ) -> OrchestratorStatus:
        with self._lock:
            self.status()
            self._require_expected_configuration(safety)
            if requires_real_acknowledgement:
                self._require_real_hardware_acknowledgement(safety)
            self._run_manager_action(action)
            return self.status()

    def start_process(
        self,
        process_name: ProcessName,
        safety: SafetyContext,
    ) -> OrchestratorStatus:
        with self._lock:
            if process_name not in CONTROLLED_PROCESS_NAMES:
                raise OrchestratorError(REPLAY_DEDICATED_CONTROL_ERROR)
            current = self.status()
            self._require_expected_configuration(safety)
            if process_name is ProcessName.STACK and self.node_manager.runtime is Runtime.REAL:
                self._require_real_hardware_acknowledgement(safety)
            if process_name is ProcessName.STACK and current.replay.running:
                raise OrchestratorError(REPLAY_ACTIVE_ERROR)
            if process_name is not ProcessName.STACK:
                self._require_ready_stack(current)
            self._run_manager_action(lambda: self.node_manager.start(process_name))
            return self.status()

    def stop_process(self, process_name: ProcessName) -> OrchestratorStatus:
        with self._lock:
            if process_name not in CONTROLLED_PROCESS_NAMES:
                raise OrchestratorError(REPLAY_DEDICATED_CONTROL_ERROR)
            current = self.status()
            if process_name is ProcessName.STACK:
                self._stop_stack()
            else:
                if current.orchestrator.mode is PROCESS_MODES[process_name]:
                    self.orchestrator_client.request_idle()
                    if not self._wait_for_mode(Mode.IDLE):
                        logger.warning("Timed out waiting for the orchestrator to enter idle")
                self.node_manager.stop(process_name)
            return self.status()

    def request_mode(self, request: OrchestratorRequest) -> OrchestratorStatus:
        with self._lock:
            current = self.status()
            self._require_ready_stack(current)

            if request.mode is Mode.HOMING:
                preset = request.parameters[OrchestratorParameter.PRESET]
                assert isinstance(preset, HomingPreset)
                robot_config = ROBOT_CONFIGS[self.node_manager.robot]
                self.orchestrator_client.request_homing(HOMING_TARGETS[preset](robot_config))
                self._select_parameterized_request(request)
                return self.status()

            teleop_process = TELEOP_PROCESSES.get(request.mode)
            if teleop_process is not None and not current.processes[teleop_process].running:
                raise OrchestratorError(
                    f"Start {process_display_name(teleop_process)} before activating its mode."
                )

            mode_requests: dict[Mode, Callable[[], None]] = {
                Mode.IDLE: self.orchestrator_client.request_idle,
                Mode.KEYBOARD: self.orchestrator_client.request_keyboard,
                Mode.OCULUS: self.orchestrator_client.request_oculus,
            }
            mode_requests[request.mode]()
            self._clear_parameterized_request()
            return self.status()

    def set_logging(self, action: ProcessAction) -> OrchestratorStatus:
        with self._lock:
            current = self.status()
            self._require_ready_stack(current)
            logging_actions = {
                ProcessAction.START: (
                    self.logging_monitor.start_requested,
                    self.orchestrator_client.start_logging,
                ),
                ProcessAction.STOP: (
                    self.logging_monitor.stop_requested,
                    self.orchestrator_client.stop_logging,
                ),
            }
            mark_requested, publish_request = logging_actions[action]
            mark_requested()
            try:
                publish_request()
            except Exception as exc:
                self.logging_monitor.fail(f"Could not request data logging: {exc}")
                raise
            return self.status()

    def start_replay(
        self,
        recording_id: str,
        safety: SafetyContext,
    ) -> OrchestratorStatus:
        with self._lock:
            replay_nodes_started = False
            replay_player_started = False
            try:
                current = self.status()
                self._require_expected_configuration(safety)
                if self.node_manager.runtime is Runtime.REAL:
                    self._require_real_hardware_acknowledgement(safety)
                if current.replay.running or any(
                    process.running for process in current.processes.values()
                ):
                    raise OrchestratorError(REPLAY_ACTIVE_PROCESSES_ERROR)

                recording = self.recording_catalog.get(recording_id)
                robot_config = ROBOT_CONFIGS[self.node_manager.robot]
                self.replay_manager.validate(recording, robot_config)
                self._run_manager_action(lambda: self.node_manager.start(ProcessName.REPLAY))
                replay_nodes_started = True
                self._run_manager_action(self.node_manager.wait_until_robot_ready)
                self.replay_manager.start(recording)
                replay_player_started = True
                return self.status()
            except (RecordingError, ReplayManagerError) as exc:
                if replay_nodes_started:
                    self.node_manager.stop(ProcessName.REPLAY)
                    self.orchestrator_monitor.reset()
                raise OrchestratorError(str(exc)) from exc
            except Exception:
                if replay_player_started:
                    self.replay_manager.stop()
                if replay_nodes_started:
                    self.node_manager.stop(ProcessName.REPLAY)
                    self.orchestrator_monitor.reset()
                raise

    def stop_replay(self) -> OrchestratorStatus:
        with self._lock:
            self.replay_manager.stop()
            self.node_manager.stop(ProcessName.REPLAY)
            self.orchestrator_monitor.reset()
            return self.status()

    def close(self) -> None:
        try:
            self._stop_stack()
        except Exception:
            logger.exception("Failed to stop the main stack cleanly")
            self.node_manager.close()
        finally:
            self.orchestrator_monitor.close()
            self.logging_monitor.close()
            self.replay_manager.close()

    def _stop_stack(self) -> None:
        self.replay_manager.stop()
        processes = self.node_manager.status()
        if not any(process.running for process in processes.values()):
            self.logging_monitor.reset()
            return
        if processes[ProcessName.STACK].running:
            with contextlib.suppress(Exception):
                self.orchestrator_client.request_idle()
        self.node_manager.close()
        self.orchestrator_monitor.reset()
        self.logging_monitor.reset()
        self._clear_parameterized_request()

    def _reconcile_replay(
        self,
        processes: dict[ProcessName, ProcessStatus],
    ) -> tuple[ReplayStatus, bool]:
        replay = self.replay_manager.status()
        replay_process = processes[ProcessName.REPLAY]
        if replay_process.exit_code not in {None, 0} and replay.outcome is not ReplayOutcome.FAILED:
            replay = self.replay_manager.fail(
                replay_process.last_output or "A required replay node stopped unexpectedly."
            )
            self.orchestrator_monitor.reset()
        elif replay.running and not replay_process.running:
            replay = self.replay_manager.fail("A required replay node stopped unexpectedly.")
            self.orchestrator_monitor.reset()
        elif not replay.running and replay_process.running:
            self.node_manager.stop(ProcessName.REPLAY)
            self.orchestrator_monitor.reset()
            return replay, True
        return replay, False

    def _orchestrator_state(self) -> OrchestratorState:
        mode_status = self.orchestrator_monitor.snapshot()
        mode = mode_status.mode

        request = self._parameterized_request
        if request is not None:
            if mode is request.mode:
                self._parameterized_mode_seen = True
            elif self._parameterized_mode_seen or self._parameterized_request_expired():
                self._clear_parameterized_request()
        request = self._parameterized_request

        return OrchestratorState(
            mode=mode_status.mode,
            connected=mode_status.connected,
            age_seconds=mode_status.age_seconds,
            parameters=dict(request.parameters) if request is not None else {},
        )

    def _select_parameterized_request(self, request: OrchestratorRequest) -> None:
        self._parameterized_request = request
        self._parameterized_request_at = time.monotonic()
        self._parameterized_mode_seen = False

    def _clear_parameterized_request(self) -> None:
        self._parameterized_request = None
        self._parameterized_request_at = None
        self._parameterized_mode_seen = False

    def _parameterized_request_expired(self) -> bool:
        return (
            self._parameterized_request_at is not None
            and time.monotonic() - self._parameterized_request_at
            > PARAMETERIZED_REQUEST_TIMEOUT_SECONDS
        )

    def _wait_for_mode(self, mode: Mode) -> bool:
        deadline = time.monotonic() + MODE_TRANSITION_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.orchestrator_monitor.snapshot().mode is mode:
                return True
            time.sleep(MODE_TRANSITION_POLL_INTERVAL_SECONDS)
        return False

    def _require_expected_configuration(self, safety: SafetyContext) -> None:
        if (
            safety.expected_runtime is not self.node_manager.runtime
            or safety.expected_robot is not self.node_manager.robot
        ):
            raise OrchestratorError(STALE_CONFIGURATION_ERROR)

    @staticmethod
    def _require_real_hardware_acknowledgement(safety: SafetyContext) -> None:
        if not safety.real_hardware_acknowledged:
            raise OrchestratorError(REAL_HARDWARE_ACKNOWLEDGEMENT_ERROR)

    @staticmethod
    def _require_ready_stack(status: OrchestratorStatus) -> None:
        if not status.processes[ProcessName.STACK].running:
            raise OrchestratorError("Start the main stack before using robot controls.")
        if not status.orchestrator.connected:
            raise OrchestratorError("Wait for the main stack to finish starting.")

    @staticmethod
    def _run_manager_action[Result](action: Callable[[], Result]) -> Result:
        try:
            return action()
        except NodeManagerError as exc:
            raise OrchestratorError(str(exc)) from exc
