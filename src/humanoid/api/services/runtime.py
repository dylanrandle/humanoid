"""Runtime configuration, process lifecycle and replay exclusion."""

import contextlib
from _thread import RLock
from collections.abc import Callable

from humanoid.api.services.modes import ModeService
from humanoid.api.services.recording import RecordingService
from humanoid.api.services.replay import ReplayManager, ReplayManagerError
from humanoid.config import ROBOT_CONFIGS
from humanoid.config.backend import (
    CONTROLLED_PROCESS_NAMES,
    INPUT_PROCESSES,
    PROCESS_MODES,
)
from humanoid.logger import get_logger
from humanoid.nodes.groups import process_display_name
from humanoid.nodes.manager import NodeManager, NodeManagerError
from humanoid.recording import RecordingCatalog, RecordingError
from humanoid.types.api import BackendError, BackendErrorMessage, BackendStatus, SafetyContext
from humanoid.types.orchestrator import (
    Mode,
)
from humanoid.types.process import ProcessName, ProcessStatus, Runtime
from humanoid.types.replay import ReplayOutcome, ReplayStatus
from humanoid.types.robot import RobotName
from humanoid.types.simulation import SimulationScene

logger = get_logger(__name__)


class RuntimeService:
    def __init__(  # noqa: PLR0913 - explicit application collaborators
        self,
        node_manager: NodeManager,
        mode_service: ModeService,
        recording_service: RecordingService,
        replay_manager: ReplayManager,
        recording_catalog: RecordingCatalog,
        status: Callable[[], BackendStatus],
        reset_runtime: Callable[[], None],
        stop_agent: Callable[[], None],
        lock: RLock,
        reset_motion: Callable[[], None],
        stop_motion_agent: Callable[[], None],
    ) -> None:
        self.node_manager = node_manager
        self.mode_service = mode_service
        self.recording_service = recording_service
        self.replay_manager = replay_manager
        self.recording_catalog = recording_catalog
        self._status = status
        self._reset_runtime = reset_runtime
        self._stop_agent = stop_agent
        self._lock = lock
        self._reset_motion = reset_motion
        self._stop_motion_agent = stop_motion_agent
        self._handled_stack_failure: ProcessStatus | None = None

    def reconcile(self) -> tuple[dict[ProcessName, ProcessStatus], ReplayStatus]:
        processes = self.node_manager.status()
        replay, changed = self._reconcile_replay(processes)
        if changed:
            processes = self.node_manager.status()
        stack = processes[ProcessName.STACK]
        failure = stack if stack.exit_code not in {None, 0} else None
        if failure != self._handled_stack_failure:
            if failure is not None:
                self._stop_agent()
                for name in INPUT_PROCESSES.values():
                    if processes[name].running:
                        self.node_manager.stop(name)
                processes = self.node_manager.status()
                self.mode_service.mode_monitor.reset()
                self.recording_service.finalize()
                self.mode_service.clear()
                self._reset_runtime()
            self._handled_stack_failure = failure
        return processes, replay

    def set_runtime(self, runtime: Runtime, safety: SafetyContext) -> BackendStatus:
        return self._set_configuration(lambda: self.node_manager.set_runtime(runtime), safety)

    def set_scene(self, scene: SimulationScene, safety: SafetyContext) -> BackendStatus:
        return self._set_configuration(lambda: self.node_manager.set_scene(scene), safety)

    def set_robot(self, robot: RobotName, safety: SafetyContext) -> BackendStatus:
        return self._set_configuration(lambda: self.node_manager.set_robot(robot), safety)

    def _set_configuration(
        self,
        action: Callable[[], None],
        safety: SafetyContext,
        *,
        requires_real_acknowledgement: bool = False,
    ) -> BackendStatus:
        with self._lock:
            self._status()
            self.require_expected_configuration(safety)
            if requires_real_acknowledgement:
                self._require_real_hardware_acknowledgement(safety)
            self._run_manager_action(action)
            self._stop_agent()
            self._reset_motion()
            self._reset_runtime()
            return self._status()

    def start_process(
        self,
        process_name: ProcessName,
        safety: SafetyContext,
    ) -> BackendStatus:
        with self._lock:
            if process_name not in CONTROLLED_PROCESS_NAMES:
                raise BackendError(BackendErrorMessage.REPLAY_DEDICATED_CONTROL)
            current = self._status()
            self.require_expected_configuration(safety)
            if process_name is ProcessName.STACK and self.node_manager.runtime is Runtime.REAL:
                self._require_real_hardware_acknowledgement(safety)
            if process_name is ProcessName.STACK and current.replay.running:
                raise BackendError(BackendErrorMessage.REPLAY_ACTIVE)
            if process_name is ProcessName.STACK:
                if not current.processes[ProcessName.STACK].running:
                    self._reset_motion()
                    self._reset_runtime()
            else:
                self.require_ready_stack(current)
                if (
                    not current.processes[process_name].running
                    and current.orchestrator.mode is PROCESS_MODES[process_name]
                ):
                    self.mode_service.orchestrator_client.request_idle()
                    if not self.mode_service.wait_for_mode(Mode.IDLE):
                        raise BackendError(
                            f"Select Idle before restarting {process_display_name(process_name)}."
                        )
            self._run_manager_action(lambda: self.node_manager.start(process_name))
            return self._status()

    def stop_process(self, process_name: ProcessName) -> BackendStatus:
        with self._lock:
            if process_name not in CONTROLLED_PROCESS_NAMES:
                raise BackendError(BackendErrorMessage.REPLAY_DEDICATED_CONTROL)
            current = self._status()
            if process_name is ProcessName.AUTONOMY:
                self._stop_motion_agent()
            if process_name is ProcessName.STACK:
                self.stop_stack()
            else:
                if current.orchestrator.mode is PROCESS_MODES[process_name]:
                    self.mode_service.orchestrator_client.request_idle()
                    if not self.mode_service.wait_for_mode(Mode.IDLE):
                        logger.warning("Timed out waiting for the orchestrator to enter idle")
                self.node_manager.stop(process_name)
            return self._status()

    def start_replay(
        self,
        recording_id: str,
        safety: SafetyContext,
    ) -> BackendStatus:
        with self._lock:
            replay_nodes_started = False
            replay_player_started = False
            try:
                current = self._status()
                self.require_expected_configuration(safety)
                if self.node_manager.runtime is Runtime.REAL:
                    self._require_real_hardware_acknowledgement(safety)
                if current.replay.running or any(
                    process.running for process in current.processes.values()
                ):
                    raise BackendError(BackendErrorMessage.REPLAY_ACTIVE_PROCESSES)

                recording = self.recording_catalog.get(recording_id)
                robot_config = ROBOT_CONFIGS[self.node_manager.robot]
                self.replay_manager.validate(recording, robot_config)
                self._reset_runtime()
                self._run_manager_action(lambda: self.node_manager.start(ProcessName.REPLAY))
                replay_nodes_started = True
                self._run_manager_action(self.node_manager.wait_until_robot_ready)
                self.replay_manager.start(recording)
                replay_player_started = True
                return self._status()
            except (RecordingError, ReplayManagerError) as exc:
                if replay_nodes_started:
                    self.node_manager.stop(ProcessName.REPLAY)
                    self.mode_service.mode_monitor.reset()
                raise BackendError(str(exc)) from exc
            except Exception:
                if replay_player_started:
                    self.replay_manager.stop()
                if replay_nodes_started:
                    self.node_manager.stop(ProcessName.REPLAY)
                    self.mode_service.mode_monitor.reset()
                raise

    def stop_replay(self) -> BackendStatus:
        with self._lock:
            self.replay_manager.stop()
            self.node_manager.stop(ProcessName.REPLAY)
            self.mode_service.mode_monitor.reset()
            self._reset_runtime()
            return self._status()

    def stop_stack(self) -> None:
        self._stop_agent()
        self.replay_manager.stop()
        processes = self.node_manager.status()
        if not any(process.running for process in processes.values()):
            self.recording_service.finalize()
            self._reset_motion()
            self._reset_runtime()
            return
        if processes[ProcessName.STACK].running:
            with contextlib.suppress(Exception):
                self.mode_service.orchestrator_client.request_idle()
        self.recording_service.finalize()
        self.node_manager.close()
        self._reset_motion()
        self.mode_service.mode_monitor.reset()
        self.mode_service.clear()
        self._reset_runtime()

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
            self.mode_service.mode_monitor.reset()
        elif replay.running and not replay_process.running:
            replay = self.replay_manager.fail("A required replay node stopped unexpectedly.")
            self.mode_service.mode_monitor.reset()
        elif not replay.running and replay_process.running:
            self.node_manager.stop(ProcessName.REPLAY)
            self.mode_service.mode_monitor.reset()
            self._reset_runtime()
            return replay, True
        return replay, False

    def require_expected_configuration(self, safety: SafetyContext) -> None:
        if (
            safety.expected_runtime is not self.node_manager.runtime
            or safety.expected_robot is not self.node_manager.robot
        ):
            raise BackendError(BackendErrorMessage.STALE_CONFIGURATION)

    @staticmethod
    def _require_real_hardware_acknowledgement(safety: SafetyContext) -> None:
        if not safety.real_hardware_acknowledged:
            raise BackendError(BackendErrorMessage.REAL_HARDWARE_ACKNOWLEDGEMENT)

    @staticmethod
    def require_ready_stack(status: BackendStatus) -> None:
        if not status.processes[ProcessName.STACK].running:
            raise BackendError("Start the main stack before using robot controls.")
        if not status.orchestrator.connected:
            raise BackendError("Wait for the main stack to finish starting.")

    @staticmethod
    def _run_manager_action[Result](action: Callable[[], Result]) -> Result:
        try:
            return action()
        except NodeManagerError as exc:
            raise BackendError(str(exc)) from exc
