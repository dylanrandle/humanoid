"""Agent task admission using explicit backend capabilities."""

from _thread import RLock
from collections.abc import Callable

from humanoid.agent.harness.console import AgentConsole
from humanoid.api.services.runtime import RuntimeService
from humanoid.types.agent import AgentExecution, AgentPanelStatus, AgentPhase, AgentTask
from humanoid.types.api import BackendError, BackendStatus, SafetyContext
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode
from humanoid.types.process import ProcessName
from humanoid.types.robot import RobotName


class AgentTaskService:
    def __init__(  # noqa: PLR0913 - narrow application capabilities
        self,
        *,
        status: Callable[[], BackendStatus],
        require_configuration: Callable[[SafetyContext], None],
        observation: Callable[[], Observation],
        runtime_id: Callable[[], str],
        motion_session_id: Callable[[], str],
        release_motion: Callable[[], None],
        lock: RLock,
    ) -> None:
        self._status = status
        self._require_configuration = require_configuration
        self._observation = observation
        self._runtime_id = runtime_id
        self._motion_session_id = motion_session_id
        self._release_motion = release_motion
        self._lock = lock
        self.console = AgentConsole()
        self._task_mode: Mode | None = None

    def status(self) -> AgentPanelStatus:
        return self.console.status()

    def start(self) -> AgentPanelStatus:
        return self.console.start()

    def run(
        self, instruction: str, model: str, mode: Mode, safety: SafetyContext
    ) -> AgentPanelStatus:
        if not instruction.strip() or not model.strip():
            raise ValueError("Provide an instruction and model for the agent.")
        with self._lock:
            self._require_configuration(safety)
            current = self._status()
            RuntimeService.require_ready_stack(current)
            if current.robot is not RobotName.TRISKEL:
                raise BackendError("Select Triskel before running an agent task.")
            if mode is Mode.AUTONOMY and not current.processes[ProcessName.AUTONOMY].running:
                raise BackendError("Start API before requesting motion.")
            if mode not in (Mode.IDLE, Mode.AUTONOMY) or mode is not current.orchestrator.mode:
                raise BackendError("Select Idle for observation or API for motion.")
            if self._observation().mode is not mode:
                raise BackendError("Control mode changed. Refresh and retry.")
            execution = AgentExecution(
                AgentTask(instruction.strip()),
                model,
                mode,
                self._runtime_id(),
                current.robot,
                motion_session_id=self._motion_session_id() if mode is Mode.AUTONOMY else None,
            )
            result = self.console.start(execution)
            self._task_mode = mode
            return result

    def stop(self, *, tasks_only: bool = False) -> AgentPanelStatus:
        with self._lock:
            running = self.console.status().phase is AgentPhase.RUNNING
            result = self.console.stop(tasks_only=tasks_only)
            if not tasks_only and running:
                self._release_motion()
            return result

    def stop_motion_task(self) -> None:
        with self._lock:
            if self._task_mode is Mode.AUTONOMY:
                self.stop(tasks_only=True)
