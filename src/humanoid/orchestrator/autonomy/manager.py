"""UI-owned task selection, RAX process lifecycle, and realtime episode control."""

import contextlib
import hmac
import json
import os
import secrets
import subprocess
import threading
import uuid
from typing import TYPE_CHECKING

from humanoid.config import ROBOT_CONFIGS
from humanoid.integrations.rax.execution import RaxExecution
from humanoid.integrations.rax.runner import default_command
from humanoid.integrations.rax.server import dispatch
from humanoid.integrations.rax.session import RealtimeSession
from humanoid.robots.base import Robot
from humanoid.types.autonomy import AutonomyStatus
from humanoid.types.orchestrator import Mode, OrchestratorError, SafetyContext
from humanoid.types.process import ProcessName, Runtime
from humanoid.types.tasks import SimulationTaskSelection, TaskEvaluation

if TYPE_CHECKING:
    from humanoid.orchestrator.service import OrchestratorService


class AutonomyManager:
    """Own UI task selection, runner lifecycle, and simulation episode boundaries.

    These blocking process/reset operations stay outside RemoteControlPolicy.step().
    RaxExecution submits commands to the managed policy node; the runner
    receives only authenticated RPC, never a simulator object.

    The local Docker runner is the default. HUMANOID_RAX_COMMAND can override it
    with operator-supplied JSON argv, never HTTP input or generated agent code.
    """

    def __init__(self, service: "OrchestratorService"):
        self.service = service
        self.task: str | None = None
        self._process: subprocess.Popen | None = None
        self._session: RealtimeSession | None = None
        self._token = ""
        self._rpc_lock = threading.Lock()
        self._message = "Select a task. Build the local RAX runner to enable autonomy."

    @staticmethod
    def _command() -> list[str]:
        configured = os.environ.get("HUMANOID_RAX_COMMAND")
        if configured is None:
            return default_command()
        value = json.loads(configured)
        if not isinstance(value, list) or any(not isinstance(arg, str) or not arg for arg in value):
            raise OrchestratorError(
                "HUMANOID_RAX_COMMAND must be a JSON array of command arguments."
            )
        return value

    def status(self) -> AutonomyStatus:
        running = self._process is not None and self._process.poll() is None
        if self._process is not None and not running:
            code = self._process.returncode
            self._message = f"RAX runner exited with code {code}."
            self.stop(preserve_message=True)
        return AutonomyStatus(
            task=self.task,
            tasks=("toy_car_cleanup",),
            running=running,
            ready=bool(self._command()),
            message=self._message,
        )

    def select(self, task: str, safety: SafetyContext) -> AutonomyStatus:
        service = self.service
        with service._lock:
            service.status()
            service._require_expected_configuration(safety)
            if self._process is not None:
                raise OrchestratorError("Stop autonomy before selecting a task.")
            if task not in self.status().tasks:
                raise OrchestratorError("Unknown autonomy task.")
            if service.node_manager.runtime is not Runtime.SIM:
                raise OrchestratorError(
                    "Task setup currently supports simulation; hardware is not qualified yet."
                )
            service._run_manager_action(
                lambda: service.node_manager.set_simulation_task(
                    SimulationTaskSelection(name=task, seed=0)
                )
            )
            self.task = task
            self._message = (
                "Task selected. Start the stack to preview, "
                "or start autonomy when the runner is ready."
            )
            return self.status()

    def start(self, safety: SafetyContext) -> AutonomyStatus:
        service = self.service
        with service._lock:
            service.status()
            service._require_expected_configuration(safety)
            if service.node_manager.runtime is not Runtime.SIM:
                raise OrchestratorError("Hardware autonomy has not been qualified yet.")
            if self.task is None or self._process is not None:
                raise OrchestratorError("Select a task and stop any previous autonomy run first.")
            command = self._command()
            if not command:
                raise OrchestratorError("The Linux RAX runner is not configured yet.")
            self._token = secrets.token_urlsafe(32)
            self._session = RealtimeSession(self, self.task)
            environment = {
                **os.environ,
                "HUMANOID_RAX_TOKEN": self._token,
                "HUMANOID_RAX_TASK": self.task,
            }
            try:
                # Inherit the operator process log; never invoke a shell with agent input.
                self._process = subprocess.Popen(command, env=environment, start_new_session=True)
            except Exception:
                self._session = None
                self._token = ""
                raise
            self._message = (
                "RAX is starting. Robot observations and actions use the realtime stack."
            )
            return self.status()

    def stop(self, *, preserve_message: bool = False) -> AutonomyStatus:
        self._token = ""
        if self._session is not None and self._session.execution is not None:
            self._session.execution.cancel()
        self.service.orchestrator_client.request_idle()
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        with self._rpc_lock:
            if self._session is not None:
                self._session.close()
                self._session = None
        if not preserve_message:
            self._message = "Autonomy stopped."
        return AutonomyStatus(
            task=self.task,
            tasks=("toy_car_cleanup",),
            running=False,
            ready=bool(self._command()),
            message=self._message,
        )

    def rpc(self, authorization: str, payload: dict):
        if not self._token or not hmac.compare_digest(authorization, f"Bearer {self._token}"):
            raise PermissionError("Invalid autonomy supervisor credential.")
        with self._rpc_lock:
            if self._session is None:
                raise RuntimeError("No active autonomy run.")
            return dispatch(self._session, payload)

    def start_episode(self, *, formal: bool, seed: int) -> str:
        service = self.service
        assert self.task is not None
        with service._lock:
            if service.node_manager.runtime is not Runtime.SIM:
                raise OrchestratorError("Episode resets are simulation-only.")
            service._stop_stack()
            service.node_manager.set_simulation_task(
                SimulationTaskSelection(
                    name=self.task,
                    formal=formal,
                    seed=seed,
                )
            )
            try:
                service.node_manager.start(ProcessName.STACK)
                service.node_manager.wait_until_robot_ready()
                if self._process is None:
                    raise OrchestratorError("Autonomy was stopped during episode startup.")
                service.orchestrator_client.request_autonomy()
                if not service._wait_for_mode(Mode.AUTONOMY):
                    raise OrchestratorError("Timed out activating autonomy.")
                service.node_manager.start(ProcessName.AUTONOMY)
            except BaseException:
                service._stop_stack()
                raise
            self._message = (
                "Running formal evaluation." if formal else "Exploring the selected task."
            )
            return uuid.uuid4().hex

    def create_execution(self) -> RaxExecution:
        manager = self.service.node_manager
        endpoint = manager.remote_control_endpoint
        if endpoint is None:
            raise RuntimeError("The autonomy policy node has not been started.")
        return RaxExecution(Robot(ROBOT_CONFIGS[manager.robot]), endpoint)

    def evaluate_episode(self) -> TaskEvaluation:
        evaluation = self.service.node_manager.task_evaluation()
        if evaluation is None:
            raise RuntimeError("No task evaluation has arrived from the simulation node.")
        return evaluation

    def close_episode(self) -> None:
        with contextlib.suppress(Exception):
            self.service.orchestrator_client.request_idle()
        self.service.node_manager.stop(ProcessName.AUTONOMY)
