"""Own one stoppable worker and expose its progress to every dashboard tab."""

import multiprocessing
import threading
from dataclasses import replace
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess

from humanoid.agent.harness.worker import work
from humanoid.types.agent import AgentPanelStatus, AgentPhase, AgentTask
from humanoid.types.orchestrator import OrchestratorError


class AgentConsole:
    def __init__(self) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._lock = threading.RLock()
        self._state = AgentPanelStatus()
        self._final_update: AgentPanelStatus | None = None
        self._process: BaseProcess | None = None
        self._channel: Connection | None = None

    def status(self) -> AgentPanelStatus:
        with self._lock:
            if self._process is None:
                return self._state
            # Check exit first so any final write is visible before reaping.
            exited = not self._process.is_alive()
            assert self._channel is not None
            while self._channel.poll():
                try:
                    update: AgentPanelStatus = self._channel.recv()
                except EOFError:
                    break
                if update.phase in (AgentPhase.COMPLETED, AgentPhase.FAILED):
                    # A terminal phase tells the dashboard it can start the next operation.
                    self._final_update = update
                else:
                    self._state = update
            if exited:
                if self._final_update is not None:
                    self._state = self._final_update
                self._release()
                if self._state.phase in (AgentPhase.CONNECTING, AgentPhase.RUNNING):
                    self._state = replace(
                        self._state,
                        phase=AgentPhase.FAILED,
                        message="The agent worker exited unexpectedly.",
                        login_code="",
                        login_url="",
                    )
            return self._state

    def start(self, task: AgentTask | None = None) -> AgentPanelStatus:
        with self._lock:
            current = self.status()
            if self._process is not None:
                raise OrchestratorError("An agent operation is already active.")
            if task and task.model not in current.models:
                raise OrchestratorError("Connect to ChatGPT and select an available model.")
            initial_status = (
                AgentPanelStatus(phase=AgentPhase.CONNECTING, message="Connecting to ChatGPT…")
                if task is None
                else AgentPanelStatus(
                    phase=AgentPhase.RUNNING,
                    models=current.models,
                    model=task.model,
                    instruction=task.instruction,
                    message="Reading the robot and camera…",
                )
            )
            incoming, outgoing = self._context.Pipe(duplex=False)
            process = self._context.Process(
                target=work, args=(outgoing, task, initial_status), daemon=True
            )
            try:
                process.start()
            except BaseException:
                incoming.close()
                raise
            finally:
                outgoing.close()
            self._process, self._channel = process, incoming
            self._state = initial_status
            return self._state

    def stop(self, *, tasks_only: bool = False) -> AgentPanelStatus:
        with self._lock:
            current = self.status()
            if self._process is None or (tasks_only and current.phase is not AgentPhase.RUNNING):
                return current
            self._process.terminate()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join()
            self._release()
            self._state = replace(
                current,
                phase=AgentPhase.STOPPED,
                message="Stopped by operator.",
                login_url="",
                login_code="",
            )
            return self._state

    def _release(self) -> None:
        assert self._channel is not None and self._process is not None
        self._process.join()
        self._process.close()
        self._channel.close()
        self._channel = self._process = None
        self._final_update = None
