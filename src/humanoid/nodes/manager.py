"""Process lifecycle management for humanoid node groups."""

import contextlib
import os
import signal
import threading
import time
from collections.abc import Sequence
from multiprocessing import get_context
from multiprocessing.process import BaseProcess
from typing import cast

from humanoid.constants import (
    ROBOT_ENVIRONMENT_VARIABLE,
    RUNTIME_ENVIRONMENT_VARIABLE,
    Topic,
)
from humanoid.logger import get_logger
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.base import Node
from humanoid.nodes.groups import (
    NODE_GROUPS,
    PROCESS_ORDER,
    PROCESS_STOP_ORDER,
)
from humanoid.types.node import ManagedNodeGroup, NodeGroup, ProcessContext
from humanoid.types.process import ProcessName, ProcessStatus, Runtime
from humanoid.types.robot import RobotName

logger = get_logger(__name__)

DEFAULT_STATE_TIMEOUT_SECONDS = 10.0
DEFAULT_STATE_POLL_INTERVAL_SECONDS = 0.1
DEFAULT_STOP_TIMEOUT_SECONDS = 8.0
TERMINATE_TIMEOUT_SECONDS = 2.0
POSIX_OS_NAME = "posix"
CHILD_PROCESS_CONTEXT = "spawn"


class NodeManagerError(Exception):
    """An expected node lifecycle failure."""


class NodeManager:
    """Starts, monitors, and stops imported node classes as logical groups."""

    def __init__(
        self,
        runtime: Runtime | None = None,
        robot: RobotName | None = None,
        state_timeout_seconds: float = DEFAULT_STATE_TIMEOUT_SECONDS,
        state_poll_interval_seconds: float = DEFAULT_STATE_POLL_INTERVAL_SECONDS,
    ):
        self.runtime = runtime if runtime is not None else Runtime.from_environment()
        self.robot = robot if robot is not None else RobotName.from_environment()
        self.state_timeout_seconds = state_timeout_seconds
        self.state_poll_interval_seconds = state_poll_interval_seconds
        self._process_context = cast(ProcessContext, get_context(CHILD_PROCESS_CONTEXT))
        self._groups: dict[ProcessName, ManagedNodeGroup] = {}
        self._lock = threading.RLock()

    def set_runtime(self, runtime: Runtime) -> None:
        with self._lock:
            self._require_configuration_change_allowed_locked()
            self.runtime = runtime

    def set_robot(self, robot: RobotName) -> None:
        with self._lock:
            self._require_configuration_change_allowed_locked()
            self.robot = robot

    def start(self, name: ProcessName) -> ProcessStatus:
        definition = NODE_GROUPS[name]
        with self._lock:
            if (
                definition.allowed_runtimes is not None
                and self.runtime not in definition.allowed_runtimes
            ):
                raise NodeManagerError(
                    f"{definition.display_name} is not available in {self.runtime.value} runtime."
                )
            if self._status_locked(name).running:
                raise NodeManagerError(f"{definition.display_name} is already running.")

            group = ManagedNodeGroup(
                runtime=self.runtime,
                robot=self.robot,
                started_monotonic=time.monotonic(),
            )
            self._groups[name] = group
            try:
                self._start_nodes(group, definition.nodes)
            except Exception as exc:
                message = f"Could not start {definition.display_name}: {exc}"
                group.failure_exit_code = 1
                group.last_output = message
                group.stop_event.set()
                self._stop_processes(group.processes)
                group.ended_monotonic = time.monotonic()
                raise NodeManagerError(message) from exc

            if definition.deferred_nodes:
                startup_thread = threading.Thread(
                    target=self._start_deferred_nodes,
                    args=(definition, group),
                    name=f"{name}-startup",
                    daemon=True,
                )
                group.startup_thread = startup_thread
                startup_thread.start()

            return self._status_locked(name)

    def stop(self, name: ProcessName) -> ProcessStatus:
        self._stop_group(name)
        with self._lock:
            return self._status_locked(name)

    def status(self) -> dict[ProcessName, ProcessStatus]:
        failures: list[tuple[ProcessName, tuple[int, str]]] = []
        with self._lock:
            for name in PROCESS_ORDER:
                failure = self._partial_failure_locked(name)
                if failure is not None:
                    failures.append((name, failure))
        for name, failure in failures:
            self._stop_group(name, failure=failure)
        with self._lock:
            return {name: self._status_locked(name) for name in PROCESS_ORDER}

    def close(self) -> None:
        for name in PROCESS_STOP_ORDER:
            try:
                self.stop(name)
            except Exception:
                logger.exception("Failed to stop %s while closing the node manager", name)

    def wait_until_robot_ready(self) -> None:
        """Wait until the selected driver publishes its first robot state."""
        try:
            self._wait_for_robot_state(threading.Event())
        except RuntimeError as exc:
            raise NodeManagerError(str(exc)) from exc

    def _start_deferred_nodes(
        self,
        definition: NodeGroup,
        group: ManagedNodeGroup,
    ) -> None:
        try:
            if not self._wait_for_robot_state(group.stop_event):
                return
            with self._lock:
                if group.stop_event.is_set() or self._groups.get(definition.name) is not group:
                    return
                self._start_nodes(group, definition.deferred_nodes)
        except Exception as exc:
            message = f"{definition.display_name} failed to start: {exc}"
            logger.exception(message)
            self._stop_group(definition.name, failure=(1, message))

    def _start_nodes(
        self,
        group: ManagedNodeGroup,
        nodes: Sequence[type[Node]],
    ) -> None:
        for node in nodes:
            process = self._process_context.Process(target=node.main, name=node.__name__)
            with _node_environment(group.runtime, group.robot):
                process.start()
            group.processes.append(process)

    def _require_configuration_change_allowed_locked(self) -> None:
        if any(self._status_locked(name).running for name in PROCESS_ORDER):
            raise NodeManagerError(
                "Stop the stack and teleop processes before changing configuration."
            )

    def _wait_for_robot_state(self, stop_event: threading.Event) -> bool:
        subscriber = Subscriber(topics=[Topic.ROBOT_STATE])
        deadline = time.monotonic() + self.state_timeout_seconds
        try:
            while not stop_event.is_set():
                if subscriber.receive(Topic.ROBOT_STATE) is not None:
                    return True
                if time.monotonic() >= deadline:
                    raise RuntimeError("Timed out waiting for robot state")
                stop_event.wait(self.state_poll_interval_seconds)
            return False
        finally:
            subscriber.close()

    def _stop_group(
        self,
        name: ProcessName,
        failure: tuple[int, str] | None = None,
    ) -> None:
        with self._lock:
            group = self._groups.get(name)
            if group is None:
                return
            if failure is not None:
                group.failure_exit_code, group.last_output = failure
            else:
                group.stop_requested = True
            group.stop_event.set()
            if group.shutdown_started:
                return
            group.shutdown_started = True
            processes = list(group.processes)
            startup_thread = group.startup_thread

        if startup_thread is not None and startup_thread is not threading.current_thread():
            startup_thread.join(timeout=1.0)
        self._stop_processes(processes)

        with self._lock:
            group.ended_monotonic = time.monotonic()

    @staticmethod
    def _stop_processes(
        processes: Sequence[BaseProcess],
        timeout: float = DEFAULT_STOP_TIMEOUT_SECONDS,
    ) -> None:
        for process in processes:
            NodeManager._interrupt(process)

        deadline = time.monotonic() + timeout
        for process in processes:
            remaining = max(0.0, deadline - time.monotonic())
            process.join(timeout=remaining)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=TERMINATE_TIMEOUT_SECONDS)
            if process.is_alive():
                process.kill()
                process.join(timeout=TERMINATE_TIMEOUT_SECONDS)

    @staticmethod
    def _interrupt(process: BaseProcess) -> None:
        if not process.is_alive() or process.pid is None:
            return
        if os.name == POSIX_OS_NAME:
            with contextlib.suppress(ProcessLookupError):
                os.kill(process.pid, signal.SIGINT)
        else:
            process.terminate()

    def _status_locked(self, name: ProcessName) -> ProcessStatus:
        group = self._groups.get(name)
        if group is None:
            return _stopped_status()

        alive_processes = [process for process in group.processes if process.is_alive()]
        startup_running = (
            group.startup_thread is not None
            and group.startup_thread.is_alive()
            and not group.shutdown_started
        )
        running = (bool(alive_processes) or startup_running) and group.failure_exit_code is None
        if not running and group.ended_monotonic is None:
            group.ended_monotonic = time.monotonic()

        if running:
            exit_code = None
        elif group.failure_exit_code is not None:
            exit_code = group.failure_exit_code
        elif group.stop_requested:
            exit_code = 0
        else:
            exit_codes = [
                process.exitcode for process in group.processes if process.exitcode is not None
            ]
            exit_code = next((code for code in exit_codes if code != 0), 0 if exit_codes else None)

        last_output = group.last_output
        if not running and last_output is None and exit_code not in {None, 0}:
            failed_process = next(
                (process for process in group.processes if process.exitcode not in {None, 0}),
                None,
            )
            if failed_process is not None:
                last_output = f"{failed_process.name} exited with code {failed_process.exitcode}"

        end = group.ended_monotonic or time.monotonic()
        return ProcessStatus(
            running=running,
            pid=alive_processes[0].pid if alive_processes else None,
            exit_code=exit_code,
            runtime=group.runtime,
            uptime_seconds=round(end - group.started_monotonic, 1),
            last_output=last_output,
        )

    def _partial_failure_locked(self, name: ProcessName) -> tuple[int, str] | None:
        group = self._groups.get(name)
        if (
            group is None
            or group.stop_requested
            or group.shutdown_started
            or group.failure_exit_code is not None
        ):
            return None

        failed_process = next(
            (process for process in group.processes if process.exitcode is not None),
            None,
        )
        if failed_process is None:
            return None

        siblings_running = any(process.is_alive() for process in group.processes)
        startup_running = group.startup_thread is not None and group.startup_thread.is_alive()
        if not siblings_running and not startup_running:
            return None

        exit_code = failed_process.exitcode or 1
        message = f"{failed_process.name} exited unexpectedly with code {failed_process.exitcode}"
        return exit_code, message


def _stopped_status() -> ProcessStatus:
    return ProcessStatus(
        running=False,
        pid=None,
        exit_code=None,
        runtime=None,
        uptime_seconds=None,
        last_output=None,
    )


@contextlib.contextmanager
def _node_environment(runtime: Runtime, robot: RobotName):
    values = {
        RUNTIME_ENVIRONMENT_VARIABLE: runtime,
        ROBOT_ENVIRONMENT_VARIABLE: robot,
    }
    previous_values = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value
