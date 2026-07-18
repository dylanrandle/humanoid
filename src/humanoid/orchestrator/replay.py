"""Lifecycle management for the external LCM log player."""

import os
import signal
import subprocess
import tempfile
import threading
from typing import BinaryIO

from humanoid.constants import DEFAULT_LCM_URL
from humanoid.orchestrator.constants import (
    LCM_LOGPLAYER_COMMAND,
    REPLAY_CHANNEL_PATTERN,
    REPLAY_STDERR_TAIL_BYTES,
    REPLAY_STOP_TIMEOUT_SECONDS,
)
from humanoid.recording import serialize_robot_config
from humanoid.types.replay import RecordingBundle, ReplayOutcome, ReplayStatus
from humanoid.types.robot import RobotConfig, RobotName


class ReplayManagerError(Exception):
    """An expected replay lifecycle failure."""


class ReplayManager:
    """Starts, observes, and stops one ``lcm-logplayer`` process."""

    def __init__(self, lcm_url: str = DEFAULT_LCM_URL):
        self.lcm_url = lcm_url
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr: BinaryIO | None = None
        self._source: RecordingBundle | None = None
        self._outcome: ReplayOutcome | None = None
        self._exit_code: int | None = None
        self._last_output: str | None = None

    @staticmethod
    def validate(source: RecordingBundle, expected_robot_config: RobotConfig) -> None:
        """Validate that a recording matches the selected robot configuration."""
        if not source.log_path.is_file():
            raise ReplayManagerError("The selected LCM log file is unavailable.")
        expected_robot = RobotName(expected_robot_config.name)
        if source.robot is not expected_robot:
            raise ReplayManagerError(
                f"This recording is for {source.robot.value}, not {expected_robot.value}."
            )
        if source.robot_config != serialize_robot_config(expected_robot_config):
            raise ReplayManagerError(
                "The recorded robot configuration does not match the selected configuration."
            )

    def start(self, source: RecordingBundle) -> ReplayStatus:
        with self._lock:
            current = self._status_locked()
            if current.running:
                raise ReplayManagerError("A replay is already running.")

            self._source = source
            self._outcome = None
            self._exit_code = None
            self._last_output = None
            if not source.log_path.is_file():
                message = "The selected LCM log file is unavailable."
                self._record_start_failure_locked(message)
                raise ReplayManagerError(message)
            command = [
                LCM_LOGPLAYER_COMMAND,
                "-l",
                self.lcm_url,
                "-e",
                REPLAY_CHANNEL_PATTERN,
                str(source.log_path),
            ]
            try:
                # The file stays open for the subprocess lifetime so stderr cannot block it.
                self._stderr = tempfile.TemporaryFile()  # noqa: SIM115
                self._process = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=self._stderr,
                )
            except FileNotFoundError as exc:
                message = "lcm-logplayer was not found in PATH."
                self._record_start_failure_locked(message)
                raise ReplayManagerError(message) from exc
            except OSError as exc:
                message = f"Could not start lcm-logplayer: {exc}"
                self._record_start_failure_locked(message)
                raise ReplayManagerError(message) from exc
            except Exception:
                self._close_stderr_locked()
                raise
            return self._status_locked()

    def status(self) -> ReplayStatus:
        with self._lock:
            return self._status_locked()

    def stop(self) -> ReplayStatus:
        with self._lock:
            current = self._status_locked()
            if not current.running:
                return current
            if not self._interrupt_process_locked():
                return self._status_locked()
            self._outcome = ReplayOutcome.STOPPED
            self._last_output = None
            return self._status_locked()

    def fail(self, message: str) -> ReplayStatus:
        with self._lock:
            self._status_locked()
            self._interrupt_process_locked()
            self._outcome = ReplayOutcome.FAILED
            self._exit_code = 1
            self._last_output = message
            return self._status_locked()

    def close(self) -> None:
        self.stop()

    def _status_locked(self) -> ReplayStatus:
        if self._process is not None:
            exit_code = self._process.poll()
            if exit_code is not None:
                self._finish_process_locked(exit_code)
        return ReplayStatus(
            running=self._process is not None,
            file_name=self._source.file_name if self._source is not None else None,
            outcome=self._outcome,
            exit_code=self._exit_code,
            last_output=self._last_output,
        )

    def _interrupt_process_locked(self) -> bool:
        process = self._process
        if process is None:
            return False
        exit_code = process.poll()
        if exit_code is not None:
            self._finish_process_locked(exit_code)
            return False
        process.send_signal(signal.SIGINT)
        try:
            exit_code = process.wait(timeout=REPLAY_STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            exit_code = process.wait()
        self._process = None
        self._exit_code = exit_code
        self._close_stderr_locked()
        return True

    def _finish_process_locked(self, exit_code: int) -> None:
        self._process = None
        self._exit_code = exit_code
        diagnostic = self._read_stderr_locked()
        if exit_code == 0:
            self._outcome = ReplayOutcome.COMPLETED
            self._last_output = None
        else:
            self._outcome = ReplayOutcome.FAILED
            self._last_output = diagnostic or f"lcm-logplayer exited with code {exit_code}."

    def _record_start_failure_locked(self, message: str) -> None:
        self._outcome = ReplayOutcome.FAILED
        self._exit_code = 1
        self._last_output = message
        self._close_stderr_locked()

    def _read_stderr_locked(self) -> str | None:
        stderr = self._stderr
        self._stderr = None
        if stderr is None:
            return None
        try:
            stderr.flush()
            size = stderr.seek(0, os.SEEK_END)
            stderr.seek(max(0, size - REPLAY_STDERR_TAIL_BYTES))
            return stderr.read().decode(errors="replace").strip() or None
        finally:
            stderr.close()

    def _close_stderr_locked(self) -> None:
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
