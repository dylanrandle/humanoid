"""API-owned LCM recording lifecycle, ownership and clean finalization."""

import signal
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from humanoid.config import ROBOT_CONFIG
from humanoid.config.recording import (
    ALL_CHANNELS_PATTERN,
    LCM_LOGGER_COMMAND,
    RECORDING_POLL_SECONDS,
    SUBPROCESS_WAIT_TIMEOUT_SEC,
)
from humanoid.constants import DEFAULT_LCM_URL
from humanoid.logger import get_logger
from humanoid.recording import DEFAULT_RECORDING_ROOT, RecordingCatalog
from humanoid.types.api import ServiceConflict
from humanoid.types.recording import RecordingOwner, RecordingState, RecordingStatus
from humanoid.types.robot import RobotConfig

logger = get_logger(__name__)


class RecordingService:
    def __init__(
        self,
        log_dir: str | Path = DEFAULT_RECORDING_ROOT,
        channel_regex: str = ALL_CHANNELS_PATTERN,
        lcm_url: str = DEFAULT_LCM_URL,
        robot_config: RobotConfig = ROBOT_CONFIG,
    ) -> None:
        self.log_dir = log_dir
        self.channel_regex = channel_regex
        self.lcm_url = lcm_url
        self.robot_config = robot_config
        self.recording_catalog = RecordingCatalog(log_dir)
        self.process: subprocess.Popen[bytes] | None = None
        self.recording_status = RecordingStatus(timestamp=time.time(), state=RecordingState.STOPPED)
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._worker: threading.Thread | None = None
        self._owner: RecordingOwner | None = None
        self._finished: dict[RecordingOwner, RecordingStatus] = {}

    def start(self, *, owner: RecordingOwner | None = None) -> RecordingStatus:
        with self._lock:
            if self._closed.is_set():
                raise ServiceConflict("Recording service is closed.")
            self._report_unexpected_exit()
            if owner is not None and owner in self._finished:
                raise ServiceConflict("This episode's recording has already finished.")
            if self._owner != owner and (self._owner is not None or self.process is not None):
                raise ServiceConflict("The recorder is already in use by another owner.")
            if self.process is not None:
                return self.recording_status
            if self._owner is not None:
                raise ServiceConflict(
                    self.recording_status.error or "Episode recording was interrupted."
                )
            self._start_recording()
            if self.recording_status.state is RecordingState.RUNNING:
                self._owner = owner
                if self._worker is None:
                    self._worker = threading.Thread(target=self._monitor, daemon=True)
                    self._worker.start()
            return self.recording_status

    def stop(self, *, owner: RecordingOwner | None = None) -> RecordingStatus:
        with self._lock:
            if owner is not None and owner in self._finished:
                return self._finished[owner]
            if self._owner != owner:
                raise ServiceConflict("This caller does not own the current recording.")
            self._report_unexpected_exit()
            if self.process is not None:
                self._stop_recording()
            if owner is not None:
                self._finished[owner] = self.recording_status
            self._owner = None
            return self.recording_status

    def snapshot(self) -> RecordingStatus:
        with self._lock:
            self._report_unexpected_exit()
            return self.recording_status

    def configure(self, robot_config: RobotConfig) -> None:
        with self._lock:
            if robot_config is self.robot_config:
                return
            if self.process is not None or self._owner is not None:
                raise ServiceConflict("Stop recording before changing robot configuration.")
            self.robot_config = robot_config

    def finalize(self) -> RecordingStatus:
        """Backend shutdown finalizes even an episode-owned recording."""
        with self._lock:
            return self.stop(owner=self._owner)

    def reconcile_runtime(self, runtime_id: str) -> None:
        """Finalize an owned episode when its robot runtime has ended."""
        with self._lock:
            if self._owner is not None and self._owner.runtime_id != runtime_id:
                self.stop(owner=self._owner)

    def _monitor(self) -> None:
        while not self._closed.wait(RECORDING_POLL_SECONDS):
            self.snapshot()

    def close(self) -> None:
        self._closed.set()
        self.finalize()
        if self._worker is not None:
            self._worker.join(timeout=SUBPROCESS_WAIT_TIMEOUT_SEC)

    @contextmanager
    def recording(self) -> Iterator[str]:
        started = self.start()
        if started.state is not RecordingState.RUNNING or not started.file_name:
            raise ServiceConflict(started.error or "The recorder did not start.")
        try:
            yield started.file_name
        finally:
            stopped = self.stop()
            if stopped.state is not RecordingState.STOPPED:
                raise ServiceConflict(stopped.error or "The recording did not finalize.")

    def _start_recording(self) -> None:
        """Spawn the lcm-logger subprocess."""
        if self.process is not None:
            if self.process.poll() is None:
                logger.warning("lcm-logger is already running.")
                return
            logger.warning("Previous lcm-logger process exited; starting a new one.")
            self.process = None

        self._set_status(RecordingState.STARTING)

        try:
            recording = self.recording_catalog.create(self.robot_config)
        except (OSError, TypeError, ValueError) as exc:
            error = f"Could not prepare the recording directory: {exc}"
            logger.error(error)
            self._set_status(RecordingState.FAILED, error=error)
            return
        filename = str(recording.log_path)

        cmd = [LCM_LOGGER_COMMAND]
        if self.lcm_url:
            cmd.extend(["-l", self.lcm_url])
        if self.channel_regex:
            cmd.extend(["-c", self.channel_regex])
        cmd.append(filename)

        logger.info(f"Starting recording: Running command {' '.join(cmd)}")
        try:
            self.process = subprocess.Popen(cmd)
            exit_code = self.process.poll()
            if exit_code is not None:
                self.process = None
                self._set_status(
                    RecordingState.FAILED,
                    file_name=filename,
                    error=f"lcm-logger exited during startup with code {exit_code}.",
                )
                return
            logger.info("lcm-logger subprocess spawned successfully.")
            self._set_status(RecordingState.RUNNING, file_name=filename)
        except FileNotFoundError:
            error = f"LCM recorder not found at {LCM_LOGGER_COMMAND}."
            logger.error(error)
            self._set_status(RecordingState.FAILED, error=error)
        except Exception as exc:
            error = f"Failed to spawn lcm-logger: {exc}"
            logger.error(error)
            self._set_status(RecordingState.FAILED, error=error)

    def _stop_recording(self) -> None:
        """Terminate the lcm-logger subprocess cleanly."""
        if self.process is None:
            logger.warning("No active recording process to stop.")
            self._set_status(RecordingState.STOPPED)
            return

        logger.info("Stopping recording: Sending SIGINT to lcm-logger...")
        self._set_status(
            RecordingState.STOPPING,
            file_name=self.recording_status.file_name,
        )
        file_name = self.recording_status.file_name
        error: str | None = None
        try:
            # Send SIGINT so lcm-logger flushes buffers and exits cleanly
            self.process.send_signal(signal.SIGINT)
            exit_code = self.process.wait(timeout=SUBPROCESS_WAIT_TIMEOUT_SEC)
            if exit_code == 0:
                logger.info("lcm-logger subprocess terminated cleanly.")
            else:
                error = f"lcm-logger exited with code {exit_code} while stopping."
        except subprocess.TimeoutExpired:
            logger.warning("lcm-logger did not exit cleanly. Killing subprocess...")
            self.process.kill()
            self.process.wait()
            error = "lcm-logger did not stop cleanly and was killed; the log may be incomplete."
        except Exception as exc:
            error = f"Error stopping lcm-logger: {exc}"
        finally:
            self.process = None
        if error is not None:
            logger.error(error)
            self._set_status(
                RecordingState.FAILED,
                file_name=file_name,
                error=error,
            )
        else:
            self._set_status(RecordingState.STOPPED, file_name=file_name)

    def _report_unexpected_exit(self) -> None:
        if self.process is None:
            return
        exit_code = self.process.poll()
        if exit_code is None:
            return
        file_name = self.recording_status.file_name
        self.process = None
        self._set_status(
            RecordingState.FAILED,
            file_name=file_name,
            error=f"lcm-logger exited unexpectedly with code {exit_code}.",
        )

    def _set_status(
        self,
        state: RecordingState,
        *,
        file_name: str | None = None,
        error: str | None = None,
    ) -> None:
        self.recording_status = RecordingStatus(
            timestamp=time.time(),
            state=state,
            file_name=file_name,
            error=error,
        )
