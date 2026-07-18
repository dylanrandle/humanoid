"""Robot logger node that controls the lcm-logger subprocess.

Listens for START_LOGGING and STOP_LOGGING events and launches/terminates
lcm-logger to record LCM network traffic.
"""

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

from humanoid.config import ROBOT_CONFIG
from humanoid.constants import DEFAULT_LCM_URL, Topic
from humanoid.logger import get_logger
from humanoid.middleware.publisher import Publisher
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.base import Node
from humanoid.recording import DEFAULT_RECORDING_ROOT, RecordingCatalog
from humanoid.types.logging import LoggingState, LoggingStatus
from humanoid.types.orchestrator import EventKind
from humanoid.types.robot import RobotConfig

logger = get_logger(__name__)

DEFAULT_RATE_HZ = 10.0
SUBPROCESS_WAIT_TIMEOUT_SEC = 2.0
EXIT_SUCCESS = 0
LCM_LOGGER_COMMAND = "lcm-logger"
ALL_CHANNELS_PATTERN = ".*"


class RobotLoggerNode(Node):
    """Node that launches lcm-logger in a subprocess upon receiving START_LOGGING."""

    rate_hz: float = DEFAULT_RATE_HZ

    def __init__(
        self,
        log_dir: str | Path = DEFAULT_RECORDING_ROOT,
        channel_regex: str = ALL_CHANNELS_PATTERN,
        lcm_url: str = DEFAULT_LCM_URL,
        rate_hz: float = DEFAULT_RATE_HZ,
        robot_config: RobotConfig = ROBOT_CONFIG,
    ):
        """Initialize the robot logger."""
        self.rate_hz = rate_hz
        self.log_dir = log_dir
        self.channel_regex = channel_regex
        self.lcm_url = lcm_url
        self.robot_config = robot_config
        self.recording_catalog = RecordingCatalog(log_dir)

        self.subscriber = Subscriber(topics=[Topic.ORCHESTRATOR_EVENT], url=lcm_url)
        self.publisher = Publisher(url=lcm_url)
        self.process: subprocess.Popen[bytes] | None = None
        self.logging_status = LoggingStatus(
            timestamp=time.time(),
            state=LoggingState.STOPPED,
        )

    def setup(self) -> None:
        logger.info(
            f"RobotLoggerNode started. Monitoring {Topic.ORCHESTRATOR_EVENT.value} "
            f"at {self.rate_hz} Hz"
        )
        self._publish_status(LoggingState.STOPPED)

    def step(self) -> None:
        """Check for START_LOGGING and STOP_LOGGING events."""
        self._report_unexpected_exit()
        while True:
            event = self.subscriber.receive(Topic.ORCHESTRATOR_EVENT)
            if event is None:
                break

            kind = event.kind
            if kind is EventKind.START_LOGGING:
                self._start_logging()
            elif kind is EventKind.STOP_LOGGING:
                self._stop_logging()

    def _start_logging(self) -> None:
        """Spawn the lcm-logger subprocess."""
        if self.process is not None:
            if self.process.poll() is None:
                logger.warning("lcm-logger is already running.")
                return
            logger.warning("Previous lcm-logger process exited; starting a new one.")
            self.process = None

        self._publish_status(LoggingState.STARTING)

        try:
            recording = self.recording_catalog.create(self.robot_config)
        except (OSError, TypeError, ValueError) as exc:
            error = f"Could not prepare the recording directory: {exc}"
            logger.error(error)
            self._publish_status(LoggingState.FAILED, error=error)
            return
        filename = str(recording.log_path)

        cmd = [LCM_LOGGER_COMMAND]
        if self.lcm_url:
            cmd.extend(["-l", self.lcm_url])
        if self.channel_regex:
            cmd.extend(["-c", self.channel_regex])
        cmd.append(filename)

        logger.info(f"Starting logging: Running command {' '.join(cmd)}")
        try:
            self.process = subprocess.Popen(cmd)
            exit_code = self.process.poll()
            if exit_code is not None:
                self.process = None
                self._publish_status(
                    LoggingState.FAILED,
                    file_name=filename,
                    error=f"lcm-logger exited during startup with code {exit_code}.",
                )
                return
            logger.info("lcm-logger subprocess spawned successfully.")
            self._publish_status(LoggingState.RUNNING, file_name=filename)
        except FileNotFoundError:
            error = "lcm-logger was not found in PATH."
            logger.error(error)
            self._publish_status(LoggingState.FAILED, error=error)
        except Exception as exc:
            error = f"Failed to spawn lcm-logger: {exc}"
            logger.error(error)
            self._publish_status(LoggingState.FAILED, error=error)

    def _stop_logging(self) -> None:
        """Terminate the lcm-logger subprocess cleanly."""
        if self.process is None:
            logger.warning("No active logging process to stop.")
            self._publish_status(LoggingState.STOPPED)
            return

        logger.info("Stopping logging: Sending SIGINT to lcm-logger...")
        self._publish_status(
            LoggingState.STOPPING,
            file_name=self.logging_status.file_name,
        )
        file_name = self.logging_status.file_name
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
            self._publish_status(
                LoggingState.FAILED,
                file_name=file_name,
                error=error,
            )
        else:
            self._publish_status(LoggingState.STOPPED, file_name=file_name)

    def _report_unexpected_exit(self) -> None:
        if self.process is None:
            return
        exit_code = self.process.poll()
        if exit_code is None:
            return
        file_name = self.logging_status.file_name
        self.process = None
        self._publish_status(
            LoggingState.FAILED,
            file_name=file_name,
            error=f"lcm-logger exited unexpectedly with code {exit_code}.",
        )

    def _publish_status(
        self,
        state: LoggingState,
        *,
        file_name: str | None = None,
        error: str | None = None,
    ) -> None:
        self.logging_status = LoggingStatus(
            timestamp=time.time(),
            state=state,
            file_name=file_name,
            error=error,
        )
        self.publisher.publish(self.logging_status, Topic.LOGGING_STATUS)

    def on_close(self) -> None:
        """Cleanup subscriber and active subprocess."""
        self._stop_logging()
        self.subscriber.close()


def main():
    parser = argparse.ArgumentParser(description="Lifecycle wrapper node for lcm-logger")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(DEFAULT_RECORDING_ROOT),
        help="Directory to save the LCM logs in",
    )
    parser.add_argument(
        "--channel-regex",
        type=str,
        default=ALL_CHANNELS_PATTERN,
        help="Channel regex string to log (default: .* for all channels)",
    )
    parser.add_argument(
        "--lcm-url",
        type=str,
        default=DEFAULT_LCM_URL,
        help="LCM URL to bind to",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_RATE_HZ,
        help="Check loop rate in Hz",
    )
    args = parser.parse_args()

    try:
        RobotLoggerNode(
            log_dir=args.log_dir,
            channel_regex=args.channel_regex,
            lcm_url=args.lcm_url,
            rate_hz=args.rate,
        ).run()
    except KeyboardInterrupt:
        logger.info("RobotLoggerNode interrupted by keyboard")
    sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    main()
