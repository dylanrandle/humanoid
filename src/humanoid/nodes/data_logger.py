"""Data Logger Node that controls the lcm-logger subprocess.

Listens for START_LOGGING and STOP_LOGGING events and launches/terminates
lcm-logger to record LCM network traffic.
"""

import argparse
import datetime
import signal
import subprocess
import sys
from pathlib import Path

from humanoid.constants import DEFAULT_LCM_URL, Topic
from humanoid.logger import get_logger
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.base import Node
from humanoid.types.orchestrator import EventKind

logger = get_logger(__name__)

DEFAULT_RATE_HZ = 10.0
SUBPROCESS_WAIT_TIMEOUT_SEC = 2.0
EXIT_SUCCESS = 0


class DataLoggerNode(Node):
    """Node that launches lcm-logger in a subprocess upon receiving START_LOGGING."""

    rate_hz: float = DEFAULT_RATE_HZ

    def __init__(
        self,
        log_dir: str = "logs",
        channel_regex: str = ".*",
        lcm_url: str = DEFAULT_LCM_URL,
        rate_hz: float = DEFAULT_RATE_HZ,
    ):
        """Initialize the data logger node."""
        self.rate_hz = rate_hz
        self.log_dir = log_dir
        self.channel_regex = channel_regex
        self.lcm_url = lcm_url

        self.subscriber = Subscriber(topics=[Topic.ORCHESTRATOR_EVENT])
        self.process: subprocess.Popen | None = None

    def setup(self) -> None:
        logger.info(
            f"DataLoggerNode started. Monitoring {Topic.ORCHESTRATOR_EVENT.value} "
            f"at {self.rate_hz} Hz"
        )

    def step(self) -> None:
        """Check for START_LOGGING and STOP_LOGGING events."""
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
            logger.warning("lcm-logger is already running.")
            return

        # Ensure the log directory exists
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # Generate a timestamped output filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = str(Path(self.log_dir) / f"lcmlog_{timestamp}")

        # Build lcm-logger command
        cmd = ["lcm-logger", "-i"]
        if self.lcm_url:
            cmd.extend(["-l", self.lcm_url])
        if self.channel_regex:
            cmd.extend(["-c", self.channel_regex])
        cmd.append(filename)

        logger.info(f"Starting logging: Running command {' '.join(cmd)}")
        try:
            self.process = subprocess.Popen(cmd)
            logger.info("lcm-logger subprocess spawned successfully.")
        except FileNotFoundError:
            logger.error("lcm-logger utility not found in PATH.")
        except Exception as e:
            logger.error(f"Failed to spawn lcm-logger: {e}")

    def _stop_logging(self) -> None:
        """Terminate the lcm-logger subprocess cleanly."""
        if self.process is None:
            logger.warning("No active logging process to stop.")
            return

        logger.info("Stopping logging: Sending SIGINT to lcm-logger...")
        try:
            # Send SIGINT so lcm-logger flushes buffers and exits cleanly
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=SUBPROCESS_WAIT_TIMEOUT_SEC)
            logger.info("lcm-logger subprocess terminated cleanly.")
        except subprocess.TimeoutExpired:
            logger.warning("lcm-logger did not exit cleanly. Killing subprocess...")
            self.process.kill()
            self.process.wait()
            logger.info("lcm-logger subprocess killed.")
        except Exception as e:
            logger.error(f"Error stopping logging process: {e}")
        finally:
            self.process = None

    def on_close(self) -> None:
        """Cleanup subscriber and active subprocess."""
        self._stop_logging()
        self.subscriber.close()


def main():
    parser = argparse.ArgumentParser(description="Lifecycle wrapper node for lcm-logger")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save the LCM logs in",
    )
    parser.add_argument(
        "--channel-regex",
        type=str,
        default=".*",
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
        DataLoggerNode(
            log_dir=args.log_dir,
            channel_regex=args.channel_regex,
            lcm_url=args.lcm_url,
            rate_hz=args.rate,
        ).run()
    except KeyboardInterrupt:
        logger.info("DataLoggerNode interrupted by keyboard")
    sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    main()
