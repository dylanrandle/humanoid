"""Recorder process defaults."""

from pathlib import Path

import lcm

LCM_LOGGER_COMMAND = str((Path(lcm.LCM_BIN_DIR) / "lcm-logger").resolve())
ALL_CHANNELS_PATTERN = ".*"
RECORDING_POLL_SECONDS = 0.1
SUBPROCESS_WAIT_TIMEOUT_SEC = 2.0
RECORDING_FEEDBACK_POLL_SECONDS = 0.5
