"""Log configuration for the operator console."""

from pathlib import Path

from humanoid.utils.paths import find_data_root

LOG_FILE_NAME = "humanoid.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 2
LOG_RETAINED_ENTRIES = 200
LOG_INITIAL_TAIL_BYTES = 64 * 1024


def get_log_path() -> Path:
    """Return the persistent log file shared by the console and node processes."""
    return find_data_root(__file__) / "logs" / LOG_FILE_NAME
