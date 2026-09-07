"""Filesystem path helpers."""

import os
import sys
from pathlib import Path

HUMANOID_DATA_DIR_ENVIRONMENT_VARIABLE = "HUMANOID_DATA_DIR"


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the nearest ancestor containing a ``.git`` marker."""
    path = Path.cwd() if start is None else Path(start)
    path = path.resolve()
    if path.is_file():
        path = path.parent

    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate

    raise FileNotFoundError(f"Could not find a repository root from {path}.")


def find_data_root(start: str | Path | None = None) -> Path:
    """Find the persistent data root in development and installed environments."""
    configured_root = os.getenv(HUMANOID_DATA_DIR_ENVIRONMENT_VARIABLE)
    if configured_root:
        return Path(configured_root).expanduser().resolve()

    try:
        return find_repo_root(start)
    except FileNotFoundError:
        if sys.prefix != sys.base_prefix:
            return Path(sys.prefix).resolve().parent
        return Path.cwd().resolve()
