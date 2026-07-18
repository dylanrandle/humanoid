"""Filesystem path helpers."""

from pathlib import Path


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
