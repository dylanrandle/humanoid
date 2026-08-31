"""Small rosbag2 recording catalog with Triskel compatibility metadata."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1
MANIFEST_NAME = "triskel.json"
BAG_DIRECTORY_NAME = "bag"


@dataclass(frozen=True)
class Recording:
    id: str
    robot: str
    runtime: str
    created_at: str
    directory: Path

    @property
    def bag_path(self) -> Path:
        return self.directory / BAG_DIRECTORY_NAME

    def public(self) -> dict[str, str]:
        return {
            "id": self.id,
            "robot": self.robot,
            "runtime": self.runtime,
            "created_at": self.created_at,
        }


class RecordingCatalog:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser()

    def create(self, *, robot: str, runtime: str) -> Recording:
        self.root.mkdir(parents=True, exist_ok=True)
        created_at = datetime.now(UTC)
        stem = f"triskel_{created_at.strftime('%Y%m%d_%H%M%S_%f')}"
        directory = self.root / stem
        suffix = 0
        while directory.exists():
            suffix += 1
            directory = self.root / f"{stem}_{suffix}"
        directory.mkdir()
        recording = Recording(
            id=directory.name,
            robot=robot,
            runtime=runtime,
            created_at=created_at.isoformat(),
            directory=directory,
        )
        manifest = asdict(recording)
        manifest.pop("directory")
        manifest["schema_version"] = SCHEMA_VERSION
        (directory / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return recording

    def list(self) -> Sequence[Recording]:
        if not self.root.is_dir():
            return []
        recordings = [recording for path in self.root.iterdir() if (recording := self._load(path))]
        return sorted(recordings, key=lambda item: item.created_at, reverse=True)

    def get(self, recording_id: str) -> Recording:
        if not recording_id or Path(recording_id).name != recording_id:
            raise ValueError("Select a valid recording.")
        recording = self._load(self.root / recording_id)
        if recording is None:
            raise ValueError("The selected recording is unavailable or incomplete.")
        return recording

    @staticmethod
    def _load(directory: Path) -> Recording | None:
        if (
            not directory.is_dir()
            or not (directory / BAG_DIRECTORY_NAME / "metadata.yaml").is_file()
        ):
            return None
        try:
            manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
            if manifest.get("schema_version") != SCHEMA_VERSION:
                return None
            if manifest.get("id") != directory.name:
                return None
            return Recording(
                id=directory.name,
                robot=str(manifest["robot"]),
                runtime=str(manifest["runtime"]),
                created_at=str(manifest["created_at"]),
                directory=directory,
            )
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return None
