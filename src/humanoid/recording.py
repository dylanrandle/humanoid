"""Server-managed LCM recording bundles."""

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from itertools import count
from pathlib import Path

import numpy as np

from humanoid.logger import get_logger
from humanoid.types.replay import JsonValue, RecordingBundle, RecordingSummary
from humanoid.types.robot import RobotConfig, RobotName
from humanoid.utils.paths import find_repo_root

logger = get_logger(__name__)

DEFAULT_RECORDING_ROOT = find_repo_root(__file__) / "logs"
RECORDING_DIRECTORY_PREFIX = "recording_"
RECORDING_LOG_FILENAME = "recording.lcm"
RECORDING_MANIFEST_FILENAME = "robot.json"
RECORDING_SCHEMA_VERSION = 1


class RecordingError(Exception):
    """A recording bundle is missing, malformed, or incompatible."""


class RecordingCatalog:
    """Creates and discovers recording bundles below one trusted root."""

    def __init__(self, root: Path | str = DEFAULT_RECORDING_ROOT):
        self.root = Path(root)

    def create(self, robot_config: RobotConfig) -> RecordingBundle:
        self.root.mkdir(parents=True, exist_ok=True)
        created_at = datetime.now(UTC)
        base_id = f"{RECORDING_DIRECTORY_PREFIX}{created_at.strftime('%Y%m%d_%H%M%S_%f')}"
        for suffix in count():
            recording_id = base_id if suffix == 0 else f"{base_id}_{suffix}"
            directory = self.root / recording_id
            try:
                directory.mkdir()
                break
            except FileExistsError:
                continue

        manifest_path = directory / RECORDING_MANIFEST_FILENAME
        manifest = {
            "schema_version": RECORDING_SCHEMA_VERSION,
            "recording_id": recording_id,
            "created_at": created_at.isoformat(),
            "robot": RobotName(robot_config.name).value,
            "robot_config": serialize_robot_config(robot_config),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        return self._load(recording_id, require_log=False)

    def list(self) -> list[RecordingSummary]:
        if not self.root.is_dir():
            return []
        try:
            directories = list(self.root.iterdir())
        except OSError as exc:
            logger.warning("Could not inspect recording directory %s: %s", self.root, exc)
            return []
        recordings: list[RecordingSummary] = []
        for directory in directories:
            if not directory.is_dir() or not directory.name.startswith(RECORDING_DIRECTORY_PREFIX):
                continue
            if not (directory / RECORDING_LOG_FILENAME).is_file():
                continue
            try:
                bundle = self._load(directory.name)
            except RecordingError as exc:
                logger.warning("Ignoring invalid recording bundle %s: %s", directory, exc)
                continue
            recordings.append(
                RecordingSummary(
                    id=bundle.id,
                    robot=bundle.robot,
                    created_at=bundle.created_at,
                )
            )
        return sorted(recordings, key=lambda recording: recording.created_at, reverse=True)

    def get(self, recording_id: str) -> RecordingBundle:
        return self._load(recording_id)

    def _load(self, recording_id: str, *, require_log: bool = True) -> RecordingBundle:
        if not recording_id or Path(recording_id).name != recording_id:
            raise RecordingError("Select a valid recording.")
        directory = self.root / recording_id
        if not directory.is_dir():
            raise RecordingError("The selected recording is unavailable.")

        manifest_path = directory / RECORDING_MANIFEST_FILENAME
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RecordingError("The selected recording has no readable robot config.") from exc
        if not isinstance(manifest, dict):
            raise RecordingError("The selected recording has an invalid robot config.")
        if manifest.get("schema_version") != RECORDING_SCHEMA_VERSION:
            raise RecordingError("The selected recording uses an unsupported config format.")
        if manifest.get("recording_id") != recording_id:
            raise RecordingError("The selected recording config does not match its directory.")

        try:
            robot = RobotName(str(manifest["robot"]))
            created_at = str(manifest["created_at"])
            robot_config = manifest["robot_config"]
        except (KeyError, ValueError) as exc:
            raise RecordingError("The selected recording has an invalid robot config.") from exc
        if not isinstance(robot_config, dict):
            raise RecordingError("The selected recording has an invalid robot config.")

        log_path = directory / RECORDING_LOG_FILENAME
        if require_log and not log_path.is_file():
            raise RecordingError("The selected recording has no LCM log.")
        return RecordingBundle(
            id=recording_id,
            directory=directory,
            log_path=log_path,
            manifest_path=manifest_path,
            robot=robot,
            created_at=created_at,
            robot_config=robot_config,
        )


def serialize_robot_config(robot_config: RobotConfig) -> dict[str, JsonValue]:
    """Convert a RobotConfig into a stable, JSON-compatible snapshot."""
    value = _json_value(robot_config)
    if not isinstance(value, dict):
        raise TypeError("Robot configuration did not serialize to an object.")
    return value


def _json_value(value: object) -> JsonValue:
    if isinstance(value, Enum):
        result = _json_value(value.value)
    elif value is None or isinstance(value, bool | int | float | str):
        result = value
    elif isinstance(value, np.ndarray):
        # NumPy's typing cannot express that tolist() recursively returns JSON values.
        result = _json_value(value.tolist())  # ty: ignore[no-matching-overload]
    elif isinstance(value, np.integer | np.floating):
        result = value.item()
    elif is_dataclass(value) and not isinstance(value, type):
        result = {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    elif isinstance(value, Mapping):
        result = {str(key): _json_value(item) for key, item in value.items()}
    elif isinstance(value, list | tuple):
        result = [_json_value(item) for item in value]
    else:
        raise TypeError(f"Unsupported robot configuration value: {type(value).__name__}")
    return result
