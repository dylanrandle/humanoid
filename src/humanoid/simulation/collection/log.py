"""Audit raw LCM logs before admitting successful episodes to training data."""

import json
import struct
from contextlib import closing
from io import BytesIO
from pathlib import Path

import lcm
from PIL import Image

from humanoid.config.collection import COLLECTION_SCHEMA_VERSION
from humanoid.constants import Topic
from humanoid.types.collection import CollectionConfig, EpisodeOutcome
from humanoid.types.lcm import (
    collection_episode_t,
    robot_joint_command_t,
    robot_state_t,
    simulation_sample_t,
)
from humanoid.types.process import Runtime

CLOCK_TOLERANCE_NS = 2


def contains_episode_event(
    path: Path, episode_id: str, outcome: EpisodeOutcome, offset: int = 0
) -> bool:
    """Read a flushed boundary, optionally scanning only a recent portion of the log."""
    if not path.exists() or not path.stat().st_size:
        return False
    try:
        with closing(lcm.EventLog(str(path), "r")) as log:
            log.seek(offset)
            while (event := log.read_next_event()) is not None:
                if event.channel == Topic.COLLECTION_EPISODE.value:
                    message = collection_episode_t.decode(event.data)
                    if message.episode_id == episode_id and message.outcome == outcome.value:
                        return True
    except (OSError, ValueError, struct.error):
        return False
    return False


class _LogAudit:
    """Streaming audit state; image payloads are never accumulated in memory."""

    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        self.errors: set[str] = set()
        self.started: collection_episode_t | None = None
        self.terminal: collection_episode_t | None = None
        self.samples = self.states = self.commands = 0
        self.last_sample_time = -1
        self.last_control_time: dict[str, int] = {}
        self.validate_observations = False
        self.camera_names: list[str] | None = None

    def consume(self, channel: str, data: bytes) -> None:
        if channel == Topic.COLLECTION_EPISODE.value:
            self.boundary(collection_episode_t.decode(data))
        elif channel in (Topic.ROBOT_STATE.value, Topic.ROBOT_JOINT_COMMAND.value):
            self.control(channel, data)
        elif channel == Topic.SIMULATION_SAMPLE.value:
            self.sample(simulation_sample_t.decode(data))

    def boundary(self, message: collection_episode_t) -> None:
        if message.episode_id != self.episode_id:
            self.errors.add("Mixed episode identities.")
        metadata = json.loads(message.metadata_json)
        if message.runtime != Runtime.SIM.value or message.clock != "host_monotonic":
            self.errors.add("Simulation audit requires a simulation episode and clock.")
        if metadata["schema_version"] != COLLECTION_SCHEMA_VERSION:
            self.errors.add("Unsupported collection schema version.")
        if message.outcome == EpisodeOutcome.STARTED.value:
            if self.started is not None:
                self.errors.add("Duplicate episode start.")
            CollectionConfig(**metadata["collection"])
            self.started = message
            names = metadata.get("camera_names")
            if names is not None:
                if (
                    not isinstance(names, list)
                    or not names
                    or any(not isinstance(name, str) or not name for name in names)
                    or len(set(names)) != len(names)
                ):
                    self.errors.add("Invalid declared camera identities.")
                else:
                    self.camera_names = names
            self.validate_observations = (
                metadata.get("observation_validation") == "task_checks_per_sample"
            )
        elif message.outcome in (
            EpisodeOutcome.SUCCEEDED,
            EpisodeOutcome.FAILED,
            EpisodeOutcome.ABORTED,
        ):
            if self.terminal is not None:
                self.errors.add("Duplicate episode end.")
            self.terminal = message
        elif message.outcome != EpisodeOutcome.READY.value:
            self.errors.add("Unknown episode outcome.")

    def control(self, channel: str, data: bytes) -> None:
        # The ordinary stack keeps publishing before and after an episode.
        if self.started is None or self.terminal is not None:
            return
        is_state = channel == Topic.ROBOT_STATE.value
        wire = robot_state_t.decode(data) if is_state else robot_joint_command_t.decode(data)
        if wire.timestamp <= self.last_control_time.get(channel, -1):
            self.errors.add("Non-monotonic robot stream.")
        self.last_control_time[channel] = wire.timestamp
        if is_state:
            self.states += 1
        else:
            self.commands += 1

    def sample(self, sample: simulation_sample_t) -> None:
        if self.started is None or self.terminal is not None:
            self.errors.add("Sample outside episode boundaries.")
        if sample.episode_id != self.episode_id or sample.sequence != self.samples:
            self.errors.add("Missing, duplicated, or mixed samples.")
        if sample.action_index != -1:
            self.errors.add("Asynchronous samples must not claim a synchronized action index.")
        if (
            sample.timestamp != sample.state.timestamp
            or sample.command.timestamp > sample.timestamp
        ):
            self.errors.add("Sample clock mismatch.")
        if sample.timestamp <= self.last_sample_time or (
            self.started is not None and sample.timestamp < self.started.timestamp
        ):
            self.errors.add("Non-monotonic or stale sample.")
        self.last_sample_time = sample.timestamp
        self.observations(sample)
        self.samples += 1

    def observations(self, sample: simulation_sample_t) -> None:
        if not sample.num_cameras or any(not c.frame.rgb_size for c in sample.cameras):
            self.errors.add("Missing RGB observations.")
        names = [observation.frame.name for observation in sample.cameras]
        if len(set(names)) != len(names):
            self.errors.add("Duplicate camera observations.")
        if self.camera_names is not None and set(names) != set(self.camera_names):
            self.errors.add("Camera observations do not match the declared camera set.")
        for observation in sample.cameras:
            camera = observation.frame
            if camera.timestamp != sample.timestamp:
                self.errors.add("Camera capture clock mismatch.")
            if not camera.name or not camera.frame_id:
                self.errors.add("Missing camera identity or optical frame.")
            with Image.open(BytesIO(camera.rgb_jpeg)) as image:
                if (
                    image.size != (camera.width, camera.height)
                    or image.format != "JPEG"
                    or image.mode != "RGB"
                ):
                    self.errors.add("Camera payload must be RGB JPEG with matching dimensions.")
                image.load()
        annotation = sample.task_annotation
        if (
            annotation.episode_id != sample.episode_id
            or annotation.timestamp != sample.timestamp
            or (
                self.started is not None
                and (annotation.task, annotation.task_version)
                != (self.started.task, self.started.task_version)
            )
        ):
            self.errors.add("Task annotation does not match its episode/sample.")
        labels = json.loads(annotation.labels_json)
        if not isinstance(labels, dict):
            self.errors.add("Task annotation labels must be a JSON object.")
        for item in sample.objects:
            if (
                not item.has_bbox
                or not item.has_velocity
                or item.frame_id != "world"
                or abs(item.timestamp - sample.timestamp) > CLOCK_TOLERANCE_NS
            ):
                self.errors.add("Incomplete or misaligned simulation object state.")
        if self.validate_observations and (
            not isinstance(labels, dict)
            or labels.get("observation_valid") is not True
            or labels.get("observation_errors") != []
        ):
            self.errors.add("Task observation requirements failed or are missing from a sample.")

    def finish(self) -> tuple[str, ...]:
        if self.started is None or self.terminal is None:
            self.errors.add("Missing episode boundary.")
        if self.terminal is not None:
            if not self.states or (self.terminal.action_count > 0 and not self.commands):
                self.errors.add("Missing robot state or command stream.")
            if not self.samples or self.samples != self.terminal.sample_count:
                self.errors.add("Sample count does not match the terminal event.")
            if self.terminal.timestamp < self.last_sample_time:
                self.errors.add("Terminal clock precedes the last observation.")
            if self.started is not None and (
                self.started.task,
                self.started.task_version,
                json.loads(self.started.metadata_json).get("seed"),
            ) != (
                self.terminal.task,
                self.terminal.task_version,
                json.loads(self.terminal.metadata_json).get("seed"),
            ):
                self.errors.add("Episode task identity changed.")
            if self.started is not None and json.loads(self.started.metadata_json).get(
                "camera_names"
            ) != json.loads(self.terminal.metadata_json).get("camera_names"):
                self.errors.add("Episode camera identities changed.")
        return tuple(sorted(self.errors))


def validate_collection_log(path: Path, episode_id: str) -> tuple[str, ...]:
    """Require boundaries, continuous sample sequences, and valid observation clocks.

    LCM delivery is not assumed reliable. Incomplete logs remain available for
    diagnosis and must never be selected as successful training episodes.
    """
    audit = _LogAudit(episode_id)
    try:
        with closing(lcm.EventLog(str(path), "r")) as log:
            while (event := log.read_next_event()) is not None:
                audit.consume(event.channel, event.data)
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError, struct.error) as error:
        audit.errors.add(f"Unreadable or malformed log: {error}")
    return audit.finish()
