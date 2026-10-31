"""Shared collection streams round-trip through middleware and raw LCM logs."""

import json
from contextlib import closing
from io import BytesIO
from unittest.mock import patch

import lcm
import numpy as np
import pinocchio as pin
import pytest
from PIL import Image

from humanoid.config.collection import DEFAULT_COLLECTION_SETTINGS
from humanoid.constants import Topic
from humanoid.middleware.publisher import Publisher
from humanoid.middleware.subscriber import Subscriber
from humanoid.types.camera import CameraFrame
from humanoid.types.collection import (
    CollectionAction,
    CollectionEpisode,
    CollectionRequest,
    CollectionRunState,
    CollectionRunStatus,
    EpisodeOutcome,
)
from humanoid.types.logging import LoggingRequest
from humanoid.types.object_state import ObjectState
from humanoid.types.process import Runtime
from humanoid.types.simulation import SimulationReset, SimulationResetStatus
from humanoid.types.task_annotation import TaskAnnotation


@pytest.mark.parametrize(
    "topic,message",
    [
        (Topic.SIMULATION_RESET, SimulationReset(1.5, "scene", "cleanup", 24)),
        (Topic.SIMULATION_RESET_STATUS, SimulationResetStatus(1.5, "scene")),
        (Topic.SIMULATION_RESET_STATUS, SimulationResetStatus(1.5, "scene", "Not idle")),
        (Topic.LOGGING_REQUEST, LoggingRequest(1.5, True, "logs/selected")),
        (Topic.LOGGING_REQUEST, LoggingRequest(1.5, False)),
        (
            Topic.COLLECTION_REQUEST,
            CollectionRequest(1.5, "run", CollectionAction.START, DEFAULT_COLLECTION_SETTINGS),
        ),
        (Topic.COLLECTION_REQUEST, CollectionRequest(1.5, "run", CollectionAction.STOP)),
        (
            Topic.COLLECTION_STATUS,
            CollectionRunStatus(
                1.5,
                CollectionRunState.RUNNING,
                "run",
                "cleanup",
                1,
                3,
                attempts=2,
                seed=8,
                phase="pickup",
            ),
        ),
    ],
)
def test_operator_collection_messages_round_trip_through_transport(topic, message):
    with patch("lcm.LCM") as bus, patch.object(Subscriber, "_lcm_spin"):
        subscriber = Subscriber([topic])
        try:
            publisher = Publisher()
            bus.return_value.publish.side_effect = subscriber._handle_message
            publisher.publish(message, topic)
            assert subscriber.receive(topic) == message
        finally:
            subscriber.close()


@pytest.mark.parametrize("runtime", [Runtime.SIM, Runtime.REAL])
@pytest.mark.parametrize("task", ["cleanup", "stack_blocks"])
def test_shared_messages_record_and_replay_without_simulation_fields(tmp_path, runtime, task):
    clock = "episode_simulation" if runtime is Runtime.SIM else "host_monotonic"
    timestamp = 0.125 if runtime is Runtime.SIM else 734.125
    episode = CollectionEpisode(
        timestamp, "demo", task, 1, runtime, clock, EpisodeOutcome.STARTED, -1, -1, "{}"
    )
    rgb = BytesIO()
    Image.fromarray(np.full((2, 4, 3), 100, dtype=np.uint8)).save(rgb, format="JPEG")
    camera = CameraFrame(
        timestamp,
        "wrist",
        "wrist_optical",
        4,
        2,
        np.array([[400.0, 0, 1.5], [0, 401.0, 0.5], [0, 0, 1]]),
        rgb.getvalue(),
    )
    # A real pose estimator need not supply a bounding box or velocity estimate.
    item = ObjectState(
        timestamp,
        "block_0",
        "wrist_optical",
        pin.SE3(pin.exp3(np.array([0.1, 0.2, 0.3])), np.array([0.05, -0.1, 0.4])),
    )
    annotation = TaskAnnotation(timestamp, "demo", task, 1, '{"target":"block_0"}')
    messages = {
        Topic.COLLECTION_EPISODE: episode,
        Topic.CAMERA_FRAME: camera,
        Topic.OBJECT_STATE: item,
        Topic.TASK_ANNOTATION: annotation,
    }
    path = tmp_path / "recording.lcm"
    with patch("lcm.LCM") as bus, patch.object(Subscriber, "_lcm_spin"):
        publisher = Publisher()
        with closing(lcm.EventLog(str(path), "w")) as log:
            bus.return_value.publish.side_effect = lambda channel, data: log.write_event(
                123456789, channel, data
            )
            for topic, message in messages.items():
                publisher.publish(message, topic)
        subscriber = Subscriber(list(messages))
        try:
            with closing(lcm.EventLog(str(path), "r")) as log:
                while (event := log.read_next_event()) is not None:
                    subscriber._handle_message(event.channel, event.data)
            assert subscriber.receive(Topic.COLLECTION_EPISODE) == episode
            assert json.loads(episode.metadata_json) == {}  # No mandatory simulation seed.
            assert subscriber.receive(Topic.TASK_ANNOTATION) == annotation
            frame = subscriber.receive(Topic.CAMERA_FRAME)
            assert frame is not None
            assert frame.timestamp == timestamp  # Capture time, not the LCM event's time.
            assert frame.name == camera.name and frame.frame_id == camera.frame_id
            assert (frame.width, frame.height) == (camera.width, camera.height)
            assert frame.intrinsics == pytest.approx(camera.intrinsics)
            assert frame.rgb_jpeg == camera.rgb_jpeg
            state = subscriber.receive(Topic.OBJECT_STATE)
            assert state is not None
            assert state.timestamp == timestamp and state.name == item.name
            assert state.frame_id == "wrist_optical"
            assert state.pose.homogeneous == pytest.approx(item.pose.homogeneous)
            assert state.bbox_center is state.bbox_half_size is None
            assert state.linear_velocity is state.angular_velocity is None
        finally:
            subscriber.close()
