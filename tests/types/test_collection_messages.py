"""Wire round-trips and corruption/missing-event rejection for collection data."""

import json
from contextlib import closing
from dataclasses import replace
from io import BytesIO

import lcm
import numpy as np
import pinocchio as pin
import pytest
from PIL import Image

from humanoid.config.collection import COLLECTION_SCHEMA_VERSION
from humanoid.constants import Topic
from humanoid.simulation.collection.log import validate_collection_log
from humanoid.types.camera import CameraFrame
from humanoid.types.collection import (
    CollectionConfig,
    CollectionEpisode,
    CollectionReport,
    EpisodeOutcome,
    SimulationCameraObservation,
    SimulationSample,
)
from humanoid.types.lcm import collection_episode_t, simulation_sample_t
from humanoid.types.lcm.collection import (
    episode_from_lcm,
    episode_to_lcm,
    sample_from_lcm,
    sample_to_lcm,
)
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.object_state import ObjectState
from humanoid.types.process import Runtime
from humanoid.types.robot import RobotJointCommand, RobotState
from humanoid.types.task_annotation import TaskAnnotation


def camera_frame(timestamp=0.0):
    rgb = BytesIO()
    Image.fromarray(np.zeros((2, 4, 3), dtype=np.uint8)).save(rgb, format="JPEG")
    return SimulationCameraObservation(
        CameraFrame(timestamp, "wrist", "wrist_optical", 4, 2, np.eye(3), rgb.getvalue()),
        pin.SE3.Identity(),
    )


def sample(index=0, sequence=0):
    timestamp = index * 0.02
    state = RobotState(timestamp, np.arange(3, dtype=float), np.zeros(3), np.full(3, 25.0))
    command = RobotJointCommand(timestamp, np.ones(3), np.zeros(3))
    item = ObjectState(
        timestamp,
        "toy",
        "world",
        pin.SE3.Identity(),
        np.zeros(3),
        np.array([0.01, 0.02, 0.03]),
        np.zeros(3),
        np.zeros(3),
    )
    return SimulationSample(
        timestamp,
        "episode",
        sequence,
        -1,
        state,
        command,
        (camera_frame(timestamp),),
        (item,),
        TaskAnnotation(timestamp, "episode", "cleanup", 1, '{"phase":"grasp"}'),
    )


def boundary(outcome):
    finished = outcome != EpisodeOutcome.STARTED
    return CollectionEpisode(
        0.12 if finished else 0.0,
        "episode",
        "cleanup",
        1,
        Runtime.SIM,
        "host_monotonic",
        outcome,
        2 if finished else 0,
        6 if finished else 0,
        json.dumps(
            {
                "schema_version": COLLECTION_SCHEMA_VERSION,
                "seed": 7,
                "observation_modalities": ["rgb"],
                "camera_names": ["wrist"],
                "collection": {"control_hz": 50, "sample_hz": 10},
            }
        ),
    )


@pytest.mark.parametrize("action_index", [0, 205, 410, 820, 845, 1715, 3430])
def test_collection_wire_preserves_images_obb_and_action_alignment(action_index):
    original = sample(action_index)
    decoded = sample_from_lcm(simulation_sample_t.decode(sample_to_lcm(original).encode()))
    assert decoded.sequence == original.sequence
    assert decoded.timestamp == pytest.approx(original.timestamp, abs=1e-9)
    assert decoded.timestamp == decoded.state.timestamp == decoded.command.timestamp
    assert decoded.state.joint_positions == pytest.approx(original.state.joint_positions)
    assert decoded.command.joint_velocities == pytest.approx(original.command.joint_velocities)
    assert decoded.cameras[0].frame.rgb_jpeg == original.cameras[0].frame.rgb_jpeg
    assert decoded.cameras[0].frame.intrinsics == pytest.approx(
        original.cameras[0].frame.intrinsics
    )
    assert decoded.cameras[0].frame.timestamp == decoded.timestamp
    assert decoded.cameras[0].frame.frame_id == "wrist_optical"
    assert decoded.cameras[0].world_from_optical.homogeneous == pytest.approx(
        original.cameras[0].world_from_optical.homogeneous
    )
    assert decoded.objects[0].corners == pytest.approx(original.objects[0].corners)
    assert decoded.objects[0].timestamp == decoded.timestamp
    assert decoded.objects[0].frame_id == "world"
    assert decoded.task_annotation == replace(original.task_annotation, timestamp=decoded.timestamp)
    episode = boundary(EpisodeOutcome.SUCCEEDED)
    assert (
        episode_from_lcm(collection_episode_t.decode(episode_to_lcm(episode).encode())) == episode
    )


def events():
    result = [(Topic.COLLECTION_EPISODE, episode_to_lcm(boundary(EpisodeOutcome.STARTED)).encode())]
    for index in range(6):
        item = sample(index, index // 5)
        result += [
            (Topic.ROBOT_STATE, LCMConverter.robot_state_to_lcm(item.state).encode()),
            (
                Topic.ROBOT_JOINT_COMMAND,
                LCMConverter.robot_joint_command_to_lcm(item.command).encode(),
            ),
        ]
        if index % 5 == 0:
            result.append((Topic.SIMULATION_SAMPLE, sample_to_lcm(item).encode()))
    result.append(
        (Topic.COLLECTION_EPISODE, episode_to_lcm(boundary(EpisodeOutcome.SUCCEEDED)).encode())
    )
    return result


def write_log(path, payloads):
    with closing(lcm.EventLog(str(path), "w")) as log:
        for topic, data in payloads:
            log.write_event(0, topic.value, data)


def test_audit_accepts_complete_aligned_lcm_episode(tmp_path):
    path = tmp_path / "recording.lcm"
    write_log(path, events())
    assert validate_collection_log(path, "episode") == ()


@pytest.mark.parametrize("corruption", [None, "missing", "duplicate", "unexpected", "end"])
def test_audit_requires_every_declared_camera_in_each_sample(tmp_path, corruption):
    names = ["wrist", "base_1_2", "base_2_3", "base_3_1"]
    payloads = []
    for topic, original_data in events():
        data = original_data
        if topic is Topic.COLLECTION_EPISODE:
            wire = collection_episode_t.decode(data)
            metadata = json.loads(wire.metadata_json)
            metadata["camera_names"] = names
            if corruption == "end" and wire.outcome == EpisodeOutcome.SUCCEEDED:
                metadata["camera_names"] = names[:-1]
            wire.metadata_json = json.dumps(metadata)
            data = wire.encode()
        elif topic is Topic.SIMULATION_SAMPLE:
            original = sample_from_lcm(simulation_sample_t.decode(data))
            current = names.copy()
            if corruption == "missing":
                current.pop()
            elif corruption == "duplicate":
                current[-1] = current[0]
            elif corruption == "unexpected":
                current[-1] = "unconfigured"
            cameras = tuple(
                replace(
                    original.cameras[0],
                    frame=replace(original.cameras[0].frame, name=name, frame_id=f"{name}_optical"),
                )
                for name in current
            )
            data = sample_to_lcm(replace(original, cameras=cameras)).encode()
        payloads.append((topic, data))
    path = tmp_path / "recording.lcm"
    write_log(path, payloads)
    errors = validate_collection_log(path, "episode")
    if corruption is None:
        assert errors == ()
    else:
        assert any("camera" in error.lower() for error in errors)


@pytest.mark.parametrize(
    "field",
    [
        "camera_clock",
        "task_clock",
        "task_id",
        "task_version",
        "episode_id",
        "object_clock",
        "object_frame",
        "object_bbox",
    ],
)
def test_audit_rejects_misaligned_shared_observations(tmp_path, field):
    payloads = events()
    wire = simulation_sample_t.decode(payloads[3][1])
    if field == "camera_clock":
        wire.cameras[0].frame.timestamp += 10_000_000
    elif field == "task_clock":
        wire.task_annotation.timestamp += 10_000_000
    elif field == "task_id":
        wire.task_annotation.task = "different_task"
    elif field == "task_version":
        wire.task_annotation.task_version += 1
    elif field == "episode_id":
        wire.task_annotation.episode_id = "different_episode"
    elif field == "object_clock":
        wire.objects[0].timestamp += 10_000_000
    elif field == "object_frame":
        wire.objects[0].frame_id = "camera"
    else:
        wire.objects[0].has_bbox = False
    payloads[3] = (Topic.SIMULATION_SAMPLE, wire.encode())
    path = tmp_path / "recording.lcm"
    write_log(path, payloads)
    assert validate_collection_log(path, "episode")


def test_simulation_auditor_does_not_apply_its_timing_rules_to_real_episodes(tmp_path):
    payloads = events()
    start = replace(boundary(EpisodeOutcome.STARTED), runtime=Runtime.REAL, clock="host_monotonic")
    payloads[0] = (Topic.COLLECTION_EPISODE, episode_to_lcm(start).encode())
    path = tmp_path / "recording.lcm"
    write_log(path, payloads)
    assert "Simulation audit requires a simulation episode and clock." in validate_collection_log(
        path, "episode"
    )


@pytest.mark.parametrize("quality", ["valid", "occluded", "failed", "missing", "malformed"])
def test_success_requires_recorded_observation_checks_when_enabled(tmp_path, quality):
    payloads = []
    for topic, original in events():
        data = original
        if topic is Topic.COLLECTION_EPISODE:
            message = collection_episode_t.decode(data)
            metadata = json.loads(message.metadata_json)
            metadata["observation_validation"] = "task_checks_per_sample"
            message.metadata_json = json.dumps(metadata)
            data = message.encode()
        elif topic is Topic.SIMULATION_SAMPLE:
            message = simulation_sample_t.decode(data)
            labels = {"observation_valid": True, "observation_errors": []}
            if quality == "occluded":
                labels["views"] = {"toy": {"visible": False, "camera": "wrist"}}
            if message.sequence == 0:
                if quality == "failed":
                    labels = {"observation_valid": False, "observation_errors": ["invalid sensor"]}
                elif quality == "missing":
                    labels = {}
            message.task_annotation.labels_json = (
                "null" if quality == "malformed" else json.dumps(labels)
            )
            data = message.encode()
        payloads.append((topic, data))
    path = tmp_path / "recording.lcm"
    write_log(path, payloads)
    errors = validate_collection_log(path, "episode")
    assert bool(errors) == (quality not in {"valid", "occluded"})
    if errors:
        assert "Task observation requirements failed or are missing from a sample." in errors


def test_audit_rejects_previous_collection_schema(tmp_path):
    payloads = events()
    start = boundary(EpisodeOutcome.STARTED)
    metadata = json.loads(start.metadata_json)
    metadata["schema_version"] = COLLECTION_SCHEMA_VERSION - 1
    start = replace(start, metadata_json=json.dumps(metadata))
    payloads[0] = (Topic.COLLECTION_EPISODE, episode_to_lcm(start).encode())
    path = tmp_path / "recording.lcm"
    write_log(path, payloads)
    assert "Unsupported collection schema version." in validate_collection_log(path, "episode")


@pytest.mark.parametrize("corruption", ["empty", "grayscale", "png", "dimensions", "truncated"])
def test_audit_requires_decodable_rgb_jpeg_with_matching_dimensions(tmp_path, corruption):
    payloads = events()
    item = sample()
    camera = item.cameras[0].frame
    if corruption == "empty":
        camera = replace(camera, rgb_jpeg=b"")
    elif corruption == "dimensions":
        camera = replace(camera, width=camera.width + 1)
    elif corruption == "truncated":
        camera = replace(camera, rgb_jpeg=camera.rgb_jpeg[:-20])
    else:
        buffer = BytesIO()
        shape = (2, 4) if corruption == "grayscale" else (2, 4, 3)
        image_format = "JPEG" if corruption == "grayscale" else "PNG"
        Image.fromarray(np.zeros(shape, dtype=np.uint8)).save(buffer, format=image_format)
        camera = replace(camera, rgb_jpeg=buffer.getvalue())
    payloads[3] = (
        Topic.SIMULATION_SAMPLE,
        sample_to_lcm(replace(item, cameras=(replace(item.cameras[0], frame=camera),))).encode(),
    )
    path = tmp_path / "recording.lcm"
    write_log(path, payloads)
    assert validate_collection_log(path, "episode")


@pytest.mark.parametrize("dropped", [0, 3, -1])
def test_audit_rejects_missing_boundaries_control_or_camera_samples(tmp_path, dropped):
    payloads = events()
    del payloads[dropped]
    path = tmp_path / "recording.lcm"
    write_log(path, payloads)
    assert validate_collection_log(path, "episode")


@pytest.mark.parametrize(
    "change", ["wrong_id", "wrong_clock", "future_action", "missing_camera", "duplicate"]
)
def test_audit_rejects_misalignment_and_duplicate_frames(tmp_path, change):
    payloads = events()
    item = sample()
    if change == "wrong_id":
        item = replace(item, episode_id="different")
    elif change == "wrong_clock":
        item = replace(item, timestamp=0.01)
    elif change == "future_action":
        item.command.timestamp = 1.0
    elif change == "missing_camera":
        item = replace(item, cameras=())
    else:
        payloads.insert(4, payloads[3])
    payloads[3] = (Topic.SIMULATION_SAMPLE, sample_to_lcm(item).encode())
    path = tmp_path / "recording.lcm"
    write_log(path, payloads)
    assert validate_collection_log(path, "episode")


def test_training_eligibility_requires_both_success_and_valid_recording(tmp_path):
    report = CollectionReport(tmp_path, EpisodeOutcome.SUCCEEDED, True, "ok", 2, 6)
    assert report.eligible_for_training
    assert not replace(report, valid=False).eligible_for_training
    assert not replace(report, outcome=EpisodeOutcome.FAILED).eligible_for_training
    assert not replace(report, outcome=EpisodeOutcome.ABORTED).eligible_for_training


def test_audit_rejects_an_aligned_sample_from_the_wrong_control_interval(tmp_path):
    payloads = events()
    # Move sample 0 after action 1 and change all nested timestamps to match it.
    # Pairwise alignment alone cannot detect this incorrect action_index label.
    payloads.pop(3)
    payloads.insert(5, (Topic.SIMULATION_SAMPLE, sample_to_lcm(sample(1, 0)).encode()))
    wire = simulation_sample_t.decode(payloads[5][1])
    wire.action_index = 0
    payloads[5] = (Topic.SIMULATION_SAMPLE, wire.encode())
    path = tmp_path / "recording.lcm"
    write_log(path, payloads)
    assert validate_collection_log(path, "episode")


@pytest.mark.parametrize("corruption", ["rates", "wire", "image"])
def test_malformed_payloads_are_rejected_without_crashing_the_audit(tmp_path, corruption):
    payloads = events()
    if corruption == "rates":
        start = boundary(EpisodeOutcome.STARTED)
        metadata = json.loads(start.metadata_json)
        metadata["collection"]["control_hz"] = "oops"
        start = replace(start, metadata_json=json.dumps(metadata))
        payloads[0] = (Topic.COLLECTION_EPISODE, episode_to_lcm(start).encode())
    elif corruption == "wire":
        payloads[3] = (Topic.SIMULATION_SAMPLE, payloads[3][1][:20])
    else:
        item = sample()
        camera = replace(item.cameras[0].frame, rgb_jpeg=b"invalid image")
        item = replace(item, cameras=(replace(item.cameras[0], frame=camera),))
        payloads[3] = (Topic.SIMULATION_SAMPLE, sample_to_lcm(item).encode())
    path = tmp_path / "recording.lcm"
    write_log(path, payloads)
    assert validate_collection_log(path, "episode")


@pytest.mark.parametrize(
    "kwargs", [{"sample_hz": 0}, {"control_hz": 0}, {"max_sim_seconds": float("inf")}]
)
def test_invalid_collection_timing_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CollectionConfig(**kwargs)
