"""Conversions for shared episode boundaries and privileged simulation samples."""

import numpy as np
import pinocchio as pin

from humanoid.types.collection import (
    CollectionEpisode,
    EpisodeOutcome,
    SimulationCameraObservation,
    SimulationSample,
)
from humanoid.types.lcm import (
    collection_episode_t,
    simulation_camera_observation_t,
    simulation_sample_t,
)
from humanoid.types.lcm.camera import camera_frame_from_lcm, camera_frame_to_lcm
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.lcm.object_state import object_state_from_lcm, object_state_to_lcm
from humanoid.types.lcm.task_annotation import task_annotation_from_lcm, task_annotation_to_lcm
from humanoid.types.process import Runtime


def _quaternion(pose: pin.SE3) -> list[float]:
    q = pin.Quaternion(pose.rotation)
    return [q.w, q.x, q.y, q.z]


def _pose(position, quaternion) -> pin.SE3:
    return pin.SE3(pin.Quaternion(*quaternion).toRotationMatrix(), np.asarray(position))


def sample_to_lcm(sample: SimulationSample) -> simulation_sample_t:
    message = simulation_sample_t()
    message.timestamp = round(sample.timestamp * 1e9)
    message.episode_id = sample.episode_id
    message.sequence = sample.sequence
    message.action_index = sample.action_index
    message.state = LCMConverter.robot_state_to_lcm(sample.state)
    message.command = LCMConverter.robot_joint_command_to_lcm(sample.command)
    message.num_cameras = len(sample.cameras)
    message.cameras = []
    for camera in sample.cameras:
        wire = simulation_camera_observation_t()
        wire.frame = camera_frame_to_lcm(camera.frame)
        wire.position = camera.world_from_optical.translation.tolist()
        wire.quaternion = _quaternion(camera.world_from_optical)
        message.cameras.append(wire)
    message.num_objects = len(sample.objects)
    message.objects = [object_state_to_lcm(item) for item in sample.objects]
    message.task_annotation = task_annotation_to_lcm(sample.task_annotation)
    return message


def sample_from_lcm(message: simulation_sample_t) -> SimulationSample:
    return SimulationSample(
        timestamp=message.timestamp / 1e9,
        episode_id=message.episode_id,
        sequence=message.sequence,
        action_index=message.action_index,
        state=LCMConverter.robot_state_from_lcm(message.state),
        command=LCMConverter.robot_joint_command_from_lcm(message.command),
        cameras=tuple(
            SimulationCameraObservation(
                frame=camera_frame_from_lcm(c.frame),
                world_from_optical=_pose(c.position, c.quaternion),
            )
            for c in message.cameras
        ),
        objects=tuple(object_state_from_lcm(o) for o in message.objects),
        task_annotation=task_annotation_from_lcm(message.task_annotation),
    )


def episode_to_lcm(episode: CollectionEpisode) -> collection_episode_t:
    message = collection_episode_t()
    message.timestamp = round(episode.timestamp * 1e9)
    message.episode_id, message.task, message.task_version = (
        episode.episode_id,
        episode.task,
        episode.task_version,
    )
    message.runtime = episode.runtime.value
    message.clock = episode.clock
    message.outcome = episode.outcome.value
    message.sample_count, message.action_count = episode.sample_count, episode.action_count
    message.metadata_json = episode.metadata_json
    return message


def episode_from_lcm(message: collection_episode_t) -> CollectionEpisode:
    return CollectionEpisode(
        message.timestamp / 1e9,
        message.episode_id,
        message.task,
        message.task_version,
        Runtime(message.runtime),
        message.clock,
        EpisodeOutcome(message.outcome),
        message.sample_count,
        message.action_count,
        message.metadata_json,
    )
