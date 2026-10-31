import lcm

from humanoid.constants import DEFAULT_LCM_URL, TOPIC_TO_TYPE, Topic
from humanoid.logger import get_logger
from humanoid.types.actuator import ActuatorHealthReport
from humanoid.types.camera import CameraFrame
from humanoid.types.collection import (
    CollectionEpisode,
    CollectionRequest,
    CollectionRunStatus,
    SimulationSample,
)
from humanoid.types.homing import HomingTarget
from humanoid.types.lcm.camera import camera_frame_to_lcm
from humanoid.types.lcm.collection import episode_to_lcm, sample_to_lcm
from humanoid.types.lcm.collection_control import request_to_lcm, status_to_lcm
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.lcm.logging import logging_request_to_lcm
from humanoid.types.lcm.object_state import object_state_to_lcm
from humanoid.types.lcm.simulation_control import (
    reset_status_to_lcm,
    reset_to_lcm,
    state_to_lcm,
)
from humanoid.types.lcm.task_annotation import task_annotation_to_lcm
from humanoid.types.logging import LoggingRequest, LoggingStatus
from humanoid.types.middleware import AcceptedTypes
from humanoid.types.node import NodeRateSample
from humanoid.types.object_state import ObjectState
from humanoid.types.orchestrator import OrchestratorEvent, OrchestratorMode
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)
from humanoid.types.scene import SimulationSceneState
from humanoid.types.simulation import (
    SimulationReset,
    SimulationResetStatus,
    SimulationState,
)
from humanoid.types.task_annotation import TaskAnnotation

logger = get_logger(__name__)


class Publisher:
    def __init__(self, url: str = DEFAULT_LCM_URL):
        self.lc = lcm.LCM(url)
        self.url = url

    def publish(self, data: AcceptedTypes, topic: Topic) -> None:  # noqa: PLR0912 - explicit wire types
        expected_type = TOPIC_TO_TYPE.get(topic)
        if expected_type is not type(data):
            raise TypeError(f"Topic {topic} expects {expected_type}, but got {type(data).__name__}")

        if isinstance(data, ActuatorHealthReport):
            lcm_data = LCMConverter.actuator_health_report_to_lcm(data)
        elif isinstance(data, NodeRateSample):
            lcm_data = LCMConverter.node_rate_sample_to_lcm(data)
        elif isinstance(data, RobotJointCommand):
            lcm_data = LCMConverter.robot_joint_command_to_lcm(data)
        elif isinstance(data, RobotState):
            lcm_data = LCMConverter.robot_state_to_lcm(data)
        elif isinstance(data, RobotToolCommand):
            lcm_data = LCMConverter.robot_tool_command_to_lcm(data)
        elif isinstance(data, RobotBaseCommand):
            lcm_data = LCMConverter.robot_base_command_to_lcm(data)
        elif isinstance(data, OrchestratorMode):
            lcm_data = LCMConverter.orchestrator_mode_to_lcm(data)
        elif isinstance(data, OrchestratorEvent):
            lcm_data = LCMConverter.orchestrator_event_to_lcm(data)
        elif isinstance(data, HomingTarget):
            lcm_data = LCMConverter.homing_target_to_lcm(data)
        elif isinstance(data, LoggingStatus):
            lcm_data = LCMConverter.logging_status_to_lcm(data)
        elif isinstance(data, SimulationSceneState):
            lcm_data = LCMConverter.simulation_scene_to_lcm(data)
        elif isinstance(data, SimulationReset):
            lcm_data = reset_to_lcm(data)
        elif isinstance(data, SimulationResetStatus):
            lcm_data = reset_status_to_lcm(data)
        elif isinstance(data, SimulationState):
            lcm_data = state_to_lcm(data)
        elif isinstance(data, LoggingRequest):
            lcm_data = logging_request_to_lcm(data)
        elif isinstance(data, SimulationSample):
            lcm_data = sample_to_lcm(data)
        elif isinstance(data, CollectionEpisode):
            lcm_data = episode_to_lcm(data)
        elif isinstance(data, CollectionRequest):
            lcm_data = request_to_lcm(data)
        elif isinstance(data, CollectionRunStatus):
            lcm_data = status_to_lcm(data)
        elif isinstance(data, CameraFrame):
            lcm_data = camera_frame_to_lcm(data)
        elif isinstance(data, ObjectState):
            lcm_data = object_state_to_lcm(data)
        elif isinstance(data, TaskAnnotation):
            lcm_data = task_annotation_to_lcm(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        self.lc.publish(topic.value, lcm_data.encode())
