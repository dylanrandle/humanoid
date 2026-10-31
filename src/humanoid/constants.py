from enum import Enum

from humanoid.types.actuator import ActuatorHealthReport
from humanoid.types.camera import CameraFrame
from humanoid.types.collection import (
    CollectionEpisode,
    CollectionRequest,
    CollectionRunStatus,
    SimulationSample,
)
from humanoid.types.homing import HomingTarget
from humanoid.types.logging import LoggingRequest, LoggingStatus
from humanoid.types.node import NodeRateSample
from humanoid.types.object_state import ObjectState
from humanoid.types.orchestrator import OrchestratorEvent, OrchestratorMode
from humanoid.types.process import Runtime
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotName,
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

DEFAULT_LCM_URL = "udpm://239.255.76.67:7667?ttl=1"
RUNTIME_ENVIRONMENT_VARIABLE = "HUMANOID_RUNTIME"
ROBOT_ENVIRONMENT_VARIABLE = "HUMANOID_ROBOT"
OCULUS_IP_ENVIRONMENT_VARIABLE = "HUMANOID_OCULUS_IP"
DEFAULT_HUMANOID_RUNTIME = Runtime.SIM
DEFAULT_HUMANOID_ROBOT = RobotName.TRISKEL


class Topic(Enum):
    # Per-node loop-rate telemetry consumed by the operator console.
    NODE_RATE = "NODE/RATE"

    # Per-actuator feedback health published by the real robot driver.
    ACTUATOR_HEALTH = "ACTUATOR/HEALTH"

    # Robot state (single publisher: robot driver)
    ROBOT_STATE = "ROBOT/STATE"

    # Shared observation and collection messages, usable in simulation and real.
    CAMERA_FRAME = "CAMERA/FRAME"
    OBJECT_STATE = "OBJECT/STATE"
    TASK_ANNOTATION = "TASK/ANNOTATION"
    COLLECTION_EPISODE = "COLLECTION/EPISODE"
    COLLECTION_REQUEST = "COLLECTION/REQUEST"
    COLLECTION_STATUS = "COLLECTION/STATUS"

    # Simulation-only visualization and autonomous data collection.
    SIMULATION_SCENE = "SIMULATION/SCENE"
    SIMULATION_RESET = "SIMULATION/RESET"
    SIMULATION_RESET_STATUS = "SIMULATION/RESET_STATUS"
    SIMULATION_STATE = "SIMULATION/STATE"
    LOGGING_REQUEST = "LOGGING/REQUEST"
    SIMULATION_SAMPLE = "SIMULATION/SAMPLE"

    # Final command topics — consumed by robot driver / controller. Orchestrator routes
    # exactly one per-source topic to each of these.
    ROBOT_JOINT_COMMAND = "ROBOT/JOINT_COMMAND"
    ROBOT_TOOL_COMMAND = "ROBOT/TOOL_COMMAND"
    ROBOT_BASE_COMMAND = "ROBOT/BASE_COMMAND"

    # Per-source policy / controller outputs. Multiple topics can carry the same
    # message type — the orchestrator selects which one feeds the final topic.
    CONTROLLER_JOINT_COMMAND = "CONTROLLER/JOINT_COMMAND"
    HOMING_JOINT_COMMAND = "HOMING/JOINT_COMMAND"
    OCULUS_TOOL_COMMAND = "OCULUS/TOOL_COMMAND"
    OCULUS_BASE_COMMAND = "OCULUS/BASE_COMMAND"
    KEYBOARD_TOOL_COMMAND = "KEYBOARD/TOOL_COMMAND"
    KEYBOARD_BASE_COMMAND = "KEYBOARD/BASE_COMMAND"
    SYSTEM_TOOL_COMMAND = "SYSTEM/TOOL_COMMAND"
    SYSTEM_BASE_COMMAND = "SYSTEM/BASE_COMMAND"

    # Orchestrator broadcasts its currently active mode here.
    ORCHESTRATOR_MODE = "ORCHESTRATOR/MODE"

    # Events published to the orchestrator (request_*, complete).
    ORCHESTRATOR_EVENT = "ORCHESTRATOR/EVENT"

    # Current lcm-logger lifecycle, published by RobotLoggerNode.
    LOGGING_STATUS = "LOGGING/STATUS"

    # Homing target published by requesters for the homing node.
    HOMING_TARGET = "HOMING/TARGET"


TOPIC_TO_TYPE: dict[Topic, type] = {
    Topic.NODE_RATE: NodeRateSample,
    Topic.ACTUATOR_HEALTH: ActuatorHealthReport,
    Topic.ROBOT_STATE: RobotState,
    Topic.CAMERA_FRAME: CameraFrame,
    Topic.OBJECT_STATE: ObjectState,
    Topic.TASK_ANNOTATION: TaskAnnotation,
    Topic.SIMULATION_SCENE: SimulationSceneState,
    Topic.SIMULATION_RESET: SimulationReset,
    Topic.SIMULATION_RESET_STATUS: SimulationResetStatus,
    Topic.SIMULATION_STATE: SimulationState,
    Topic.LOGGING_REQUEST: LoggingRequest,
    Topic.SIMULATION_SAMPLE: SimulationSample,
    Topic.COLLECTION_EPISODE: CollectionEpisode,
    Topic.COLLECTION_REQUEST: CollectionRequest,
    Topic.COLLECTION_STATUS: CollectionRunStatus,
    Topic.ROBOT_JOINT_COMMAND: RobotJointCommand,
    Topic.ROBOT_TOOL_COMMAND: RobotToolCommand,
    Topic.ROBOT_BASE_COMMAND: RobotBaseCommand,
    Topic.CONTROLLER_JOINT_COMMAND: RobotJointCommand,
    Topic.HOMING_JOINT_COMMAND: RobotJointCommand,
    Topic.OCULUS_TOOL_COMMAND: RobotToolCommand,
    Topic.OCULUS_BASE_COMMAND: RobotBaseCommand,
    Topic.KEYBOARD_TOOL_COMMAND: RobotToolCommand,
    Topic.KEYBOARD_BASE_COMMAND: RobotBaseCommand,
    Topic.SYSTEM_TOOL_COMMAND: RobotToolCommand,
    Topic.SYSTEM_BASE_COMMAND: RobotBaseCommand,
    Topic.ORCHESTRATOR_MODE: OrchestratorMode,
    Topic.ORCHESTRATOR_EVENT: OrchestratorEvent,
    Topic.LOGGING_STATUS: LoggingStatus,
    Topic.HOMING_TARGET: HomingTarget,
}
