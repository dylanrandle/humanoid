from enum import Enum

from humanoid.types.homing import HomingTarget
from humanoid.types.logging import LoggingStatus
from humanoid.types.orchestrator import OrchestratorEvent, OrchestratorMode
from humanoid.types.process import Runtime
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotName,
    RobotState,
    RobotToolCommand,
)

DEFAULT_LCM_URL = "udpm://239.255.76.67:7667?ttl=1"
RUNTIME_ENVIRONMENT_VARIABLE = "HUMANOID_RUNTIME"
ROBOT_ENVIRONMENT_VARIABLE = "HUMANOID_ROBOT"
DEFAULT_HUMANOID_RUNTIME = Runtime.SIM
DEFAULT_HUMANOID_ROBOT = RobotName.ELROBOT_MOBILE


class Topic(Enum):
    # Robot state (single publisher: robot driver)
    ROBOT_STATE = "ROBOT/STATE"

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

    # Orchestrator broadcasts its currently active mode here.
    ORCHESTRATOR_MODE = "ORCHESTRATOR/MODE"

    # Events published to the orchestrator (request_*, complete).
    ORCHESTRATOR_EVENT = "ORCHESTRATOR/EVENT"

    # Current lcm-logger lifecycle, published by RobotLoggerNode.
    LOGGING_STATUS = "LOGGING/STATUS"

    # Homing target published by requesters for the homing node.
    HOMING_TARGET = "HOMING/TARGET"


TOPIC_TO_TYPE: dict[Topic, type] = {
    Topic.ROBOT_STATE: RobotState,
    Topic.ROBOT_JOINT_COMMAND: RobotJointCommand,
    Topic.ROBOT_TOOL_COMMAND: RobotToolCommand,
    Topic.ROBOT_BASE_COMMAND: RobotBaseCommand,
    Topic.CONTROLLER_JOINT_COMMAND: RobotJointCommand,
    Topic.HOMING_JOINT_COMMAND: RobotJointCommand,
    Topic.OCULUS_TOOL_COMMAND: RobotToolCommand,
    Topic.OCULUS_BASE_COMMAND: RobotBaseCommand,
    Topic.KEYBOARD_TOOL_COMMAND: RobotToolCommand,
    Topic.KEYBOARD_BASE_COMMAND: RobotBaseCommand,
    Topic.ORCHESTRATOR_MODE: OrchestratorMode,
    Topic.ORCHESTRATOR_EVENT: OrchestratorEvent,
    Topic.LOGGING_STATUS: LoggingStatus,
    Topic.HOMING_TARGET: HomingTarget,
}
