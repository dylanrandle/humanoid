from enum import Enum

from humanoid.types.orchestrator import OrchestratorMode
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)

DEFAULT_LCM_URL = "udpm://239.255.76.67:7667?ttl=1"


class Topic(Enum):
    # Robot state (single publisher: robot driver)
    ROBOT_STATE = "ROBOT_STATE"

    # Final command topics — consumed by robot driver / OSC. Orchestrator routes
    # exactly one per-source topic to each of these.
    ROBOT_JOINT_COMMAND = "ROBOT_JOINT_COMMAND"
    ROBOT_TOOL_COMMAND = "ROBOT_TOOL_COMMAND"
    ROBOT_BASE_COMMAND = "ROBOT_BASE_COMMAND"

    # Per-source policy / controller outputs. Multiple topics can carry the same
    # message type — the orchestrator selects which one feeds the final topic.
    CONTROLLER_JOINT_COMMAND = "CONTROLLER/ROBOT_JOINT_COMMAND"
    HOMING_JOINT_COMMAND = "HOMING/ROBOT_JOINT_COMMAND"
    OCULUS_TOOL_COMMAND = "OCULUS/ROBOT_TOOL_COMMAND"
    OCULUS_BASE_COMMAND = "OCULUS/ROBOT_BASE_COMMAND"
    KEYBOARD_TOOL_COMMAND = "KEYBOARD/ROBOT_TOOL_COMMAND"
    KEYBOARD_BASE_COMMAND = "KEYBOARD/ROBOT_BASE_COMMAND"

    # Orchestrator broadcasts its currently active mode here.
    ORCHESTRATOR_MODE = "ORCHESTRATOR_MODE"


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
}
