import os
import secrets
from enum import Enum

from humanoid.types.actuator import ActuatorHealthReport
from humanoid.types.camera import CameraImage, CameraTopic, RegisteredDepth
from humanoid.types.homing import HomingTarget
from humanoid.types.logging import LoggingStatus
from humanoid.types.node import NodeRateSample
from humanoid.types.orchestrator import OrchestratorEvent, OrchestratorMode
from humanoid.types.process import Runtime
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotName,
    RobotState,
    RobotToolCommand,
)
from humanoid.types.simulation import SimulationState

LCM_RECEIVE_BUFFER_BYTES = 8 * 1024 * 1024


def local_lcm_url(group: str, port: int) -> str:
    return f"mpudpm://{group}:{port}?ttl=0&recv_buf_size={LCM_RECEIVE_BUFFER_BYTES}"


def private_lcm_url() -> str:
    """Allocate a random local bus independently of the operator's LCM_DEFAULT_URL."""
    group = f"239.192.{secrets.randbelow(256)}.{secrets.randbelow(256)}"
    # Multiport LCM uses 500 consecutive ports starting at the supplied port.
    return local_lcm_url(group, 49152 + secrets.randbelow(16384 - 500 + 1))


DEFAULT_LCM_URL = os.environ.get("LCM_DEFAULT_URL", local_lcm_url("239.255.76.67", 7667))
RUNTIME_ENVIRONMENT_VARIABLE = "HUMANOID_RUNTIME"
SCENE_ENVIRONMENT_VARIABLE = "HUMANOID_SCENE"
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
    SIMULATION_STATE = "SIMULATION/STATE"
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
    SYSTEM_TOOL_COMMAND = "SYSTEM/TOOL_COMMAND"
    AUTONOMY_BASE_COMMAND = "AUTONOMY/BASE_COMMAND"
    AUTONOMY_TOOL_COMMAND = "AUTONOMY/TOOL_COMMAND"

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
    Topic.SIMULATION_STATE: SimulationState,
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
    Topic.AUTONOMY_TOOL_COMMAND: RobotToolCommand,
    Topic.AUTONOMY_BASE_COMMAND: RobotBaseCommand,
    Topic.ORCHESTRATOR_MODE: OrchestratorMode,
    Topic.ORCHESTRATOR_EVENT: OrchestratorEvent,
    Topic.LOGGING_STATUS: LoggingStatus,
    Topic.HOMING_TARGET: HomingTarget,
}


type Channel = Topic | CameraTopic[CameraImage] | CameraTopic[RegisteredDepth]


def message_type(channel: Channel) -> type:
    return channel.message_type if isinstance(channel, CameraTopic) else TOPIC_TO_TYPE[channel]
