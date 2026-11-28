import os
import secrets
from enum import Enum, IntEnum, StrEnum

from humanoid.types.actuator import ActuatorHealthReport
from humanoid.types.camera import CameraImage, CameraTopic, RegisteredDepth
from humanoid.types.homing import HomingTarget
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


class Host(StrEnum):
    """Network host defaults; environment and CLI settings can override them."""

    LCM_MULTICAST = "239.255.76.67"
    LOCALHOST = "localhost"
    LOOPBACK = "127.0.0.1"
    ROBOT = "triskel"


class Port(IntEnum):
    """Service port defaults; DYNAMIC asks the OS to allocate a free listener."""

    API = 8765
    DASHBOARD_GATEWAY = 443
    DYNAMIC = 0
    LCM = 7667
    QUEST_GATEWAY = 8443
    QUEST_SERVER = 8012


LCM_RECEIVE_BUFFER_BYTES = 8 * 1024 * 1024


def local_lcm_url(group: str, port: int) -> str:
    return f"mpudpm://{group}:{port}?ttl=0&recv_buf_size={LCM_RECEIVE_BUFFER_BYTES}"


def private_lcm_url() -> str:
    """Allocate a random local bus independently of the operator's LCM_DEFAULT_URL."""
    group = f"239.192.{secrets.randbelow(256)}.{secrets.randbelow(256)}"
    # The multicast group isolates each bus. Keep its 500 channel ports outside
    # the ephemeral range: Linux's outgoing sockets can otherwise claim a port
    # before LCM binds it, even when the multicast group is unique.
    return local_lcm_url(group, Port.LCM)


class EnvironmentVariable(StrEnum):
    """Humanoid environment-variable names for runtime code and tooling."""

    API_URL = "HUMANOID_API_URL"
    DASHBOARD_PORT = "HUMANOID_DASHBOARD_PORT"
    DATA_DIR = "HUMANOID_DATA_DIR"
    LOCAL_DASHBOARD_PORT = "HUMANOID_LOCAL_DASHBOARD_PORT"
    PUBLIC_HOST = "HUMANOID_PUBLIC_HOST"
    QUEST_PORT = "HUMANOID_QUEST_PORT"
    ROBOT = "HUMANOID_ROBOT"
    ROBOT_DIR = "HUMANOID_ROBOT_DIR"
    ROBOT_HOST = "HUMANOID_ROBOT_HOST"
    RUNTIME = "HUMANOID_RUNTIME"
    SCENE = "HUMANOID_SCENE"


DEFAULT_LCM_URL = os.environ.get("LCM_DEFAULT_URL", local_lcm_url(Host.LCM_MULTICAST, Port.LCM))
DEFAULT_HUMANOID_RUNTIME = Runtime.SIM
DEFAULT_HUMANOID_ROBOT = RobotName.TRISKEL
DEFAULT_ROBOT_DIR = "/opt/humanoid"


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
    QUEST_TOOL_COMMAND = "QUEST/TOOL_COMMAND"
    QUEST_BASE_COMMAND = "QUEST/BASE_COMMAND"
    KEYBOARD_TOOL_COMMAND = "KEYBOARD/TOOL_COMMAND"
    KEYBOARD_BASE_COMMAND = "KEYBOARD/BASE_COMMAND"
    SYSTEM_TOOL_COMMAND = "SYSTEM/TOOL_COMMAND"
    AUTONOMY_BASE_COMMAND = "AUTONOMY/BASE_COMMAND"
    AUTONOMY_TOOL_COMMAND = "AUTONOMY/TOOL_COMMAND"

    # Orchestrator broadcasts its currently active mode here.
    ORCHESTRATOR_MODE = "ORCHESTRATOR/MODE"

    # Events published to the orchestrator (request_*, complete).
    ORCHESTRATOR_EVENT = "ORCHESTRATOR/EVENT"

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
    Topic.QUEST_TOOL_COMMAND: RobotToolCommand,
    Topic.QUEST_BASE_COMMAND: RobotBaseCommand,
    Topic.KEYBOARD_TOOL_COMMAND: RobotToolCommand,
    Topic.KEYBOARD_BASE_COMMAND: RobotBaseCommand,
    Topic.SYSTEM_TOOL_COMMAND: RobotToolCommand,
    Topic.AUTONOMY_TOOL_COMMAND: RobotToolCommand,
    Topic.AUTONOMY_BASE_COMMAND: RobotBaseCommand,
    Topic.ORCHESTRATOR_MODE: OrchestratorMode,
    Topic.ORCHESTRATOR_EVENT: OrchestratorEvent,
    Topic.HOMING_TARGET: HomingTarget,
}


type Channel = Topic | CameraTopic[CameraImage] | CameraTopic[RegisteredDepth]


def message_type(channel: Channel) -> type:
    return channel.message_type if isinstance(channel, CameraTopic) else TOPIC_TO_TYPE[channel]
