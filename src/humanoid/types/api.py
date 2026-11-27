"""Shared API contract, backend status, and application boundary types."""

from dataclasses import dataclass
from enum import StrEnum
from http import HTTPStatus

from humanoid.types.actuator import ActuatorHealthStatus
from humanoid.types.host import HostResourceStatus
from humanoid.types.node import NodeRateStatus
from humanoid.types.orchestrator import OrchestratorState
from humanoid.types.process import ProcessName, ProcessStatus, Runtime
from humanoid.types.recording import RecordingStatus
from humanoid.types.replay import RecordingSummary, ReplayStatus
from humanoid.types.robot import RobotName
from humanoid.types.simulation import SimulationScene


class ApiError(Exception):
    """An expected API failure that is safe to return to the caller."""

    def __init__(self, message: str, status: HTTPStatus) -> None:
        super().__init__(message)
        self.status = status


class BackendError(Exception):
    """An expected backend operation failure."""


class BackendErrorMessage(StrEnum):
    """Shared messages for expected backend operation failures."""

    EXTERNAL_STACK = "Another stack is already broadcasting. Stop it before using this console."
    STALE_CONFIGURATION = "Configuration changed. Refresh the console and try again."
    REAL_HARDWARE_ACKNOWLEDGEMENT = "Real hardware acknowledgement is required."
    REPLAY_ACTIVE_PROCESSES = "Stop the stack and input processes before starting replay."
    REPLAY_ACTIVE = "Stop replay before starting the main stack."
    REPLAY_DEDICATED_CONTROL = "Use the dedicated replay controls to manage replay."


class ServiceConflict(BackendError, RuntimeError):
    """The current service state prevents the requested operation."""


@dataclass(frozen=True)
class BackendStatus:
    """Aggregate operator-console status produced by BackendApplication."""

    runtime: Runtime
    robot: RobotName
    robots: list[RobotName]
    scene: SimulationScene
    scenes: list[SimulationScene]
    processes: dict[ProcessName, ProcessStatus]
    node_rates: list[NodeRateStatus]
    host_resources: HostResourceStatus | None
    actuator_health: ActuatorHealthStatus
    recording: RecordingStatus
    recordings: list[RecordingSummary]
    replay: ReplayStatus
    orchestrator: OrchestratorState


@dataclass(frozen=True)
class SafetyContext:
    """Configuration observed and hardware risk acknowledged by an operator."""

    expected_runtime: Runtime
    expected_robot: RobotName
    real_hardware_acknowledged: bool = False


class ApiRoute(StrEnum):
    ROOT = "/api"
    STATE = f"{ROOT}/state"
    CAMERA = f"{ROOT}/camera"
    CAPTURE = f"{ROOT}/capture"
    RELEASE_AUTONOMY = f"{ROOT}/release_autonomy"
    MOTIONS = f"{ROOT}/motions"
    MOTION = f"{MOTIONS}/<motion_id>"
    MOTION_CANCEL = f"{MOTION}/cancel"
    STATUS = f"{ROOT}/status"
    LOGS = f"{ROOT}/logs"
    CAMERAS = f"{ROOT}/cameras"
    STREAMS = f"{ROOT}/streams"
    STREAM = f"{STREAMS}/<session_id>"
    RUNTIME = f"{ROOT}/runtime"
    SCENE = f"{ROOT}/scene"
    ROBOT = f"{ROOT}/robot"
    PROCESSES = f"{ROOT}/processes/<name>/<action>"
    RECORDINGS = f"{ROOT}/recordings"
    RECORDING_ACTION = f"{RECORDINGS}/<action>"
    RECORDING_TOGGLE = f"{RECORDINGS}/toggle"
    REPLAY = f"{ROOT}/replay/<action>"
    ORCHESTRATOR = f"{ROOT}/orchestrator"
    AGENT = f"{ROOT}/agent"
    AGENT_CONNECT = f"{AGENT}/connect"
    AGENT_RUN = f"{AGENT}/run"
    AGENT_STOP = f"{AGENT}/stop"


class ApiKey(StrEnum):
    """Shared JSON field and query parameter names for API clients and the backend."""

    AFTER = "after"
    AGE_SECONDS = "age_seconds"
    CAMERA_MODEL = "camera_model"
    DEPTH = "depth"
    DEPTH_FAR = "depth_far"
    DEPTH_SCALE = "depth_scale"
    DEPTH_VIEW = "depth_view"
    DEVICE_TIMESTAMP = "device_timestamp"
    DISTORTION_COEFFICIENTS = "distortion_coefficients"
    ENCODING = "encoding"
    EXPECTED_ROBOT = "expected_robot"
    EXPECTED_RUNTIME = "expected_runtime"
    FRAME_ID = "frame_id"
    FRAME_NUMBER = "frame_number"
    INTRINSICS = "intrinsics"
    MODE = "mode"
    PARAMETERS = "parameters"
    PNG_BASE64 = "png_base64"
    REAL_HARDWARE_ACKNOWLEDGED = "real_hardware_acknowledged"
    RECORDING = "recording"
    REGISTERED = "registered"
    RGB = "rgb"
    RGB_FRAME_NUMBER = "rgb_frame_number"
    ROBOT = "robot"
    RUNTIME = "runtime"
    SCENE = "scene"
    SERIAL_NUMBER = "serial_number"
    TIMESTAMP = "timestamp"
    TIMESTAMP_DOMAIN = "timestamp_domain"


@dataclass(frozen=True)
class ApiEndpoint:
    url: str
