"""Shared orchestrator configuration constants."""

from humanoid.constants import Topic
from humanoid.types.orchestrator import Mode
from humanoid.types.process import ProcessName

PARAMETERIZED_REQUEST_TIMEOUT_SECONDS = 2.0
MODE_TRANSITION_TIMEOUT_SECONDS = 1.0
MODE_TRANSITION_POLL_INTERVAL_SECONDS = 0.01
LOGGING_ACKNOWLEDGEMENT_TIMEOUT_SECONDS = 2.0
REPLAY_STOP_TIMEOUT_SECONDS = 2.0
REPLAY_STDERR_TAIL_BYTES = 8 * 1024
LCM_LOGPLAYER_COMMAND = "lcm-logplayer"
REPLAY_CHANNELS = (
    Topic.ROBOT_JOINT_COMMAND,
    Topic.ROBOT_TOOL_COMMAND,
    Topic.ROBOT_BASE_COMMAND,
    Topic.ORCHESTRATOR_MODE,
)
REPLAY_CHANNEL_PATTERN = "|".join(topic.value for topic in REPLAY_CHANNELS)
TELEOP_PROCESSES: dict[Mode, ProcessName] = {
    Mode.KEYBOARD: ProcessName.KEYBOARD,
    Mode.OCULUS: ProcessName.OCULUS,
}
PROCESS_MODES = {process: mode for mode, process in TELEOP_PROCESSES.items()}
CONTROLLED_PROCESS_NAMES = frozenset({ProcessName.STACK, *TELEOP_PROCESSES.values()})
EXTERNAL_STACK_ERROR = "Another stack is already broadcasting. Stop it before using this console."
STALE_CONFIGURATION_ERROR = "Configuration changed. Refresh the console and try again."
REAL_HARDWARE_ACKNOWLEDGEMENT_ERROR = "Real hardware acknowledgement is required."
REPLAY_ACTIVE_PROCESSES_ERROR = "Stop the stack and teleop processes before starting replay."
REPLAY_ACTIVE_ERROR = "Stop replay before starting the main stack."
REPLAY_DEDICATED_CONTROL_ERROR = "Use the dedicated replay controls to manage replay."
