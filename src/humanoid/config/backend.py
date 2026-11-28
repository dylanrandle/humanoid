"""Backend lifecycle and mode-request defaults."""

from humanoid.constants import Topic
from humanoid.types.orchestrator import Mode
from humanoid.types.process import ProcessName

PARAMETERIZED_REQUEST_TIMEOUT_SECONDS = 2.0
MODE_TRANSITION_TIMEOUT_SECONDS = 1.0
MODE_TRANSITION_POLL_INTERVAL_SECONDS = 0.01
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
INPUT_PROCESSES: dict[Mode, ProcessName] = {
    Mode.KEYBOARD: ProcessName.KEYBOARD,
    Mode.QUEST: ProcessName.QUEST,
    Mode.AUTONOMY: ProcessName.AUTONOMY,
}
PROCESS_MODES = {process: mode for mode, process in INPUT_PROCESSES.items()}
CONTROLLED_PROCESS_NAMES = frozenset({ProcessName.STACK, *INPUT_PROCESSES.values()})
