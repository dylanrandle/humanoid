"""One trial episode through the stack's robot API, on simulation or hardware."""

import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack, closing, contextmanager
from pathlib import Path

from humanoid.agent.harness.recording import record_lcm_episode
from humanoid.agent.harness.robot import RobotTools
from humanoid.api.clients.robot import RobotClient
from humanoid.config.agent import TRIAL_CAPTURE_POLL_INTERVAL_SECONDS, TRIAL_CAPTURE_TIMEOUT_SECONDS
from humanoid.config.motion import DEFAULT_MOTION_TIMEOUT_SECONDS
from humanoid.recording import DEFAULT_RECORDING_ROOT
from humanoid.robots.base import Robot
from humanoid.simulation.trial_stack import TrialStack
from humanoid.types.agent import TimeBudget
from humanoid.types.api import ApiEndpoint
from humanoid.types.camera import HardwareCameraConfig
from humanoid.types.motion import ControlOperation, MotionResult
from humanoid.types.observation import CameraCaptures, Observation
from humanoid.types.orchestrator import Mode
from humanoid.types.process import Runtime
from humanoid.types.robot import RobotConfig
from humanoid.types.tools import ToolError
from humanoid.types.trial import EpisodeScorer, SimulationTask, TaskScore, TrialTraceEntry


class Episode:
    def __init__(
        self,
        client: RobotClient,
        robot_config: RobotConfig,
        *,
        backend: Runtime,
        score_fn: EpisodeScorer | None = None,
    ):
        self.robot_config = robot_config
        self.client = client
        self.backend = backend
        self.score_fn = score_fn
        self.trace: list[TrialTraceEntry] = []
        self.timer: TimeBudget | None = None
        self._released = False
        self._recording = ExitStack()
        self.lcm_recording: str | None = None

    def start_recording(self) -> None:
        self.lcm_recording = self._recording.enter_context(
            record_lcm_episode(self.client.endpoint.url, self.client.runtime_id)
        )

    def release(self) -> None:
        if not self._released:
            try:
                self.client.release_autonomy()
                self._released = True
            finally:
                self._recording.close()

    def observation(self) -> Observation:
        return self.client.observation()

    def capture(self) -> CameraCaptures:
        deadline = time.monotonic() + TRIAL_CAPTURE_TIMEOUT_SECONDS
        while True:
            if self.timer:
                self.timer.remaining()
            try:
                captures = self.client.capture()
                for name, mount in self.robot_config.cameras.items():
                    config = mount.hardware if self.backend is Runtime.REAL else mount.simulation
                    if config is None:
                        continue
                    if name not in captures:
                        raise RuntimeError(f"Waiting for the {name} RGB-D camera.")
                    frames = captures[name][1]
                    if frames.registered is None:
                        raise RuntimeError(f"No registered RGB-D available for {name}.")
                    model = (
                        config.expected_model
                        if isinstance(config, HardwareCameraConfig)
                        else config.model
                    )
                    if frames.rgb.frame_id != config.rgb_frame or (
                        model and frames.registered.camera_model != model
                    ):
                        raise ValueError(f"Unexpected {name} camera model or optical frame.")
                    if (
                        isinstance(config, HardwareCameraConfig)
                        and config.serial_number
                        and frames.rgb.serial_number != config.serial_number
                    ):
                        raise ValueError(f"Unexpected {name} camera serial.")
                return captures
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
            if time.monotonic() >= deadline:
                raise RuntimeError("No registered RGB-D capture available.")
            time.sleep(TRIAL_CAPTURE_POLL_INTERVAL_SECONDS)

    def execute(
        self,
        operation: ControlOperation,
        arguments: Mapping[str, object],
        *,
        timeout: float = DEFAULT_MOTION_TIMEOUT_SECONDS,
    ) -> MotionResult:
        call = {"operation": operation, "arguments": dict(arguments)}
        if self.timer:
            timeout = min(timeout, self.timer.remaining())
        try:
            result = self.client.execute(operation, arguments, timeout=timeout)
        except (RuntimeError, ValueError) as exc:
            self.trace.append(
                {"command": call, "result": ToolError(operation=operation, error=str(exc))}
            )
            if self.timer:
                self.timer.remaining()
            raise
        self.trace.append({"command": call, "result": result})
        return result

    def completed_score(self) -> TaskScore | None:
        if self.score_fn is not None and self.score_fn().success:
            settled = self.score_fn(settle_seconds=0.5)
            if settled.success:
                return settled
        return None

    def final_score(self) -> TaskScore:
        if self.score_fn is not None:
            return self.score_fn(settle_seconds=0.5)
        answer = input("Did the robot complete the task? [y/n] ").strip().lower()
        while answer not in {"y", "n", "yes", "no"}:
            answer = input("Enter y or n: ").strip().lower()
        return TaskScore(answer in {"y", "yes"}, {"score_source": "operator"})


type EpisodeFactory = Callable[[int], AbstractContextManager[tuple[Episode, RobotTools]]]


@contextmanager
def open_episode(  # noqa: PLR0913 - task and backend attachment settings
    seed: int,
    *,
    task: SimulationTask,
    robot_config: RobotConfig,
    backend: Runtime,
    endpoint: ApiEndpoint,
    score_fn: EpisodeScorer | None = None,
    record: bool = False,
    recording_root: Path = DEFAULT_RECORDING_ROOT,
) -> Iterator[tuple[Episode, RobotTools]]:
    robot = Robot(robot_config)
    with ExitStack() as resources:
        if backend is Runtime.SIM:
            stack = resources.enter_context(
                closing(
                    TrialStack(
                        task,
                        seed,
                        robot_config=robot_config,
                        recording_root=recording_root if record else None,
                    )
                )
            )
            client = stack.client
            episode = Episode(
                client,
                robot_config,
                backend=backend,
                score_fn=stack.score if score_fn is None else score_fn,
            )
            episode.lcm_recording = stack.lcm_recording
            resources.callback(episode.release)
        else:
            if not any(mount.hardware for mount in robot_config.cameras.values()):
                raise ValueError(
                    f"The vision trial requires a hardware RGB-D camera on {robot_config.name}."
                )
            client = resources.enter_context(closing(RobotClient(endpoint)))
            client.wait_until_ready()
            input(
                f"Set up physical episode {seed}, home the robot, and select API. "
                "Press Enter when ready. "
            )
            episode = Episode(client, robot_config, backend=backend, score_fn=score_fn)
            resources.callback(episode.release)
            if any(
                measured.mode is not Mode.AUTONOMY for measured, _ in episode.capture().values()
            ):
                raise RuntimeError("Select API on the robot before starting the episode.")
            if record:
                episode.start_recording()
        tools = resources.enter_context(closing(RobotTools(episode, robot)))
        yield episode, tools
