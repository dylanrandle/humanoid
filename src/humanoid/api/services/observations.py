"""Robot state and synchronized camera captures, independent of motion execution."""

import threading
import time
from collections import deque

from humanoid.config.observations import OBSERVATION_CONFIG, get_observation_subscriptions
from humanoid.config.selection import ROBOT_CONFIG
from humanoid.constants import DEFAULT_LCM_URL
from humanoid.environment.realtime import ActionTopics, RealtimeEnvironment
from humanoid.types.api import ServiceConflict
from humanoid.types.camera import CameraFrames, DepthView
from humanoid.types.observation import CameraFrameCaptures, Observation, ObservationConfig
from humanoid.types.robot import RobotConfig, RobotState


class ObservationService:
    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        *,
        lcm_url: str = DEFAULT_LCM_URL,
        config: ObservationConfig = OBSERVATION_CONFIG,
        collect: bool = True,
    ) -> None:
        self.config = config
        self.feedback_timeout = config.feedback_timeout
        self._primary = next(iter(robot_config.cameras), "")
        self._lock = threading.Lock()
        self._observation: Observation | None = None
        self._state_history: deque[RobotState] = deque(maxlen=128)
        self._registered: dict[str, CameraFrames] = {}
        self._closed = False
        self._stop = threading.Event()
        self._env = (
            RealtimeEnvironment(
                ActionTopics(),
                lcm_url=lcm_url,
                subscriptions=get_observation_subscriptions(robot_config),
            )
            if collect
            else None
        )
        self._worker = threading.Thread(target=self._collect, daemon=True, name="observations")
        if collect:
            self._worker.start()

    def _collect(self) -> None:
        assert self._env is not None
        while not self._stop.wait(0.01):
            try:
                self.observe(self._env.observe())
            except RuntimeError:
                continue  # The API can run before any robot feedback exists.

    def close(self) -> None:
        self._stop.set()
        if self._worker.is_alive():
            self._worker.join(timeout=1)
        if self._env is not None:
            self._env.close()
        with self._lock:
            self._closed = True

    def observe(self, observation: Observation) -> None:
        with self._lock:
            self._observation = observation
            state = observation.robot_state
            if not self._state_history or state.timestamp > self._state_history[-1].timestamp:
                self._state_history.append(state)
            for name, frames in observation.camera_frames.items():
                if frames.registered is not None:
                    self._registered[name] = frames

    def observation(self, *, camera: bool = False) -> Observation:
        with self._lock:
            observation = self._observation
            if self._closed or observation is None:
                raise ServiceConflict("Robot control service is not ready.")
        age = time.perf_counter() - observation.robot_state.timestamp
        if not 0 <= age <= self.feedback_timeout:
            raise ServiceConflict("Robot feedback is stale.")
        if camera:
            frames = observation.camera_frames.get(self._primary)
            if frames is None or any(
                not 0 <= time.perf_counter() - frame.timestamp <= self.config.camera_timeout_seconds
                for frame in (frames.rgb, frames.depth)
            ):
                raise ServiceConflict("No fresh paired RGB/depth observation is available.")
        return observation

    def camera_frames(self, *, depth_view: DepthView) -> CameraFrames:
        frames = self.observation(camera=True).camera_frames[self._primary]
        assert frames is not None
        with self._lock:
            registered = self._registered.get(self._primary)
        if (
            depth_view is not DepthView.NATIVE
            and registered
            and registered.rgb.serial_number == frames.rgb.serial_number
            and 0
            <= time.perf_counter() - registered.rgb.timestamp
            <= self.config.camera_timeout_seconds
        ):
            return registered
        return frames

    def capture(self) -> CameraFrameCaptures:
        """Match each live camera to measured joint feedback on the host clock."""
        self.observation()
        with self._lock:
            current = self._observation
            if current is None or self._closed:
                raise ServiceConflict("Robot control service is not ready.")
            now = time.perf_counter()
            result = {}
            for name, latest in current.camera_frames.items():
                frames = latest
                registered = self._registered.get(name)
                if (
                    registered
                    and registered.rgb.serial_number == frames.rgb.serial_number
                    and 0 <= now - registered.rgb.timestamp <= self.config.camera_timeout_seconds
                ):
                    frames = registered
                stamp = frames.rgb.timestamp
                if not 0 <= now - stamp <= self.config.camera_timeout_seconds:
                    continue
                state = min(self._state_history, key=lambda sample: abs(sample.timestamp - stamp))
                if abs(state.timestamp - stamp) <= self.config.capture_state_tolerance:
                    result[name] = Observation(robot_state=state, mode=current.mode), frames
            if not result:
                raise ServiceConflict("No fresh camera capture with matching joint feedback.")
            return result
