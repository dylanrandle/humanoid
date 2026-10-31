"""Task extension point: reset, expert execution, privileged labels, success."""

from abc import ABC, abstractmethod

from humanoid.config import ROBOT_CONFIG, ROBOT_CONFIGS
from humanoid.simulation.client import SimulationClient
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.types.camera import RGBCameraConfig
from humanoid.types.collection import TaskResult
from humanoid.types.object_state import ObjectState
from humanoid.types.robot import RobotConfig, RobotName


class SimulationTask(ABC):
    name: str
    version: int
    label: str
    robot: RobotName | None = None
    collectable: bool = True

    def __init__(self, robot_config: RobotConfig | None = None):
        if robot_config is None:
            robot_config = ROBOT_CONFIGS[self.robot] if self.robot is not None else ROBOT_CONFIG
        if self.robot is not None and self.robot is not robot_config.name:
            raise ValueError(f"{self.label} requires the {self.robot.value} robot.")
        self.robot_config = robot_config
        self.robot = robot_config.name

    @abstractmethod
    def create_engine(self, seed: int) -> NativeMujocoEngine:
        """Build the task's complete scene and robot for preview or an episode.

        Scene composition and seeded randomization belong only here. The
        generic engine does not select or add task geometry.
        """

    @abstractmethod
    def cameras(self) -> dict[str, RGBCameraConfig]: ...

    @abstractmethod
    def objects(self, engine: NativeMujocoEngine) -> tuple[ObjectState, ...]: ...

    @abstractmethod
    def metadata(self) -> dict[str, object]:
        """JSON-serializable task settings needed to reproduce the episode."""

    @abstractmethod
    def state(self, runtime: SimulationClient) -> dict[str, object]:
        """Privileged task progress, goals, and phase labels for this instant."""

    @abstractmethod
    def run(self, runtime: SimulationClient) -> TaskResult: ...

    @abstractmethod
    def success(self, engine: NativeMujocoEngine) -> bool:
        """Independent final-state validation, also checked by the collector."""

    def observation_errors(self, runtime: SimulationClient) -> tuple[str, ...]:
        """Optional task-specific observation requirements, checked at each RGB sample."""
        return ()
