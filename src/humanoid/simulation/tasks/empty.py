"""The original robot-only scene for teleoperation and manual recording."""

from humanoid.simulation.client import SimulationClient
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.simulation.tasks.base import SimulationTask
from humanoid.types.camera import RGBCameraConfig
from humanoid.types.collection import TaskResult
from humanoid.types.object_state import ObjectState


class EmptySceneTask(SimulationTask):
    name = "empty"
    version = 1
    label = "Empty scene"
    collectable = False

    def create_engine(self, seed: int) -> NativeMujocoEngine:
        return NativeMujocoEngine(self.robot_config)

    def cameras(self) -> dict[str, RGBCameraConfig]:
        return {}

    def objects(self, engine: NativeMujocoEngine) -> tuple[ObjectState, ...]:
        return ()

    def metadata(self) -> dict[str, object]:
        return {}

    def state(self, runtime: SimulationClient) -> dict[str, object]:
        return {}

    def run(self, runtime: SimulationClient) -> TaskResult:
        return TaskResult(False, "Empty scene has no automated collection task.")

    def success(self, engine: NativeMujocoEngine) -> bool:
        return False
