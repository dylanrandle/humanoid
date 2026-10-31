"""Register scene and collection tasks for the operator console."""

from humanoid.simulation.tasks.base import SimulationTask
from humanoid.simulation.tasks.cleanup.task import CleanupTask
from humanoid.simulation.tasks.empty import EmptySceneTask

TASKS: dict[str, type[SimulationTask]] = {"empty": EmptySceneTask, "cleanup": CleanupTask}
