"""Task catalog, independent of agent launchers and robot control."""

from humanoid.tasks.toy_car_cleanup import ToyCarCleanupTask
from humanoid.types.tasks import SimulationTask


def make_task(name: str) -> SimulationTask:
    factories = {"toy_car_cleanup": ToyCarCleanupTask}
    if name not in factories:
        raise ValueError(f"Unknown task {name!r}; choose from {tuple(factories)}.")
    return factories[name]()
