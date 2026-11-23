"""Built-in tasks available to the simulated agent trial CLI."""

from argparse import ArgumentParser, Namespace
from collections.abc import Callable

from humanoid.simulation.tasks import toy_cleanup
from humanoid.types.trial import SimulationTask

TASKS: dict[str, Callable[[ArgumentParser], Callable[[Namespace], SimulationTask]]] = {
    "toy_cleanup": toy_cleanup.add_arguments
}
