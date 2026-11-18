"""LCM-backed native MuJoCo digital-twin node."""

import queue
import time
from collections.abc import Callable
from multiprocessing.queues import Queue

import numpy as np

from humanoid.config import ROBOT_CONFIG
from humanoid.config.simulation import DEFAULT_MUJOCO_SIMULATION_CONFIG
from humanoid.constants import DEFAULT_LCM_URL, Topic
from humanoid.logger import get_logger
from humanoid.middleware.publisher import Publisher
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.base import Node
from humanoid.robots.watchdog import VelocityCommandWatchdog
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.tasks import make_task
from humanoid.types.robot import RobotConfig
from humanoid.types.simulation import MujocoSimulationConfig
from humanoid.types.tasks import SimulationTaskSelection, TaskEvaluation

logger = get_logger(__name__)

DEFAULT_COMMAND_TIMEOUT_SECONDS = 0.25
SUBSTEP_INTEGRAL_TOLERANCE = 1e-9


class SimulationNode(Node):
    """Replace the hardware driver with native physics behind the same LCM topics."""

    def __init__(  # noqa: PLR0913 - physics, transport, and timing are independently injectable
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        simulation_config: MujocoSimulationConfig = DEFAULT_MUJOCO_SIMULATION_CONFIG,
        engine: NativeMujocoEngine | None = None,
        command_timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.perf_counter,
        lcm_url: str = DEFAULT_LCM_URL,
        task_selection: SimulationTaskSelection | None = None,
        evaluation_queue: Queue[TaskEvaluation] | None = None,
    ) -> None:
        if engine is not None and task_selection is not None:
            raise ValueError("Task scenes must be built by SimulationNode, not an injected engine.")
        self.task = make_task(task_selection.name) if task_selection is not None else None
        self._evaluation_queue = evaluation_queue
        self.engine = engine or NativeMujocoEngine(
            robot_config,
            simulation_config,
            configure_scene=None if self.task is None else self.task.build_scene,
        )
        if self.task is not None and task_selection is not None:
            seed = int(
                np.random.SeedSequence(
                    [int(task_selection.formal), task_selection.seed]
                ).generate_state(1)[0]
            )
            self.task.reset(self.engine.model, self.engine.data, seed)
        self.rate_hz = simulation_config.publish_rate_hz
        substeps = 1.0 / (self.rate_hz * self.engine.physics_timestep)
        rounded_substeps = round(substeps)
        if (
            rounded_substeps <= 0
            or not abs(substeps - rounded_substeps) < SUBSTEP_INTEGRAL_TOLERANCE
        ):
            raise ValueError(
                "MuJoCo publish period must be an integer multiple of the physics timestep."
            )
        self.substeps = rounded_substeps
        self._clock = clock
        self._command_watchdog = VelocityCommandWatchdog(
            self._stop_velocity_actuators,
            command_timeout_seconds,
            clock,
        )
        self.subscriber = Subscriber(topics=[Topic.ROBOT_JOINT_COMMAND], url=lcm_url)
        self.publisher = Publisher(url=lcm_url)

    def setup(self) -> None:
        logger.info(
            "MuJoCo digital twin ready: %s joints, %.4f s timestep, %s substeps",
            len(self.engine.binding.joints),
            self.engine.physics_timestep,
            self.substeps,
        )

    def step(self) -> None:
        command = self.subscriber.receive(Topic.ROBOT_JOINT_COMMAND)
        if command is not None:
            normalized = self.engine.apply_joint_command(command)
            self._command_watchdog.observe_command(
                velocity_active=bool(normalized.joint_velocities.any())
            )
        elif self._command_watchdog.stop_if_stale():
            logger.warning("Robot command watchdog stopped simulated velocity actuators")

        if self.task is None:
            self.engine.step(self.substeps)
        else:
            for _ in range(self.substeps):
                self.engine.step()
                evaluation = self.task.evaluate(self.engine.model, self.engine.data)
            self._publish_evaluation(evaluation)
        robot_state = self.engine.read_robot_state(timestamp=self._clock())
        self.publisher.publish(robot_state, topic=Topic.ROBOT_STATE)
        frames = self.engine.read_camera_frames(timestamp=robot_state.timestamp)
        if frames is not None:
            self.publisher.publish(frames.rgb, topic=Topic.CAMERA_RGB)
            self.publisher.publish(frames.depth, topic=Topic.CAMERA_DEPTH)

    def _publish_evaluation(self, evaluation: TaskEvaluation) -> None:
        """Private operator IPC; object truth never enters robot observations."""
        if self._evaluation_queue is None:
            return
        try:
            self._evaluation_queue.put_nowait(evaluation)
        except queue.Full:
            try:
                self._evaluation_queue.get_nowait()
                self._evaluation_queue.put_nowait(evaluation)
            except (queue.Empty, queue.Full):
                pass

    def _stop_velocity_actuators(self) -> None:
        self.engine.stop_velocity_actuators()

    def on_close(self) -> None:
        try:
            self._command_watchdog.stop()
        finally:
            try:
                self.engine.close()
            finally:
                self.subscriber.close()


def main() -> None:
    SimulationNode().run()


if __name__ == "__main__":
    main()
