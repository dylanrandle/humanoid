"""LCM-backed native MuJoCo digital-twin node."""

import time
from collections.abc import Callable

from humanoid.config import ROBOT_CONFIG
from humanoid.config.simulation import DEFAULT_MUJOCO_SIMULATION_CONFIG
from humanoid.constants import DEFAULT_LCM_URL, Topic
from humanoid.logger import get_logger
from humanoid.middleware.publisher import Publisher
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.base import Node
from humanoid.robots.watchdog import VelocityCommandWatchdog
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.types.robot import RobotConfig
from humanoid.types.simulation import MujocoScene, MujocoSimulationConfig
from humanoid.visualizers.mujoco import (
    MujocoSimulationVisualizer,
    relaunch_with_mjpython_on_macos,
)

logger = get_logger(__name__)

DEFAULT_COMMAND_TIMEOUT_SECONDS = 0.25
SUBSTEP_INTEGRAL_TOLERANCE = 1e-9


class MujocoSimulationNode(Node):
    """Replace the hardware driver with native physics behind the same LCM topics."""

    def __init__(  # noqa: PLR0913 - physics, transport, and timing are independently injectable
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        simulation_config: MujocoSimulationConfig = DEFAULT_MUJOCO_SIMULATION_CONFIG,
        engine: NativeMujocoEngine | None = None,
        visualizer: MujocoSimulationVisualizer | None = None,
        scene: MujocoScene | None = None,
        command_timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.perf_counter,
        lcm_url: str = DEFAULT_LCM_URL,
    ) -> None:
        self.scene = scene if scene is not None else MujocoScene.from_environment()
        self.engine = engine or NativeMujocoEngine(
            robot_config,
            simulation_config,
            scene=self.scene,
        )
        self.visualizer = visualizer
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
        if self.visualizer is None:
            self.visualizer = MujocoSimulationVisualizer(self.engine.model, self.engine.data)
        self.visualizer.initialize()
        logger.info(
            "MuJoCo digital twin ready: %s scene, %s joints, %.4f s timestep, %s substeps",
            self.scene,
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

        self.engine.step(self.substeps)
        if self.visualizer is not None:
            self.visualizer.sync()
        robot_state = self.engine.read_robot_state(timestamp=self._clock())
        self.publisher.publish(robot_state, topic=Topic.ROBOT_STATE)

    def _stop_velocity_actuators(self) -> None:
        self.engine.stop_velocity_actuators()

    def on_close(self) -> None:
        try:
            self._command_watchdog.stop()
        finally:
            try:
                if self.visualizer is not None:
                    self.visualizer.close()
            finally:
                self.subscriber.close()

    @classmethod
    def main(cls, *args, **kwargs) -> None:
        relaunch_with_mjpython_on_macos(__name__)
        super().main(*args, **kwargs)


def main() -> None:
    MujocoSimulationNode.main()


if __name__ == "__main__":
    main()
