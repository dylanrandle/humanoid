"""LCM-backed native MuJoCo digital-twin node."""

import math
import time
from collections.abc import Callable
from contextlib import ExitStack

from humanoid.config import ROBOT_CONFIG
from humanoid.config.simulation import DEFAULT_MUJOCO_SIMULATION_CONFIG
from humanoid.constants import DEFAULT_LCM_URL, Topic
from humanoid.logger import get_logger
from humanoid.middleware.publisher import Publisher
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.base import Node
from humanoid.robots.watchdog import VelocityCommandWatchdog
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.simulation.render_stream import SimulationRenderStream, publish_camera
from humanoid.types.robot import RobotConfig
from humanoid.types.simulation import MujocoSimulationConfig

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
    ) -> None:
        self.engine = engine or NativeMujocoEngine(robot_config, simulation_config)
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
        self._viewer_interval = max(1, math.ceil(self.rate_hz / 30))
        self._viewer_ticks = 0
        self._clock = clock
        self._command_watchdog = VelocityCommandWatchdog(
            self._stop_velocity_actuators,
            command_timeout_seconds,
            clock,
        )
        self.subscriber = Subscriber(topics=[Topic.ROBOT_JOINT_COMMAND], url=lcm_url)
        self.publisher = Publisher(url=lcm_url)
        self._render_streams: list[SimulationRenderStream] = []
        with ExitStack() as startup:
            self._streams = startup
            for name, camera in self.engine.cameras.items():
                self.add_render_stream(
                    SimulationRenderStream(
                        name,
                        camera.model,
                        camera.config.fps,
                        publish_camera(name, camera, lcm_url),
                    )
                )
            self._streams = startup.pop_all()

    def add_render_stream(self, stream: SimulationRenderStream) -> None:
        self._streams.callback(stream.close)
        self._render_streams.append(stream)

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

        self.engine.step(self.substeps)
        robot_state = self.engine.read_robot_state(timestamp=self._clock())
        self.publisher.publish(robot_state, topic=Topic.ROBOT_STATE)
        self._viewer_ticks += 1
        due = [stream for stream in self._render_streams if stream.clock.due(self.engine.data.time)]
        if due or self._viewer_ticks >= self._viewer_interval:
            state = self.engine.simulation_state()
            for stream in due:
                stream.submit(state, robot_state.timestamp)
            if self._viewer_ticks >= self._viewer_interval:
                self.publisher.publish(state, topic=Topic.SIMULATION_STATE)
                self._viewer_ticks = 0

    def _stop_velocity_actuators(self) -> None:
        self.engine.stop_velocity_actuators()

    def on_close(self) -> None:
        with ExitStack() as cleanup:
            cleanup.callback(self.subscriber.close)
            cleanup.callback(self.engine.close)
            cleanup.callback(self._streams.close)
            self._command_watchdog.stop()


def main() -> None:
    SimulationNode().run()


if __name__ == "__main__":
    main()
