"""LCM-backed native MuJoCo driver with task-defined scene resets."""

import time
from collections.abc import Callable

import mujoco
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
from humanoid.simulation.tasks.registry import TASKS
from humanoid.types.logging import LoggingState, LoggingStatus
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import RobotConfig, RobotJointCommand
from humanoid.types.simulation import (
    MujocoSimulationConfig,
    SimulationReset,
    SimulationResetStatus,
    SimulationState,
)

logger = get_logger(__name__)
DEFAULT_COMMAND_TIMEOUT_SECONDS = 0.25
SCENE_PUBLISH_RATE_HZ = 20.0
OBSERVATION_RATE_HZ = 50.0
RESET_STATUS_MAX_AGE = 2.0


class MujocoSimulationNode(Node):
    """Replace hardware with physics behind the normal command/state topics."""

    def __init__(  # noqa: PLR0913 - injectable physics, transport, and watchdog
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
        self._clock = clock
        self._configure_timing(self.engine)
        self._ticks = 0
        self._scene_interval = max(1, round(self.rate_hz / SCENE_PUBLISH_RATE_HZ))
        self._observation_interval = max(1, round(self.rate_hz / OBSERVATION_RATE_HZ))
        self._reset_id = ""
        self._reset_status: SimulationResetStatus | None = None
        self._accept_commands_after = float("-inf")
        self._applied: RobotJointCommand | None = None
        self._mode: Mode | None = None
        self._logging: LoggingStatus | None = None
        self._mode_received = self._logging_received = float("-inf")
        self._command_watchdog = VelocityCommandWatchdog(
            self._stop_velocity_actuators, command_timeout_seconds, clock
        )
        self.subscriber = Subscriber(
            [
                Topic.ROBOT_JOINT_COMMAND,
                Topic.SIMULATION_RESET,
                Topic.ORCHESTRATOR_MODE,
                Topic.LOGGING_STATUS,
            ],
            url=lcm_url,
        )
        self.publisher = Publisher(url=lcm_url)

    def _configure_timing(self, engine: NativeMujocoEngine) -> None:
        substeps = 1 / (self.rate_hz * engine.physics_timestep)
        if substeps < 1:
            raise ValueError("Physics timestep must not exceed the driver publish period.")
        self.substeps = substeps
        self._substep_remainder = 0.0

    def setup(self) -> None:
        logger.info("MuJoCo driver ready at %s Hz", self.rate_hz)

    def _read_status(self) -> None:
        mode = self.subscriber.receive(Topic.ORCHESTRATOR_MODE)
        if mode is not None:
            self._mode, self._mode_received = mode.mode, time.monotonic()
        logging = self.subscriber.receive(Topic.LOGGING_STATUS)
        if logging is not None:
            self._logging, self._logging_received = logging, time.monotonic()

    def _require_idle(self) -> None:
        self._read_status()
        if (
            self._mode is not Mode.IDLE
            or time.monotonic() - self._mode_received > RESET_STATUS_MAX_AGE
        ):
            raise ValueError("Scene resets require a connected, Idle orchestrator.")
        if (
            self._logging is None
            or self._logging.state not in {LoggingState.STOPPED, LoggingState.FAILED}
            or time.monotonic() - self._logging_received > RESET_STATUS_MAX_AGE
        ):
            raise ValueError("Scene resets require a connected, stopped logger.")

    def _reset(self, request: SimulationReset) -> None:
        if self._reset_status is None or request.request_id != self._reset_status.request_id:
            error = ""
            try:
                self._require_idle()
                task = TASKS[request.task](robot_config=self.engine.robot.config)
                if task.robot is not self.engine.robot.config.name or request.seed < 0:
                    raise ValueError("The scene must match the running robot and use a valid seed.")
                engine = task.create_engine(request.seed)
                self._require_idle()  # Model compilation may outlast an operator mode change.
                self._configure_timing(engine)
                self.stop()
                self.engine = engine
                self._reset_id = request.request_id
                self._accept_commands_after = self._clock()
                self._applied = RobotJointCommand(
                    self._accept_commands_after,
                    engine.read_robot_state(self._accept_commands_after).joint_positions,
                    np.zeros(engine.robot.model.nv),
                )
                while self.subscriber.receive(Topic.ROBOT_JOINT_COMMAND) is not None:
                    pass
                self.publisher.publish(
                    engine.read_scene_state(self._clock()), Topic.SIMULATION_SCENE
                )
            except (KeyError, ValueError, RuntimeError) as exc:
                error = str(exc)
            self._reset_status = SimulationResetStatus(self._clock(), request.request_id, error)
        self.publisher.publish(self._reset_status, Topic.SIMULATION_RESET_STATUS)

    def step(self) -> None:
        self._read_status()
        request = self.subscriber.receive(Topic.SIMULATION_RESET)
        if request is not None:
            self._reset(request)
        command = self.subscriber.receive(Topic.ROBOT_JOINT_COMMAND)
        if command is not None and command.timestamp >= self._accept_commands_after:
            normalized = self.engine.apply_joint_command(command)
            self._applied = RobotJointCommand(
                self._clock(), normalized.joint_positions.copy(), normalized.joint_velocities.copy()
            )
            self._command_watchdog.observe_command(
                velocity_active=bool(normalized.joint_velocities.any())
            )
        elif self._command_watchdog.stop_if_stale():
            logger.warning("Robot command watchdog stopped simulated velocity actuators")
        # Keep the node's normal rate when a task changes the physics timestep.
        self._substep_remainder += self.substeps
        count = int(self._substep_remainder + 1e-9)
        self._substep_remainder -= count
        self.engine.step(count)
        state = self.engine.read_robot_state(timestamp=self._clock())
        self.publisher.publish(state, topic=Topic.ROBOT_STATE)
        self._ticks += 1
        # A fresh driver has no task scene. During replay, the log owns this
        # channel; broadcasting empty scenes here would erase its geometry.
        if self._reset_id and self._ticks % self._scene_interval == 0:
            mujoco.mj_forward(self.engine.model, self.engine.data)
            self.publisher.publish(
                self.engine.read_scene_state(state.timestamp), Topic.SIMULATION_SCENE
            )
        if self._reset_id and self._ticks % self._observation_interval == 0:
            assert self._applied is not None
            self.publisher.publish(
                SimulationState(
                    state.timestamp,
                    self._reset_id,
                    self.engine.data.time,
                    self.engine.data.qpos.copy(),
                    self.engine.data.qvel.copy(),
                    state,
                    self._applied,
                    "MuJoCo reported a physics warning."
                    if any(w.number for w in self.engine.data.warning)
                    else "",
                ),
                Topic.SIMULATION_STATE,
            )

    def _stop_velocity_actuators(self) -> None:
        self.engine.stop_velocity_actuators()
        if self._applied is not None:
            self._applied = RobotJointCommand(
                self._clock(),
                self._applied.joint_positions,
                np.zeros(self.engine.robot.model.nv),
            )

    def stop(self) -> None:
        self._command_watchdog.stop()

    def on_close(self) -> None:
        try:
            self.stop()
        finally:
            self.subscriber.close()


def main() -> None:
    MujocoSimulationNode().run()


if __name__ == "__main__":
    main()
