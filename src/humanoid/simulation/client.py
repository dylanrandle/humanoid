"""Collection client for the ordinary, independently running simulation stack."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import pinocchio as pin

from humanoid.config import ROBOT_CONFIG
from humanoid.config.collection import DEFAULT_COLLECTION_CONFIG
from humanoid.constants import DEFAULT_LCM_URL, Topic
from humanoid.middleware.publisher import Publisher
from humanoid.middleware.subscriber import Subscriber
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.types.collection import CollectionCancelled, CollectionConfig
from humanoid.types.logging import LoggingState, LoggingStatus
from humanoid.types.orchestrator import EventKind, Mode, OrchestratorEvent
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotConfig,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)
from humanoid.types.simulation import SimulationReset, SimulationState
from humanoid.utils.loop import next_deadline, sleep_until

if TYPE_CHECKING:
    from humanoid.simulation.tasks.base import SimulationTask

POLL_SECONDS = 0.01
RETRY_SECONDS = 0.25
STATUS_MAX_AGE = 2.0


class SimulationClient:
    """Publish policy targets and observe the simulator; never step physics or nodes."""

    def __init__(self, robot_config: RobotConfig = ROBOT_CONFIG, lcm_url: str = DEFAULT_LCM_URL):
        self.robot_config = robot_config
        self.robot_name = robot_config.name
        self.publisher = Publisher(url=lcm_url)
        self.subscriber = Subscriber(
            [
                Topic.SIMULATION_STATE,
                Topic.SIMULATION_RESET_STATUS,
                Topic.ORCHESTRATOR_MODE,
                Topic.LOGGING_STATUS,
            ],
            url=lcm_url,
        )
        self.config = DEFAULT_COLLECTION_CONFIG
        self.engine: NativeMujocoEngine
        self.snapshot: SimulationState | None = None
        self.mode: Mode | None = None
        self.logging_status: LoggingStatus | None = None
        self.before_step: Callable[[], None] | None = None
        self.on_tick: Callable[[], None] | None = None
        self.observe: Callable[[RobotState, RobotJointCommand, int], None] | None = None
        self._reset_id = ""
        self._mode_received = self._snapshot_received = float("-inf")
        self._origin = 0.0
        self._next_command = 0.0
        self._base_reference: pin.SE3 | None = None
        self.action_count = 0
        self.phase, self.target = "reset", ""

    @property
    def period(self) -> float:
        return 1 / self.config.control_hz

    @property
    def time(self) -> float:
        return max(0.0, self.snapshot.sim_seconds - self._origin) if self.snapshot else 0.0

    def state(self) -> RobotState:
        if self.snapshot is None:
            raise RuntimeError("No observation from the selected scene.")
        # Experts may edit an IK seed derived from this state. Keep the received
        # observation intact for subsequent planning and recording.
        return deepcopy(self.snapshot.state)

    def poll(self) -> None:
        mode = self.subscriber.receive(Topic.ORCHESTRATOR_MODE)
        if mode is not None:
            self.mode, self._mode_received = mode.mode, time.monotonic()
        status = self.subscriber.receive(Topic.LOGGING_STATUS)
        if status is not None:
            self.logging_status = status
        snapshot = self.subscriber.receive(Topic.SIMULATION_STATE)
        if snapshot is not None and self._reset_id:
            if snapshot.reset_id != self._reset_id:
                # Old packets can arrive while a reset is being acknowledged.
                return
            engine = self.engine
            if (
                snapshot.qpos.shape != engine.data.qpos.shape
                or snapshot.qvel.shape != engine.data.qvel.shape
            ):
                raise ValueError("The simulator scene does not match the selected task.")
            engine.data.qpos[:] = snapshot.qpos
            engine.data.qvel[:] = snapshot.qvel
            engine.data.time = snapshot.sim_seconds
            mujoco.mj_forward(engine.model, engine.data)
            self.snapshot, self._snapshot_received = snapshot, time.monotonic()

    def _check_cancelled(self) -> None:
        if self.before_step is not None:
            self.before_step()

    def require_idle(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout
        while True:
            self._check_cancelled()
            self.poll()
            if (
                time.monotonic() - self._mode_received <= STATUS_MAX_AGE
                and self.logging_status is not None
                and time.time() - self.logging_status.timestamp <= STATUS_MAX_AGE
            ):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("The stack's orchestrator or logger is unavailable.")
            time.sleep(POLL_SECONDS)
        if self.mode is not Mode.IDLE:
            raise ValueError("Select Idle before loading a task scene or starting collection.")
        if self.logging_status.state not in {LoggingState.STOPPED, LoggingState.FAILED}:
            raise ValueError("Stop manual recording before loading a task scene or collecting.")

    def reset(self, task: SimulationTask, seed: int, config: CollectionConfig) -> None:
        self.config = config
        self.require_idle()
        if task.robot is not self.robot_name:
            raise ValueError("The task must match the robot selected in the running stack.")
        # This copy is only for FK, privileged geometry queries, and rendering.
        # Every qpos/qvel update comes from the driver; no local physics is run.
        self.engine = task.create_engine(seed)
        self._reset_id = str(uuid.uuid4())
        self.snapshot = None
        self.action_count = 0
        self.phase, self.target = "reset", ""
        request = SimulationReset(time.perf_counter(), self._reset_id, task.name, seed)
        deadline = time.monotonic() + config.startup_timeout
        next_retry = 0.0
        acknowledged = False
        while time.monotonic() < deadline:
            self._check_cancelled()
            if time.monotonic() >= next_retry:
                self.publisher.publish(request, Topic.SIMULATION_RESET)
                next_retry = time.monotonic() + RETRY_SECONDS
            self.poll()
            status = self.subscriber.receive(Topic.SIMULATION_RESET_STATUS)
            if status is not None and status.request_id == request.request_id:
                if status.error:
                    raise RuntimeError(status.error)
                acknowledged = True
            if acknowledged and self.snapshot is not None:
                self._origin = self.snapshot.sim_seconds
                self._base_reference = self.engine.robot.get_base_pose(self.state().joint_positions)
                if self._base_reference is not None:
                    self._base_reference = self._base_reference.copy()
                return
            time.sleep(POLL_SECONDS)
        raise TimeoutError("The simulator did not acknowledge the scene reset.")

    def activate(self) -> None:
        self._request_mode(EventKind.REQUEST_SYSTEM, Mode.SYSTEM, self.config.startup_timeout)
        assert self.snapshot is not None
        self._origin = self.snapshot.sim_seconds
        self._next_command = time.perf_counter()

    def _request_mode(self, event: EventKind, mode: Mode, timeout: float) -> None:
        requested = time.monotonic()
        deadline, next_retry = requested + timeout, 0.0
        while time.monotonic() < deadline:
            if mode is Mode.SYSTEM:
                self._check_cancelled()
            if time.monotonic() >= next_retry:
                self.publisher.publish(
                    OrchestratorEvent(time.perf_counter(), event), Topic.ORCHESTRATOR_EVENT
                )
                next_retry = time.monotonic() + RETRY_SECONDS
            self.poll()
            if self.mode is mode and self._mode_received > requested:
                return
            time.sleep(POLL_SECONDS)
        raise TimeoutError(f"The orchestrator did not enter {mode.value} mode.")

    def step(self, positions: np.ndarray, velocities: np.ndarray | None = None) -> None:
        self._check_cancelled()
        sleep_until(self._next_command)
        self.poll()
        if min(self._snapshot_received, self._mode_received) < time.monotonic() - STATUS_MAX_AGE:
            raise TimeoutError("The simulator or orchestrator stopped publishing.")
        if self.mode is not Mode.SYSTEM:
            raise CollectionCancelled("Collection mode was interrupted.")
        if self.snapshot is not None and self.snapshot.error:
            raise RuntimeError(self.snapshot.error)
        if self.time >= self.config.max_sim_seconds:
            raise TimeoutError("Episode simulation time budget exhausted.")
        robot = self.engine.robot
        timestamp = time.perf_counter()
        self.publisher.publish(
            RobotToolCommand(
                timestamp,
                robot.get_tool_command_pose(positions).copy(),
                positions[robot.get_gripper_position_indices()].copy(),
            ),
            Topic.SYSTEM_TOOL_COMMAND,
        )
        if self._base_reference is not None:
            root = robot.get_root_v_slice()
            assert root is not None
            if velocities is not None:
                pose = robot.get_base_pose(self.state().joint_positions)
                assert pose is not None
                velocity = velocities[root]
                self._base_reference.translation += (
                    pose.rotation @ np.r_[velocity[:2], 0.0] * self.period
                )
                self._base_reference.rotation = (
                    pin.exp3(np.array([0.0, 0.0, velocity[2] * self.period]))
                    @ self._base_reference.rotation
                )
            self.publisher.publish(
                RobotBaseCommand(timestamp, self._base_reference.copy()), Topic.SYSTEM_BASE_COMMAND
            )
        self.action_count += 1
        if self.observe is not None:
            assert self.snapshot is not None
            self.observe(self.state(), self.snapshot.command, self.action_count)
        if self.on_tick is not None:
            self.on_tick()
        self._next_command = next_deadline(self._next_command, self.period, time.perf_counter())

    def stop(self) -> None:
        self._request_mode(EventKind.REQUEST_IDLE, Mode.IDLE, self.config.finalization_timeout)

    def close(self) -> None:
        self.subscriber.close()
