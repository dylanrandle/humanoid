"""Execute agent motion commands on the normal policy loop."""

import copy
import math
import threading
import time
from concurrent.futures import Future

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray

from humanoid.config.agent import get_agent_policy_config
from humanoid.policy.base import Policy
from humanoid.robots.base import Robot
from humanoid.types.action import Action
from humanoid.types.agent import (
    AgentMotion,
    AgentPolicyConfig,
    CommandCancelledError,
    CommandExpiredError,
    MotionCommand,
    MotionResult,
    PendingCommand,
)
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode


class AgentPolicy(Policy):
    """Turn bounded agent commands into Actions without waiting on callers.

    submit() and cancel() may be called by an API thread. Validation and motion
    execution use the next observation on the policy loop; the transport never
    publishes robot commands. Only one pending or active command is accepted.
    """

    mode = Mode.AUTONOMY

    def __init__(self, robot: Robot, *, config: AgentPolicyConfig | None = None) -> None:
        self._lock = threading.RLock()
        self._pending_command: PendingCommand | None = None
        self._active_result: Future[MotionResult] | None = None
        self.robot = robot
        self.config = config or get_agent_policy_config()
        bounds = np.asarray(robot.get_gripper_limits()).reshape(-1, 2)
        self.gripper_speed = (bounds[:, 1] - bounds[:, 0]) / self.config.gripper_close_time
        self.step_id = 0
        self._motion: AgentMotion | None = None
        self._hold_pending = True
        self._gripper_target: NDArray[np.float64] | None = None
        self._hold_tool: pin.SE3 | None = None

    def submit(
        self, command: MotionCommand, *, deadline: float | None = None
    ) -> Future[MotionResult]:
        """Copy and queue a command; validate it against the next observation."""
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("Command deadline must be finite.")
        command = copy.deepcopy(command)
        with self._lock:
            command_pending = self._pending_command is not None
            command_running = self._active_result is not None
            if command_pending or command_running:
                raise RuntimeError("A policy command is already running.")
            result: Future[MotionResult] = Future()
            self._pending_command = PendingCommand(command, result, deadline)
            return result

    def step(self, observation: Observation) -> Action:
        with self._lock:
            try:
                self.validate_observation(observation)
            except (ValueError, RuntimeError) as error:
                self._interrupt(error)
                return Action()
            self._start_pending_command(observation)
            if self._active_result is None:
                return self.idle_action(observation)
            try:
                action, result = self.advance_command(observation)
            except (ValueError, RuntimeError) as error:
                self._interrupt(error)
                return Action()
            if result is not None:
                future = self._active_result
                self._active_result = None
                future.set_result(result)
            return action

    def _start_pending_command(self, observation: Observation) -> None:
        pending = self._pending_command
        self._pending_command = None
        if pending is None or not pending.result.set_running_or_notify_cancel():
            return
        try:
            command_expired = (
                pending.deadline is not None and time.perf_counter() >= pending.deadline
            )
            if command_expired:
                raise CommandExpiredError("Policy command expired before execution.")
            self.start_command(pending.command, observation)
        except (ValueError, RuntimeError) as error:
            pending.result.set_exception(error)
        else:
            self._active_result = pending.result

    def _interrupt(self, error: Exception) -> None:
        pending, active = self._pending_command, self._active_result
        self._pending_command = None
        self._active_result = None
        self.reset_command()
        if pending is not None and pending.result.set_running_or_notify_cancel():
            pending.result.set_exception(error)
        if active is not None:
            active.set_exception(error)

    def reset(self) -> None:
        with self._lock:
            self._interrupt(CommandCancelledError("Policy command was interrupted."))

    def cancel(self) -> None:
        """Interrupt movement while preserving the current gripper squeeze target."""
        with self._lock:
            gripper = self._gripper_target
            self._interrupt(CommandCancelledError("Policy command was cancelled."))
            self._gripper_target = gripper

    def validate_observation(self, observation: Observation) -> None:
        if observation.mode is not self.mode:
            raise RuntimeError("Select this policy mode in the UI before issuing motion commands.")
        q = observation.robot_state.joint_positions
        if q.shape != (self.robot.model.nq,) or not np.isfinite(q).all():
            raise RuntimeError("Joint feedback does not match the configured robot.")
        age = time.perf_counter() - observation.robot_state.timestamp
        if not 0 <= age <= self.config.feedback_timeout:
            raise RuntimeError("Robot feedback is stale; no motion is permitted.")

    def start_command(self, command: MotionCommand, observation: Observation) -> None:
        duration = command.duration
        if (
            type(duration) not in (int, float)
            or not math.isfinite(duration)
            or not 0 < duration <= self.config.max_duration
        ):
            raise ValueError(f"duration must be in (0, {self.config.max_duration}] seconds.")
        initial, target, base_target, gripper = self._targets(command, observation)
        self._motion = AgentMotion(
            duration=duration,
            started_at=time.perf_counter(),
            initial=initial,
            target=target,
            base_target=base_target,
            gripper=gripper,
            initial_gripper=self._grip_reference(observation),
        )

    def advance_command(self, observation: Observation) -> tuple[Action, MotionResult | None]:
        motion = self._motion
        assert motion is not None
        elapsed = time.perf_counter() - motion.started_at
        grip_time = min(elapsed, motion.duration)
        self._gripper_target = motion.initial_gripper + np.clip(
            motion.gripper - motion.initial_gripper,
            -grip_time * self.gripper_speed,
            grip_time * self.gripper_speed,
        )
        self.step_id += 1
        if elapsed >= motion.duration:
            result = self._result(
                observation.robot_state.joint_positions,
                motion,
                elapsed,
            )
            self._motion = None
            return self._hold(observation), result
        alpha = elapsed / motion.duration
        displacement = np.linalg.norm(motion.target.translation - motion.initial.translation)
        rotation = np.linalg.norm(pin.log3(motion.initial.rotation.T @ motion.target.rotation))
        velocity = self.robot.config.tool.velocity_limits
        if displacement > 0:
            alpha = min(alpha, elapsed * velocity.linear / displacement)
        if rotation > 0:
            alpha = min(alpha, elapsed * velocity.angular / rotation)
        tool = pin.SE3(
            motion.initial.rotation
            @ pin.exp3(alpha * pin.log3(motion.initial.rotation.T @ motion.target.rotation)),
            motion.initial.translation
            + alpha * (motion.target.translation - motion.initial.translation),
        )
        measured_base = self.robot.get_base_pose(observation.robot_state.joint_positions)
        base = (
            self._bounded_base_target(measured_base, motion.base_target)
            if measured_base is not None and motion.base_target is not None
            else None
        )
        return Action(
            tool_pose=tool, base_pose=base, gripper_positions=self._gripper_target.copy()
        ), None

    def idle_action(self, observation: Observation) -> Action:
        if self._hold_pending:
            return self._hold(observation)
        assert self._hold_tool is not None and self._gripper_target is not None
        # Maintain squeeze while the command producer thinks; let the base source expire.
        return Action(
            tool_pose=self._hold_tool.copy(), gripper_positions=self._gripper_target.copy()
        )

    def reset_command(self) -> None:
        self._motion = None
        self._hold_pending = True
        self._gripper_target = None
        self._hold_tool = None

    def _grip_reference(self, observation: Observation) -> NDArray[np.float64]:
        if self._gripper_target is not None:
            return self._gripper_target.copy()
        return observation.robot_state.joint_positions[
            self.robot.get_gripper_position_indices()
        ].copy()

    def _hold(self, observation: Observation) -> Action:
        self._hold_pending = False
        q = observation.robot_state.joint_positions
        self._hold_tool = self.robot.get_tool_command_pose(q).copy()
        self._gripper_target = self._grip_reference(observation)
        base = self.robot.get_base_pose(q)
        return Action(
            tool_pose=self._hold_tool.copy(),
            base_pose=None if base is None else base.copy(),
            gripper_positions=self._gripper_target.copy(),
        )

    def _bounded_base_target(self, measured: pin.SE3, target: pin.SE3) -> pin.SE3:
        config = self.robot.config.omniwheel_base_config
        assert config is not None
        base_config = self.robot.config.base
        assert base_config is not None
        velocity = base_config.velocity_limits
        local = measured.inverse() * target
        delta = local.translation.copy()
        distance = np.linalg.norm(delta[:2])
        maximum = velocity.linear * config.dt / config.position_gain
        if distance > maximum:
            delta[:2] *= maximum / distance
        yaw = math.atan2(local.rotation[1, 0], local.rotation[0, 0])
        yaw_limit = velocity.angular * config.dt / config.orientation_gain
        return measured * pin.SE3(
            pin.rpy.rpyToMatrix(0, 0, np.clip(yaw, -yaw_limit, yaw_limit)), delta
        )

    def _result(
        self, measured: NDArray[np.float64], motion: AgentMotion, elapsed: float
    ) -> MotionResult:
        tool = self.robot.get_tool_command_pose(measured)
        base = self.robot.get_base_pose(measured)
        pairs = [(tool, motion.target)]
        if base is not None and motion.base_target is not None:
            pairs.append((base, motion.base_target))
        position_error = max(
            float(np.linalg.norm(actual.translation - goal.translation)) for actual, goal in pairs
        )
        orientation_error = max(
            float(np.linalg.norm(pin.log3(actual.rotation.T @ goal.rotation)))
            for actual, goal in pairs
        )
        gripper_error = float(
            np.max(
                np.abs(measured[self.robot.get_gripper_position_indices()] - motion.gripper),
                initial=0.0,
            )
        )
        reached = (
            position_error <= self.config.position_tolerance
            and orientation_error <= self.config.orientation_tolerance
            and gripper_error <= self.config.orientation_tolerance
        )
        return {
            "reached": reached,
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "gripper_error_rad": gripper_error,
            "elapsed_seconds": elapsed,
            "step_id": self.step_id,
        }

    def _targets(
        self, command: MotionCommand, observation: Observation
    ) -> tuple[pin.SE3, pin.SE3, pin.SE3 | None, NDArray[np.float64]]:
        q = observation.robot_state.joint_positions
        initial_tool = self.robot.get_tool_command_pose(q).copy()
        base_pose = self.robot.get_base_pose(q)
        base_target = None if base_pose is None else base_pose.copy()
        target = initial_tool.copy()
        gripper = self._grip_reference(observation)
        if command.tool_pose is not None:
            target = command.tool_pose.copy()
            rotation_valid = np.allclose(
                target.rotation.T @ target.rotation, np.eye(3), atol=1e-5
            ) and np.isclose(np.linalg.det(target.rotation), 1.0, atol=1e-5)
            if not np.isfinite(target.homogeneous).all() or not rotation_valid:
                raise ValueError("Tool target must be a finite rigid transform.")
            if np.any(target.translation < self.config.tool_workspace_min) or np.any(
                target.translation > self.config.tool_workspace_max
            ):
                raise ValueError("Tool target is outside the configured workspace.")
            if (
                np.linalg.norm(target.translation - initial_tool.translation)
                > self.config.max_tool_translation
            ):
                raise ValueError("Tool displacement exceeds the per-command limit.")
            if (
                np.linalg.norm(pin.log3(initial_tool.rotation.T @ target.rotation))
                > self.config.max_tool_rotation
            ):
                raise ValueError("Tool rotation exceeds the per-command limit.")
        if command.base_delta is not None:
            if base_pose is None:
                raise ValueError("move_base requires a mobile robot.")
            delta = self._vector(command.base_delta, 3)
            if (
                np.linalg.norm(delta[:2]) > self.config.max_base_translation
                or abs(delta[2]) > self.config.max_base_rotation
            ):
                raise ValueError("Base displacement exceeds the per-command limit.")
            base_target = base_pose * pin.SE3(
                pin.rpy.rpyToMatrix(0, 0, delta[2]), np.array([*delta[:2], 0])
            )
        if command.gripper_positions is not None:
            if not len(gripper):
                raise ValueError("set_gripper requires a configured gripper.")
            gripper = self._vector(command.gripper_positions, len(gripper))
            bounds = np.asarray(self.robot.get_gripper_limits())
            if np.any(gripper < bounds[:, 0]) or np.any(gripper > bounds[:, 1]):
                raise ValueError("Gripper positions exceed joint limits.")
        return initial_tool, target, base_target, gripper

    @staticmethod
    def _vector(value: NDArray[np.float64], length: int) -> NDArray[np.float64]:
        if value.shape != (length,) or value.dtype.kind not in "fi" or not np.isfinite(value).all():
            raise ValueError(f"Expected a finite numeric vector of length {length}.")
        return value.copy()
