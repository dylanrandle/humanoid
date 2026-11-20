"""Camera-guided pickup Policy with asynchronous perception and bounded actions."""

import threading
import time
from concurrent.futures import Future
from dataclasses import replace
from functools import cached_property

import numpy as np
import pinocchio as pin

from humanoid.logger import get_logger
from humanoid.manipulation.pick_motion import (
    GRIPPER_COMMAND_TOLERANCE,
    MAX_TRACKING_ERROR,
    advance_pose,
    reached,
)
from humanoid.manipulation.pick_perception import confirm_scene, perceive
from humanoid.manipulation.preview import render_preview
from humanoid.policy.base import Policy
from humanoid.robots.base import Robot
from humanoid.serving.client import ModelClient
from humanoid.types.action import Action
from humanoid.types.camera import CameraInferencePreview
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode
from humanoid.types.pick import (
    GraspGenXConfig,
    ObjectMask,
    PickConfig,
    PickPerceptionPreview,
    PickPhase,
    PickPlan,
    PickPreview,
    PickStatus,
)
from humanoid.types.robot import RobotConfig


class PickPolicy(Policy):
    mode = Mode.SYSTEM

    def __init__(
        self,
        robot_config: RobotConfig,
        config: PickConfig,
        grasp_config: GraspGenXConfig,
        models: ModelClient,
    ) -> None:
        self.robot_config = robot_config
        self.robot = Robot(robot_config)
        self.config = config
        self.grasp_config = grasp_config
        self.models = models
        self.phase = PickPhase.READY
        self.message = "Enter an object to pick"
        self.description = ""
        self._future: Future[tuple[PickPlan, ObjectMask]] | None = None
        self._initial: Observation | None = None
        self._plan: PickPlan | None = None
        self._pose: pin.SE3 | None = None
        self._base: pin.SE3 | None = None
        self._gripper = 0.0
        self._waypoint = 0
        self._deadline = 0.0
        self._arrived: float | None = None
        self._feedback: dict[str, tuple[float | None, float]] = {}
        self._preview_lock = threading.Lock()
        self._preview_generation = 0
        self._preview: PickPreview | None = None

    def preview(self, depth_far: float) -> CameraInferencePreview | None:
        with self._preview_lock:
            preview = self._preview
        if preview is None:
            return None
        if depth_far not in preview.depths:
            raise ValueError("Unsupported depth preview range")
        age = max(0.0, time.perf_counter() - preview.timestamp)
        return CameraInferencePreview(
            rgb=replace(preview.rgb, age_seconds=age),
            depth=replace(preview.depths[depth_far], age_seconds=age),
            object_description=preview.object_description,
            mask_count=preview.mask_count,
            grasp_count=preview.grasp_count,
            selected=preview.selected,
        )

    def _invalidate_preview(self, *, clear: bool) -> int:
        with self._preview_lock:
            self._preview_generation += 1
            if clear:
                self._preview = None
            return self._preview_generation

    def _update_preview(self, preview: PickPerceptionPreview, generation: int) -> None:
        with self._preview_lock:
            if generation != self._preview_generation:
                return
        try:
            rendered = render_preview(preview)
        except Exception:
            # Visualization must never change the outcome of control/preflight.
            get_logger(__name__).exception("Could not render pickup preview")
            return
        with self._preview_lock:
            if generation == self._preview_generation:
                self._preview = rendered

    @cached_property
    def perception_robot(self) -> Robot:
        # Construct once on the perception thread. Reuse meshes across tasks,
        # with separate Pinocchio data from the realtime policy's robot.
        return Robot(self.robot_config)

    @property
    def active(self) -> bool:
        return self.phase in {
            PickPhase.PLANNING,
            PickPhase.APPROACH,
            PickPhase.GRASP,
            PickPhase.SQUEEZE,
            PickPhase.LIFT,
        }

    @property
    def busy(self) -> bool:
        return self.active or (self._future is not None and not self._future.done())

    def status(self, *, models_ready: bool) -> PickStatus:
        return PickStatus(
            phase=self.phase,
            message=self.message,
            object_description=self.description,
            busy=self.busy,
            models_ready=models_ready,
        )

    def start(self, description: str, observation: Observation) -> None:
        if self.busy:
            raise RuntimeError("A pickup request is still in progress")
        description = description.strip()
        if not any(character.isalnum() for character in description):
            raise ValueError("Enter an object name or description")
        if observation.mode is not self.mode:
            raise RuntimeError("Pickup requires SYSTEM mode")
        self.require_camera(observation)
        generation = self._invalidate_preview(clear=True)
        self.description = description
        self._initial = observation
        self._plan = None
        self._waypoint = 0
        self._arrived = None
        self._feedback.clear()
        q = observation.robot_state.joint_positions
        self._pose = self.robot.get_tool_command_pose(q).copy()
        base = self.robot.get_base_pose(q)
        self._base = base.copy() if base is not None else None
        self._gripper = float(q[self.robot.get_gripper_position_indices()[0]])
        self.phase = PickPhase.PLANNING
        self.message = f'Finding a reachable grasp for "{description}"'
        future: Future[tuple[PickPlan, ObjectMask]] = Future()
        self._future = future

        def plan() -> None:
            try:
                result = perceive(
                    observation,
                    description,
                    self.perception_robot,
                    self.config,
                    self.grasp_config,
                    self.models,
                    lambda preview: self._update_preview(preview, generation),
                )
                future.set_result(result)
            except Exception as error:
                # A traceback retains point clouds and C++ geometry from failed
                # planning frames. Keep the error message, not those frames.
                future.set_exception(RuntimeError(str(error)))

        # Inference has bounded socket timeouts. A cancelled/old result cannot
        # reactivate this policy; only start() may enter PLANNING.
        threading.Thread(target=plan, name="pick-perception", daemon=True).start()

    def reset(self) -> None:
        if self.active:
            self.cancel("Control mode changed; pickup cancelled")

    def cancel(self, message: str = "Pickup cancelled", *, clear_preview: bool = True) -> None:
        self._invalidate_preview(clear=clear_preview)
        self.phase = PickPhase.CANCELLED
        self.message = message
        self._plan = None
        self._pose = None

    def fail(self, message: str) -> None:
        self.cancel(message, clear_preview=False)
        self.phase = PickPhase.FAILED

    def step(self, observation: Observation) -> Action:
        if not self.active:
            return Action()
        try:
            self._check_feedback(observation)
            if self.phase is PickPhase.PLANNING:
                assert self._future is not None
                if self._future.done():
                    future, self._future = self._future, None
                    error = future.exception()
                    if error is not None:
                        raise RuntimeError(str(error))
                    plan, mask = future.result()
                    if plan.grasp.width is None:
                        raise RuntimeError("Gripper width is unresolved")
                    self.require_camera(observation)
                    assert self._initial is not None
                    confirm_scene(self._initial, observation, plan, mask, self.robot, self.config)
                    self._plan = plan
                    self._select_waypoint()
                return self._action()
            return self._advance(observation)
        except Exception as error:
            self.fail(str(error))
            return Action()

    def require_camera(self, observation: Observation) -> None:
        frames = observation.camera_frames
        if frames is None:
            raise RuntimeError("Waiting for paired RGB and depth frames")
        if (
            abs(observation.robot_state.timestamp - frames.rgb.timestamp)
            > self.config.image_max_age
        ):
            raise RuntimeError("Camera frames are stale; waiting for fresh RGB-D")

    def _check_feedback(self, observation: Observation) -> None:
        now = time.monotonic()
        command = observation.robot_joint_command
        for name, timestamp in (
            ("Robot state", observation.robot_state.timestamp),
            ("Orchestrator", observation.mode_timestamp),
            ("Controller", command.timestamp if command is not None else None),
        ):
            previous, received = self._feedback.get(name, (None, now))
            if timestamp is not None and (previous is None or timestamp > previous):
                received = now
            self._feedback[name] = (timestamp, received)
            if now - received > self.config.feedback_timeout:
                raise RuntimeError(f"{name} feedback stopped")
        q = observation.robot_state.joint_positions
        measured = self.robot.get_tool_command_pose(q)
        assert self._pose is not None
        if np.linalg.norm(measured.translation - self._pose.translation) > MAX_TRACKING_ERROR:
            raise RuntimeError("Arm is not tracking the commanded trajectory")
        base = self.robot.get_base_pose(q)
        if (
            self._base is not None
            and base is not None
            and not reached(base, self._base, self.config)
        ):
            raise RuntimeError("Base moved after perception")

    def _select_waypoint(self) -> None:
        assert self._plan is not None
        waypoint = self._plan.waypoints[self._waypoint]
        self.phase = PickPhase(waypoint.name)
        self.message = f"{waypoint.name.capitalize()}: {self.description}"
        self._deadline = time.monotonic() + self.config.phase_timeout
        self._arrived = None

    def _advance(self, observation: Observation) -> Action:
        assert self._plan is not None and self._pose is not None
        now = time.monotonic()
        if now > self._deadline:
            raise RuntimeError(f"Timed out during {self.phase}")
        waypoint = self._plan.waypoints[self._waypoint]
        measured = self.robot.get_tool_command_pose(observation.robot_state.joint_positions)
        dt = 1 / self.config.rate_hz
        self._pose = advance_pose(self._pose, waypoint.pose, dt, self.config)
        self._gripper += float(
            np.clip(
                waypoint.gripper - self._gripper,
                -self.config.gripper_speed * dt,
                self.config.gripper_speed * dt,
            )
        )
        settled = reached(measured, waypoint.pose, self.config)
        settled &= abs(self._gripper - waypoint.gripper) < GRIPPER_COMMAND_TOLERANCE
        self._arrived = (now if self._arrived is None else self._arrived) if settled else None
        if self._arrived is not None and now - self._arrived >= waypoint.hold_seconds:
            self._waypoint += 1
            if self._waypoint == len(self._plan.waypoints):
                self.phase = PickPhase.COMPLETED
                self.message = "Lift completed; check that the object was retained"
            else:
                self._select_waypoint()
        return self._action()

    def _action(self) -> Action:
        return Action(
            tool_pose=self._pose.copy() if self._pose is not None else None,
            gripper_positions=np.array([self._gripper]),
        )
