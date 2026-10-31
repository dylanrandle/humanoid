"""Bounded arm and mobile-base motion primitives for privileged experts."""

import numpy as np
import pinocchio as pin

from humanoid.simulation.client import SimulationClient
from humanoid.simulation.tasks.cleanup.kinematics import ArmKinematics
from humanoid.simulation.tasks.cleanup.navigation import wrap_angle
from humanoid.simulation.tasks.cleanup.visibility import CleanupView
from humanoid.types.homing import HomingPreset
from humanoid.types.tasks.cleanup import (
    CleanupConfig,
    CleanupSceneConfig,
    CleanupViewConfig,
    MobileCleanupConfig,
    NavigationPlan,
    TargetVisibility,
)

STOPPED_GRIPPER_SPEED = 0.02
GRIPPER_SETTLE_SECONDS = 0.2


class CleanupMotion:
    def __init__(
        self,
        runtime: SimulationClient,
        config: CleanupConfig,
        mobile: MobileCleanupConfig,
        scene: CleanupSceneConfig,
        view_config: CleanupViewConfig,
    ):
        self.runtime = runtime
        self.engine = runtime.engine
        self.config = config
        self.mobile = mobile
        self.scene = scene
        self.kinematics = ArmKinematics(self.engine.robot, config)
        self.view = CleanupView(self.engine, scene, view_config)
        self.view_target = ""
        self.carried_target = ""
        self.root_v = self.engine.robot.get_root_v_slice()
        self.root_q = self.engine.robot.get_root_q_slice()
        if self.root_q is None or self.root_v is None:
            raise ValueError("Cleanup requires Triskel's unlocked mobile base.")
        self.command_q = self.positions()
        self.gripper_target = config.open_position
        self.base_target = self.base_pose()[:2].copy()
        self.base_yaw_target = self.base_pose()[2]
        self.face_target: np.ndarray | None = None
        self.navigation_plan: NavigationPlan | None = None
        self.planned_distance = 0.0
        # Translation is limited in the world frame, independent of base rotation.
        self.base_velocity = np.zeros(3)
        self.base_distance = 0.0

    def positions(self) -> np.ndarray:
        return self.runtime.state().joint_positions

    def base_pose(self) -> np.ndarray:
        x, y, cosine, sine = self.positions()[self.root_q]
        return np.array([x, y, np.arctan2(sine, cosine)])

    def _tick(self, q: np.ndarray) -> None:
        pose = self.base_pose()
        yaw = pose[2]
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        if self.face_target is not None:
            self.base_yaw_target = self.facing_yaw(self.face_target)
        yaw_error = wrap_angle(self.base_yaw_target - yaw)
        wanted = np.r_[
            self.mobile.position_gain * (self.base_target - pose[:2]),
            self.mobile.yaw_gain * yaw_error,
        ]
        speed = np.linalg.norm(wanted[:2])
        if speed > self.mobile.max_speed:
            wanted[:2] *= self.mobile.max_speed / speed
        wanted[2] = np.clip(wanted[2], -self.mobile.max_yaw_speed, self.mobile.max_yaw_speed)
        change = wanted[:2] - self.base_velocity[:2]
        self.base_velocity[:2] += change * min(
            1.0,
            self.mobile.max_acceleration * self.runtime.period / max(np.linalg.norm(change), 1e-12),
        )
        angular_change = self.mobile.max_yaw_acceleration * self.runtime.period
        self.base_velocity[2] += np.clip(
            wanted[2] - self.base_velocity[2], -angular_change, angular_change
        )
        body_velocity = np.r_[rotation.T @ self.base_velocity[:2], self.base_velocity[2]]
        velocity = np.zeros(self.engine.robot.model.nv)
        velocity[self.root_v] = body_velocity
        q = q.copy()
        self.engine.robot.set_gripper_positions(q, np.array([self.gripper_target]))
        self.runtime.step(q, velocity)
        self.command_q = q.copy()
        self.base_distance += float(np.linalg.norm(self.base_pose()[:2] - pose[:2]))

    def observed_targets(self) -> set[str]:
        targets = {self.view_target} if self.view_target else set()
        if self.carried_target and self.gripper_target == self.config.closed_position:
            targets.add(self.carried_target)
        return targets

    def views(self) -> dict[str, TargetVisibility]:
        return {
            name: self.view.best_view(
                name,
                held=name == self.carried_target,
            )
            for name in sorted(self.observed_targets())
        }

    def command(self, target: np.ndarray, duration: float) -> None:
        start = self.command_q.copy()
        steps = max(1, round(duration / self.runtime.period))
        for index in range(steps):
            self._tick(start + (index + 1) / steps * (target - start))

    def hold(self, duration: float) -> None:
        self.command(self.command_q, duration)

    def arm(self, positions: tuple[float, ...], *, require_tool_pose: bool = True) -> None:
        start = self.positions()
        target = start.copy()
        target[self.kinematics.indices] = positions
        duration = max(1.0, 2 * float(np.max(np.abs(target - start))) / self.config.joint_speed)
        for fraction in np.linspace(0, 1, max(2, round(duration / self.runtime.period))):
            smooth = fraction * fraction * (3 - 2 * fraction)
            self._tick(start + smooth * (target - start))
        self.command(target, 0.3)
        if not require_tool_pose:
            return
        # A redundant arm can reach this tool target with different joint
        # angles under the shared OSC. Check the commanded frame rather than
        # requiring the planner's particular IK solution.
        robot = self.engine.robot
        expected = robot.get_tool_command_pose(target).copy()
        deadline = self.runtime.time + self.config.motion_timeout
        while True:
            actual = robot.get_tool_command_pose(self.positions())
            if (
                np.linalg.norm(actual.translation - expected.translation)
                < self.config.position_tolerance
                and np.linalg.norm(pin.log3(actual.rotation @ expected.rotation.T))
                < self.config.orientation_tolerance
            ):
                break
            if self.runtime.time >= deadline:
                raise ValueError("Arm tool-pose tracking failed.")
            self.command(target, self.runtime.period)

    def park(self) -> None:
        """Retract the empty arm to its home pose without aiming a camera."""
        if self.carried_target:
            return
        target = self.engine.robot.config.homing_presets[HomingPreset.HOME]
        self.arm(tuple(target[self.kinematics.indices]))

    def acquire(self, names: list[str]) -> str:
        """Choose the nearest destination from the expert's current scene state."""
        self.base_target = self.base_pose()[:2].copy()
        self.base_yaw_target = self.base_pose()[2]
        self.view_target = min(
            names,
            key=lambda name: np.linalg.norm(
                (
                    np.asarray(self.scene.bin_position)
                    if name == "cleanup_bin"
                    else self.engine.data.body(name).xpos[:2]
                )
                - self.base_target
            ),
        )
        return self.view_target

    def move(self, pose: pin.SE3) -> None:
        start_q = self.positions()
        start = self.kinematics.grasp_pose(start_q)
        angle = pin.log3(pose.rotation @ start.rotation.T)
        distance = float(np.linalg.norm(pose.translation - start.translation))
        duration = max(
            0.5, 1.6 * distance / self.config.cartesian_speed, np.linalg.norm(angle) / 0.6
        )
        if duration > self.config.motion_timeout:
            raise ValueError("Cartesian motion exceeds its time budget.")
        command = start_q.copy()
        path = []
        for fraction in np.linspace(0, 1, max(2, round(duration / self.runtime.period))):
            smooth = fraction * fraction * (3 - 2 * fraction)
            waypoint = pin.SE3(
                pin.exp3(smooth * angle) @ start.rotation,
                start.translation + smooth * (pose.translation - start.translation),
            )
            next_q = self.kinematics.solve(waypoint, command)
            step_duration = max(
                self.runtime.period,
                float(np.max(np.abs(next_q - command))) / self.config.joint_speed,
            )
            path.append((next_q, step_duration))
            command = next_q
        if sum(duration for _, duration in path) > self.config.motion_timeout:
            raise ValueError("Joint motion exceeds its time budget.")
        for command, step_duration in path:
            self.command(command, step_duration)
        self.command(command, 0.35)
        actual = self.kinematics.grasp_pose(self.positions())
        if (
            np.linalg.norm(actual.translation - pose.translation) >= self.config.position_tolerance
            or np.linalg.norm(pin.log3(actual.rotation @ pose.rotation.T))
            >= self.config.orientation_tolerance
        ):
            raise ValueError("Cartesian target tracking failed.")

    def reach(self, pose: pin.SE3) -> None:
        # Retracted viewing postures can lie on a different IK branch from the
        # vertical grasp. Solve the endpoint before executing a smooth joint
        # transfer; descent and lift still use Cartesian paths.
        seed = self.positions()
        seed[self.kinematics.indices] = self.config.grasp_seed_arm_positions
        target = self.kinematics.solve(pose, seed)
        self.arm(tuple(target[self.kinematics.indices]))

    def grip(self, closed: bool) -> None:
        self.gripper_target = self.config.closed_position if closed else self.config.open_position
        self.command(self.positions(), 0.6)
        # A contacted object can stop the jaws short of the requested position.
        # Wait for measured motion to settle before lifting or withdrawing.
        indices = self.engine.robot.get_joint_velocity_indices(
            self.engine.robot.get_gripper_joint_indices()
        )
        deadline = self.runtime.time + self.config.motion_timeout
        settled_since = self.runtime.time
        while self.runtime.time - settled_since < GRIPPER_SETTLE_SECONDS:
            if (
                np.max(np.abs(self.runtime.state().joint_velocities[indices]))
                > STOPPED_GRIPPER_SPEED
            ):
                settled_since = self.runtime.time
            if self.runtime.time >= deadline:
                raise ValueError("Gripper motion did not settle.")
            self.hold(self.runtime.period)

    def aperture(self) -> float:
        index = self.engine.robot.get_gripper_position_indices()[0]
        return max(
            0.0,
            float(self.config.open_width - self.config.width_per_radian * self.positions()[index]),
        )

    def facing_yaw(self, target: np.ndarray) -> float:
        toward = target - self.base_pose()[:2]
        front = np.asarray(self.mobile.pickup_offset)
        return wrap_angle(float(np.arctan2(toward[1], toward[0]) - np.arctan2(front[1], front[0])))

    def drive(self, plan: NavigationPlan) -> None:
        self.navigation_plan = plan
        self.planned_distance += plan.length
        started = self.runtime.time
        self.face_target = plan.target.copy()
        # Turn in place before translating. A circular swept footprint makes
        # this safe at every planned pose, including when carrying a toy.
        self.base_target = self.base_pose()[:2].copy()
        self._settle_base(started, self.mobile.position_tolerance)
        for index, waypoint in enumerate(plan.waypoints):
            self.base_target = waypoint.copy()
            tolerance = (
                self.mobile.position_tolerance
                if index == len(plan.waypoints) - 1
                else self.mobile.waypoint_tolerance
            )
            self._settle_base(started, tolerance)
            self.hold(0.3)
        self.face_target = None
        self.base_yaw_target = plan.yaw

    def _settle_base(self, started: float, tolerance: float) -> None:
        while True:
            pose = self.base_pose()
            yaw = (
                self.facing_yaw(self.face_target)
                if self.face_target is not None
                else self.base_yaw_target
            )
            if (
                np.linalg.norm(pose[:2] - self.base_target) <= tolerance
                and abs(wrap_angle(yaw - pose[2])) <= self.mobile.yaw_tolerance
            ):
                return
            if self.runtime.time - started > self.mobile.navigation_timeout:
                raise TimeoutError("Base navigation time budget exhausted.")
            self.hold(self.runtime.period)


def grasp_rotation(yaw: float) -> np.ndarray:
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.array([[cosine, 0, -sine], [sine, 0, cosine], [0, -1, 0]])
