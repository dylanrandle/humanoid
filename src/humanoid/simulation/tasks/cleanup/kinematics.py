"""URDF-only inverse kinematics; the scene and simulator are not inputs."""

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

from humanoid.robots.base import Robot
from humanoid.types.tasks.cleanup import CleanupConfig

MAX_IK_RESIDUAL = 0.015


class ArmKinematics:
    """Solve for the physical grasp center while holding the mobile base fixed."""

    def __init__(self, robot: Robot, config: CleanupConfig):
        self.robot = robot
        self.config = config
        self.indices = robot.get_joint_position_indices(robot.get_arm_joint_indices())

    def grasp_pose(self, q: np.ndarray) -> pin.SE3:
        tool = self.robot.get_tool_pose(q).copy()
        return tool * pin.SE3(np.eye(3), np.asarray(self.config.grasp_offset))

    def solve(self, target: pin.SE3, seed: np.ndarray) -> np.ndarray:
        def residual(arm: np.ndarray) -> np.ndarray:
            q = seed.copy()
            q[self.indices] = arm
            pose = self.grasp_pose(q)
            return np.concatenate(
                [
                    5 * (pose.translation - target.translation),
                    pin.log3(target.rotation @ pose.rotation.T),
                    0.01 * (arm - seed[self.indices]),
                ]
            )

        solution = least_squares(
            residual,
            np.clip(
                seed[self.indices],
                self.robot.model.lowerPositionLimit[self.indices] + 0.002,
                self.robot.model.upperPositionLimit[self.indices] - 0.002,
            ),
            bounds=(
                self.robot.model.lowerPositionLimit[self.indices] + 0.002,
                self.robot.model.upperPositionLimit[self.indices] - 0.002,
            ),
            max_nfev=80,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
        )
        if np.linalg.norm(residual(solution.x)[:6]) > MAX_IK_RESIDUAL:
            raise ValueError("Grasp waypoint is outside the calibrated arm workspace.")
        q = seed.copy()
        q[self.indices] = solution.x
        return q
