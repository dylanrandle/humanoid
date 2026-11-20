"""Bounded pickup motion through the existing asynchronous LCM controller."""

import numpy as np
import pinocchio as pin

from humanoid.controllers.operational_space import OperationalSpaceController
from humanoid.robots.base import Robot
from humanoid.types.pick import PickConfig, PickPlan

MAX_TRACKING_ERROR = 0.05
GRIPPER_COMMAND_TOLERANCE = 1e-6


def advance_pose(current: pin.SE3, target: pin.SE3, dt: float, config: PickConfig) -> pin.SE3:
    displacement = target.translation - current.translation
    angle = pin.log3(current.rotation.T @ target.rotation)
    linear_scale = min(1.0, config.linear_speed * dt / max(np.linalg.norm(displacement), 1e-12))
    angular_scale = min(1.0, config.angular_speed * dt / max(np.linalg.norm(angle), 1e-12))
    return pin.SE3(
        current.rotation @ pin.exp3(angle * angular_scale),
        current.translation + displacement * linear_scale,
    )


def reached(current: pin.SE3, target: pin.SE3, config: PickConfig) -> bool:
    return bool(
        np.linalg.norm(current.translation - target.translation) < config.position_tolerance
        and np.linalg.norm(pin.log3(current.rotation.T @ target.rotation))
        < config.orientation_tolerance
    )


def check_reachability(plan: PickPlan, robot: Robot, q: np.ndarray, config: PickConfig) -> None:
    """Dry-run the same OSC with joint limits/self-collisions before requesting motion.

    This is a kinematic check, not a scene collision or grasp-physics simulation.
    """
    controller = OperationalSpaceController(robot, robot.config.operational_space_config)
    q = q.copy()
    dt = controller.config.dt
    for waypoint in plan.waypoints:
        if waypoint.name == "squeeze":
            continue
        command = robot.get_tool_command_pose(q).copy()
        for _ in range(int(config.phase_timeout / dt)):
            controller.update_state(q)
            command = advance_pose(command, waypoint.pose, dt, config)
            q = controller.compute_control(command, dt=dt).q
            if reached(robot.get_tool_command_pose(q), waypoint.pose, config):
                break
        else:
            raise RuntimeError(f"Kinematic preflight could not reach {waypoint.name}")
