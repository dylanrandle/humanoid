"""Image-driven, top-down approach and squeeze for one cube on level ground.

The policy consumes native RGB/depth plus joint state, never scene/object poses.
It assumes rectified pinhole images, a known cube size/ground height, and a
stationary base/cube. Red-top detection needs a visible top; depth-side detection
fits vertical faces without a color requirement. It does not claim grasp success.
"""

import numpy as np
import pinocchio as pin

from humanoid.config.cube_pick import CUBE_PICK_CONFIG
from humanoid.policy.base import Policy
from humanoid.policy.cube_geometry import detect_depth_cube
from humanoid.robots.base import Robot
from humanoid.types.action import Action
from humanoid.types.camera import CameraFrames
from humanoid.types.cube_pick import (
    CubeDetection,
    CubeDetectionMethod,
    CubePickConfig,
    CubePickPhase,
)
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode

MIN_RED_INTENSITY = 60
RED_DOMINANCE = 1.8
DISTORTION_TOLERANCE = 1e-10


def detect_cube(
    frames: CameraFrames,
    robot: Robot,
    q: np.ndarray,
    config: CubePickConfig,
) -> CubeDetection | None:
    """Register native depth into the RGB field of view, then fit the configured cube."""
    rgb, depth = frames.rgb, frames.depth
    for image in (rgb, depth):
        if any(abs(c) > DISTORTION_TOLERANCE for c in image.intrinsics.distortion_coefficients):
            raise ValueError("Cube detection requires rectified pinhole images.")
    world_depth = robot.get_frame_pose(depth.frame_id, q).copy()
    world_rgb = robot.get_frame_pose(rgb.frame_id, q).copy()
    base = robot.get_base_pose(q)
    command_depth = world_depth if base is None else base.inverse() * world_depth
    rgb_depth = world_rgb.inverse() * world_depth
    rows, cols = np.nonzero(depth.data)
    z = depth.data[rows, cols].astype(float) * depth.depth_scale
    intr = depth.intrinsics
    points = np.column_stack(((cols - intr.cx) * z / intr.fx, (rows - intr.cy) * z / intr.fy, z))
    color_points = points @ rgb_depth.rotation.T + rgb_depth.translation
    forward = color_points[:, 2] > 0
    points, color_points = points[forward], color_points[forward]
    intr = rgb.intrinsics
    u = np.rint(intr.fx * color_points[:, 0] / color_points[:, 2] + intr.cx).astype(int)
    v = np.rint(intr.fy * color_points[:, 1] / color_points[:, 2] + intr.cy).astype(int)
    inside = (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    points = points[inside]
    if config.detection_method is CubeDetectionMethod.DEPTH_SIDES:
        command_points = points @ command_depth.rotation.T + command_depth.translation
        return detect_depth_cube(command_points, command_depth.translation, config)
    colors = rgb.data[v[inside], u[inside]].astype(float)
    red = (colors[:, 0] > MIN_RED_INTENSITY) & (colors[:, 0] > RED_DOMINANCE * colors[:, 1])
    red &= colors[:, 0] > RED_DOMINANCE * colors[:, 2]
    points = points[red] @ command_depth.rotation.T + command_depth.translation
    top_z = config.ground_height + config.cube_size
    points = points[np.abs(points[:, 2] - top_z) < config.top_tolerance]
    if len(points) < config.minimum_points:
        return None

    # A square's yaw is ambiguous modulo 90 degrees; choose the minimum-area
    # bounding rectangle in [-45, 45). Using the top avoids front-face bias.
    best = None
    for yaw in np.deg2rad(np.arange(-45, 45, 1)):
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        local = points[:, :2] @ rotation
        lower, upper = np.quantile(local, [0.005, 0.995], axis=0)
        extent = upper - lower
        area = float(np.prod(extent))
        if best is None or area < best[0]:
            best = (area, yaw, rotation, lower, upper)
    assert best is not None
    _, yaw, rotation, lower, upper = best
    extent = upper - lower
    if np.any(extent < config.cube_size * 0.4) or np.any(extent > config.cube_size * 1.3):
        return None
    local_center = (lower + upper) / 2
    # At this low camera height the far edge can fall between depth scanlines.
    # Recover missing top extent from the known size and camera-facing edge.
    camera_xy = command_depth.translation[:2] @ rotation
    facing_center = np.where(
        camera_xy < local_center, lower + config.cube_size / 2, upper - config.cube_size / 2
    )
    local_center = np.where(extent < config.cube_size * 0.9, facing_center, local_center)
    xy = rotation @ local_center
    center = np.array([*xy, config.ground_height + config.cube_size / 2])
    if np.any(center < config.workspace_min) or np.any(center > config.workspace_max):
        return None
    return CubeDetection(center=center, yaw=float(yaw), point_count=len(points))


class CubePickPolicy(Policy):
    """Acquire once, approach, descend, and close; bounded waits fail open.

    Acquisition is latched before the arm occludes the camera. Motion phases
    advance on measured tool pose, not elapsed time. Completion means the
    squeeze command was held for the configured duration, not a confirmed grasp.
    """

    mode = Mode.SYSTEM

    def __init__(self, robot: Robot, config: CubePickConfig = CUBE_PICK_CONFIG):
        self.robot = robot
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.phase = CubePickPhase.SEARCH
        self.detection: CubeDetection | None = None
        self.failure_reason: str | None = None
        self._phase_start: float | None = None
        self._target: pin.SE3 | None = None
        self._hold: pin.SE3 | None = None

    @property
    def is_done(self) -> bool:
        return self.phase is CubePickPhase.DONE

    def step(self, observation: Observation) -> Action:
        now = observation.robot_state.timestamp
        q = observation.robot_state.joint_positions
        measured = self.robot.get_tool_command_pose(q).copy()
        if self._phase_start is None:
            self._phase_start = now
            self._hold = measured.copy()
        if self.phase is CubePickPhase.FAILED:
            return Action(
                tool_pose=self._hold, gripper_positions=np.array([self.config.open_position])
            )
        if now - self._phase_start > self.config.phase_timeout and not self.is_done:
            self.failure_reason = f"Timed out in {self.phase.value}"
            self.phase = CubePickPhase.FAILED
            self._hold = measured.copy()
            return Action(
                tool_pose=self._hold, gripper_positions=np.array([self.config.open_position])
            )
        if self.phase is CubePickPhase.SEARCH:
            frames = observation.camera_frames
            if frames is not None and all(
                0 <= now - image.timestamp <= self.config.image_max_age
                for image in (frames.rgb, frames.depth)
            ):
                self.detection = detect_cube(frames, self.robot, q, self.config)
            if self.detection is None:
                return Action(
                    tool_pose=self._hold, gripper_positions=np.array([self.config.open_position])
                )
            # +Y along the fingers points down; +X closes across a cube edge.
            rotation = pin.rpy.rpyToMatrix(0, 0, self.detection.yaw) @ pin.rpy.rpyToMatrix(
                -np.pi / 2, 0, 0
            )
            position = self.detection.center - rotation @ np.asarray(self.config.grip_offset)
            position[2] += self.config.approach_clearance
            self._target = pin.SE3(rotation, position)
            self.phase = CubePickPhase.APPROACH
            self._phase_start = now
        assert self._target is not None
        reached = (
            np.linalg.norm(measured.translation - self._target.translation)
            < self.config.position_tolerance
            and np.linalg.norm(pin.log3(measured.rotation.T @ self._target.rotation))
            < self.config.orientation_tolerance
        )
        if reached and self.phase is CubePickPhase.APPROACH:
            self._target.translation[2] -= self.config.approach_clearance
            self.phase = CubePickPhase.DESCEND
            self._phase_start = now
        elif reached and self.phase is CubePickPhase.DESCEND:
            self.phase = CubePickPhase.SQUEEZE
            self._phase_start = now
        elif (
            self.phase is CubePickPhase.SQUEEZE
            and now - self._phase_start >= self.config.squeeze_seconds
        ):
            self.phase = CubePickPhase.DONE
        closed = self.phase in (CubePickPhase.SQUEEZE, CubePickPhase.DONE)
        return Action(
            tool_pose=self._target.copy(),
            gripper_positions=np.array(
                [
                    self.config.closed_position if closed else self.config.open_position,
                ]
            ),
        )
