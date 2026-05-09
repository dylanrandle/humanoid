import argparse
import time

import numpy as np
import pinocchio as pin

from humanoid.config import ROBOT_CONFIG
from humanoid.constants import Topic
from humanoid.logger import get_logger
from humanoid.middleware.lcm import Publisher, Subscriber
from humanoid.robots.base import Robot
from humanoid.types.robot import RobotConfig, RobotToolCommand

logger = get_logger(__name__)


def smooth_interpolation(t: float) -> float:
    """Compute smooth interpolation parameter using a cubic polynomial.

    This provides smooth acceleration and deceleration.

    Args:
        t: Interpolation parameter in [0, 1]

    Returns:
        Smoothed interpolation parameter in [0, 1]
    """
    # Smooth step function: 3t^2 - 2t^3
    return 3 * t**2 - 2 * t**3


def interpolate_se3(
    pose_start: pin.SE3,
    pose_goal: pin.SE3,
    t: float,
) -> pin.SE3:
    """Interpolate between two SE3 poses using SLERP for rotation and linear for translation.

    Args:
        pose_start: Starting pose
        pose_goal: Goal pose
        t: Interpolation parameter in [0, 1]

    Returns:
        Interpolated SE3 pose
    """
    # Linear interpolation for translation
    position = (1 - t) * pose_start.translation + t * pose_goal.translation

    # SLERP (Spherical Linear Interpolation) for rotation
    # Convert rotations to quaternions
    quat_start = pin.Quaternion(pose_start.rotation)
    quat_goal = pin.Quaternion(pose_goal.rotation)

    # Perform SLERP
    quat_interp = quat_start.slerp(t, quat_goal)

    # Create interpolated pose
    return pin.SE3(quat_interp.toRotationMatrix(), position)


def generate_pose_trajectory(
    pose_start: pin.SE3,
    pose_goal: pin.SE3,
    speed: float,
    dt: float,
    min_duration: float = 0.1,
) -> list[pin.SE3]:
    """Generate a smooth end effector trajectory from start to goal pose.

    Args:
        pose_start: Starting end effector pose
        pose_goal: Goal end effector pose
        speed: Maximum end effector velocity in meters per second
        dt: Time step for trajectory discretization
        min_duration: Minimum duration in seconds (default: 0.1)

    Returns:
        List of SE3 poses representing the trajectory
    """
    # Calculate the distance to travel
    translation_distance = np.linalg.norm(pose_goal.translation - pose_start.translation)

    # Calculate duration based on speed and distance
    # Ensure minimum duration to avoid division by zero
    duration = max(translation_distance / speed, min_duration)

    num_steps = int(duration / dt)
    trajectory = []

    for step in range(num_steps + 1):
        # Compute smooth interpolation parameter
        s = step / num_steps
        t = smooth_interpolation(s)

        # Interpolate pose
        pose = interpolate_se3(pose_start, pose_goal, t)
        trajectory.append(pose)

    return trajectory


def move_to_pose(  # noqa: PLR0913, PLR0915
    goal_pose: pin.SE3 | None = None,
    goal_x: float | None = None,
    goal_y: float | None = None,
    goal_z: float | None = None,
    goal_roll: float | None = None,
    goal_pitch: float | None = None,
    goal_yaw: float | None = None,
    robot_config: RobotConfig = ROBOT_CONFIG,
    speed: float = 0.1,
    publish_rate: float = 100.0,
    timeout_ms: int = 1000,
) -> None:
    """Move the robot smoothly to a goal pose in end effector space.

    The robot always starts from its current pose (read from robot state).

    Args:
        goal_pose: Goal end effector pose (SE3). If provided, other goal parameters are ignored.
        goal_x: Goal X position in meters. If None, maintains starting X position.
        goal_y: Goal Y position in meters. If None, maintains starting Y position.
        goal_z: Goal Z position in meters. If None, maintains starting Z position.
        goal_roll: Goal roll angle in radians. If None, maintains starting orientation.
        goal_pitch: Goal pitch angle in radians. If None, maintains starting orientation.
        goal_yaw: Goal yaw angle in radians. If None, maintains starting orientation.
        robot_config: Robot configuration
        speed: Maximum end effector velocity in meters per second (default: 0.1)
        publish_rate: Rate at which to publish commands in Hz
        timeout_ms: Timeout in milliseconds for reading robot state
    """
    # Load robot configuration
    robot_name = robot_config.name
    end_effector_frame = robot_config.tool_frame
    dt = 1 / publish_rate

    logger.info(f"Moving {robot_name} end effector to goal pose...")
    logger.info(f"End effector frame: {end_effector_frame}")

    subscriber = Subscriber([Topic.ROBOT_STATE])
    robot_state = subscriber.receive(Topic.ROBOT_STATE, timeout=timeout_ms)
    subscriber.close()

    if robot_state is None:
        raise RuntimeError(f"Could not read current robot position within {timeout_ms}ms")

    q_start = robot_state.joint_positions

    robot = Robot(robot_config)
    start_pose = robot.get_frame_pose(end_effector_frame, q_start)

    logger.info("Start pose:")
    logger.info(f"  Position: {start_pose.translation}")
    logger.info(f"  Rotation:\n{start_pose.rotation}")

    # Build goal pose if not provided directly
    if goal_pose is None:
        # Use starting position for any unspecified coordinates
        x = goal_x if goal_x is not None else start_pose.translation[0]
        y = goal_y if goal_y is not None else start_pose.translation[1]
        z = goal_z if goal_z is not None else start_pose.translation[2]
        goal_position = np.array([x, y, z])

        # Extract starting orientation as RPY
        start_rpy = pin.rpy.matrixToRpy(start_pose.rotation)

        # Use starting orientation for any unspecified angles
        roll = goal_roll if goal_roll is not None else start_rpy[0]
        pitch = goal_pitch if goal_pitch is not None else start_rpy[1]
        yaw = goal_yaw if goal_yaw is not None else start_rpy[2]

        # Build goal pose
        goal_rotation = pin.rpy.rpyToMatrix(roll, pitch, yaw)
        goal_pose = pin.SE3(goal_rotation, goal_position)

    logger.info("Goal pose:")
    logger.info(f"  Position: {goal_pose.translation}")
    logger.info(f"  Rotation:\n{goal_pose.rotation}")

    # Calculate distance and duration
    translation_distance = np.linalg.norm(goal_pose.translation - start_pose.translation)
    duration = max(translation_distance / speed, 0.1)

    logger.info(f"Translation distance: {translation_distance:.3f} m")
    logger.info(f"Speed: {speed} m/s")
    logger.info(f"Calculated duration: {duration:.2f}s")
    logger.info(f"Publish rate: {publish_rate} Hz")

    trajectory = generate_pose_trajectory(start_pose, goal_pose, speed, dt)
    logger.info(f"Generated {len(trajectory)} waypoints")

    publisher = Publisher()

    # Execute trajectory and publish commands
    logger.info("Executing trajectory...")
    publish_period = 1.0 / publish_rate
    last_publish_time = 0.0

    start_time = time.time()

    for step, pose in enumerate(trajectory):
        current_time = time.time() - start_time

        # Publish at the specified rate
        if current_time - last_publish_time >= publish_period:
            # Create and publish RobotToolCommand
            command = RobotToolCommand(
                timestamp=current_time,
                pose=pose,
            )

            publisher.publish(command)
            last_publish_time = current_time

            # Log progress every 0.5 seconds
            if step % int(0.5 / dt) == 0:
                progress = (step / len(trajectory)) * 100
                logger.info(f"t={current_time:.2f}s ({progress:.1f}%)")

        # Sleep to maintain trajectory timing
        time.sleep(dt)

    logger.info("Motion complete!")
    logger.info("Final pose:")
    logger.info(f"  Position: {goal_pose.translation}")
    logger.info(f"  Rotation:\n{goal_pose.rotation}")


def main():
    """Run the move to pose script."""

    parser = argparse.ArgumentParser(
        description="Move robot smoothly to goal pose in end effector space"
    )
    parser.add_argument(
        "--x",
        type=float,
        default=None,
        help="Goal X position in meters (default: maintain starting X position)",
    )
    parser.add_argument(
        "--y",
        type=float,
        default=None,
        help="Goal Y position in meters (default: maintain starting Y position)",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=None,
        help="Goal Z position in meters (default: maintain starting Z position)",
    )
    parser.add_argument(
        "--roll",
        type=float,
        default=None,
        help="Goal orientation roll angle in radians (default: maintain starting orientation)",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=None,
        help="Goal orientation pitch angle in radians (default: maintain starting orientation)",
    )
    parser.add_argument(
        "--yaw",
        type=float,
        default=None,
        help="Goal orientation yaw angle in radians (default: maintain starting orientation)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.1,
        help="Maximum end effector velocity in meters per second (default: 0.1)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=100.0,
        help="Command publish rate in Hz (default: 100.0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1000,
        help="Timeout in milliseconds for reading robot state (default: 1000)",
    )

    args = parser.parse_args()

    # Pass position and orientation components separately to allow move_to_pose to handle defaults
    move_to_pose(
        goal_x=args.x,
        goal_y=args.y,
        goal_z=args.z,
        goal_roll=args.roll,
        goal_pitch=args.pitch,
        goal_yaw=args.yaw,
        speed=args.speed,
        publish_rate=args.rate,
        timeout_ms=args.timeout,
    )


if __name__ == "__main__":
    main()
