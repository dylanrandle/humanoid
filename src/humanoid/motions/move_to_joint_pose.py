import argparse
import time

import numpy as np

from humanoid.config import ROBOT_CONFIG
from humanoid.constants import Topic
from humanoid.logger import get_logger
from humanoid.middleware.lcm import Publisher, Subscriber
from humanoid.types.robot import RobotConfig, RobotJointCommand

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


def generate_joint_trajectory(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    speed: float,
    dt: float,
    min_duration: float = 0.1,
) -> list[np.ndarray]:
    """Generate a smooth joint space trajectory from start to goal.

    Args:
        q_start: Starting joint configuration
        q_goal: Goal joint configuration
        speed: Maximum joint velocity in radians per second
        dt: Time step for trajectory discretization
        min_duration: Minimum duration in seconds (default: 0.1)

    Returns:
        List of joint configurations representing the trajectory
    """
    # Calculate the maximum joint displacement
    joint_displacement = np.abs(q_goal - q_start)
    max_displacement = np.max(joint_displacement)

    # Calculate duration based on speed and distance
    # Ensure minimum duration to avoid division by zero
    duration = max(max_displacement / speed, min_duration)

    num_steps = int(duration / dt)
    trajectory = []

    for step in range(num_steps + 1):
        # Compute smooth interpolation parameter
        s = step / num_steps
        t = smooth_interpolation(s)

        # Linear interpolation in joint space
        q = (1 - t) * q_start + t * q_goal
        trajectory.append(q)

    return trajectory


def move_to_joint_pose(  # noqa: PLR0913
    target_position: np.ndarray,
    robot_config: RobotConfig = ROBOT_CONFIG,
    speed: float = 1.0,
    dt: float = 0.01,
    publish_rate: float = 100.0,
    timeout_ms: int = 1000,
) -> None:
    """Move the robot smoothly to a target joint position.

    Args:
        target_position: Target joint configuration to move to
        robot_config: Robot configuration (default: ROBOT_CONFIG)
        speed: Maximum joint velocity in radians per second (default: 1.0)
        dt: Time step for trajectory generation
        publish_rate: Rate at which to publish commands in Hz
        timeout_ms: Timeout in milliseconds for reading robot state
    """
    # Load robot configuration
    robot_name = robot_config.name

    logger.info(f"Moving {robot_name} to target position...")
    logger.info(f"Target position: {target_position}")

    subscriber = Subscriber([Topic.ROBOT_STATE])
    robot_state = subscriber.receive(Topic.ROBOT_STATE, timeout=timeout_ms)
    subscriber.close()

    if robot_state is None:
        raise RuntimeError(f"Could not read current robot position within {timeout_ms}ms")

    q_start = robot_state.joint_positions

    # Calculate distance and duration
    joint_displacement = np.abs(target_position - q_start)
    max_displacement = np.max(joint_displacement)
    duration = max(max_displacement / speed, 0.1)

    logger.info(f"Start position: {q_start}")
    logger.info(f"Max joint displacement: {max_displacement:.3f} rad")
    logger.info(f"Speed: {speed} rad/s")
    logger.info(f"Calculated duration: {duration:.2f}s")
    logger.info(f"Publish rate: {publish_rate} Hz")

    trajectory = generate_joint_trajectory(q_start, target_position, speed, dt)
    logger.info(f"Generated {len(trajectory)} waypoints")

    publisher = Publisher()

    # Execute trajectory and publish commands
    logger.info("Executing trajectory...")
    publish_period = 1.0 / publish_rate
    last_publish_time = 0.0

    start_time = time.time()

    for step, q in enumerate(trajectory):
        current_time = time.time() - start_time

        # Publish at the specified rate
        if current_time - last_publish_time >= publish_period:
            # Create and publish RobotJointCommand with np.ndarray
            command = RobotJointCommand(
                timestamp=current_time,
                joint_positions=q,
                # NOTE: we don't move velocity-controlled joints
                joint_velocities=np.zeros_like(q),
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
    logger.info(f"Final position: {target_position}")


def main():
    parser = argparse.ArgumentParser(description="Move robot smoothly to a target joint position")
    parser.add_argument(
        "-p",
        "--position",
        type=str,
        required=True,
        choices=["home", "rest"],
        help="Target position to move to: 'home' or 'rest'",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Maximum joint velocity in radians per second (default: 1.0)",
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

    # Map position names to robot configuration attributes
    position_map = {
        "home": ROBOT_CONFIG.home_position,
        "rest": ROBOT_CONFIG.rest_position,
    }

    target_position = position_map[args.position]

    move_to_joint_pose(
        target_position=target_position,
        speed=args.speed,
        publish_rate=args.rate,
        timeout_ms=args.timeout,
    )


if __name__ == "__main__":
    main()
