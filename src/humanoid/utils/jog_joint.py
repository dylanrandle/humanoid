import argparse
import sys
import time

import numpy as np
from pynput import keyboard

from humanoid.config import ROBOT_CONFIG
from humanoid.constants import Topic
from humanoid.logger import get_logger
from humanoid.middleware.publisher import Publisher
from humanoid.middleware.subscriber import Subscriber
from humanoid.robots.base import Robot
from humanoid.types.robot import RobotConfig, RobotJointCommand

logger = get_logger(__name__)

# Jog step size in radians
DEFAULT_STEP_SIZE = 0.05  # ~2.86 degrees
DEFAULT_PUBLISH_RATE = 100.0  # Hz


def jog_joint(  # noqa: PLR0915
    joint_idx: int,
    step_size: float = DEFAULT_STEP_SIZE,
    publish_rate: float = DEFAULT_PUBLISH_RATE,
    robot_config: RobotConfig = ROBOT_CONFIG,
    timeout_ms: int = 1000,
):
    """Jog a joint using arrow keys with global keyboard capture.

    This function allows interactive control of a single joint by publishing
    RobotJointCommand messages. All other joints maintain their current positions.

    Args:
        joint_idx: Index of the joint to jog (0-based)
        step_size: Step size in radians for each key press
        publish_rate: Rate at which to publish commands in Hz
        robot_config: Robot configuration
        timeout_ms: Timeout in milliseconds for reading robot state
    """
    if joint_idx not in robot_config.joint_idx_to_servo_id:
        valid = sorted(robot_config.joint_idx_to_servo_id.keys())
        raise ValueError(f"Invalid joint index: {joint_idx}. Valid indices are: {valid}.")

    robot = Robot(robot_config)
    num_joints = len(robot_config.joint_idx_to_servo_id)
    position_idx = robot.joint_idx_to_position_idx(joint_idx)

    logger.info(f"Starting jog mode for joint {joint_idx}")
    logger.info(f"Robot: {robot_config.name}")
    logger.info(f"Number of actuated joints: {num_joints}")
    logger.info(f"Step size: {step_size:.4f} rad (~{np.rad2deg(step_size):.2f} degrees)")
    logger.info(f"Publish rate: {publish_rate} Hz")
    logger.info("Controls:")
    logger.info("  LEFT ARROW  - Decrease joint angle (counter-clockwise)")
    logger.info("  RIGHT ARROW - Increase joint angle (clockwise)")
    logger.info("  ESC or 'q'  - Quit")
    logger.info("\nNote: Keyboard input works globally (terminal doesn't need focus)")

    # Read initial robot state
    subscriber = Subscriber([Topic.ROBOT_STATE])
    robot_state = subscriber.receive(Topic.ROBOT_STATE, timeout=timeout_ms)
    subscriber.close()

    if robot_state is None:
        raise RuntimeError(f"Could not read current robot state within {timeout_ms}ms")

    # Initialize joint positions from current state
    current_positions = robot_state.joint_positions.copy()

    logger.info(f"Initial joint positions: {current_positions}")
    logger.info(
        f"Initial joint {joint_idx} position: {current_positions[position_idx]:.4f} rad "
        f"({np.rad2deg(current_positions[position_idx]):.2f} deg)"
    )
    logger.info("Ready! Use arrow keys to jog the joint.\n")

    publisher = Publisher()
    publish_period = 1.0 / publish_rate
    last_publish_time = time.time()

    def publish_command():
        """Publish the current joint command."""
        nonlocal last_publish_time
        current_time = time.time()

        # Rate limiting
        if current_time - last_publish_time >= publish_period:
            command = RobotJointCommand(
                timestamp=current_time,
                joint_positions=current_positions,
            )
            publisher.publish(command, topic=Topic.ROBOT_JOINT_COMMAND)
            last_publish_time = current_time

    def on_press(key):
        try:
            if key == keyboard.Key.left:
                # Decrease joint angle (counter-clockwise)
                current_positions[position_idx] -= step_size
                publish_command()
                logger.info(
                    f"← Joint {joint_idx}: {current_positions[position_idx]:.4f} rad "
                    f"({np.rad2deg(current_positions[position_idx]):.2f} deg)"
                )
            elif key == keyboard.Key.right:
                # Increase joint angle (clockwise)
                current_positions[position_idx] += step_size
                publish_command()
                logger.info(
                    f"→ Joint {joint_idx}: {current_positions[position_idx]:.4f} rad "
                    f"({np.rad2deg(current_positions[position_idx]):.2f} deg)"
                )
            elif key == keyboard.Key.esc or (hasattr(key, "char") and key.char == "q"):
                logger.info("Exiting jog mode...")
                return False  # Stop listener
        except Exception as e:
            logger.error(f"Error during jog: {e}")

    try:
        # Start listening for keyboard events
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

        logger.info(
            f"Final joint {joint_idx} position: {current_positions[position_idx]:.4f} rad "
            f"({np.rad2deg(current_positions[position_idx]):.2f} deg)"
        )
        logger.info(f"Final joint positions: {current_positions}")

    except KeyboardInterrupt:
        logger.info("Exiting")


def main():
    parser = argparse.ArgumentParser(
        description="Jog a robot joint using arrow keys by publishing RobotJointCommand",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Jog joint 0 using default step size
  python -m humanoid.motions.jog --joint-idx 0

  # Jog joint 2 using custom step size
  python -m humanoid.motions.jog --joint-idx 2 --step-size 0.1

  # Jog joint 1 with custom publish rate
  python -m humanoid.motions.jog --joint-idx 1 --rate 50
        """,
    )
    parser.add_argument(
        "--joint-idx",
        type=int,
        required=True,
        help="Index of the joint to jog (0-based)",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=DEFAULT_STEP_SIZE,
        help=f"Step size in radians (default: {DEFAULT_STEP_SIZE})",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_PUBLISH_RATE,
        help=f"Command publish rate in Hz (default: {DEFAULT_PUBLISH_RATE})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1000,
        help="Timeout in milliseconds for reading robot state (default: 1000)",
    )

    args = parser.parse_args()

    # Validate step size
    if args.step_size <= 0:
        logger.error(f"Invalid step size: {args.step_size}. Must be positive.")
        sys.exit(1)

    # Validate publish rate
    if args.rate <= 0:
        logger.error(f"Invalid publish rate: {args.rate}. Must be positive.")
        sys.exit(1)

    # Validate timeout
    if args.timeout <= 0:
        logger.error(f"Invalid timeout: {args.timeout}. Must be positive.")
        sys.exit(1)

    jog_joint(
        joint_idx=args.joint_idx,
        step_size=args.step_size,
        publish_rate=args.rate,
        timeout_ms=args.timeout,
    )


if __name__ == "__main__":
    main()
