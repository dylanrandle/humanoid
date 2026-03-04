import argparse
import sys
import threading
import time

import numpy as np
import pinocchio as pin
from pynput import keyboard

from humanoid.config import ROBOT_CONFIG
from humanoid.constants import Topic
from humanoid.logger import get_logger
from humanoid.middleware.lcm import Publisher, Subscriber
from humanoid.robots.base import Robot
from humanoid.types.robot import RobotConfig, RobotToolCommand

logger = get_logger(__name__)

# Jog step sizes
DEFAULT_TRANSLATION_STEP = 0.01  # 1 cm
DEFAULT_ROTATION_STEP = 0.05  # ~2.86 degrees
DEFAULT_PUBLISH_RATE = 100.0  # Hz


def jog_tool_pose(  # noqa: PLR0915
    translation_step: float = DEFAULT_TRANSLATION_STEP,
    rotation_step: float = DEFAULT_ROTATION_STEP,
    publish_rate: float = DEFAULT_PUBLISH_RATE,
    robot_config: RobotConfig = ROBOT_CONFIG,
    timeout_ms: int = 1000,
):
    """Jog the end-effector pose using keyboard with global keyboard capture.

    This function allows interactive control of the robot's end-effector pose
    by publishing RobotToolCommand messages. The robot starts from its current pose.

    Controls:
        - W/S: Move forward/backward (X axis)
        - A/D: Move left/right (Y axis)
        - Q/E: Move up/down (Z axis)
        - I/K: Pitch up/down
        - J/L: Yaw left/right
        - U/O: Roll counter-clockwise/clockwise
        - ESC or 'x': Quit

    Args:
        translation_step: Step size in meters for translation
        rotation_step: Step size in radians for rotation
        publish_rate: Rate at which to publish commands in Hz
        robot_config: Robot configuration
        timeout_ms: Timeout in milliseconds for reading robot state
    """
    robot_name = robot_config.name
    end_effector_frame = robot_config.end_effector_frame

    logger.info(f"Starting task-space jog mode for {robot_name}")
    logger.info(f"End effector frame: {end_effector_frame}")
    logger.info(f"Translation step: {translation_step:.4f} m ({translation_step * 100:.2f} cm)")
    logger.info(
        f"Rotation step: {rotation_step:.4f} rad (~{np.rad2deg(rotation_step):.2f} degrees)"
    )
    logger.info(f"Publish rate: {publish_rate} Hz")
    logger.info("\nControls:")
    logger.info("  Translation:")
    logger.info("    W/S - Move forward/backward (X axis)")
    logger.info("    A/D - Move left/right (Y axis)")
    logger.info("    Q/E - Move up/down (Z axis)")
    logger.info("  Rotation:")
    logger.info("    I/K - Pitch up/down")
    logger.info("    J/L - Yaw left/right")
    logger.info("    U/O - Roll counter-clockwise/clockwise")
    logger.info("  ESC or 'x' - Quit")
    logger.info("\nNote: Keyboard input works globally (terminal doesn't need focus)")

    # Read initial robot state
    subscriber = Subscriber([Topic.ROBOT_STATE])
    robot_state = subscriber.receive(Topic.ROBOT_STATE, timeout=timeout_ms)
    subscriber.close()

    if robot_state is None:
        raise RuntimeError(f"Could not read current robot state within {timeout_ms}ms")

    # Get initial end-effector pose
    robot = Robot.from_name(robot_name)
    current_pose = robot.get_frame_pose(end_effector_frame, robot_state.joint_positions)

    logger.info("\nInitial end-effector pose:")
    logger.info(f"  Position: {current_pose.translation}")
    rpy = pin.rpy.matrixToRpy(current_pose.rotation)
    logger.info(f"  Orientation (RPY): {rpy} rad")
    logger.info(f"  Orientation (RPY): {np.rad2deg(rpy)} deg")
    logger.info("\nReady! Use keyboard to jog the end-effector.\n")

    publisher = Publisher()
    publish_period = 1.0 / publish_rate

    # Thread-safe lock for accessing current_pose
    pose_lock = threading.Lock()

    # Flag to control the publishing thread
    running = threading.Event()
    running.set()

    def publish_loop():
        """Continuously publish the current tool command at the desired rate."""
        while running.is_set():
            current_time = time.time()

            # Acquire lock to safely read current_pose
            with pose_lock:
                command = RobotToolCommand(
                    timestamp=current_time,
                    pose=current_pose,
                )

            publisher.publish(command)

            # Sleep to maintain the desired publish rate
            time.sleep(publish_period)

    def log_pose():
        """Log the current pose."""
        rpy = pin.rpy.matrixToRpy(current_pose.rotation)
        logger.info(f"Position: {current_pose.translation}")
        logger.info(f"Orientation (RPY): {np.rad2deg(rpy)} deg")

    def on_press(key):  # noqa: PLR0912, PLR0915
        nonlocal current_pose

        try:
            # Get the key character if available
            key_char = None
            if hasattr(key, "char"):
                key_char = key.char

            # Translation controls
            if key_char == "w":
                # Move forward (positive X)
                with pose_lock:
                    current_pose.translation[0] += translation_step
                logger.info(f"↑ Forward (X+{translation_step:.4f}m)")
                log_pose()
            elif key_char == "s":
                # Move backward (negative X)
                with pose_lock:
                    current_pose.translation[0] -= translation_step
                logger.info(f"↓ Backward (X-{translation_step:.4f}m)")
                log_pose()
            elif key_char == "a":
                # Move left (negative Y)
                with pose_lock:
                    current_pose.translation[1] -= translation_step
                logger.info(f"← Left (Y-{translation_step:.4f}m)")
                log_pose()
            elif key_char == "d":
                # Move right (positive Y)
                with pose_lock:
                    current_pose.translation[1] += translation_step
                logger.info(f"→ Right (Y+{translation_step:.4f}m)")
                log_pose()
            elif key_char == "q":
                # Move up (positive Z)
                with pose_lock:
                    current_pose.translation[2] += translation_step
                logger.info(f"⬆ Up (Z+{translation_step:.4f}m)")
                log_pose()
            elif key_char == "e":
                # Move down (negative Z)
                with pose_lock:
                    current_pose.translation[2] -= translation_step
                logger.info(f"⬇ Down (Z-{translation_step:.4f}m)")
                log_pose()

            # Rotation controls (applied in the current frame)
            elif key_char == "i":
                # Pitch up (positive rotation around Y axis)
                rotation = pin.utils.rotate("y", rotation_step)
                with pose_lock:
                    current_pose.rotation = current_pose.rotation @ rotation
                logger.info(f"⤴ Pitch up (+{np.rad2deg(rotation_step):.2f}°)")
                log_pose()
            elif key_char == "k":
                # Pitch down (negative rotation around Y axis)
                rotation = pin.utils.rotate("y", -rotation_step)
                with pose_lock:
                    current_pose.rotation = current_pose.rotation @ rotation
                logger.info(f"⤵ Pitch down (-{np.rad2deg(rotation_step):.2f}°)")
                log_pose()
            elif key_char == "j":
                # Yaw left (positive rotation around Z axis)
                rotation = pin.utils.rotate("z", rotation_step)
                with pose_lock:
                    current_pose.rotation = current_pose.rotation @ rotation
                logger.info(f"↶ Yaw left (+{np.rad2deg(rotation_step):.2f}°)")
                log_pose()
            elif key_char == "l":
                # Yaw right (negative rotation around Z axis)
                rotation = pin.utils.rotate("z", -rotation_step)
                with pose_lock:
                    current_pose.rotation = current_pose.rotation @ rotation
                logger.info(f"↷ Yaw right (-{np.rad2deg(rotation_step):.2f}°)")
                log_pose()
            elif key_char == "u":
                # Roll counter-clockwise (positive rotation around X axis)
                rotation = pin.utils.rotate("x", rotation_step)
                with pose_lock:
                    current_pose.rotation = current_pose.rotation @ rotation
                logger.info(f"↺ Roll CCW (+{np.rad2deg(rotation_step):.2f}°)")
                log_pose()
            elif key_char == "o":
                # Roll clockwise (negative rotation around X axis)
                rotation = pin.utils.rotate("x", -rotation_step)
                with pose_lock:
                    current_pose.rotation = current_pose.rotation @ rotation
                logger.info(f"↻ Roll CW (-{np.rad2deg(rotation_step):.2f}°)")
                log_pose()
            elif key_char == "x" or key == keyboard.Key.esc:
                logger.info("Exiting jog mode...")
                running.clear()  # Signal the publish thread to stop
                return False  # Stop listener

        except Exception as e:
            logger.error(f"Error during jog: {e}")

    try:
        # Start the publishing thread
        publish_thread = threading.Thread(target=publish_loop, daemon=True)
        publish_thread.start()

        # Start listening for keyboard events
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

        # Stop the publishing thread
        running.clear()
        publish_thread.join(timeout=1.0)

        logger.info("\nFinal end-effector pose:")
        with pose_lock:
            logger.info(f"  Position: {current_pose.translation}")
            rpy = pin.rpy.matrixToRpy(current_pose.rotation)
        logger.info(f"  Orientation (RPY): {rpy} rad")
        logger.info(f"  Orientation (RPY): {np.rad2deg(rpy)} deg")

    except KeyboardInterrupt:
        logger.info("Exiting")
        running.clear()


def main():
    parser = argparse.ArgumentParser(
        description="Jog robot end-effector pose using keyboard by publishing RobotToolCommand",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Jog end-effector using default step sizes
  uv run python -m humanoid.motions.jog_tool_pose

  # Jog with custom translation step (2cm)
  uv run python -m humanoid.motions.jog_tool_pose --translation-step 0.02

  # Jog with custom rotation step (5 degrees)
  uv run python -m humanoid.motions.jog_tool_pose --rotation-step 0.087

  # Jog with custom publish rate
  uv run python -m humanoid.motions.jog_tool_pose --rate 50

Controls:
  Translation:
    W/S - Move forward/backward (X axis)
    A/D - Move left/right (Y axis)
    Q/E - Move up/down (Z axis)
  Rotation:
    I/K - Pitch up/down
    J/L - Yaw left/right
    U/O - Roll counter-clockwise/clockwise
  ESC or 'x' - Quit
        """,
    )
    parser.add_argument(
        "--translation-step",
        type=float,
        default=DEFAULT_TRANSLATION_STEP,
        help=f"Translation step size in meters (default: {DEFAULT_TRANSLATION_STEP})",
    )
    parser.add_argument(
        "--rotation-step",
        type=float,
        default=DEFAULT_ROTATION_STEP,
        help=f"Rotation step size in radians (default: {DEFAULT_ROTATION_STEP})",
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

    # Validate translation step
    if args.translation_step <= 0:
        logger.error(f"Invalid translation step: {args.translation_step}. Must be positive.")
        sys.exit(1)

    # Validate rotation step
    if args.rotation_step <= 0:
        logger.error(f"Invalid rotation step: {args.rotation_step}. Must be positive.")
        sys.exit(1)

    # Validate publish rate
    if args.rate <= 0:
        logger.error(f"Invalid publish rate: {args.rate}. Must be positive.")
        sys.exit(1)

    # Validate timeout
    if args.timeout <= 0:
        logger.error(f"Invalid timeout: {args.timeout}. Must be positive.")
        sys.exit(1)

    jog_tool_pose(
        translation_step=args.translation_step,
        rotation_step=args.rotation_step,
        publish_rate=args.rate,
        timeout_ms=args.timeout,
    )


if __name__ == "__main__":
    main()
