"""Keyboard teleoperation policy for controlling robot end-effector pose."""

import threading
from typing import Any

import numpy as np
import pinocchio as pin
from pynput import keyboard

from humanoid.config import ROBOT_CONFIG
from humanoid.logger import get_logger
from humanoid.policy.base import Policy
from humanoid.robots.base import Robot
from humanoid.types.action import Action
from humanoid.types.observation import Observation
from humanoid.types.robot import RobotConfig

logger = get_logger(__name__)

DEFAULT_TRANSLATION_STEP = 0.005  # 0.5 cm
DEFAULT_ROTATION_STEP = 0.05  # ~2.86 degrees


class KeyboardTeleopPolicy(Policy):
    """Policy that allows keyboard control of the robot's end-effector pose.

    This policy captures keyboard input globally and updates the target end-effector
    pose based on user input. The pose is continuously published as actions.

    Controls:
        - W/S: Move right/left (Y axis)
        - A/D: Move backward/forward (X axis)
        - Q/E: Move down/up (Z axis)
        - I/K: Roll clockwise/counter-clockwise
        - J/L: Yaw left/right
        - U/O: Pitch down/up
        - ESC or 'x': Quit (sets running flag to False)

    Args:
        translation_step: Step size in meters for translation (default: 0.01)
        rotation_step: Step size in radians for rotation (default: 0.05)
        robot_config: Robot configuration (default: ROBOT_CONFIG)
        verbose: Whether to log pose updates (default: True)
    """

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        translation_step: float = DEFAULT_TRANSLATION_STEP,
        rotation_step: float = DEFAULT_ROTATION_STEP,
        verbose: bool = True,
    ):
        """Initialize the keyboard teleoperation policy."""
        self.translation_step = translation_step
        self.rotation_step = rotation_step
        self.robot_config = robot_config
        self.verbose = verbose

        # Robot instance for forward kinematics
        self.robot = Robot(robot_config)

        # Current target pose (will be initialized on first observation)
        self.current_pose: pin.SE3 | None = None

        # Thread-safe lock for accessing current_pose
        self.pose_lock = threading.Lock()

        # Flag to track if policy is running
        self.running = True

        # Keyboard listener
        self.listener: keyboard.Listener | None = None

        # Log configuration
        if self.verbose:
            logger.info(f"KeyboardTeleopPolicy initialized for {robot_config.name}")
            logger.info(f"End effector frame: {robot_config.end_effector_frame}")
            logger.info(
                f"Translation step: {translation_step:.4f} m ({translation_step * 100:.2f} cm)"
            )
            logger.info(
                f"Rotation step: {rotation_step:.4f} rad (~{np.rad2deg(rotation_step):.2f} degrees)"
            )
            logger.info("\nControls:")
            logger.info("  Translation:")
            logger.info("    W/S - Move right/left (Y axis)")
            logger.info("    A/D - Move backward/forward (X axis)")
            logger.info("    Q/E - Move down/up (Z axis)")
            logger.info("  Rotation:")
            logger.info("    I/K - Roll clockwise/counter-clockwise")
            logger.info("    J/L - Yaw left/right")
            logger.info("    U/O - Pitch down/up")
            logger.info("  ESC or 'x' - Quit")
            logger.info("\nNote: Keyboard input works globally (terminal doesn't need focus)")

    def reset(self) -> None:
        """Reset policy state."""
        with self.pose_lock:
            self.current_pose = None
        self.running = True

        # Stop existing listener if any
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

    def start_listener(self) -> None:
        """Start the keyboard listener if not already running."""
        if self.listener is None or not self.listener.running:
            self.listener = keyboard.Listener(on_press=self._on_press)
            self.listener.start()

    def stop_listener(self) -> None:
        """Stop the keyboard listener."""
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

    def _log_pose(self) -> None:
        """Log the current pose (must be called with pose_lock held)."""
        if self.current_pose is not None and self.verbose:
            rpy = pin.rpy.matrixToRpy(self.current_pose.rotation)
            logger.info(f"Position: {self.current_pose.translation}")
            logger.info(f"Orientation (RPY): {np.rad2deg(rpy)} deg")

    def _on_press(self, key: Any) -> bool:  # noqa: PLR0912, PLR0915
        """Handle keyboard press events.

        Args:
            key: The key that was pressed

        Returns:
            False to stop the listener, True to continue
        """
        if self.current_pose is None:
            return True

        try:
            # Get the key character if available
            key_char = None
            if hasattr(key, "char"):
                key_char = key.char

            # Translation controls
            if key_char == "w":
                # Move right (positive Y)
                with self.pose_lock:
                    self.current_pose.translation[1] += self.translation_step
                    if self.verbose:
                        logger.info(f"→ Right (Y+{self.translation_step:.4f}m)")
                        self._log_pose()
            elif key_char == "s":
                # Move left (negative Y)
                with self.pose_lock:
                    self.current_pose.translation[1] -= self.translation_step
                    if self.verbose:
                        logger.info(f"← Left (Y-{self.translation_step:.4f}m)")
                        self._log_pose()
            elif key_char == "a":
                # Move backward (negative X)
                with self.pose_lock:
                    self.current_pose.translation[0] -= self.translation_step
                    if self.verbose:
                        logger.info(f"↓ Backward (X-{self.translation_step:.4f}m)")
                        self._log_pose()
            elif key_char == "d":
                # Move forward (positive X)
                with self.pose_lock:
                    self.current_pose.translation[0] += self.translation_step
                    if self.verbose:
                        logger.info(f"↑ Forward (X+{self.translation_step:.4f}m)")
                        self._log_pose()
            elif key_char == "q":
                # Move down (negative Z)
                with self.pose_lock:
                    self.current_pose.translation[2] -= self.translation_step
                    if self.verbose:
                        logger.info(f"⬇ Down (Z-{self.translation_step:.4f}m)")
                        self._log_pose()
            elif key_char == "e":
                # Move up (positive Z)
                with self.pose_lock:
                    self.current_pose.translation[2] += self.translation_step
                    if self.verbose:
                        logger.info(f"⬆ Up (Z+{self.translation_step:.4f}m)")
                        self._log_pose()

            # Rotation controls (applied in the current frame)
            elif key_char == "i":
                # Roll clockwise (negative rotation around X axis)
                rotation = pin.utils.rotate("x", -self.rotation_step)
                with self.pose_lock:
                    self.current_pose.rotation = self.current_pose.rotation @ rotation
                    if self.verbose:
                        logger.info(f"↻ Roll CW (-{np.rad2deg(self.rotation_step):.2f}°)")
                        self._log_pose()
            elif key_char == "k":
                # Roll counter-clockwise (positive rotation around X axis)
                rotation = pin.utils.rotate("x", self.rotation_step)
                with self.pose_lock:
                    self.current_pose.rotation = self.current_pose.rotation @ rotation
                    if self.verbose:
                        logger.info(f"↺ Roll CCW (+{np.rad2deg(self.rotation_step):.2f}°)")
                        self._log_pose()
            elif key_char == "j":
                # Yaw left (positive rotation around Z axis)
                rotation = pin.utils.rotate("z", self.rotation_step)
                with self.pose_lock:
                    self.current_pose.rotation = self.current_pose.rotation @ rotation
                    if self.verbose:
                        logger.info(f"↶ Yaw left (+{np.rad2deg(self.rotation_step):.2f}°)")
                        self._log_pose()
            elif key_char == "l":
                # Yaw right (negative rotation around Z axis)
                rotation = pin.utils.rotate("z", -self.rotation_step)
                with self.pose_lock:
                    self.current_pose.rotation = self.current_pose.rotation @ rotation
                    if self.verbose:
                        logger.info(f"↷ Yaw right (-{np.rad2deg(self.rotation_step):.2f}°)")
                        self._log_pose()
            elif key_char == "u":
                # Pitch down (negative rotation around Y axis)
                rotation = pin.utils.rotate("y", -self.rotation_step)
                with self.pose_lock:
                    self.current_pose.rotation = self.current_pose.rotation @ rotation
                    if self.verbose:
                        logger.info(f"⤵ Pitch down (-{np.rad2deg(self.rotation_step):.2f}°)")
                        self._log_pose()
            elif key_char == "o":
                # Pitch up (positive rotation around Y axis)
                rotation = pin.utils.rotate("y", self.rotation_step)
                with self.pose_lock:
                    self.current_pose.rotation = self.current_pose.rotation @ rotation
                    if self.verbose:
                        logger.info(f"⤴ Pitch up (+{np.rad2deg(self.rotation_step):.2f}°)")
                        self._log_pose()
            elif key_char == "x" or key == keyboard.Key.esc:
                if self.verbose:
                    logger.info("Exiting keyboard teleop mode...")
                self.running = False
                return False  # Stop listener

        except Exception as e:
            logger.error(f"Error during keyboard teleop: {e}")

        return True

    def __call__(self, observation: Observation) -> Action:
        """Generate an action given an observation.

        On the first call, initializes the target pose from the current robot state.
        Subsequently returns the current target pose as modified by keyboard input.

        Args:
            observation: Current observation from the environment

        Returns:
            Action containing the target tool pose
        """
        # Initialize pose on first call
        if self.current_pose is None:
            with self.pose_lock:
                self.current_pose = self.robot.get_frame_pose(
                    self.robot_config.end_effector_frame,
                    observation.robot_state.joint_positions,
                )

            if self.verbose:
                logger.info("\nInitial end-effector pose:")
                logger.info(f"  Position: {self.current_pose.translation}")
                rpy = pin.rpy.matrixToRpy(self.current_pose.rotation)
                logger.info(f"  Orientation (RPY): {rpy} rad")
                logger.info(f"  Orientation (RPY): {np.rad2deg(rpy)} deg")
                logger.info("\nReady! Use keyboard to jog the end-effector.\n")

            # Start keyboard listener
            self.start_listener()

        # Return current target pose
        with self.pose_lock:
            # Create a copy of the pose to avoid race conditions
            target_pose = pin.SE3(self.current_pose.rotation, self.current_pose.translation)

        return Action(tool_pose=target_pose)

    def __del__(self) -> None:
        """Cleanup when policy is destroyed."""
        self.stop_listener()
