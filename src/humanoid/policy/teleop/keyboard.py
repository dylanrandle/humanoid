"""Keyboard teleoperation policy for controlling robot end-effector pose."""

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import pinocchio as pin
from pynput import keyboard

from humanoid.config import ROBOT_CONFIG
from humanoid.logger import get_logger
from humanoid.policy.teleop.base import BaseTeleopPolicy
from humanoid.types.action import Action
from humanoid.types.observation import Observation
from humanoid.types.robot import RobotConfig

logger = get_logger(__name__)

DEFAULT_TRANSLATION_STEP = 0.005  # 0.5 cm
DEFAULT_ROTATION_STEP = 0.05  # ~2.86 degrees
DEFAULT_GRIPPER_STEP_PCT = 0.025  # 2.5% of gripper range per keypress
DEFAULT_BASE_TRANSLATION_STEP = 0.01  # 1 cm per keypress
DEFAULT_BASE_ROTATION_STEP = 0.05  # ~2.86 degrees per keypress


@dataclass
class KeyboardTeleopPolicyConfig:
    """Tunable parameters for KeyboardTeleopPolicy.

    Args:
        translation_step: Step size in meters for end-effector translation.
        rotation_step: Step size in radians for end-effector rotation.
        gripper_step_pct: Percentage of gripper range per keypress (e.g. 0.025 = 2.5%).
        base_translation_step: Step size in meters for base translation.
        base_rotation_step: Step size in radians for base yaw.
        verbose: Whether to log pose updates.
    """

    translation_step: float = DEFAULT_TRANSLATION_STEP
    rotation_step: float = DEFAULT_ROTATION_STEP
    gripper_step_pct: float = DEFAULT_GRIPPER_STEP_PCT
    base_translation_step: float = DEFAULT_BASE_TRANSLATION_STEP
    base_rotation_step: float = DEFAULT_BASE_ROTATION_STEP
    verbose: bool = True


class KeyboardTeleopPolicy(BaseTeleopPolicy):
    """Policy that allows keyboard control of the robot's end-effector pose and gripper.

    This policy captures keyboard input globally and updates the target end-effector
    pose and gripper position based on user input. The pose is continuously published as actions.

    Controls:
        - W/S: Move right/left (Y axis)
        - A/D: Move backward/forward (X axis)
        - Q/E: Move down/up (Z axis)
        - I/K: Roll clockwise/counter-clockwise
        - J/L: Yaw left/right
        - U/O: Pitch down/up
        - [/]: Close/open gripper
        - Arrow Up/Down: Move base forward/backward (X axis)
        - Arrow Left/Right: Strafe base left/right (Y axis)
        - ,/. : Yaw base left/right (Z axis)
        - ESC or 'x': Quit (sets running flag to False)

    Args:
        robot_config: Robot configuration (default: ROBOT_CONFIG)
        config: Tunable policy parameters (default: KeyboardTeleopPolicyConfig())
    """

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        config: KeyboardTeleopPolicyConfig | None = None,
    ):
        """Initialize the keyboard teleoperation policy."""
        if config is None:
            config = KeyboardTeleopPolicyConfig()
        super().__init__(robot_config=robot_config, verbose=config.verbose)

        self.config = config
        self.translation_step = config.translation_step
        self.rotation_step = config.rotation_step
        self.base_translation_step = config.base_translation_step
        self.base_rotation_step = config.base_rotation_step

        gripper_range = self.gripper_max - self.gripper_min
        self.gripper_step = gripper_range * config.gripper_step_pct
        if self.verbose and robot_config.gripper_joint_indices:
            logger.info(
                f"Gripper step: {self.gripper_step:.4f} "
                f"({self.gripper_step * 1000:.2f}mm, "
                f"{config.gripper_step_pct * 100:.1f}% of range)"
            )

        # Current target pose (will be initialized on first observation)
        self.current_pose: pin.SE3 | None = None

        # Current gripper positions (will be initialized on first observation)
        # Track gripper positions based on gripper_joint_indices from config
        self.gripper_positions: np.ndarray | None = None

        # Current target base pose (will be initialized on first observation if base_frame set)
        self.current_base_pose: pin.SE3 | None = None

        # Thread-safe lock for accessing current_pose, gripper_positions, and base pose
        self.lock = threading.Lock()

        # Flag to track if policy is running
        self.running = True

        # Keyboard listener
        self.listener: keyboard.Listener | None = None

        if self.verbose:
            self._log_init_info()

    def _log_init_info(self) -> None:
        """Log initialization info and controls."""
        logger.info(f"KeyboardTeleopPolicy initialized for {self.robot_config.name}")
        logger.info(f"End effector frame: {self.robot_config.tool_frame}")
        if self.robot_config.gripper_joint_indices:
            logger.info(f"Gripper joint indices: {self.robot_config.gripper_joint_indices}")
        logger.info(
            f"Translation step: {self.translation_step:.4f} m "
            f"({self.translation_step * 100:.2f} cm)"
        )
        logger.info(
            f"Rotation step: {self.rotation_step:.4f} rad "
            f"(~{np.rad2deg(self.rotation_step):.2f} degrees)"
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
        logger.info("  Gripper:")
        logger.info("    [ - Close gripper")
        logger.info("    ] - Open gripper")
        if self.robot_config.base_frame is not None:
            logger.info("  Base:")
            logger.info("    Up/Down arrow - Move base forward/backward (X axis)")
            logger.info("    Left/Right arrow - Strafe base left/right (Y axis)")
            logger.info("    , / . - Yaw base left/right")
        logger.info("  ESC or 'x' - Quit")
        logger.info("\nNote: Keyboard input works globally (terminal doesn't need focus)")

    def reset(self) -> None:
        """Reset policy state."""
        with self.lock:
            self.current_pose = None
            self.gripper_positions = None
            self.current_base_pose = None
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
        """Log the current pose and gripper positions (must be called with pose_lock held)."""
        if self.current_pose is not None and self.verbose:
            rpy = pin.rpy.matrixToRpy(self.current_pose.rotation)
            logger.info(f"Position: {self.current_pose.translation}")
            logger.info(f"Orientation (RPY): {np.rad2deg(rpy)} deg")
            if self.gripper_positions is not None:
                logger.info(f"Gripper positions: {self.gripper_positions}")

    def _log_base_pose(self) -> None:
        """Log the current base pose (must be called with lock held)."""
        if self.current_base_pose is not None and self.verbose:
            rpy = pin.rpy.matrixToRpy(self.current_base_pose.rotation)
            logger.info(f"Base position: {self.current_base_pose.translation}")
            logger.info(f"Base orientation (RPY): {np.rad2deg(rpy)} deg")

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
                with self.lock:
                    self.current_pose.translation[1] += self.translation_step
                    if self.verbose:
                        logger.info(f"→ Right (Y+{self.translation_step:.4f}m)")
                        self._log_pose()
            elif key_char == "s":
                # Move left (negative Y)
                with self.lock:
                    self.current_pose.translation[1] -= self.translation_step
                    if self.verbose:
                        logger.info(f"← Left (Y-{self.translation_step:.4f}m)")
                        self._log_pose()
            elif key_char == "a":
                # Move backward (negative X)
                with self.lock:
                    self.current_pose.translation[0] -= self.translation_step
                    if self.verbose:
                        logger.info(f"↓ Backward (X-{self.translation_step:.4f}m)")
                        self._log_pose()
            elif key_char == "d":
                # Move forward (positive X)
                with self.lock:
                    self.current_pose.translation[0] += self.translation_step
                    if self.verbose:
                        logger.info(f"↑ Forward (X+{self.translation_step:.4f}m)")
                        self._log_pose()
            elif key_char == "q":
                # Move down (negative Z)
                with self.lock:
                    self.current_pose.translation[2] -= self.translation_step
                    if self.verbose:
                        logger.info(f"⬇ Down (Z-{self.translation_step:.4f}m)")
                        self._log_pose()
            elif key_char == "e":
                # Move up (positive Z)
                with self.lock:
                    self.current_pose.translation[2] += self.translation_step
                    if self.verbose:
                        logger.info(f"⬆ Up (Z+{self.translation_step:.4f}m)")
                        self._log_pose()

            # Rotation controls (applied in the current frame)
            elif key_char == "i":
                # Roll clockwise (negative rotation around X axis)
                rotation = pin.utils.rotate("x", -self.rotation_step)
                with self.lock:
                    self.current_pose.rotation = self.current_pose.rotation @ rotation
                    if self.verbose:
                        logger.info(f"↻ Roll CW (-{np.rad2deg(self.rotation_step):.2f}°)")
                        self._log_pose()
            elif key_char == "k":
                # Roll counter-clockwise (positive rotation around X axis)
                rotation = pin.utils.rotate("x", self.rotation_step)
                with self.lock:
                    self.current_pose.rotation = self.current_pose.rotation @ rotation
                    if self.verbose:
                        logger.info(f"↺ Roll CCW (+{np.rad2deg(self.rotation_step):.2f}°)")
                        self._log_pose()
            elif key_char == "j":
                # Yaw left (positive rotation around Z axis)
                rotation = pin.utils.rotate("z", self.rotation_step)
                with self.lock:
                    self.current_pose.rotation = self.current_pose.rotation @ rotation
                    if self.verbose:
                        logger.info(f"↶ Yaw left (+{np.rad2deg(self.rotation_step):.2f}°)")
                        self._log_pose()
            elif key_char == "l":
                # Yaw right (negative rotation around Z axis)
                rotation = pin.utils.rotate("z", -self.rotation_step)
                with self.lock:
                    self.current_pose.rotation = self.current_pose.rotation @ rotation
                    if self.verbose:
                        logger.info(f"↷ Yaw right (-{np.rad2deg(self.rotation_step):.2f}°)")
                        self._log_pose()
            elif key_char == "u":
                # Pitch down (negative rotation around Y axis)
                rotation = pin.utils.rotate("y", -self.rotation_step)
                with self.lock:
                    self.current_pose.rotation = self.current_pose.rotation @ rotation
                    if self.verbose:
                        logger.info(f"⤵ Pitch down (-{np.rad2deg(self.rotation_step):.2f}°)")
                        self._log_pose()
            elif key_char == "o":
                # Pitch up (positive rotation around Y axis)
                rotation = pin.utils.rotate("y", self.rotation_step)
                with self.lock:
                    self.current_pose.rotation = self.current_pose.rotation @ rotation
                    if self.verbose:
                        logger.info(f"⤴ Pitch up (+{np.rad2deg(self.rotation_step):.2f}°)")
                        self._log_pose()
            elif key_char == "[":
                # Close gripper (decrease gripper position)
                with self.lock:
                    if self.gripper_positions is not None:
                        # Decrease gripper position with URDF-based limits
                        self.gripper_positions[0] = max(
                            self.gripper_min, self.gripper_positions[0] - self.gripper_step
                        )
                        if self.verbose:
                            logger.info(
                                f"⊏ Close gripper (-{self.gripper_step:.4f}, "
                                f"-{self.gripper_step * 1000:.2f}mm)"
                            )
                            logger.info(f"Gripper position: {self.gripper_positions[0]:.4f}")
            elif key_char == "]":
                # Open gripper (increase gripper position)
                with self.lock:
                    if self.gripper_positions is not None:
                        # Increase gripper position with URDF-based limits
                        self.gripper_positions[0] = min(
                            self.gripper_max, self.gripper_positions[0] + self.gripper_step
                        )
                        if self.verbose:
                            logger.info(
                                f"⊐ Open gripper (+{self.gripper_step:.4f}, "
                                f"+{self.gripper_step * 1000:.2f}mm)"
                            )
                            logger.info(f"Gripper position: {self.gripper_positions[0]:.4f}")
            # Base controls (only active if base_frame configured)
            elif key == keyboard.Key.up:
                with self.lock:
                    if self.current_base_pose is not None:
                        self.current_base_pose.translation[0] += self.base_translation_step
                        if self.verbose:
                            logger.info(f"⇧ Base forward (X+{self.base_translation_step:.4f}m)")
                            self._log_base_pose()
            elif key == keyboard.Key.down:
                with self.lock:
                    if self.current_base_pose is not None:
                        self.current_base_pose.translation[0] -= self.base_translation_step
                        if self.verbose:
                            logger.info(f"⇩ Base backward (X-{self.base_translation_step:.4f}m)")
                            self._log_base_pose()
            elif key == keyboard.Key.left:
                with self.lock:
                    if self.current_base_pose is not None:
                        self.current_base_pose.translation[1] += self.base_translation_step
                        if self.verbose:
                            logger.info(f"⇦ Base strafe left (Y+{self.base_translation_step:.4f}m)")
                            self._log_base_pose()
            elif key == keyboard.Key.right:
                with self.lock:
                    if self.current_base_pose is not None:
                        self.current_base_pose.translation[1] -= self.base_translation_step
                        if self.verbose:
                            logger.info(
                                f"⇨ Base strafe right (Y-{self.base_translation_step:.4f}m)"
                            )
                            self._log_base_pose()
            elif key_char == ",":
                # Yaw base left (positive rotation around Z)
                rotation = pin.utils.rotate("z", self.base_rotation_step)
                with self.lock:
                    if self.current_base_pose is not None:
                        self.current_base_pose.rotation = self.current_base_pose.rotation @ rotation
                        if self.verbose:
                            logger.info(
                                f"↶ Base yaw left (+{np.rad2deg(self.base_rotation_step):.2f}°)"
                            )
                            self._log_base_pose()
            elif key_char == ".":
                # Yaw base right (negative rotation around Z)
                rotation = pin.utils.rotate("z", -self.base_rotation_step)
                with self.lock:
                    if self.current_base_pose is not None:
                        self.current_base_pose.rotation = self.current_base_pose.rotation @ rotation
                        if self.verbose:
                            logger.info(
                                f"↷ Base yaw right (-{np.rad2deg(self.base_rotation_step):.2f}°)"
                            )
                            self._log_base_pose()
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

        On the first call, initializes the target pose and gripper positions
        from the current robot state.

        Subsequently returns the current target pose as modified by keyboard input.

        Args:
            observation: Current observation from the environment

        Returns:
            Action containing the target tool pose
        """
        # Initialize pose and gripper positions on first call
        if self.current_pose is None:
            with self.lock:
                self.current_pose = self._get_current_tool_pose(observation)

                # Initialize gripper positions from current state
                self.gripper_positions = self._get_current_gripper_positions(observation)

                # Initialize base pose from current state if base_frame is configured
                base_pose = self._get_current_base_pose(observation)
                if base_pose is not None:
                    # Copy so subsequent FK calls don't mutate our target
                    self.current_base_pose = pin.SE3(base_pose.rotation, base_pose.translation)

            if self.verbose:
                logger.info("\nInitial end-effector pose:")
                logger.info(f"  Position: {self.current_pose.translation}")
                rpy = pin.rpy.matrixToRpy(self.current_pose.rotation)
                logger.info(f"  Orientation (RPY): {rpy} rad")
                logger.info(f"  Orientation (RPY): {np.rad2deg(rpy)} deg")
                if self.gripper_positions is not None:
                    logger.info(f"  Gripper positions: {self.gripper_positions}")
                if self.current_base_pose is not None:
                    logger.info("\nInitial base pose:")
                    logger.info(f"  Position: {self.current_base_pose.translation}")
                    base_rpy = pin.rpy.matrixToRpy(self.current_base_pose.rotation)
                    logger.info(f"  Orientation (RPY): {np.rad2deg(base_rpy)} deg")
                logger.info("\nReady! Use keyboard to jog the end-effector.\n")

            # Start keyboard listener
            self.start_listener()

        # Return current target pose and gripper positions
        with self.lock:
            # Create a copy of the pose to avoid race conditions
            target_pose = pin.SE3(self.current_pose.rotation, self.current_pose.translation)
            # Copy gripper positions if available
            gripper_positions_copy = (
                self.gripper_positions.copy() if self.gripper_positions is not None else None
            )
            base_pose_copy = (
                pin.SE3(self.current_base_pose.rotation, self.current_base_pose.translation)
                if self.current_base_pose is not None
                else None
            )

        return Action(
            tool_pose=target_pose,
            gripper_positions=gripper_positions_copy,
            base_pose=base_pose_copy,
        )

    def __del__(self) -> None:
        """Cleanup when policy is destroyed."""
        self.stop_listener()
