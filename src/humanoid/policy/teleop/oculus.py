"""Oculus VR teleoperation policy for controlling robot end-effector pose."""

import time

import numpy as np
import pinocchio as pin
from oculus_reader import OculusReader

from humanoid.config import ROBOT_CONFIG
from humanoid.logger import get_logger
from humanoid.policy.base import Policy
from humanoid.robots.base import Robot
from humanoid.types.action import Action
from humanoid.types.observation import Observation
from humanoid.types.robot import RobotConfig

logger = get_logger(__name__)

# Oculus controller data keys
RIGHT_CONTROLLER_KEY = "r"
RIGHT_TRIGGER_KEY = "rightTrig"
A_BUTTON_KEY = "A"


class OculusTeleopPolicy(Policy):
    """Policy that uses Oculus VR controllers to control the robot's end-effector.

    This policy uses the right controller's pose to command the tool pose and the
    right trigger to command the gripper width.

    Controls:
        - Right controller pose: Controls end-effector position and orientation
        - Right trigger: Controls gripper width (0.0 = closed, 1.0 = fully open)

    Args:
        robot_config: Robot configuration (default: ROBOT_CONFIG)
        scale_translation: Scale factor for controller translation (default: 1.0)
        scale_rotation: Scale factor for controller rotation (default: 1.0)
        verbose: Whether to log pose updates (default: True)
    """

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        scale_translation: float = 1.0,
        scale_rotation: float = 1.0,
        verbose: bool = True,
    ):
        """Initialize the Oculus teleoperation policy."""
        self.robot_config = robot_config
        self.scale_translation = scale_translation
        self.scale_rotation = scale_rotation
        self.verbose = verbose

        # Robot instance for forward kinematics
        self.robot = Robot(robot_config)

        # Oculus reader
        self.reader = OculusReader()

        while not all(self.reader.get_transformations_and_buttons()):
            logger.info("Waiting for Oculus data...")
            time.sleep(0.1)

        # Reference poses (set on first observation or reset when 'A' button is pressed)
        self.reference_controller_pose: np.ndarray | None = None
        self.reference_tool_pose: pin.SE3 | None = None

        # Gripper limits
        if robot_config.gripper_joint_indices:
            # For Oculus teleop, we only support commanding a single gripper joint
            assert len(robot_config.gripper_joint_indices) == 1, (
                f"OculusTeleopPolicy only supports 1 gripper joint, "
                f"but {len(robot_config.gripper_joint_indices)} were specified"
            )

            # Get limits for the gripper joint
            gripper_idx = robot_config.gripper_joint_indices[0]
            self.gripper_min = self.robot.model.lowerPositionLimit[gripper_idx]
            self.gripper_max = self.robot.model.upperPositionLimit[gripper_idx]
            gripper_range = self.gripper_max - self.gripper_min

            if self.verbose:
                logger.info(
                    f"Gripper joint {gripper_idx}: "
                    f"[{self.gripper_min:.4f}, {self.gripper_max:.4f}] "
                    f"(range: {gripper_range:.4f}, {gripper_range * 1000:.2f}mm)"
                )
        else:
            # No gripper joints
            self.gripper_min = 0.0
            self.gripper_max = 0.0

        # Log configuration
        if self.verbose:
            logger.info(f"OculusTeleopPolicy initialized for {robot_config.name}")
            logger.info(f"End effector frame: {robot_config.end_effector_frame}")
            if robot_config.gripper_joint_indices:
                logger.info(f"Gripper joint indices: {robot_config.gripper_joint_indices}")
            logger.info(f"Translation scale: {scale_translation:.2f}")
            logger.info(f"Rotation scale: {scale_rotation:.2f}")
            logger.info("\nControls:")
            logger.info("  Right controller pose -> End-effector pose")
            logger.info("  Right trigger -> Gripper width (0.0=closed, 1.0=open)")
            logger.info("  'A' button -> Reset reference poses (allows free controller movement)")

    def reset(self) -> None:
        """Reset policy state."""
        self.reference_controller_pose = None
        self.reference_tool_pose = None

    def _has_valid_controller_data(self, transforms: dict, buttons: dict) -> bool:
        """Check if controller data is valid and contains required keys.

        Args:
            transforms: Dictionary of controller transformations
            buttons: Dictionary of button states

        Returns:
            True if data is valid, False otherwise
        """
        return bool(
            transforms
            and RIGHT_CONTROLLER_KEY in transforms
            and buttons
            and RIGHT_TRIGGER_KEY in buttons
            and A_BUTTON_KEY in buttons
        )

    def _get_current_gripper_positions(self, observation: Observation) -> np.ndarray | None:
        """Get current gripper positions from observation.

        Args:
            observation: Current observation from the environment

        Returns:
            Array of gripper positions or None if no gripper joints
        """
        if self.robot_config.gripper_joint_indices:
            gripper_idx = self.robot_config.gripper_joint_indices[0]
            gripper_position = observation.robot_state.joint_positions[gripper_idx]
            return np.array([gripper_position])
        return None

    def _initialize_reference_poses(
        self, right_controller_pose: np.ndarray, observation: Observation
    ) -> None:
        """Initialize reference poses from current controller and robot state.

        Args:
            right_controller_pose: Current right controller pose (4x4 matrix)
            observation: Current observation from the environment
        """
        self.reference_controller_pose = right_controller_pose.copy()
        self.reference_tool_pose = self.robot.get_frame_pose(
            self.robot_config.end_effector_frame,
            observation.robot_state.joint_positions,
        )

        if self.verbose:
            logger.info("\nReference poses initialized:")
            logger.info(f"  Tool position: {self.reference_tool_pose.translation}")
            rpy = pin.rpy.matrixToRpy(self.reference_tool_pose.rotation)
            logger.info(f"  Tool orientation (RPY): {np.rad2deg(rpy)} deg")
            logger.info("\nReady! Move the right controller to control the robot.\n")

    def _handle_uninitialized_state(self, observation: Observation) -> Action:
        """Handle case when controller data is not yet available or 'A' button is pressed.

        Returns an action based on the current robot state to maintain position
        until valid controller data is received or while 'A' button is held.

        Args:
            observation: Current observation from the environment

        Returns:
            Action to maintain current pose
        """
        current_tool_pose = self.robot.get_frame_pose(
            self.robot_config.end_effector_frame,
            observation.robot_state.joint_positions,
        )
        gripper_positions = self._get_current_gripper_positions(observation)
        return Action(tool_pose=current_tool_pose, gripper_positions=gripper_positions)

    def __call__(self, observation: Observation) -> Action:
        """Generate an action given an observation.

        On the first call, initializes the reference poses from the current robot state
        and controller pose. When 'A' button is pressed, resets reference poses and holds
        current position. When 'A' is released, new reference poses are set on next call.

        Subsequently returns the target pose based on the controller's pose relative to
        the reference pose, with configurable scaling for translation and rotation.

        Args:
            observation: Current observation from the environment

        Returns:
            Action containing the target tool pose and gripper positions
        """
        # Get current controller data
        transforms, buttons = self.reader.get_transformations_and_buttons()

        # Check if we have valid data (OculusReader may return empty dicts on startup)
        if not self._has_valid_controller_data(transforms, buttons):
            return self._handle_uninitialized_state(observation)

        # Reset reference poses if A button is pressed
        if buttons[A_BUTTON_KEY]:
            self.reset()
            return self._handle_uninitialized_state(observation)

        # Extract right controller pose (4x4 transformation matrix)
        right_controller_pose = transforms[RIGHT_CONTROLLER_KEY]

        # Initialize reference poses on first call
        if self.reference_controller_pose is None or self.reference_tool_pose is None:
            self._initialize_reference_poses(right_controller_pose, observation)

        assert self.reference_controller_pose is not None, "Missing reference controller pose!"

        # Compute relative transformation from reference controller pose to current
        # T_current = T_ref * T_delta
        # T_delta = T_ref^-1 * T_current
        ref_controller_SE3 = pin.SE3(
            self.reference_controller_pose[:3, :3],
            self.reference_controller_pose[:3, 3],
        )
        current_controller_SE3 = pin.SE3(
            right_controller_pose[:3, :3],
            right_controller_pose[:3, 3],
        )

        delta_controller = ref_controller_SE3.inverse() * current_controller_SE3

        # Apply scaling to translation
        scaled_translation = delta_controller.translation * self.scale_translation

        # Apply scaling to rotation (scale the rotation vector)
        if self.scale_rotation != 1.0:
            # Convert rotation to axis-angle, scale, and convert back
            rotation_vec = pin.log3(delta_controller.rotation)
            scaled_rotation_vec = rotation_vec * self.scale_rotation
            scaled_rotation = pin.exp3(scaled_rotation_vec)
        else:
            scaled_rotation = delta_controller.rotation

        # Create scaled delta transformation
        scaled_delta = pin.SE3(scaled_rotation, scaled_translation)

        # Apply delta to reference tool pose
        target_pose = self.reference_tool_pose * scaled_delta

        # Get gripper position from right trigger
        # rightTrig is a tuple with one element (trigger value from 0.0 to 1.0)
        trigger_value = buttons[RIGHT_TRIGGER_KEY][0]

        # Map trigger value to gripper position
        # trigger 0.0 (not pressed) -> gripper_min (closed)
        # trigger 1.0 (fully pressed) -> gripper_max (open)
        if self.robot_config.gripper_joint_indices:
            gripper_position = self.gripper_min + trigger_value * (
                self.gripper_max - self.gripper_min
            )
            gripper_positions = np.array([gripper_position])
        else:
            gripper_positions = None

        return Action(tool_pose=target_pose, gripper_positions=gripper_positions)
