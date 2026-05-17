"""Oculus VR teleoperation policy for controlling robot end-effector pose."""

import time
from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin
from oculus_reader import OculusReader

from humanoid.config import ROBOT_CONFIG
from humanoid.logger import get_logger
from humanoid.policy.teleop.base import BaseTeleopPolicy
from humanoid.types.action import Action
from humanoid.types.observation import Observation
from humanoid.types.robot import RobotConfig

logger = get_logger(__name__)

# Oculus controller data keys
RIGHT_CONTROLLER_KEY = "r"
RIGHT_TRIGGER_KEY = "rightTrig"
LEFT_GRIP_KEY = "LG"
RIGHT_GRIP_KEY = "RG"
LEFT_JOYSTICK_KEY = "leftJS"
RIGHT_JOYSTICK_KEY = "rightJS"


@dataclass
class OculusTeleopPolicyConfig:
    """Tunable parameters for OculusTeleopPolicy.

    Args:
        scale_translation: Scale factor for controller translation.
        scale_rotation: Scale factor for controller rotation.
        oculus_to_world_rotation: 3x3 matrix mapping Oculus-frame vectors
            into the robot's world frame. Used to reorient the controller
            pose so headset-frame motion matches the operator's intuition
            of the robot's world.
        base_translation_matrix: 2x2 matrix mapping the left joystick
            (jx, jy) in [-1, 1]^2 to a per-tick (dx, dy) translation in the
            base's local frame, scaled by ``base_translation_step``. The
            result is rotated into world coordinates by the base's current
            yaw, so "forward stick" always means forward along the base's
            current heading. Default identity gives stick-right (+jx) ->
            +base-local x and stick-forward (+jy) -> +base-local y.
        base_translation_step: Meters of base translation per loop tick at
            full joystick deflection.
        base_rotation_step: Radians of base yaw per loop tick at full
            right-joystick-x deflection.
        base_yaw_scale: Scalar sign applied to right-joystick x for yaw.
            Default -1 makes stick-right yield -yaw and stick-left +yaw.
        base_deadzone: Per-axis joystick magnitude below which input is
            treated as zero.
        verbose: Whether to log pose updates.
    """

    scale_translation: float = 1.0
    scale_rotation: float = 1.0
    oculus_to_world_rotation: np.ndarray = field(
        # 180 degree rotation about x
        default_factory=lambda: np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0], [0.0, 0.0, -1.0]])
    )
    base_translation_matrix: np.ndarray = field(default_factory=lambda: np.eye(2))
    base_translation_step: float = 0.003
    base_rotation_step: float = 0.02
    base_yaw_scale: float = -1.0
    base_deadzone: float = 0.1
    verbose: bool = True


class OculusTeleopPolicy(BaseTeleopPolicy):
    """Policy that uses Oculus VR controllers to control the robot.

    The right controller's pose commands the tool pose, the right index trigger
    commands the gripper, and (when the robot has a base frame) the joysticks
    command base translation and yaw. Motion is gated by a grip-trigger
    dead-man so the robot only moves while the operator is actively engaged.

    Controls:
        - Right controller pose: End-effector position and orientation
        - Right index trigger: Gripper width (0.0 = closed, 1.0 = fully open)
        - Grip trigger (either hand): Dead-man switch -- motion is only
          commanded while held; releasing both grips clears the controller
          and base reference poses, which re-anchor on re-engage
        - Left joystick: Base XY translation (see ``base_translation_matrix``)
        - Right joystick X: Base yaw (see ``base_yaw_scale``)

    The raw controller pose is remapped from the Oculus frame into the robot's
    world frame via ``oculus_to_world_rotation`` before computing the delta.

    Args:
        robot_config: Robot configuration (default: ROBOT_CONFIG)
        config: Tunable policy parameters (default: OculusTeleopPolicyConfig())
    """

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        config: OculusTeleopPolicyConfig | None = None,
    ):
        """Initialize the Oculus teleoperation policy."""
        if config is None:
            config = OculusTeleopPolicyConfig()
        super().__init__(robot_config=robot_config, verbose=config.verbose)

        self.config = config
        self.scale_translation = config.scale_translation
        self.scale_rotation = config.scale_rotation
        self.oculus_to_world = pin.SE3(config.oculus_to_world_rotation, np.zeros(3))

        # Oculus reader
        self.reader = OculusReader()

        while not all(self.reader.get_transformations_and_buttons()):
            logger.info("Waiting for Oculus data...")
            time.sleep(0.1)

        # Reference poses (set on first engaged call; cleared whenever the dead-man releases)
        self.reference_controller_pose: np.ndarray | None = None
        self.reference_tool_pose: pin.SE3 | None = None
        self.reference_base_pose: pin.SE3 | None = None

        if self.verbose:
            self.log_configuration()

    def log_configuration(self):
        logger.info(f"OculusTeleopPolicy initialized for {self.robot_config.name}")
        logger.info(f"End effector frame: {self.robot_config.tool_frame}")
        if self.robot_config.gripper_joint_indices:
            logger.info(f"Gripper joint indices: {self.robot_config.gripper_joint_indices}")
        logger.info(f"Translation scale: {self.scale_translation:.2f}")
        logger.info(f"Rotation scale: {self.scale_rotation:.2f}")
        logger.info(f"Oculus->world rotation:\n{self.oculus_to_world.rotation}")
        logger.info("\nControls:")
        logger.info("  Right controller pose -> End-effector pose")
        logger.info("  Right trigger -> Gripper width (0.0=closed, 1.0=open)")
        logger.info(
            "  Grip trigger (either hand) -> Dead-man switch (hold to move; release to freeze)"
        )
        if self.robot_config.base_frame is not None:
            logger.info("  Left joystick -> Base XY in base frame (default: jx -> +x, jy -> +y)")
            logger.info(f"  Right joystick X -> Base yaw (sign {self.config.base_yaw_scale:+.0f})")

    def reset(self) -> None:
        """Reset policy state."""
        self.reference_controller_pose = None
        self.reference_tool_pose = None
        self.reference_base_pose = None

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
            and LEFT_GRIP_KEY in buttons
            and RIGHT_GRIP_KEY in buttons
        )

    def _read_joystick(self, buttons: dict, key: str) -> tuple[float, float]:
        """Read a joystick (x, y) pair with the configured deadzone applied.

        Args:
            buttons: Dictionary of button states from OculusReader
            key: Joystick key, e.g. ``leftJS`` or ``rightJS``

        Returns:
            Tuple of (x, y) in [-1, 1], with axes inside the deadzone returned as 0.
        """
        if key not in buttons:
            return 0.0, 0.0
        jx, jy = buttons[key]
        deadzone = self.config.base_deadzone
        if abs(jx) < deadzone:
            jx = 0.0
        if abs(jy) < deadzone:
            jy = 0.0
        return float(jx), float(jy)

    def _initialize_reference_poses(
        self, right_controller_pose: np.ndarray, observation: Observation
    ) -> None:
        """Initialize reference poses from current controller and robot state.

        Args:
            right_controller_pose: Current right controller pose (4x4 matrix)
            observation: Current observation from the environment
        """
        self.reference_controller_pose = right_controller_pose.copy()
        self.reference_tool_pose = self._get_current_tool_pose(observation)
        if self.robot_config.base_frame is not None:
            base_pose = self._get_current_base_pose(observation)
            assert base_pose is not None, "base_frame configured but FK returned None"
            self.reference_base_pose = pin.SE3(
                base_pose.rotation.copy(), base_pose.translation.copy()
            )

        if self.verbose:
            logger.info("\nReference poses initialized:")
            logger.info(f"  Tool position: {self.reference_tool_pose.translation}")
            rpy = pin.rpy.matrixToRpy(self.reference_tool_pose.rotation)
            logger.info(f"  Tool orientation (RPY): {np.rad2deg(rpy)} deg")
            if self.reference_base_pose is not None:
                logger.info(f"  Base position: {self.reference_base_pose.translation}")
            logger.info("\nReady! Move the right controller to control the robot.\n")

    def __call__(self, observation: Observation) -> Action:
        """Generate an action given an observation.

        On the first call (with a grip-trigger dead-man held), initializes the reference
        poses from the current robot state and controller pose. While neither grip is
        held, the policy clears its reference poses and holds the current robot position;
        new reference poses are re-established the next time a grip is pressed.

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
            return self._hold_current_pose_action(observation)

        # Dead-man switch: either grip trigger must be held to command motion.
        # Releasing both clears the reference poses so they re-anchor on re-engage.
        if not (buttons[LEFT_GRIP_KEY] or buttons[RIGHT_GRIP_KEY]):
            self.reset()
            return self._hold_current_pose_action(observation)

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

        # Remap the delta from the Oculus frame into the robot's world frame
        # via similarity transform with the configured rotation.
        delta_controller = self.oculus_to_world * delta_controller * self.oculus_to_world.inverse()

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

        base_pose_command: pin.SE3 | None = None
        if self.robot_config.base_frame is not None:
            assert self.reference_base_pose is not None, "Missing reference base pose!"

            left_jx, left_jy = self._read_joystick(buttons, LEFT_JOYSTICK_KEY)
            right_jx, _ = self._read_joystick(buttons, RIGHT_JOYSTICK_KEY)

            delta_xy_base = (
                self.config.base_translation_matrix
                @ np.array([left_jx, left_jy])
                * self.config.base_translation_step
            )
            # Translate in the base's local frame: rotate by current base
            # orientation so "forward stick" follows the base's heading.
            delta_xyz_world = self.reference_base_pose.rotation @ np.array(
                [delta_xy_base[0], delta_xy_base[1], 0.0]
            )
            self.reference_base_pose.translation[0] += delta_xyz_world[0]
            self.reference_base_pose.translation[1] += delta_xyz_world[1]

            delta_yaw = self.config.base_yaw_scale * right_jx * self.config.base_rotation_step
            if delta_yaw != 0.0:
                yaw_rot = pin.utils.rotate("z", delta_yaw)
                self.reference_base_pose.rotation = self.reference_base_pose.rotation @ yaw_rot

            base_pose_command = pin.SE3(
                self.reference_base_pose.rotation.copy(),
                self.reference_base_pose.translation.copy(),
            )

        return Action(
            tool_pose=target_pose,
            gripper_positions=gripper_positions,
            base_pose=base_pose_command,
        )
