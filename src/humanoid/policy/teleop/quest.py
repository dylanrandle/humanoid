"""Quest VR teleoperation policy for controlling robot end-effector pose."""

import time
from collections.abc import Callable
from copy import deepcopy

import numpy as np
import pinocchio as pin

from humanoid.api.clients.quest import QuestClient
from humanoid.config import ROBOT_CONFIG
from humanoid.config.teleop import TELEOP_SUBSCRIPTIONS
from humanoid.logger import get_logger
from humanoid.orchestrator.client import OrchestratorClient
from humanoid.policy.teleop.base import BaseTeleopPolicy
from humanoid.policy.teleop.motion import CartesianPoseLimiter, low_pass_pose
from humanoid.types.action import Action
from humanoid.types.homing import HomingPreset
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode
from humanoid.types.recording import RecordingState, RecordingStatus
from humanoid.types.robot import RobotConfig
from humanoid.types.teleop import QuestInputSnapshot, QuestTeleopPolicyConfig

logger = get_logger(__name__)

# Quest controller data keys
LEFT_CONTROLLER_KEY = "l"
RIGHT_CONTROLLER_KEY = "r"
LEFT_GRIP_KEY = "LG"
RIGHT_GRIP_KEY = "RG"
LEFT_JOYSTICK_KEY = "leftJS"
RIGHT_JOYSTICK_KEY = "rightJS"
A_BUTTON_KEY = "A"
B_BUTTON_KEY = "B"
X_BUTTON_KEY = "X"
Y_BUTTON_KEY = "Y"


class QuestTeleopPolicy(BaseTeleopPolicy):
    """Policy that uses Quest VR controllers to control the robot.

    The right controller's pose commands the tool pose, the A and B buttons
    command the gripper (push-and-hold), and (when the robot has a base frame)
    the joysticks command base translation and yaw. All commands are gated by
    a grip-trigger dead-man so the robot only moves while the operator is
    actively engaged.

    Controls:
        - Right controller pose: End-effector position and orientation
        - A button (hold): Close gripper; release to freeze at current width
        - B button (hold): Open gripper; release to freeze at current width
        - X button: Move to the configured home position
        - Y button: Toggle LCM data recording
        - Grip trigger (either hand): Dead-man switch -- all motion is
          only commanded while held; releasing both grips clears the
          controller, base, and gripper reference state, which re-anchor
          from the current robot state on re-engage
        - Left joystick: Base XY translation (see ``base_translation_matrix``)
        - Right joystick X: Base yaw (see ``base_yaw_scale``)

    Controller-pose deltas are remapped from the Quest frame into the robot's
    tool-command frame via ``quest_to_tool_command_rotation``. That command
    frame is world for fixed-base robots and the configured base frame for
    mobile robots.

    Args:
        robot_config: Robot configuration (default: ROBOT_CONFIG)
        config: Optional policy parameters and observation subscriptions.
        orchestrator_client: Client used for homing requests
    """

    mode = Mode.QUEST

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        config: QuestTeleopPolicyConfig | None = None,
        orchestrator_client: OrchestratorClient | None = None,
        recording_request: Callable[[], bool] | None = None,
        recording_status: Callable[[], RecordingStatus | None] | None = None,
    ):
        """Initialize the Quest teleoperation policy."""
        if config is None:
            config = QuestTeleopPolicyConfig(subscriptions=TELEOP_SUBSCRIPTIONS)
        super().__init__(robot_config=robot_config, verbose=config.verbose)

        self.config = config
        self.tool_translation_scale = config.tool_translation_scale
        self.tool_rotation_scale = config.tool_rotation_scale
        self.base_config = robot_config.base
        self.quest_to_tool_command_rotation = config.quest_to_tool_command_rotation.copy()
        self.tool_motion_limiter = CartesianPoseLimiter(
            robot_config.tool.velocity_limits,
            config.tool_linear_acceleration_limit,
            config.tool_angular_acceleration_limit,
        )
        self.gripper_step = (
            (self.gripper_max - self.gripper_min) * self.config.dt / self.config.gripper_close_time
        )
        self.orchestrator_client = orchestrator_client or OrchestratorClient()
        self.previous_button_states: dict[str, bool] = {}
        self.recording_status = recording_status
        self.recording_request = recording_request

        self.input_client = QuestClient()
        self._input_issue: str | None = None

        # Reference poses (set on first engaged call; cleared whenever the dead-man releases)
        self._clutched_at: float | None = None
        self.filtered_controller_pose: pin.SE3 | None = None
        self.commanded_tool_pose: pin.SE3 | None = None
        self.reference_base_pose: pin.SE3 | None = None
        self._hold_action: Action | None = None

        # Integrated gripper command. Seeded from the observation on first
        # tick, then nudged by gripper_step while A or B is held.
        self.commanded_gripper_position: float | None = None

        if self.verbose:
            self.log_configuration()

    def _tracking_issue(self, snapshot: QuestInputSnapshot, now: float) -> str | None:
        captured_at = snapshot.captured_monotonic
        if captured_at is None or now - captured_at > self.config.input_timeout:
            return "controller stream timed out"
        for hand, pose_key, grip_key in (
            ("left", LEFT_CONTROLLER_KEY, LEFT_GRIP_KEY),
            ("right", RIGHT_CONTROLLER_KEY, RIGHT_GRIP_KEY),
        ):
            pose = snapshot.transforms.get(pose_key)
            if pose is None or pose.shape != (4, 4) or not np.isfinite(pose).all():
                return f"{hand} controller pose unavailable"
            if not isinstance(snapshot.buttons.get(grip_key), bool):
                return f"{hand} controller buttons unavailable"
        return None

    def _tracking_ready(self, snapshot: QuestInputSnapshot) -> bool:
        # Invalid or old controller data is always disengaged. This prevents a
        # dropped Vuer stream from replaying the last grip/joystick state.
        now = time.monotonic()
        issue = self._tracking_issue(snapshot, now)
        if issue is not None:
            if issue != self._input_issue:
                age = (
                    "no frame received"
                    if snapshot.captured_monotonic is None
                    else f"last frame captured {now - snapshot.captured_monotonic:.2f} s ago"
                )
                logger.warning("Quest %s (%s). Suspending teleop commands.", issue, age)
            if self._input_issue is None:
                self.reset()
            self._input_issue = issue
            return False
        if self._input_issue is not None:
            logger.info("Quest input recovered")
            self._input_issue = None
        return True

    def log_configuration(self):
        logger.info(f"QuestTeleopPolicy initialized for {self.robot_config.name}")
        logger.info(f"End effector frame: {self.robot_config.tool.frame}")
        if self.robot_config.gripper is not None:
            logger.info(f"Gripper joints: {self.robot_config.gripper.joint_names}")
        logger.info(f"Translation scale: {self.tool_translation_scale:.2f}")
        logger.info(f"Rotation scale: {self.tool_rotation_scale:.2f}")
        logger.info("Quest connection: Vuer over local HTTPS/WebSocket")
        logger.info(f"Quest input timeout: {self.config.input_timeout:.2f} s")
        logger.info(
            "Quest->tool-command axis mapping:\n%s",
            self.quest_to_tool_command_rotation,
        )
        logger.info("\nControls:")
        logger.info("  Right controller pose -> End-effector pose")
        logger.info("  A button (hold) -> Close gripper (release to freeze)")
        logger.info("  B button (hold) -> Open gripper (release to freeze)")
        logger.info("  X button -> Move to the configured home position")
        logger.info("  Y button -> Toggle LCM data recording")
        logger.info(
            "  Grip trigger (either hand) -> Dead-man switch "
            "(hold for any command; release to freeze)"
        )
        if self.robot_config.base is not None:
            logger.info("  Left joystick -> Base XY in base frame (default: jx -> +x, jy -> +y)")
            logger.info(f"  Right joystick X -> Base yaw (sign {self.config.base_yaw_scale:+.0f})")

    def reset(self) -> None:
        """Reset policy state."""
        self._hold_action = None
        self._clutched_at = None
        self.filtered_controller_pose = None
        self.commanded_tool_pose = None
        self.reference_base_pose = None
        self.commanded_gripper_position = None
        self.tool_motion_limiter.reset()

    def _hold_on_release(self, observation: Observation) -> Action:
        if self._hold_action is None:
            self.reset()
            self._hold_action = self._hold_current_pose_action(observation)
        return deepcopy(self._hold_action)

    def _is_rising_edge(self, buttons: dict[str, object], key: str) -> bool:
        # A missing hand cannot confirm that a previously held button was released.
        if key not in buttons:
            return False
        current_pressed = bool(buttons[key])
        previous = self.previous_button_states.get(key, False)
        self.previous_button_states[key] = current_pressed
        return current_pressed and not previous

    def _read_joystick(self, buttons: dict[str, object], key: str) -> tuple[float, float]:
        """Read a joystick (x, y) pair with the configured deadzone applied.

        Args:
            buttons: Dictionary of button states from QuestClient
            key: Joystick key, e.g. ``leftJS`` or ``rightJS``

        Returns:
            Tuple of (x, y) in [-1, 1], with axes inside the deadzone returned as 0.
        """
        match buttons.get(key):
            case (float() as jx, float() as jy) if -1 <= jx <= 1 and -1 <= jy <= 1:
                deadzone = self.config.base_deadzone
                return (jx if abs(jx) >= deadzone else 0.0, jy if abs(jy) >= deadzone else 0.0)
            case _:
                return 0.0, 0.0

    def _compute_gripper_positions(
        self, buttons: dict[str, object], observation: Observation
    ) -> np.ndarray | None:
        """Compute the gripper command from the A/B buttons.

        A closes the gripper and B opens it by one step per tick. A takes
        priority when both are held. Releasing both holds the last command.

        Returns None when the robot has no gripper joint configured.
        """
        if self.robot_config.gripper is None:
            return None

        if self.commanded_gripper_position is None:
            current = self._get_current_gripper_positions(observation)
            if current is None or len(current) == 0:
                return None
            self.commanded_gripper_position = float(
                np.clip(current[0], self.gripper_min, self.gripper_max)
            )

        close = bool(buttons.get(A_BUTTON_KEY, False))
        if close or buttons.get(B_BUTTON_KEY, False):
            self.commanded_gripper_position = self._step_gripper(
                self.commanded_gripper_position, self.gripper_step, close=close
            )

        return np.array([self.commanded_gripper_position])

    def _homing_target_preserving_gripper(
        self, target_position: np.ndarray, observation: Observation
    ) -> np.ndarray:
        """Return a copy of ``target_position`` with the gripper slots overwritten
        by the current observation, so X homing leaves the gripper where it is."""
        target = target_position.copy()
        current_gripper = self._get_current_gripper_positions(observation)
        if current_gripper is not None:
            self.robot.set_gripper_positions(target, current_gripper)
        return target

    def _initialize_reference_poses(
        self, right_controller_pose: np.ndarray, observation: Observation
    ) -> None:
        """Initialize reference poses from current controller and robot state.

        Args:
            right_controller_pose: Current right controller pose (4x4 matrix)
            observation: Current observation from the environment
        """
        self.filtered_controller_pose = pin.SE3(
            right_controller_pose[:3, :3],
            right_controller_pose[:3, 3],
        )
        self.commanded_tool_pose = self._get_current_tool_pose(observation)
        self.tool_motion_limiter.reset()
        if self.robot_config.base is not None:
            self.reference_base_pose = self._get_current_base_pose(observation)
            assert self.reference_base_pose is not None, (
                "Mobile base configured but FK returned None"
            )

    def _compute_tool_pose(self, right_controller_pose: np.ndarray) -> pin.SE3:
        """Map the controller pose to a velocity-limited tool target."""
        assert self.filtered_controller_pose is not None, "Missing controller reference"
        assert self.commanded_tool_pose is not None, "Missing tool reference"
        previous_controller = self.filtered_controller_pose
        current_controller = pin.SE3(
            right_controller_pose[:3, :3],
            right_controller_pose[:3, 3],
        )
        current_controller = low_pass_pose(
            previous_controller,
            current_controller,
            self.config.dt,
            self.config.controller_pose_filter_time_constant,
        )
        self.filtered_controller_pose = current_controller
        # Consume this hand displacement once. Any motion the robot could not follow
        # at its speed limit must not keep advancing the target after the hand stops.
        command_rotation = self.quest_to_tool_command_rotation

        # translation
        controller_translation_delta = (
            current_controller.translation - previous_controller.translation
        )
        command_translation_delta = command_rotation @ controller_translation_delta
        scaled_translation_delta = command_translation_delta * self.tool_translation_scale

        # rotation
        controller_rotation_delta = current_controller.rotation @ previous_controller.rotation.T
        command_rotation_delta = command_rotation @ controller_rotation_delta @ command_rotation.T
        target_pose = self.tool_motion_limiter.follow_displacement(
            self.commanded_tool_pose,
            scaled_translation_delta,
            pin.log3(command_rotation_delta) * self.tool_rotation_scale,
            self.config.dt,
        )
        self.commanded_tool_pose = target_pose.copy()
        return target_pose

    @property
    def is_recording(self) -> bool | None:
        status = self.recording_status() if self.recording_status is not None else None
        return None if status is None else status.state is RecordingState.RUNNING

    def step(self, observation: Observation) -> Action:
        """Generate an action given an observation.

        On the first call (with a grip-trigger dead-man held), initializes the reference
        poses from the current robot state and controller pose. While neither grip is
        held, the policy republishes the targets captured on the first disengaged tick;
        new reference poses are re-established the next time a grip is pressed.

        Subsequent calls map each filtered hand displacement into a bounded tool step.

        Args:
            observation: Current observation from the environment

        Returns:
            Action containing the target tool pose and gripper positions
        """
        # Get current controller data
        snapshot = self.input_client.get_snapshot()
        buttons = snapshot.buttons

        if not self._tracking_ready(snapshot):
            return Action()

        # Handle X button homing (edge-triggered)
        if self._is_rising_edge(buttons, X_BUTTON_KEY):
            target = self._homing_target_preserving_gripper(
                self.robot_config.homing_presets[HomingPreset.HOME], observation
            )
            self.orchestrator_client.request_homing(target)

        # Handle Y button recording toggle (edge-triggered)
        if self._is_rising_edge(buttons, Y_BUTTON_KEY) and self.recording_request is not None:
            self.recording_request()

        # Dead-man switch: either grip trigger must be held to command any
        # motion. Releasing both clears the reference poses so they re-anchor
        # on re-engage.
        if not (buttons[LEFT_GRIP_KEY] or buttons[RIGHT_GRIP_KEY]):
            return self._hold_on_release(observation)

        # Extract right controller pose (4x4 transformation matrix)
        right_controller_pose = snapshot.transforms[RIGHT_CONTROLLER_KEY]

        # The reader can observe a release and a new press between policy ticks.
        if self._clutched_at is None or (
            snapshot.last_disengaged_monotonic is not None
            and snapshot.last_disengaged_monotonic >= self._clutched_at
        ):
            self._hold_action = None
            self._initialize_reference_poses(right_controller_pose, observation)
            self.commanded_gripper_position = None
            self._clutched_at = snapshot.captured_monotonic

        target_pose = self._compute_tool_pose(right_controller_pose)
        target_velocity = self.tool_motion_limiter.command_velocity(target_pose)

        gripper_positions = self._compute_gripper_positions(buttons, observation)

        base_pose_command: pin.SE3 | None = None
        if self.robot_config.base is not None:
            assert self.reference_base_pose is not None, "Missing reference base pose!"
            assert self.base_config is not None

            left_jx, left_jy = self._read_joystick(buttons, LEFT_JOYSTICK_KEY)
            right_jx, _ = self._read_joystick(buttons, RIGHT_JOYSTICK_KEY)

            delta_xy_base = (
                self.config.base_translation_matrix
                @ np.array([left_jx, left_jy])
                * self.base_config.velocity_limits.linear
                * self.config.dt
            )
            # Translate in the base's local frame: rotate by current base
            # orientation so "forward stick" follows the base's heading.
            delta_xyz_world = self.reference_base_pose.rotation @ np.array(
                [delta_xy_base[0], delta_xy_base[1], 0.0]
            )
            self.reference_base_pose.translation[0] += delta_xyz_world[0]
            self.reference_base_pose.translation[1] += delta_xyz_world[1]

            delta_yaw = (
                self.config.base_yaw_scale
                * right_jx
                * self.base_config.velocity_limits.angular
                * self.config.dt
            )
            if delta_yaw != 0.0:
                yaw_rot = pin.utils.rotate("z", delta_yaw)
                self.reference_base_pose.rotation = self.reference_base_pose.rotation @ yaw_rot

            base_pose_command = pin.SE3(
                self.reference_base_pose.rotation.copy(),
                self.reference_base_pose.translation.copy(),
            )

        return Action(
            tool_pose=target_pose,
            tool_velocity=target_velocity,
            gripper_positions=gripper_positions,
            base_pose=base_pose_command,
        )
