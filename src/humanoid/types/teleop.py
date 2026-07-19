from dataclasses import dataclass, field

import numpy as np


@dataclass
class KeyboardTeleopPolicyConfig:
    """Tunable parameters for KeyboardTeleopPolicy.

    Per-keypress step sizes are derived from the selected robot's Cartesian
    velocity limits multiplied by ``dt``. ``dt`` also sets the teleop node's
    update rate.

    Args:
        dt: Control loop period in seconds. Used to convert the selected
            robot's velocity limits into per-keypress steps and to set the
            rate of the teleop node.
        gripper_close_time: Seconds of held input required to traverse the
            full gripper joint range. The per-keypress step is
            ``(gripper_max - gripper_min) * dt / gripper_close_time``.
        verbose: Whether to log pose updates.
    """

    dt: float = 0.01
    gripper_close_time: float = 1.0
    verbose: bool = True


@dataclass
class OculusTeleopPolicyConfig:
    """Tunable parameters for OculusTeleopPolicy.

    Args:
        dt: Control loop period in seconds. Used with the selected robot's
            tool and base velocity limits to derive per-tick steps and to set
            the rate of the teleop node.
        oculus_to_tool_command_rotation: 3x3 matrix mapping Oculus-frame vectors
            into the robot's tool-command frame: world for fixed-base robots
            and the configured base frame for mobile robots.
        tool_translation_scale: Scale factor for controller translation before
            applying the selected robot's linear tool velocity limit.
        tool_rotation_scale: Scale factor for controller rotation before
            applying the selected robot's angular tool velocity limit.
        base_translation_matrix: 2x2 matrix mapping the left joystick
            (jx, jy) in [-1, 1]^2 to a per-tick (dx, dy) translation in the
            base's local frame, scaled by the robot's linear base velocity
            limit and ``dt``.
            The result is rotated into world coordinates by the base's
            current yaw, so "forward stick" always means forward along the
            base's current heading. Default identity gives stick-right
            (+jx) -> +base-local x and stick-forward (+jy) -> +base-local y.
        base_yaw_scale: Scalar sign applied to right-joystick x for yaw.
            Default -1 makes stick-right yield -yaw and stick-left +yaw.
        base_deadzone: Per-axis joystick magnitude below which input is
            treated as zero.
        gripper_close_time: Seconds of held A/B input required to traverse
            the full gripper joint range. The per-tick step is
            ``(gripper_max - gripper_min) * dt / gripper_close_time``;
            commanded position is clamped to the joint limits.
        verbose: Whether to log pose updates.
    """

    dt: float = 0.01
    oculus_to_tool_command_rotation: np.ndarray = field(
        # 180 degree rotation about x
        default_factory=lambda: np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0], [0.0, 0.0, -1.0]])
    )
    tool_translation_scale: float = 1.0
    tool_rotation_scale: float = 1.0
    base_translation_matrix: np.ndarray = field(default_factory=lambda: np.eye(2))
    base_yaw_scale: float = -1.0
    base_deadzone: float = 0.1
    gripper_close_time: float = 1.0
    verbose: bool = True
