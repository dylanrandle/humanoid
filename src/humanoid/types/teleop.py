from dataclasses import dataclass, field

import numpy as np


@dataclass
class KeyboardTeleopPolicyConfig:
    """Tunable parameters for KeyboardTeleopPolicy.

    Per-keypress step sizes are derived as ``velocity * dt``, so ``dt`` acts
    as the conversion factor between the operator-facing velocity values and
    the displacement applied on each keypress. ``dt`` also sets the rate of
    the teleop node.

    Args:
        dt: Control loop period in seconds. Used to convert the configured
            velocities into per-keypress step sizes and to set the rate of
            the teleop node.
        tool_translation_velocity: Meters per second of tool translation per
            unit of ``dt`` (i.e. per keypress).
        tool_rotation_velocity: Radians per second of tool rotation per unit
            of ``dt`` (i.e. per keypress).
        base_translation_velocity: Meters per second of base translation per
            unit of ``dt`` (i.e. per keypress).
        base_rotation_velocity: Radians per second of base yaw per unit of
            ``dt`` (i.e. per keypress).
        gripper_close_time: Seconds of held input required to traverse the
            full gripper joint range. The per-keypress step is
            ``(gripper_max - gripper_min) * dt / gripper_close_time``.
        verbose: Whether to log pose updates.
    """

    dt: float = 0.01
    tool_translation_velocity: float = 0.5
    tool_rotation_velocity: float = np.pi
    base_translation_velocity: float = 0.5
    base_rotation_velocity: float = np.pi
    gripper_close_time: float = 1.0
    verbose: bool = True


@dataclass
class OculusTeleopPolicyConfig:
    """Tunable parameters for OculusTeleopPolicy.

    Args:
        dt: Control loop period in seconds. Used to convert the base
            translation/rotation velocities into per-tick steps and to set
            the rate of the teleop node.
        oculus_to_world_rotation: 3x3 matrix mapping Oculus-frame vectors
            into the robot's world frame. Used to reorient the controller
            pose so headset-frame motion matches the operator's intuition
            of the robot's world.
        tool_translation_scale: Scale factor for controller translation.
        tool_rotation_scale: Scale factor for controller rotation.
        base_translation_matrix: 2x2 matrix mapping the left joystick
            (jx, jy) in [-1, 1]^2 to a per-tick (dx, dy) translation in the
            base's local frame, scaled by ``base_translation_velocity * dt``.
            The result is rotated into world coordinates by the base's
            current yaw, so "forward stick" always means forward along the
            base's current heading. Default identity gives stick-right
            (+jx) -> +base-local x and stick-forward (+jy) -> +base-local y.
        base_translation_velocity: Meters per second of base translation at
            full joystick deflection.
        base_rotation_velocity: Radians per second of base yaw at full
            right-joystick-x deflection.
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
    oculus_to_world_rotation: np.ndarray = field(
        # 180 degree rotation about x
        default_factory=lambda: np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0], [0.0, 0.0, -1.0]])
    )
    tool_translation_scale: float = 1.0
    tool_rotation_scale: float = 1.0
    base_translation_matrix: np.ndarray = field(default_factory=lambda: np.eye(2))
    base_translation_velocity: float = 0.5
    base_rotation_velocity: float = np.pi
    base_yaw_scale: float = -1.0
    base_deadzone: float = 0.1
    gripper_close_time: float = 1.0
    verbose: bool = True
