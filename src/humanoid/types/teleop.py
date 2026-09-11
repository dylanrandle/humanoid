from dataclasses import dataclass, field
from ipaddress import AddressValueError, IPv4Address

import numpy as np

MIN_NETWORK_PORT = 1
MAX_NETWORK_PORT = 65_535


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
        tool_linear_acceleration_limit: Optional tool translation acceleration
            limit in m/s^2. ``None`` disables acceleration shaping.
        tool_angular_acceleration_limit: Optional tool rotation acceleration
            limit in rad/s^2. ``None`` disables acceleration shaping.
        verbose: Whether to log pose updates.
    """

    dt: float = 0.01
    gripper_close_time: float = 1.0
    tool_linear_acceleration_limit: float | None = None
    tool_angular_acceleration_limit: float | None = None
    verbose: bool = True

    def __post_init__(self) -> None:
        _validate_motion_config(
            self.dt,
            self.tool_linear_acceleration_limit,
            self.tool_angular_acceleration_limit,
        )


@dataclass
class OculusTeleopPolicyConfig:
    """Tunable parameters for OculusTeleopPolicy.

    Args:
        dt: Control loop period in seconds. Used with the selected robot's
            tool and base velocity limits to derive per-tick steps and to set
            the rate of the teleop node.
        oculus_to_tool_command_rotation: 3x3 orthogonal matrix mapping
            Oculus-frame vector components into the robot's tool-command
            frame: world for fixed-base robots and the configured base frame
            for mobile robots. Defaults to a 90-degree rotation about X:
            Oculus +Y maps to command +Z (up), and Oculus -Z maps to command
            +Y (forward).
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
        ip_address: Quest IPv4 address for wireless ADB. ``None`` preserves
            the upstream reader's USB-device behavior.
        port: Quest wireless ADB port.
        input_timeout: Maximum age in seconds of the most recently received
            controller frame. Older input is treated as disengaged.
        startup_timeout: Maximum time in seconds to wait for the first valid
            controller frame before failing with a setup error.
        tool_linear_acceleration_limit: Optional tool translation acceleration
            limit in m/s^2. ``None`` disables acceleration shaping.
        tool_angular_acceleration_limit: Optional tool rotation acceleration
            limit in rad/s^2. ``None`` disables acceleration shaping.
        controller_pose_filter_time_constant: Low-pass filter time constant for
            Oculus position and orientation in seconds. Zero disables filtering.
        verbose: Whether to log pose updates.
    """

    dt: float = 0.01
    oculus_to_tool_command_rotation: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )
    tool_translation_scale: float = 1.0
    tool_rotation_scale: float = 1.0
    base_translation_matrix: np.ndarray = field(default_factory=lambda: np.eye(2))
    base_yaw_scale: float = -1.0
    base_deadzone: float = 0.1
    gripper_close_time: float = 1.0
    tool_linear_acceleration_limit: float | None = None
    tool_angular_acceleration_limit: float | None = None
    controller_pose_filter_time_constant: float = 0.0
    ip_address: str | None = None
    port: int = 5555
    input_timeout: float = 0.3
    startup_timeout: float = 15.0
    verbose: bool = True

    def __post_init__(self) -> None:
        _validate_motion_config(
            self.dt,
            self.tool_linear_acceleration_limit,
            self.tool_angular_acceleration_limit,
        )
        if (
            not np.isfinite(self.controller_pose_filter_time_constant)
            or self.controller_pose_filter_time_constant < 0.0
        ):
            raise ValueError(
                "Controller pose filter time constant must be finite and non-negative."
            )
        if self.ip_address is not None:
            try:
                self.ip_address = str(IPv4Address(self.ip_address.strip()))
            except AddressValueError as exc:
                raise ValueError("Oculus IP address must be a valid IPv4 address.") from exc
        if not MIN_NETWORK_PORT <= self.port <= MAX_NETWORK_PORT:
            raise ValueError("Oculus ADB port must be between 1 and 65535.")
        if not np.isfinite(self.input_timeout) or self.input_timeout <= 0.0:
            raise ValueError("Oculus input timeout must be positive and finite.")
        if not np.isfinite(self.startup_timeout) or self.startup_timeout <= 0.0:
            raise ValueError("Oculus startup timeout must be positive and finite.")


def _validate_motion_config(
    dt: float,
    linear_acceleration_limit: float | None,
    angular_acceleration_limit: float | None,
) -> None:
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Teleop timestep must be positive and finite.")
    for name, value in (
        ("linear acceleration limit", linear_acceleration_limit),
        ("angular acceleration limit", angular_acceleration_limit),
    ):
        if value is not None and (not np.isfinite(value) or value <= 0.0):
            raise ValueError(f"Tool {name} must be positive and finite when configured.")


@dataclass(frozen=True)
class OculusInputSnapshot:
    """One atomically sampled controller frame and its receipt time."""

    transforms: dict[str, np.ndarray]
    buttons: dict[str, object]
    received_monotonic: float | None
