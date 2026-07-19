import os
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pinocchio as pin

from humanoid.hardware.actuators.config import ActuatorControlMode
from humanoid.hardware.config import RobotHardwareConfig
from humanoid.state_estimation.config import RobotStateEstimationConfig
from humanoid.types.controllers import OperationalSpaceConfig
from humanoid.types.homing import HomingPreset


class RobotName(StrEnum):
    PANDA = "panda"
    SO101 = "so101"
    ELROBOT = "elrobot"
    ELROBOT_MOBILE = "elrobot_mobile"

    @classmethod
    def from_environment(cls) -> "RobotName":
        """Parse a robot name, using the configured default when unset."""
        # Constants imports this enum, so resolve its environment settings lazily.
        from humanoid.constants import (  # noqa: PLC0415
            DEFAULT_HUMANOID_ROBOT,
            ROBOT_ENVIRONMENT_VARIABLE,
        )

        value = os.getenv(ROBOT_ENVIRONMENT_VARIABLE)
        if value is None or not value.strip():
            return cls(DEFAULT_HUMANOID_ROBOT)
        return cls(value.lower().strip())


class WheelType(StrEnum):
    REGULAR = "regular"
    OMNI = "omni"


@dataclass
class WheelConfig:
    frame: str
    floor_frame: str
    radius: float
    type: WheelType


@dataclass(frozen=True, kw_only=True)
class CartesianVelocityLimits:
    """Maximum linear and angular velocity for a Cartesian frame."""

    linear: float
    angular: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.linear) or self.linear <= 0.0:
            raise ValueError("Cartesian linear velocity limit must be positive and finite.")
        if not np.isfinite(self.angular) or self.angular <= 0.0:
            raise ValueError("Cartesian angular velocity limit must be positive and finite.")


DEFAULT_CARTESIAN_VELOCITY_LIMITS = CartesianVelocityLimits(linear=1.0, angular=np.pi)


@dataclass(frozen=True, kw_only=True)
class RobotToolConfig:
    """Kinematic frame and motion limits for a robot tool."""

    frame: str
    velocity_limits: CartesianVelocityLimits = DEFAULT_CARTESIAN_VELOCITY_LIMITS

    def __post_init__(self) -> None:
        if not self.frame.strip():
            raise ValueError("Robot tool frame must not be empty.")


@dataclass(frozen=True, kw_only=True)
class RobotBaseConfig:
    """Kinematic frame and motion limits for a mobile robot base."""

    frame: str
    velocity_limits: CartesianVelocityLimits = DEFAULT_CARTESIAN_VELOCITY_LIMITS

    def __post_init__(self) -> None:
        if not self.frame.strip():
            raise ValueError("Robot base frame must not be empty.")


@dataclass
class RobotJointCommand:
    timestamp: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray | None = None


@dataclass
class RobotState:
    timestamp: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    actuator_temperatures: np.ndarray


@dataclass
class RobotToolCommand:
    """Tool target in world for fixed robots or the configured base frame for mobile robots."""

    timestamp: float
    pose: pin.SE3
    gripper_positions: np.ndarray | None = None


@dataclass
class RobotBaseCommand:
    timestamp: float
    pose: pin.SE3


@dataclass
class RobotConfig:
    name: RobotName
    tool: RobotToolConfig
    homing_presets: dict[HomingPreset, np.ndarray]
    actuator_control_modes: dict[str, ActuatorControlMode]
    base: RobotBaseConfig | None = None
    wheels: list[WheelConfig] | None = None
    gripper_joint_indices: list[int] | None = None
    hardware: RobotHardwareConfig | None = None
    state_estimation: RobotStateEstimationConfig | None = None
    operational_space_config: OperationalSpaceConfig | None = None

    def __post_init__(self) -> None:
        """Own invariants spanning logical controls and physical bindings."""
        if self.homing_presets.keys() != set(HomingPreset):
            raise ValueError("Robot homing presets must define every HomingPreset.")
        preset_positions = list(self.homing_presets.values())
        if any(position.ndim != 1 for position in preset_positions):
            raise ValueError("Robot homing presets must be one-dimensional.")
        if len({position.shape for position in preset_positions}) != 1:
            raise ValueError("Robot homing presets must have matching shapes.")
        if any(not np.isfinite(position).all() for position in preset_positions):
            raise ValueError("Robot homing presets must contain only finite values.")

        root_config = self.state_estimation.root if self.state_estimation is not None else None
        if self.base is not None and root_config is None:
            raise ValueError("Robots with a mobile base require root-state estimation config.")
        if self.base is None and root_config is not None:
            raise ValueError("Fixed-base robots cannot configure root-state estimation.")
        if self.hardware is None or self.hardware.actuators is None:
            return
        if self.hardware.actuators.joints.keys() != self.actuator_control_modes.keys():
            raise ValueError("Physical actuator bindings must match the robot's controlled joints.")
