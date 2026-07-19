import os
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pinocchio as pin

from humanoid.hardware.actuators.config import ActuatorControlMode
from humanoid.hardware.config import RobotHardwareConfig
from humanoid.types.controllers import OperationalSpaceConfig
from humanoid.types.wheels import WheelConfig


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
    tool_frame: str
    home_position: np.ndarray
    rest_position: np.ndarray
    actuator_control_modes: dict[str, ActuatorControlMode]
    hardware: RobotHardwareConfig | None = None
    base_frame: str | None = None
    wheels: list[WheelConfig] | None = None
    gripper_joint_indices: list[int] | None = None
    operational_space_config: OperationalSpaceConfig | None = None

    def __post_init__(self) -> None:
        """Validate consistency between logical controls and physical bindings."""
        if self.hardware is None or self.hardware.actuators is None:
            return
        if self.hardware.actuators.joints.keys() != self.actuator_control_modes.keys():
            raise ValueError("Physical actuator bindings must match the robot's controlled joints.")
