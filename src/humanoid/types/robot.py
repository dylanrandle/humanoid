from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from humanoid.types.controllers import OperationalSpaceConfig
from humanoid.types.servo import ServoControlMode
from humanoid.types.wheels import WheelConfig


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
    motor_temperatures: np.ndarray


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
    name: str
    tool_frame: str
    home_position: np.ndarray
    rest_position: np.ndarray
    joint_idx_to_servo_id: dict[int, int]
    servo_control_modes: dict[int, ServoControlMode]
    base_frame: str | None = None
    wheels: list[WheelConfig] | None = None
    inverted_servo_ids: list[int] | None = None
    gripper_joint_indices: list[int] | None = None
    operational_space_config: OperationalSpaceConfig | None = None

    @property
    def servo_ids(self) -> list[int]:
        return list(set(self.joint_idx_to_servo_id.values()))

    @property
    def servo_id_to_joint_idx(self) -> dict[int, int]:
        return {v: k for k, v in self.joint_idx_to_servo_id.items()}
