import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

import numpy as np
import pinocchio as pin

from humanoid.hardware.config import RobotHardwareConfig
from humanoid.state_estimation.config import RobotStateEstimationConfig
from humanoid.types.actuator import ActuatorControlMode, ActuatorEffortSource, JointDriveConfig
from humanoid.types.camera import RobotCameraConfig, camera_topics
from humanoid.types.controllers import OmniwheelBaseConfig, OperationalSpaceConfig
from humanoid.types.homing import HomingPreset


class RobotName(StrEnum):
    PANDA = "panda"
    SO101 = "so101"
    TRISKEL = "triskel"

    @classmethod
    def from_environment(cls) -> "RobotName":
        """Parse a robot name, using the configured default when unset."""
        # Constants imports this enum, so resolve its environment settings lazily.
        from humanoid.constants import DEFAULT_HUMANOID_ROBOT, EnvironmentVariable  # noqa: PLC0415

        value = os.getenv(EnvironmentVariable.ROBOT)
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
class CartesianVelocity:
    """Linear and angular tool velocity expressed in its command frame."""

    linear: np.ndarray
    angular: np.ndarray

    def __post_init__(self) -> None:
        for name, value in (("linear", self.linear), ("angular", self.angular)):
            array = np.asarray(value, dtype=float)
            if array.shape != (3,):
                raise ValueError(f"Cartesian {name} velocity must have shape (3,).")
            if not np.isfinite(array).all():
                raise ValueError(f"Cartesian {name} velocity must be finite.")
            object.__setattr__(self, name, array.copy())

    @classmethod
    def zero(cls) -> "CartesianVelocity":
        """Return a stationary Cartesian velocity."""
        return cls(linear=np.zeros(3), angular=np.zeros(3))


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


@dataclass(frozen=True, kw_only=True)
class RobotGripperConfig:
    """Independently commanded gripper joints identified by URDF name."""

    joint_names: tuple[str, ...]
    units: tuple[Literal["m", "rad"], ...]
    closes_at_upper_limit: bool = False

    def __post_init__(self) -> None:
        if not self.joint_names:
            raise ValueError("Robot gripper must define at least one joint name.")
        if any(not joint_name.strip() for joint_name in self.joint_names):
            raise ValueError("Robot gripper joint names must not be empty.")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Robot gripper joint names must be unique.")
        if len(self.units) != len(self.joint_names) or any(
            u not in ("m", "rad") for u in self.units
        ):
            raise ValueError("Each gripper joint requires a position unit of 'm' or 'rad'.")


@dataclass
class RobotJointCommand:
    timestamp: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray | None = None


@dataclass(frozen=True)
class NormalizedRobotJointCommand:
    """Validated, limit-clamped generalized position and velocity targets."""

    joint_positions: np.ndarray
    joint_velocities: np.ndarray


@dataclass
class RobotState:
    timestamp: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    # Signed SI efforts, indexed and oriented like joint_velocities; NaN means unavailable.
    joint_efforts: np.ndarray | None = field(default=None, kw_only=True)
    actuator_temperatures: np.ndarray
    effort_source: ActuatorEffortSource | None = None


@dataclass
class RobotToolCommand:
    """Tool target in world for fixed robots or the configured base frame for mobile robots."""

    timestamp: float
    pose: pin.SE3
    gripper_positions: np.ndarray | None = None
    velocity: CartesianVelocity | None = None


@dataclass
class RobotBaseCommand:
    timestamp: float
    pose: pin.SE3


@dataclass(frozen=True, kw_only=True)
class GeomContactConfig:
    """Solver settings and surface friction for a collision geometry."""

    priority: int
    solref: tuple[float, float]
    solimp: tuple[float, float, float, float, float]
    sliding_friction: float | None = None
    torsional_friction: float | None = None  # m, torque per unit normal force

    def __post_init__(self) -> None:
        if self.sliding_friction is not None and (
            not np.isfinite(self.sliding_friction) or self.sliding_friction < 0
        ):
            raise ValueError("Sliding friction must be nonnegative and finite.")
        if self.torsional_friction is not None and (
            not np.isfinite(self.torsional_friction) or self.torsional_friction <= 0
        ):
            raise ValueError("Torsional friction length must be positive and finite.")


@dataclass
class RobotConfig:
    name: RobotName
    tool: RobotToolConfig
    homing_presets: dict[HomingPreset, np.ndarray]
    joint_drives: dict[str, JointDriveConfig]
    base: RobotBaseConfig | None = None
    wheels: list[WheelConfig] | None = None
    gripper: RobotGripperConfig | None = None
    hardware: RobotHardwareConfig | None = None
    state_estimation: RobotStateEstimationConfig | None = None
    operational_space_config: OperationalSpaceConfig | None = None
    omniwheel_base_config: OmniwheelBaseConfig | None = None
    cameras: dict[str, RobotCameraConfig] = field(default_factory=dict)
    geom_contacts: dict[str, GeomContactConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Own invariants spanning logical controls and physical bindings."""
        if len({camera_topics(name) for name in self.cameras}) != len(self.cameras):
            raise ValueError("Camera names must have distinct uppercase LCM topics.")
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
        if self.omniwheel_base_config is not None:
            if self.base is None or not self.wheels:
                raise ValueError(
                    "Omniwheel base control requires a mobile base and configured wheels."
                )
            if any(wheel.type is not WheelType.OMNI for wheel in self.wheels):
                raise ValueError("Omniwheel base control requires omniwheel configurations.")
        self._validate_gripper()
        if self.hardware is None or self.hardware.actuators is None:
            return
        if self.hardware.actuators.joints.keys() != self.joint_drives.keys():
            raise ValueError("Physical actuator bindings must match the robot's controlled joints.")

    @property
    def actuator_control_modes(self) -> dict[str, ActuatorControlMode]:
        """Command modes for the hardware actuator system."""
        return {name: drive.control_mode for name, drive in self.joint_drives.items()}

    def _validate_gripper(self) -> None:
        if self.gripper is None:
            return
        missing_gripper_joints = set(self.gripper.joint_names) - self.joint_drives.keys()
        if missing_gripper_joints:
            missing = ", ".join(sorted(missing_gripper_joints))
            raise ValueError(f"Gripper joints must be configured actuators: {missing}.")
        if any(
            self.joint_drives[joint_name].control_mode is not ActuatorControlMode.POSITION
            for joint_name in self.gripper.joint_names
        ):
            raise ValueError("Gripper joints must use position control.")
