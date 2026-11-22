"""Teleoperation policy configuration."""

import os

from humanoid.constants import OCULUS_IP_ENVIRONMENT_VARIABLE, Topic
from humanoid.types.observation import ObservationSubscriptions
from humanoid.types.teleop import KeyboardTeleopPolicyConfig, OculusTeleopPolicyConfig

TELEOP_SUBSCRIPTIONS = ObservationSubscriptions(
    topics=(Topic.ROBOT_STATE, Topic.ROBOT_JOINT_COMMAND, Topic.ORCHESTRATOR_MODE),
)

TELEOP_RATE_HZ = 30.0
TELEOP_DT = 1.0 / TELEOP_RATE_HZ
TOOL_LINEAR_ACCELERATION_LIMIT = 0.5
TOOL_ANGULAR_ACCELERATION_LIMIT = 2.0
OCULUS_POSE_FILTER_TIME_CONSTANT = 0.05


def get_keyboard_teleop_policy_config() -> KeyboardTeleopPolicyConfig:
    """Build the project-default keyboard teleoperation configuration."""
    return KeyboardTeleopPolicyConfig(
        subscriptions=TELEOP_SUBSCRIPTIONS,
        tool_step_scale=0.1,
        base_step_scale=1.0,
        dt=TELEOP_DT,
        tool_linear_acceleration_limit=TOOL_LINEAR_ACCELERATION_LIMIT,
        tool_angular_acceleration_limit=TOOL_ANGULAR_ACCELERATION_LIMIT,
    )


def get_oculus_teleop_policy_config() -> OculusTeleopPolicyConfig:
    """Build Oculus configuration from the current process environment."""
    value = os.getenv(OCULUS_IP_ENVIRONMENT_VARIABLE)
    ip_address = value.strip() if value is not None and value.strip() else None
    return OculusTeleopPolicyConfig(
        subscriptions=TELEOP_SUBSCRIPTIONS,
        dt=TELEOP_DT,
        ip_address=ip_address,
        tool_linear_acceleration_limit=TOOL_LINEAR_ACCELERATION_LIMIT,
        tool_angular_acceleration_limit=TOOL_ANGULAR_ACCELERATION_LIMIT,
        controller_pose_filter_time_constant=OCULUS_POSE_FILTER_TIME_CONSTANT,
    )
