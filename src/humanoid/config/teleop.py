"""Environment-selected teleoperation configuration."""

import os

from humanoid.constants import OCULUS_IP_ENVIRONMENT_VARIABLE
from humanoid.types.teleop import OculusTeleopPolicyConfig


def get_oculus_teleop_policy_config() -> OculusTeleopPolicyConfig:
    """Build Oculus configuration from the current process environment."""
    value = os.getenv(OCULUS_IP_ENVIRONMENT_VARIABLE)
    ip_address = value.strip() if value is not None and value.strip() else None
    return OculusTeleopPolicyConfig(ip_address=ip_address)
