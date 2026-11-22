"""Homing trajectory and observation defaults."""

from humanoid.constants import Topic
from humanoid.types.homing import HomingPolicyConfig
from humanoid.types.observation import ObservationSubscriptions

HOMING_POLICY_CONFIG = HomingPolicyConfig(
    speed=1.0,
    rate_hz=30.0,
    subscriptions=ObservationSubscriptions(
        topics=(
            Topic.ROBOT_STATE,
            Topic.ROBOT_JOINT_COMMAND,
            Topic.ORCHESTRATOR_MODE,
            Topic.HOMING_TARGET,
        ),
    ),
)
