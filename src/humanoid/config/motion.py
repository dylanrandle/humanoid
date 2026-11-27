"""Shared motion configuration for API and policy processes."""

from humanoid.config.selection import ROBOT_CONFIG
from humanoid.constants import Topic
from humanoid.types.middleware import PipeConfig
from humanoid.types.motion import (
    MotionLimits,
    MotionPolicyConfig,
    MotionProtocolConfig,
    MotionServiceConfig,
)
from humanoid.types.observation import ObservationSubscriptions
from humanoid.types.robot import RobotConfig

DEFAULT_MOTION_LIMITS = MotionLimits()


def get_motion_policy_config(robot_config: RobotConfig = ROBOT_CONFIG) -> MotionPolicyConfig:
    return MotionPolicyConfig(
        subscriptions=ObservationSubscriptions(
            topics=(Topic.ROBOT_STATE, Topic.ROBOT_JOINT_COMMAND, Topic.ORCHESTRATOR_MODE),
        ),
    )


def get_motion_service_config() -> MotionServiceConfig:
    return MotionServiceConfig()


DEFAULT_MOTION_TIMEOUT_SECONDS = 5.0
MOTION_PIPE_CONFIG = PipeConfig(
    capacity=64,
    max_bytes=65536,
    poll_seconds=0.05,
    join_seconds=1.0,
)
MOTION_PROTOCOL_CONFIG = MotionProtocolConfig(
    heartbeat_seconds=0.25,
    contact_timeout_seconds=2.0,
    startup_timeout_seconds=30.0,
    request_timeout_seconds=2.0,
)
