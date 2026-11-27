"""Read-only robot observation defaults, independent of any motion policy."""

from humanoid.constants import Topic
from humanoid.types.observation import ObservationConfig, ObservationSubscriptions
from humanoid.types.robot import RobotConfig

OBSERVATION_CONFIG = ObservationConfig()


def get_observation_subscriptions(robot: RobotConfig) -> ObservationSubscriptions:
    return ObservationSubscriptions(
        topics=(Topic.ROBOT_STATE, Topic.ROBOT_JOINT_COMMAND, Topic.ORCHESTRATOR_MODE),
        camera_names=tuple(robot.cameras),
    )
