"""Environment-selected robot configuration."""

from humanoid.config.robot import ROBOT_CONFIGS
from humanoid.types.robot import RobotName

ROBOT_NAME = RobotName.from_environment()
ROBOT_CONFIG = ROBOT_CONFIGS[ROBOT_NAME]
