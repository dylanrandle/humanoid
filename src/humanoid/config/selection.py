"""Environment-selected runtime and robot configuration."""

from humanoid.config.robot import ROBOT_CONFIGS
from humanoid.types.process import Runtime
from humanoid.types.robot import RobotName

IS_SIMULATION = Runtime.from_environment() is Runtime.SIM
ROBOT_NAME = RobotName.from_environment()
ROBOT_CONFIG = ROBOT_CONFIGS[ROBOT_NAME]
