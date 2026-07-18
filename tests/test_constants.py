from humanoid.constants import DEFAULT_HUMANOID_ROBOT, DEFAULT_HUMANOID_RUNTIME
from humanoid.types.process import Runtime
from humanoid.types.robot import RobotName


def test_default_humanoid_configuration_is_explicit():
    assert DEFAULT_HUMANOID_RUNTIME is Runtime.SIM
    assert DEFAULT_HUMANOID_ROBOT is RobotName.ELROBOT_MOBILE
