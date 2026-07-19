import numpy as np
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.state_estimation.root.wheel_dead_reckoning import (
    WheelDeadReckoningRootStateEstimatorConfig,
)
from humanoid.types.robot import RobotConfig, RobotName


def test_only_mobile_robot_configures_root_state_estimation():
    mobile_config = ROBOT_CONFIGS[RobotName.ELROBOT_MOBILE]
    assert mobile_config.state_estimation is not None
    assert mobile_config.state_estimation.root is not None
    assert isinstance(
        mobile_config.state_estimation.root,
        WheelDeadReckoningRootStateEstimatorConfig,
    )
    assert all(
        config.state_estimation is None
        for name, config in ROBOT_CONFIGS.items()
        if name is not RobotName.ELROBOT_MOBILE
    )


def test_mobile_robot_requires_root_state_estimation_config():
    with pytest.raises(ValueError, match="mobile base require"):
        RobotConfig(
            name=RobotName.PANDA,
            tool_frame="tool",
            home_position=np.zeros(1),
            rest_position=np.zeros(1),
            actuator_control_modes={},
            base_frame="root",
        )


def test_fixed_base_robot_rejects_root_state_estimation_config():
    mobile_estimation = ROBOT_CONFIGS[RobotName.ELROBOT_MOBILE].state_estimation
    assert mobile_estimation is not None

    with pytest.raises(ValueError, match="Fixed-base robots cannot configure"):
        RobotConfig(
            name=RobotName.PANDA,
            tool_frame="tool",
            home_position=np.zeros(1),
            rest_position=np.zeros(1),
            actuator_control_modes={},
            state_estimation=mobile_estimation,
        )
