import numpy as np
import pytest

from humanoid.config.robot.elrobot_mobile import ELROBOT_MOBILE_CONFIG
from humanoid.robots.base import Robot
from humanoid.state_estimation.root.base import RootState
from humanoid.state_estimation.root.config import RootStateEstimatorConfig
from humanoid.state_estimation.root.factory import create_root_state_estimator
from humanoid.state_estimation.root.wheel_dead_reckoning import (
    WheelDeadReckoningRootStateEstimator,
    WheelDeadReckoningRootStateEstimatorConfig,
)


def _initial_state() -> RootState:
    return RootState(
        position=np.array([0.0, 0.0, 1.0, 0.0]),
        velocity=np.zeros(3),
    )


def test_builds_configured_wheel_dead_reckoning_estimator():
    estimator = create_root_state_estimator(
        WheelDeadReckoningRootStateEstimatorConfig(),
        Robot(ELROBOT_MOBILE_CONFIG),
        _initial_state(),
    )

    assert isinstance(estimator, WheelDeadReckoningRootStateEstimator)


def test_rejects_unsupported_estimator_config():
    with pytest.raises(TypeError, match="Unsupported root-state estimator"):
        create_root_state_estimator(
            RootStateEstimatorConfig(),
            Robot(ELROBOT_MOBILE_CONFIG),
            _initial_state(),
        )
