import time

import numpy as np
import pytest

from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.policy.remote_control import RemoteControlPolicy
from humanoid.remote_control.service import RemoteControlService
from humanoid.robots.base import Robot
from humanoid.types.homing import HomingPreset
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode
from humanoid.types.remote_api import (
    RemoteControlServerConfig,
)
from humanoid.types.robot import RobotState


@pytest.fixture(scope="module")
def robot():
    return Robot(TRISKEL_CONFIG)


@pytest.fixture
def service(robot):
    policy = RemoteControlPolicy(robot)
    service = RemoteControlService(
        policy, RemoteControlServerConfig(host="127.0.0.1", port=0, token="test-token")
    )
    state = RobotState(
        timestamp=time.perf_counter(),
        joint_positions=TRISKEL_CONFIG.homing_presets[HomingPreset.HOME].copy(),
        joint_velocities=np.zeros(robot.model.nv),
        actuator_temperatures=np.zeros(robot.model.nq),
    )
    service.observe(Observation(robot_state=state, mode=Mode.AUTONOMY))
    return service
