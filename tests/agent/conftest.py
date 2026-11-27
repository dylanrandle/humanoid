import time

import numpy as np
import pytest

from humanoid.api.services.observations import ObservationService
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.motion.executor import MotionExecutor
from humanoid.policy.motion import MotionPolicy
from humanoid.robots.base import Robot
from humanoid.types.homing import HomingPreset
from humanoid.types.motion import MotionServiceConfig
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import RobotState


@pytest.fixture(scope="module")
def robot():
    return Robot(TRISKEL_CONFIG)


@pytest.fixture
def service(robot):
    policy = MotionPolicy(robot)
    service = MotionExecutor(policy, MotionServiceConfig())
    state = RobotState(
        timestamp=time.perf_counter(),
        joint_positions=TRISKEL_CONFIG.homing_presets[HomingPreset.HOME].copy(),
        joint_velocities=np.zeros(robot.model.nv),
        actuator_temperatures=np.zeros(robot.model.nq),
    )
    service.observe(Observation(robot_state=state, mode=Mode.AUTONOMY))
    yield service
    service.close()


@pytest.fixture
def observations(service):

    result = ObservationService(TRISKEL_CONFIG, collect=False)
    result.observe(service._observation)
    yield result
    result.close()
