"""Tests for the dynamic-target HomingPolicy."""

from dataclasses import replace

import numpy as np
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.hardware.actuators.config import ActuatorControlMode
from humanoid.policy.homing import HomingPolicy
from humanoid.types.observation import Observation
from humanoid.types.robot import RobotJointCommand, RobotState


def _observation(q: np.ndarray, q_cmd: np.ndarray | None = None) -> Observation:
    state = RobotState(
        timestamp=0.0,
        joint_positions=q.copy(),
        joint_velocities=np.zeros_like(q),
        actuator_temperatures=np.zeros_like(q),
    )
    joint_cmd = (
        RobotJointCommand(timestamp=0.0, joint_positions=q_cmd.copy())
        if q_cmd is not None
        else None
    )
    return Observation(robot_state=state, robot_joint_command=joint_cmd)


@pytest.fixture
def panda_config():
    return ROBOT_CONFIGS["panda"]


@pytest.fixture
def policy(panda_config) -> HomingPolicy:
    return HomingPolicy(speed=1.0, dt=0.01, robot_config=panda_config)


class TestWithoutTarget:
    def test_step_returns_empty_action_when_no_target_set(self, policy, panda_config):
        action = policy.step(_observation(panda_config.home_position))
        assert action.joint_positions is None

    def test_is_done_false_when_no_target_set(self, policy):
        assert policy.is_done is False


class TestWithTarget:
    def test_set_target_builds_trajectory_on_next_step(self, policy, panda_config):
        policy.set_target(panda_config.rest_position)
        action = policy.step(_observation(panda_config.home_position))
        assert action.joint_positions is not None

    def test_trajectory_reaches_target_for_position_controlled_joints(self, policy, panda_config):
        policy.set_target(panda_config.rest_position)
        last_q = None
        for _ in range(2000):  # plenty of steps for the trajectory to finish
            action = policy.step(_observation(panda_config.home_position))
            if action.joint_positions is not None:
                last_q = action.joint_positions
            if policy.is_done:
                break

        assert policy.is_done
        assert last_q is not None
        # On the panda config every joint is position-controlled, so the final
        # commanded q matches the target on all joints.
        np.testing.assert_allclose(last_q, panda_config.rest_position, atol=1e-9)

    def test_velocity_controlled_joints_held_at_q_start(self, panda_config):
        """Velocity-controlled joints must not be moved by homing."""
        # Build a config where one joint is velocity-controlled.
        control_modes = dict(panda_config.actuator_control_modes)
        first_joint = next(iter(control_modes))
        control_modes[first_joint] = ActuatorControlMode.VELOCITY
        mixed = replace(panda_config, actuator_control_modes=control_modes)

        policy = HomingPolicy(speed=1.0, dt=0.01, robot_config=mixed)
        q_start = panda_config.home_position.copy()
        q_start[0] = 0.42  # arbitrary start position
        policy.set_target(panda_config.rest_position)

        for _ in range(2000):
            action = policy.step(_observation(q_start))
            if policy.is_done:
                final = action.joint_positions
                break

        assert final is not None
        # Velocity-controlled joint stays at q_start; others reach the target.
        assert final[0] == pytest.approx(0.42)
        np.testing.assert_allclose(final[1:], panda_config.rest_position[1:], atol=1e-9)

    def test_set_target_replaces_previous_trajectory(self, policy, panda_config):
        policy.set_target(panda_config.home_position)
        policy.step(_observation(panda_config.rest_position))
        assert policy._trajectory  # built

        # Retarget mid-execution — trajectory should reset for the new goal.
        policy.set_target(panda_config.rest_position)
        assert policy._trajectory == []
        assert policy._step == 0

    def test_reset_keeps_target_for_reactivation(self, policy, panda_config):
        policy.set_target(panda_config.rest_position)
        policy.step(_observation(panda_config.home_position))
        policy.reset()
        # Target survives reset so the next step rebuilds the trajectory from
        # a fresh q_start.
        assert policy._target_position is not None
        action = policy.step(_observation(panda_config.home_position))
        assert action.joint_positions is not None
