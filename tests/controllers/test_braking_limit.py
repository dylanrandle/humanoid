"""Reachability of a safe stop across changing integration periods."""

import numpy as np
import pink
import pinocchio as pin
import pytest

from humanoid.controllers.constraints import BrakingAccelerationLimit
from humanoid.types.controllers import OperationalSpaceConfig


def test_joint_can_brake_and_reverse_without_an_infeasible_next_step():
    model = pin.Model()
    model.addJoint(0, pin.JointModelRY(), pin.SE3.Identity(), "joint")
    model.lowerPositionLimit[:] = -1.0
    model.upperPositionLimit[:] = 1.0
    acceleration_limit = 2.0
    margin = 0.005
    tolerance = 1e-10
    limit = BrakingAccelerationLimit(model, np.array([acceleration_limit]), margin)
    configuration = pink.Configuration(model, model.createData(), np.zeros(1))
    previous = np.zeros(1)
    random = np.random.default_rng(7)
    for step in range(800):
        dt = random.uniform(1 / 60, 1 / 15)
        limit.set_last_integration(previous, dt)
        inequalities = limit.compute_qp_inequalities(configuration, dt)
        assert inequalities is not None
        _, bounds = inequalities
        lower, upper = -bounds[1] / dt, bounds[0] / dt
        assert lower <= upper + tolerance
        desired_velocity = 1.0 if (step // 100) % 2 == 0 else -1.0
        velocity = np.array([np.clip(desired_velocity, lower, upper)])
        assert np.abs(velocity - previous).max() <= acceleration_limit * dt + tolerance
        configuration.integrate_inplace(velocity, dt)
        assert np.abs(configuration.q).max() <= 1.0 - margin + tolerance
        previous = velocity


@pytest.mark.parametrize("margin", [-0.001, np.nan, np.inf])
def test_invalid_position_margin_is_rejected(margin):
    with pytest.raises(ValueError, match="position margin"):
        OperationalSpaceConfig(joint_position_margin=margin, joint_acceleration_limit=2.0)


def test_position_margin_requires_braking_limits():
    with pytest.raises(ValueError, match="acceleration limit"):
        OperationalSpaceConfig(joint_position_margin=0.005)
