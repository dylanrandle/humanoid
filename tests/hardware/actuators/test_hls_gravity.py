from argparse import Namespace
from unittest.mock import Mock

import numpy as np
import pinocchio as pin
import pytest

from humanoid.config.actuator import HLS3930M_CURRENT_CALIBRATION
from humanoid.hardware.actuators.feetech.scripts import gravity
from humanoid.hardware.actuators.feetech.scripts.gravity import (
    load_model,
    read_configuration,
    torque_to_raw,
)


def test_current_conversion_clamps_and_preserves_sign():
    calibration = HLS3930M_CURRENT_CALIBRATION
    unit = calibration.amperes_per_unit * calibration.effort_per_ampere
    limit = 0.4
    for torque in [-2.0, -0.3, 0.0, 0.3, 2.0]:
        raw = torque_to_raw(torque, 0.4)
        assert abs(raw * unit) <= limit
        assert abs(raw * unit - np.clip(torque, -0.4, 0.4)) < unit
        assert torque_to_raw(torque, 0.4, inverted=True) == -raw
    with pytest.raises(ValueError):
        torque_to_raw(float("nan"), 0.4)


def test_all_arm_positions_are_read_without_writes():
    model = load_model()
    handler = Mock()
    handler.ReadPos.return_value = (2048, 0, 0)
    q = read_configuration(handler, model)
    assert [call.args[0] for call in handler.ReadPos.call_args_list] == list(range(1, 9))
    assert all(call[0] == "ReadPos" for call in handler.mock_calls)
    arm = model.joints[model.getJointId("arm_3")]
    gripper = model.joints[model.getJointId("gripper_1")]
    assert q[gripper.idx_q] == -q[arm.idx_q]


def test_gravity_sign_matches_potential_energy_gradient():
    model = load_model()
    data = model.createData()
    q = pin.neutral(model)
    joint = model.joints[model.getJointId("arm_3")]
    q[joint.idx_q] = 0.4
    gravity = pin.computeGeneralizedGravity(model, data, q).copy()
    q_plus, q_minus = q.copy(), q.copy()
    q_plus[joint.idx_q] += 1e-6
    q_minus[joint.idx_q] -= 1e-6

    def potential(configuration):
        # Pinocchio's energy helper does not support mimic joints; sum link energy.
        pin.forwardKinematics(model, data, configuration)
        return -sum(
            inertia.mass * model.gravity.linear.dot(placement.act(inertia.lever))
            for inertia, placement in zip(model.inertias, data.oMi, strict=True)
        )

    plus = potential(q_plus)
    minus = potential(q_minus)
    assert gravity[joint.idx_v] == pytest.approx((plus - minus) / 2e-6, abs=1e-7)


def test_feedback_failure_cleans_up_only_target(monkeypatch):
    handler, port = Mock(), Mock()
    handler.ping.return_value = (4106, 0, 0)
    handler.read1ByteTxRx.return_value = (0, 0, 0)
    handler.read2ByteTxRx.return_value = (100, 0, 0)
    handler.write1ByteTxRx.return_value = (0, 0)
    handler.write2ByteTxRx.return_value = (0, 0)
    monkeypatch.setattr(gravity.scs, "PortHandler", Mock(return_value=port))
    monkeypatch.setattr(gravity.scs, "hls", Mock(return_value=handler))
    monkeypatch.setattr(gravity, "set_mode", Mock())
    cleanup_mock = Mock()
    monkeypatch.setattr(gravity, "cleanup", cleanup_mock)
    q = pin.neutral(load_model())
    monkeypatch.setattr(
        gravity, "read_configuration", Mock(side_effect=[q, RuntimeError("lost feedback")])
    )
    args = Namespace(port="mock", baud_rate=1_000_000, enable=True, output_limit=100, duration=5)
    with pytest.raises(RuntimeError, match="lost feedback"):
        gravity.run(args)
    cleanup_mock.assert_called_once_with(handler, 3, 0, 100)
    port.closePort.assert_called_once()
    for method in (handler.write1ByteTxRx, handler.write2ByteTxRx):
        assert all(call.args[0] == gravity.TARGET_ID for call in method.call_args_list)
