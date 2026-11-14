from dataclasses import replace
from unittest.mock import Mock, patch

import pytest

from humanoid.config.robot.triskel import ACTUATOR_CONFIGS
from humanoid.hardware.actuators.feetech.config import (
    FeetechActuatorControllerConfig,
    FeetechServoType,
)
from humanoid.hardware.actuators.feetech.driver import FeetechActuatorDriver
from humanoid.hardware.actuators.system import CompositeActuatorSystem
from humanoid.types.actuator import ActuatorControlMode


def make_driver(*, inverted=False):
    actuator = replace(ACTUATOR_CONFIGS["arm_3"], inverted=inverted)
    driver = FeetechActuatorDriver(
        [ACTUATOR_CONFIGS["arm_2"], actuator],
        {2: ActuatorControlMode.POSITION, 3: ActuatorControlMode.POSITION},
        FeetechActuatorControllerConfig(port="mock"),
    )
    driver.packet_handler = Mock()
    driver.packet_handler.write2ByteTxRx.return_value = (0, 0)
    return driver


@pytest.mark.parametrize("inverted", [False, True])
def test_mixed_bus_current_command_is_clamped_and_sign_encoded(inverted):
    driver = make_driver(inverted=inverted)
    assert driver.servo_type == "sts"
    driver.write_torque({3: 2.0})
    config = ACTUATOR_CONFIGS["arm_3"]
    calibration = config.current_calibration
    assert calibration is not None
    assert config.gravity_compensation is not None
    expected = int(
        config.gravity_compensation.max_torque_nm
        / (calibration.amperes_per_unit * calibration.effort_per_ampere)
    )
    driver.packet_handler.write2ByteTxRx.assert_called_once_with(
        3, 44, expected | (0x8000 if inverted else 0)
    )
    with pytest.raises(ValueError):
        driver.write_torque({2: 0.2})
    with pytest.raises(ValueError):
        driver.write_torque({3: float("nan")})


def test_sts_cannot_enter_gravity_mode():
    actuator = replace(ACTUATOR_CONFIGS["arm_3"], servo_type=FeetechServoType.STS)
    with pytest.raises(ValueError, match="requires HLS"):
        FeetechActuatorDriver(
            [actuator],
            {3: ActuatorControlMode.POSITION},
            FeetechActuatorControllerConfig(port="mock"),
        )


def test_gravity_initialization_selects_current_mode_and_zeroes_before_enable():
    driver = make_driver()
    driver.packet_handler.read1ByteTxRx.side_effect = [(0, 0, 0), (2, 0, 0)]
    with (
        patch.object(driver, "_set_torque_enabled") as enable,
        patch.object(driver, "set_operating_mode", return_value=True) as mode,
    ):
        driver._configure_actuator(3, ActuatorControlMode.POSITION)
    mode.assert_called_once_with(3, 2)
    assert [call.args for call in enable.call_args_list] == [(3,), (3,)]
    writes = driver.packet_handler.write2ByteTxRx.call_args_list
    assert [call.args for call in writes] == [(3, 48, 100), (3, 44, 0), (3, 46, 0)]


def test_system_routes_gravity_and_rejects_position_for_same_joint():
    driver = Mock()
    system = CompositeActuatorSystem(
        {name: ACTUATOR_CONFIGS[name] for name in ["arm_2", "arm_3"]},
        dict.fromkeys(["arm_2", "arm_3"], ActuatorControlMode.POSITION),
        {"main": driver},
    )
    system.write_torques({"arm_3": -0.2})
    driver.write_torque.assert_called_once_with({3: -0.2})
    with pytest.raises(ValueError, match="gravity/current"):
        system.write_positions({"arm_3": 1.0})
    system.stop()
    driver.write_torque.assert_called_with({3: 0.0})


@pytest.mark.parametrize("zero_fails", [False, True])
def test_disconnect_disables_even_if_zero_current_write_fails(zero_fails):
    driver = make_driver()
    driver._connected = True
    events = []

    def zero(torques):
        events.append(("zero", torques))
        if zero_fails:
            raise RuntimeError("write failed")

    def disable(actuator_id, *, enabled):
        events.append(("enable", actuator_id, enabled))

    with (
        patch.object(driver, "write_torque", side_effect=zero),
        patch.object(driver, "_set_torque_enabled", side_effect=disable),
        patch("humanoid.hardware.actuators.feetech.driver.ServoController.disconnect") as close,
    ):
        if zero_fails:
            with pytest.raises(ExceptionGroup, match="stop gravity"):
                driver.disconnect()
        else:
            driver.disconnect()
    assert events == [("zero", {3: 0.0}), ("enable", 3, False)]
    close.assert_called_once()


def test_low_level_rejects_position_or_velocity_on_gravity_joint():
    driver = make_driver()
    with pytest.raises(ValueError, match="position commands"):
        driver.write_position({3: 0.1})
    with pytest.raises(ValueError, match="velocity commands"):
        driver.write_velocity({3: 0.1})
