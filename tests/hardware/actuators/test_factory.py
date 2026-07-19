from unittest.mock import patch

import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.hardware.actuators.factory import create_actuator_system
from humanoid.hardware.actuators.feetech.config import (
    FeetechActuatorConfig,
    FeetechActuatorControllerConfig,
)
from humanoid.hardware.actuators.system import CompositeActuatorSystem
from humanoid.types.actuator import ActuatorControlMode, ActuatorHardwareConfig
from humanoid.types.robot import RobotName


def test_robot_without_actuator_hardware_fails_closed_in_real_runtime():
    config = ROBOT_CONFIGS[RobotName.PANDA]

    with pytest.raises(RuntimeError, match="requires configured actuator hardware"):
        create_actuator_system(
            config.actuator_control_modes,
            None,
        )


def test_real_runtime_rejects_mismatched_control_modes_at_public_boundary():
    hardware = ActuatorHardwareConfig(
        controllers={"main": FeetechActuatorControllerConfig(port="/dev/test")},
        joints={
            "hardware_joint": FeetechActuatorConfig(
                actuator_id=1,
                controller="main",
            )
        },
    )

    with pytest.raises(ValueError, match="bindings must match"):
        create_actuator_system(
            {"different_joint": ActuatorControlMode.POSITION},
            hardware,
        )


def test_real_runtime_uses_nested_physical_actuator_hardware():
    config = ROBOT_CONFIGS[RobotName.SO101]
    assert config.hardware is not None

    with patch("humanoid.hardware.actuators.factory.FeetechActuatorDriver"):
        actuator_system = create_actuator_system(
            config.actuator_control_modes,
            config.hardware.actuators,
        )

    assert isinstance(actuator_system, CompositeActuatorSystem)


def _two_controller_hardware(left_port: str | None, right_port: str | None):
    return ActuatorHardwareConfig(
        controllers={
            "left": FeetechActuatorControllerConfig(port=left_port),
            "right": FeetechActuatorControllerConfig(port=right_port),
        },
        joints={
            "left_joint": FeetechActuatorConfig(actuator_id=1, controller="left"),
            "right_joint": FeetechActuatorConfig(actuator_id=1, controller="right"),
        },
    )


@pytest.mark.parametrize(
    ("left_port", "right_port", "message"),
    [
        (None, "/dev/right", "must specify a port"),
        ("/dev/shared", "/dev/shared", "distinct serial ports"),
    ],
)
def test_multiple_feetech_controllers_require_distinct_explicit_ports(
    left_port,
    right_port,
    message,
):
    hardware = _two_controller_hardware(left_port, right_port)
    modes = {
        "left_joint": ActuatorControlMode.POSITION,
        "right_joint": ActuatorControlMode.VELOCITY,
    }

    with pytest.raises(ValueError, match=message):
        create_actuator_system(modes, hardware)


def test_multiple_feetech_controllers_receive_distinct_connection_configs():
    hardware = _two_controller_hardware("/dev/left", "/dev/right")
    modes = {
        "left_joint": ActuatorControlMode.POSITION,
        "right_joint": ActuatorControlMode.VELOCITY,
    }

    with patch("humanoid.hardware.actuators.factory.FeetechActuatorDriver") as driver_cls:
        create_actuator_system(modes, hardware)

    passed_configs = [call.args[2] for call in driver_cls.call_args_list]
    assert {config.port for config in passed_configs} == {"/dev/left", "/dev/right"}
