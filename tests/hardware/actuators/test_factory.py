from unittest.mock import patch

import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.hardware.actuators.config import ActuatorControlMode, ActuatorHardwareConfig
from humanoid.hardware.actuators.factory import create_actuator_system
from humanoid.hardware.actuators.feetech.config import (
    FeetechActuatorConfig,
    FeetechActuatorControllerConfig,
)
from humanoid.hardware.actuators.simulation import SimulatedActuatorSystem
from humanoid.hardware.actuators.system import CompositeActuatorSystem
from humanoid.robots.base import Robot
from humanoid.types.process import Runtime
from humanoid.types.robot import RobotConfig, RobotName


@pytest.mark.parametrize("config", ROBOT_CONFIGS.values(), ids=lambda config: config.name)
def test_every_robot_can_create_a_simulated_actuator_system(config: RobotConfig):
    robot = Robot(config)
    initial_positions = {
        joint_name: float(index) for index, joint_name in enumerate(config.actuator_control_modes)
    }

    actuator_system = create_actuator_system(
        Runtime.SIM,
        config.actuator_control_modes,
        config.hardware.actuators if config.hardware is not None else None,
        initial_positions,
    )

    assert isinstance(actuator_system, SimulatedActuatorSystem)
    actuator_system.connect()
    states = actuator_system.read_states()
    actuator_system.disconnect()
    assert set(robot.actuator_joint_names) == set(config.actuator_control_modes)
    assert states.keys() == config.actuator_control_modes.keys()
    assert {joint_name: state.position for joint_name, state in states.items()} == pytest.approx(
        initial_positions
    )


def test_robot_without_actuator_hardware_fails_closed_in_real_runtime():
    config = ROBOT_CONFIGS[RobotName.PANDA]

    with pytest.raises(RuntimeError, match="requires configured actuator hardware"):
        create_actuator_system(
            Runtime.REAL,
            config.actuator_control_modes,
            None,
            dict.fromkeys(config.actuator_control_modes, 0.0),
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
            Runtime.REAL,
            {"different_joint": ActuatorControlMode.POSITION},
            hardware,
            {"different_joint": 0.0},
        )


def test_real_runtime_uses_nested_physical_actuator_hardware():
    config = ROBOT_CONFIGS[RobotName.SO101]
    assert config.hardware is not None

    with patch("humanoid.hardware.actuators.factory.FeetechActuatorDriver"):
        actuator_system = create_actuator_system(
            Runtime.REAL,
            config.actuator_control_modes,
            config.hardware.actuators,
            dict.fromkeys(config.actuator_control_modes, 0.0),
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
        create_actuator_system(Runtime.REAL, modes, hardware, dict.fromkeys(modes, 0.0))


def test_multiple_feetech_controllers_receive_distinct_connection_configs():
    hardware = _two_controller_hardware("/dev/left", "/dev/right")
    modes = {
        "left_joint": ActuatorControlMode.POSITION,
        "right_joint": ActuatorControlMode.VELOCITY,
    }

    with patch("humanoid.hardware.actuators.factory.FeetechActuatorDriver") as driver_cls:
        create_actuator_system(Runtime.REAL, modes, hardware, dict.fromkeys(modes, 0.0))

    passed_configs = [call.args[2] for call in driver_cls.call_args_list]
    assert {config.port for config in passed_configs} == {"/dev/left", "/dev/right"}
