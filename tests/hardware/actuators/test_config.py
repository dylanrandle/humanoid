import numpy as np
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.hardware.actuators.config import (
    ActuatorControlMode,
    ActuatorHardwareConfig,
)
from humanoid.hardware.actuators.feetech.config import (
    FeetechActuatorConfig,
    FeetechActuatorControllerConfig,
)
from humanoid.hardware.config import RobotHardwareConfig
from humanoid.types.robot import RobotConfig, RobotName

MAIN_CONTROLLER = "main"
EXPECTED_GRIPPER_ACTUATOR_ID = 8
EXPECTED_ACTUATOR_COUNT = 2


def _actuator(
    actuator_id: int,
    *,
    controller: str = MAIN_CONTROLLER,
) -> FeetechActuatorConfig:
    return FeetechActuatorConfig(
        actuator_id=actuator_id,
        controller=controller,
    )


def test_every_robot_defines_its_runtime_actuator_control_modes():
    assert set(ROBOT_CONFIGS) == set(RobotName)
    assert all(config.actuator_control_modes for config in ROBOT_CONFIGS.values())


def test_panda_has_no_physical_hardware_configuration():
    assert ROBOT_CONFIGS[RobotName.PANDA].hardware is None


def test_elrobot_mobile_actuator_hardware_configuration():
    config = ROBOT_CONFIGS[RobotName.ELROBOT_MOBILE]
    assert config.hardware is not None
    assert config.hardware.actuators is not None
    actuator_hardware = config.hardware.actuators

    assert MAIN_CONTROLLER in actuator_hardware.controllers
    assert [actuator_hardware.joints[f"wheel_{index}"].actuator_id for index in range(1, 4)] == [
        250,
        251,
        252,
    ]
    assert all(
        config.actuator_control_modes[f"wheel_{index}"] is ActuatorControlMode.VELOCITY
        for index in range(1, 4)
    )
    assert actuator_hardware.joints["gripper_1"].actuator_id == EXPECTED_GRIPPER_ACTUATOR_ID
    assert actuator_hardware.joints["gripper_1"].inverted is True


def test_duplicate_actuator_id_on_same_controller_is_rejected():
    with pytest.raises(ValueError, match="Duplicate actuator ID 1"):
        ActuatorHardwareConfig(
            controllers={MAIN_CONTROLLER: FeetechActuatorControllerConfig()},
            joints={"joint_a": _actuator(1), "joint_b": _actuator(1)},
        )


def test_same_actuator_id_on_different_controllers_is_allowed():
    hardware = ActuatorHardwareConfig(
        controllers={
            "left": FeetechActuatorControllerConfig(),
            "right": FeetechActuatorControllerConfig(),
        },
        joints={
            "joint_a": _actuator(1, controller="left"),
            "joint_b": _actuator(1, controller="right"),
        },
    )

    assert len(hardware.joints) == EXPECTED_ACTUATOR_COUNT


def test_unknown_controller_is_rejected():
    with pytest.raises(ValueError, match="unknown controller"):
        ActuatorHardwareConfig(
            controllers={},
            joints={"joint_a": _actuator(1)},
        )


def test_feetech_controller_rejects_invalid_connection_details():
    with pytest.raises(ValueError, match="port must not be empty"):
        FeetechActuatorControllerConfig(port="")
    with pytest.raises(ValueError, match="baud rate must be positive"):
        FeetechActuatorControllerConfig(baud_rate=0)


@pytest.mark.parametrize("actuator_id", [1, 253])
def test_feetech_actuator_accepts_id_boundaries(actuator_id):
    assert _actuator(actuator_id).actuator_id == actuator_id


@pytest.mark.parametrize("actuator_id", [0, 254])
def test_feetech_actuator_rejects_ids_outside_register_range(actuator_id):
    with pytest.raises(ValueError, match="ID must be between 1 and 253"):
        _actuator(actuator_id)


@pytest.mark.parametrize("acceleration", [0, 254])
def test_feetech_actuator_accepts_acceleration_boundaries(acceleration):
    actuator = FeetechActuatorConfig(
        actuator_id=1,
        controller=MAIN_CONTROLLER,
        max_acceleration=acceleration,
    )
    assert actuator.max_acceleration == acceleration


@pytest.mark.parametrize("acceleration", [-1, 255, 1.5])
def test_feetech_actuator_rejects_acceleration_outside_register_range(acceleration):
    with pytest.raises(ValueError, match="acceleration must be between 0 and 254"):
        FeetechActuatorConfig(
            actuator_id=1,
            controller=MAIN_CONTROLLER,
            max_acceleration=acceleration,
        )


def test_robot_config_owns_physical_binding_equality_validation():
    hardware = ActuatorHardwareConfig(
        controllers={MAIN_CONTROLLER: FeetechActuatorControllerConfig()},
        joints={"different_joint": _actuator(1)},
    )

    with pytest.raises(ValueError, match="bindings must match"):
        RobotConfig(
            name=RobotName.PANDA,
            tool_frame="tool",
            home_position=np.zeros(1),
            rest_position=np.zeros(1),
            actuator_control_modes={"joint": ActuatorControlMode.POSITION},
            hardware=RobotHardwareConfig(actuators=hardware),
        )
