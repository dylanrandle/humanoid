import numpy as np
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.hardware.actuators.feetech.config import (
    FeetechActuatorConfig,
    FeetechActuatorControllerConfig,
    FeetechPIDGains,
)
from humanoid.hardware.config import RobotHardwareConfig
from humanoid.types.actuator import (
    ActuatorControlMode,
    ActuatorHardwareConfig,
)
from humanoid.types.homing import HomingPreset
from humanoid.types.robot import (
    RobotConfig,
    RobotName,
    RobotToolConfig,
)

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


def test_triskel_actuator_hardware_configuration():
    config = ROBOT_CONFIGS[RobotName.TRISKEL]
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
    for index in range(1, 8):
        actuator = actuator_hardware.joints[f"arm_{index}"]
        assert isinstance(actuator, FeetechActuatorConfig)
        gains = actuator.position_pid
        assert gains == FeetechPIDGains(p=32, i=0, d=32)
    gripper = actuator_hardware.joints["gripper_1"]
    assert isinstance(gripper, FeetechActuatorConfig)
    assert gripper.position_pid == FeetechPIDGains(
        p=32,
        i=0,
        d=32,
    )


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


@pytest.mark.parametrize("velocity", [0.0, -1.0, np.inf, np.nan])
def test_feetech_actuator_rejects_invalid_maximum_position_velocity(velocity):
    with pytest.raises(ValueError, match="maximum position velocity"):
        FeetechActuatorConfig(
            actuator_id=1,
            controller=MAIN_CONTROLLER,
            max_position_velocity=velocity,
        )


@pytest.mark.parametrize("gain", [-1.0, np.inf, np.nan])
def test_feetech_actuator_rejects_invalid_position_tracking_error_gain(gain):
    with pytest.raises(ValueError, match="tracking-error gain"):
        FeetechActuatorConfig(
            actuator_id=1,
            controller=MAIN_CONTROLLER,
            position_tracking_error_gain=gain,
        )


@pytest.mark.parametrize(
    ("gain_name", "gain_value"),
    [("p", -1), ("i", 255), ("d", 1.5)],
)
def test_feetech_pid_rejects_invalid_register_values(gain_name, gain_value):
    gains = {"p": 32, "i": 0, "d": 32, gain_name: gain_value}

    with pytest.raises(ValueError, match=f"{gain_name.upper()} gain must be between 0 and 254"):
        FeetechPIDGains(**gains)


def test_robot_config_owns_physical_binding_equality_validation():
    hardware = ActuatorHardwareConfig(
        controllers={MAIN_CONTROLLER: FeetechActuatorControllerConfig()},
        joints={"different_joint": _actuator(1)},
    )

    with pytest.raises(ValueError, match="bindings must match"):
        RobotConfig(
            name=RobotName.PANDA,
            tool=RobotToolConfig(frame="tool"),
            homing_presets={
                HomingPreset.HOME: np.zeros(1),
                HomingPreset.REST: np.zeros(1),
            },
            actuator_control_modes={"joint": ActuatorControlMode.POSITION},
            hardware=RobotHardwareConfig(actuators=hardware),
        )
