from unittest.mock import Mock, call, patch

import pytest
from vassar_feetech_servo_sdk import ServoController

from humanoid.hardware.actuators.config import ActuatorControlMode
from humanoid.hardware.actuators.feetech.config import (
    FEETECH_ACCELERATION_MAX,
    FEETECH_ACCELERATION_MIN,
    FeetechActuatorConfig,
    FeetechActuatorControllerConfig,
    FeetechServoType,
)
from humanoid.hardware.actuators.feetech.driver import (
    ADDR_GOAL_POSITION,
    ADDR_GOAL_SPEED,
    ADDR_OPERATING_MODE,
    ADDR_TORQUE_ENABLE,
    SPEED_UNIT_RAD_S,
    FeetechActuatorDriver,
)

TEST_PORT = "/dev/test-feetech"
TEST_BAUD_RATE = 115_200
MID_POSITION = 2048
RENAMED_ACTUATOR_ID = 7
EXPECTED_ACCELERATION_GROUP_COUNT = 2


def _actuator(
    actuator_id: int = 1,
    *,
    inverted: bool = False,
    max_acceleration: int = 10,
) -> FeetechActuatorConfig:
    return FeetechActuatorConfig(
        actuator_id=actuator_id,
        controller="main",
        inverted=inverted,
        max_acceleration=max_acceleration,
    )


def _driver(
    actuators: list[FeetechActuatorConfig] | None = None,
    modes: dict[int, ActuatorControlMode] | None = None,
) -> FeetechActuatorDriver:
    actuators = actuators or [_actuator()]
    modes = modes or {actuator.actuator_id: ActuatorControlMode.POSITION for actuator in actuators}
    return FeetechActuatorDriver(
        actuators,
        modes,
        FeetechActuatorControllerConfig(
            port=TEST_PORT,
            baud_rate=TEST_BAUD_RATE,
            servo_type=FeetechServoType.STS,
        ),
    )


def test_connection_config_is_passed_to_sdk():
    driver = _driver()

    assert driver.port == TEST_PORT
    assert driver.baudrate == TEST_BAUD_RATE
    assert driver.servo_type == "sts"


def test_angle_conversion_round_trips_with_inversion():
    driver = _driver([_actuator(inverted=True)])

    raw_position = driver.angle_to_position(0.75, 1)

    assert raw_position < MID_POSITION
    assert driver.position_to_angle(raw_position, 1) == pytest.approx(0.75, abs=0.002)


def test_position_writes_are_grouped_by_configured_acceleration():
    driver = _driver([_actuator(1, max_acceleration=10), _actuator(2, max_acceleration=20)])
    driver.packet_handler = Mock()
    driver.packet_handler.SyncWritePosEx.return_value = True
    driver.packet_handler.groupSyncWrite.txPacket.return_value = 0

    driver.write_position({1: 0.1, 2: 0.2})

    assert driver.packet_handler.SyncWritePosEx.call_args_list == [
        call(1, driver.angle_to_position(0.1, 1), 32767, 10),
        call(2, driver.angle_to_position(0.2, 2), 32767, 20),
    ]
    assert (
        driver.packet_handler.groupSyncWrite.txPacket.call_count
        == EXPECTED_ACCELERATION_GROUP_COUNT
    )


def test_position_sync_packet_failure_is_raised():
    driver = _driver()
    driver.packet_handler = Mock()
    driver.packet_handler.SyncWritePosEx.return_value = True
    driver.packet_handler.groupSyncWrite.txPacket.return_value = -2
    driver.packet_handler.getTxRxResult.return_value = "transmit failed"

    with pytest.raises(RuntimeError, match="position sync write failed"):
        driver.write_position({1: 0.1})

    driver.packet_handler.groupSyncWrite.clearParam.assert_called()


@pytest.mark.parametrize(
    "acceleration",
    [FEETECH_ACCELERATION_MIN, FEETECH_ACCELERATION_MAX],
)
def test_position_acceleration_override_accepts_register_boundaries(acceleration):
    driver = _driver()
    driver.packet_handler = Mock()
    driver.packet_handler.SyncWritePosEx.return_value = True
    driver.packet_handler.groupSyncWrite.txPacket.return_value = 0

    driver.write_position({1: 0.1}, acceleration=acceleration)

    assert driver.packet_handler.SyncWritePosEx.call_args.args[-1] == acceleration


@pytest.mark.parametrize("acceleration", [-1, 255, 1.5])
def test_position_acceleration_override_rejects_invalid_register_values(acceleration):
    driver = _driver()
    driver.packet_handler = Mock()

    with pytest.raises(ValueError, match="acceleration must be between 0 and 254"):
        driver.write_position({1: 0.1}, acceleration=acceleration)

    driver.packet_handler.SyncWritePosEx.assert_not_called()


def test_velocity_conversion_applies_inversion():
    driver = _driver(
        [_actuator(inverted=True)],
        {1: ActuatorControlMode.VELOCITY},
    )
    driver.packet_handler = Mock()
    driver.packet_handler.write2ByteTxRx.return_value = (0, 0)

    driver.write_velocity({1: 2 * SPEED_UNIT_RAD_S})

    driver.packet_handler.write2ByteTxRx.assert_called_once_with(
        1,
        ADDR_GOAL_SPEED,
        0x8002,
    )


def test_velocity_communication_failure_is_raised():
    driver = _driver(modes={1: ActuatorControlMode.VELOCITY})
    driver.packet_handler = Mock()
    driver.packet_handler.write2ByteTxRx.return_value = (1, 0)

    with pytest.raises(RuntimeError, match="velocity write failed"):
        driver.write_velocity({1: 1.0})


@pytest.mark.parametrize(
    ("raw_velocity", "inverted", "expected_units"),
    [
        (2, False, 2),
        (-2, False, -2),
        (-2, True, 2),
    ],
)
def test_velocity_reads_use_sdk_signed_value(raw_velocity, inverted, expected_units):
    driver = _driver(
        [_actuator(inverted=inverted)],
        {1: ActuatorControlMode.VELOCITY},
    )
    driver.packet_handler = Mock()
    driver.packet_handler.ReadSpeed.return_value = (raw_velocity, 0, 0)

    assert driver.read_velocity(1) == pytest.approx(expected_units * SPEED_UNIT_RAD_S)


@pytest.mark.parametrize(
    ("control_mode", "safe_goal_address", "safe_goal"),
    [
        (ActuatorControlMode.POSITION, ADDR_GOAL_POSITION, 1234),
        (ActuatorControlMode.VELOCITY, ADDR_GOAL_SPEED, 0),
    ],
)
def test_connect_stages_safe_goal_before_enabling_torque(
    control_mode,
    safe_goal_address,
    safe_goal,
):
    driver = _driver(modes={1: control_mode})
    driver.packet_handler = Mock()
    driver.packet_handler.read1ByteTxRx.side_effect = [
        (0, 0, 0),
        (0 if control_mode is ActuatorControlMode.POSITION else 1, 0, 0),
        (1, 0, 0),
    ]
    driver.packet_handler.write1ByteTxRx.return_value = (0, 0)
    driver.packet_handler.write2ByteTxRx.return_value = (0, 0)
    driver.packet_handler.ReadPos.return_value = (safe_goal, 0, 0)

    with patch.object(
        ServoController,
        "connect",
        autospec=True,
        side_effect=lambda connected_driver: setattr(connected_driver, "_connected", True),
    ) as connect:
        driver.connect()

    connect.assert_called_once_with(driver)
    assert driver.packet_handler.mock_calls == [
        call.write1ByteTxRx(1, ADDR_TORQUE_ENABLE, 0),
        call.read1ByteTxRx(1, ADDR_TORQUE_ENABLE),
        call.read1ByteTxRx(1, ADDR_OPERATING_MODE),
        *([call.ReadPos(1)] if control_mode is ActuatorControlMode.POSITION else []),
        call.write2ByteTxRx(1, safe_goal_address, safe_goal),
        call.write1ByteTxRx(1, ADDR_TORQUE_ENABLE, 1),
        call.read1ByteTxRx(1, ADDR_TORQUE_ENABLE),
    ]


def test_connect_applies_and_verifies_configured_operating_mode():
    driver = _driver(modes={1: ActuatorControlMode.VELOCITY})
    driver.packet_handler = Mock()
    driver.packet_handler.read1ByteTxRx.side_effect = [
        (0, 0, 0),
        (0, 0, 0),
        (1, 0, 0),
        (1, 0, 0),
    ]
    driver.packet_handler.write1ByteTxRx.return_value = (0, 0)
    driver.packet_handler.write2ByteTxRx.return_value = (0, 0)

    with (
        patch.object(
            ServoController,
            "connect",
            autospec=True,
            side_effect=lambda connected_driver: setattr(
                connected_driver,
                "_connected",
                True,
            ),
        ),
        patch.object(driver, "set_operating_mode", return_value=True) as set_mode,
    ):
        driver.connect()

    set_mode.assert_called_once_with(1, 1)


def test_connect_rolls_back_when_configuration_fails():
    driver = _driver()
    driver.packet_handler = Mock()
    driver.packet_handler.write1ByteTxRx.return_value = (0, 0)
    driver.packet_handler.read1ByteTxRx.side_effect = [(0, 0, 0), (0, 1, 0)]

    with (
        patch.object(ServoController, "connect", autospec=True),
        patch.object(ServoController, "disconnect", autospec=True) as disconnect,
        pytest.raises(RuntimeError, match="read operating mode"),
    ):
        driver.connect()

    disconnect.assert_called_once_with(driver)


def test_id_change_retains_active_wire_id_for_disconnect_cleanup():
    driver = _driver()
    packet_handler = Mock()
    packet_handler.write1ByteTxRx.return_value = (0, 0)
    driver.packet_handler = packet_handler
    driver.port_handler = Mock()
    driver._connected = True

    with (
        patch.object(ServoController, "set_motor_id", autospec=True, return_value=True),
        patch("vassar_feetech_servo_sdk.controller.time.sleep"),
    ):
        assert driver.set_actuator_id(1, RENAMED_ACTUATOR_ID, confirm=False) is True
        driver.disconnect()

    assert driver.servo_ids is driver.actuator_ids
    assert driver.actuator_ids == [1]
    assert set(driver.actuator_configs) == {1}
    assert driver.actuator_configs[1].actuator_id == 1
    assert driver.control_modes == {1: ActuatorControlMode.POSITION}
    packet_handler.write1ByteTxRx.assert_called_once_with(
        1,
        ADDR_TORQUE_ENABLE,
        0,
    )


@pytest.mark.parametrize("new_id", [0, 254])
def test_id_change_rejects_invalid_target_before_sdk_call(new_id):
    driver = _driver()

    with (
        patch.object(ServoController, "set_motor_id", autospec=True) as set_motor_id,
        pytest.raises(ValueError, match="new actuator ID must be between 1 and 253"),
    ):
        driver.set_actuator_id(1, new_id, confirm=False)

    set_motor_id.assert_not_called()


def test_id_change_rejects_configured_target_before_sdk_call():
    driver = _driver([_actuator(1), _actuator(2)])

    with (
        patch.object(ServoController, "set_motor_id", autospec=True) as set_motor_id,
        pytest.raises(ValueError, match="ID 2 is already configured"),
    ):
        driver.set_actuator_id(1, 2, confirm=False)

    set_motor_id.assert_not_called()


def test_id_change_rejects_unknown_current_id_before_sdk_call():
    driver = _driver()

    with (
        patch.object(ServoController, "set_motor_id", autospec=True) as set_motor_id,
        pytest.raises(ValueError, match="ID 2 is not configured"),
    ):
        driver.set_actuator_id(2, RENAMED_ACTUATOR_ID, confirm=False)

    set_motor_id.assert_not_called()
