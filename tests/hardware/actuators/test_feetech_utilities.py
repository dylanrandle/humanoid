import sys
from unittest.mock import MagicMock, patch

import pytest
from pynput import keyboard

from humanoid.hardware.actuators.feetech.config import (
    FEETECH_ACTUATOR_ID_MAX,
    FEETECH_ACTUATOR_ID_MIN,
    FeetechActuatorControllerConfig,
    FeetechServoType,
)
from humanoid.hardware.actuators.feetech.configurator import FeetechActuatorConfigurator
from humanoid.hardware.actuators.feetech.scripts import (
    jog,
    read_gains,
    scan,
    set_id,
    spin,
    zero,
)
from humanoid.hardware.actuators.feetech.scripts.jog import get_jog_target
from humanoid.hardware.actuators.feetech.scripts.read_gains import read_pid_gains


@pytest.mark.parametrize(
    ("key", "expected"),
    [(keyboard.Key.left, -0.05), (keyboard.Key.right, 0.05)],
)
def test_jogging_from_zero_moves_in_both_directions(key, expected):
    assert get_jog_target(0.0, key, 0.05) == pytest.approx(expected)


def test_read_gains_returns_values_without_unused_write_arguments():
    driver = MagicMock()
    driver.packet_handler.read1ByteTxRx.side_effect = [
        (12, 0, 0),
        (3, 0, 0),
        (7, 0, 0),
    ]
    context = MagicMock()
    context.__enter__.return_value = driver
    with patch(
        "humanoid.hardware.actuators.feetech.configurator.FeetechActuatorDriver.for_actuator_ids",
        return_value=context,
    ):
        gains = read_pid_gains(1)

    assert gains == {"p": 12, "i": 3, "d": 7}


def test_set_id_probes_selected_bus_before_update():
    driver = MagicMock()
    driver.ping.return_value = True
    driver.probe.return_value = False
    driver.set_actuator_id.return_value = True
    context = MagicMock()
    context.__enter__.return_value = driver
    controller_config = FeetechActuatorControllerConfig(port="/dev/right")
    with patch(
        "humanoid.hardware.actuators.feetech.configurator.FeetechActuatorDriver.for_actuator_ids",
        return_value=context,
    ) as driver_factory:
        success = FeetechActuatorConfigurator.set_id(1, 7, controller_config)

    assert success is True
    driver_factory.assert_called_once_with(
        [1],
        controller_config=controller_config,
    )
    driver.set_actuator_id.assert_called_once_with(1, 7)
    driver.ping.assert_called_once_with(1)
    driver.probe.assert_called_once_with(7)


def test_set_id_rejects_target_that_already_responds_on_selected_bus():
    driver = MagicMock()
    driver.ping.return_value = True
    driver.probe.return_value = True
    context = MagicMock()
    context.__enter__.return_value = driver
    with patch(
        "humanoid.hardware.actuators.feetech.configurator.FeetechActuatorDriver.for_actuator_ids",
        return_value=context,
    ):
        success = FeetechActuatorConfigurator.set_id(1, 7)

    assert success is False
    driver.set_actuator_id.assert_not_called()


def test_scan_reuses_one_controller_connection():
    driver = MagicMock()
    driver.probe.side_effect = lambda actuator_id: (
        actuator_id
        in {
            FEETECH_ACTUATOR_ID_MIN,
            FEETECH_ACTUATOR_ID_MAX,
        }
    )
    context = MagicMock()
    context.__enter__.return_value = driver
    controller_config = FeetechActuatorControllerConfig(port="/dev/right")
    with patch(
        "humanoid.hardware.actuators.feetech.configurator.FeetechActuatorDriver.for_actuator_ids",
        return_value=context,
    ) as driver_factory:
        found_ids = FeetechActuatorConfigurator.scan(controller_config)

    assert found_ids == [FEETECH_ACTUATOR_ID_MIN, FEETECH_ACTUATOR_ID_MAX]
    driver_factory.assert_called_once_with([], controller_config=controller_config)
    assert driver.probe.call_count == FEETECH_ACTUATOR_ID_MAX - FEETECH_ACTUATOR_ID_MIN + 1


@pytest.mark.parametrize("acceleration", [0, 254])
def test_jog_cli_accepts_acceleration_boundaries(acceleration):
    with (
        patch.object(
            sys,
            "argv",
            ["jog", "--actuator-id", "1", "--acceleration", str(acceleration)],
        ),
        patch.object(jog, "jog_actuator") as jog_actuator,
    ):
        jog.main()

    assert jog_actuator.call_args.args[2] == acceleration


@pytest.mark.parametrize("acceleration", [-1, 255, 999])
def test_jog_cli_rejects_acceleration_outside_register_range(acceleration):
    with (
        patch.object(
            sys,
            "argv",
            ["jog", "--actuator-id", "1", "--acceleration", str(acceleration)],
        ),
        patch.object(jog, "jog_actuator") as jog_actuator,
        pytest.raises(SystemExit),
    ):
        jog.main()

    jog_actuator.assert_not_called()


@pytest.mark.parametrize(
    ("module", "operation", "arguments"),
    [
        (scan, "FeetechActuatorConfigurator.scan", []),
        (set_id, "FeetechActuatorConfigurator.set_id", ["--new-id", "2"]),
        (zero, "FeetechActuatorConfigurator.set_zero", ["--actuator-id", "1"]),
        (read_gains, "read_pid_gains", ["--actuator-id", "1"]),
        (jog, "jog_actuator", ["--actuator-id", "1"]),
        (spin, "spin_actuator", ["--actuator-id", "1"]),
    ],
)
def test_every_maintenance_cli_passes_selected_controller_config(
    module,
    operation,
    arguments,
):
    common_arguments = [
        "--port",
        "/dev/right",
        "--baud-rate",
        "115200",
        "--servo-type",
        "hls",
    ]
    with (
        patch.object(sys, "argv", ["utility", *arguments, *common_arguments]),
        patch.object(module, operation.split(".")[0]) as owner,
    ):
        method = getattr(owner, operation.split(".")[1]) if "." in operation else owner
        module.main()

    controller_config = method.call_args.args[-1]
    assert controller_config == FeetechActuatorControllerConfig(
        port="/dev/right",
        baud_rate=115_200,
        servo_type=FeetechServoType.HLS,
    )
