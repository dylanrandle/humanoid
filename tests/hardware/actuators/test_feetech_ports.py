from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vassar_feetech_servo_sdk.exceptions import PortNotFoundError

from humanoid.hardware.actuators.feetech.ports import detect_servo_port


def test_adapter_is_selected_even_when_debug_device_is_first():
    ports = [
        SimpleNamespace(device="/dev/debug", vid=0x303A, pid=0x1001),
        SimpleNamespace(device="/dev/servo", vid=0x1A86, pid=0x55D3),
    ]
    with patch("humanoid.hardware.actuators.feetech.ports.list_ports.comports", return_value=ports):
        assert detect_servo_port() == "/dev/servo"


@pytest.mark.parametrize("candidates", [["/dev/legacy"], ["/dev/one", "/dev/two"]])
def test_unknown_adapters_require_an_unambiguous_choice(candidates):
    with (
        patch("humanoid.hardware.actuators.feetech.ports.list_ports.comports", return_value=[]),
        patch("humanoid.hardware.actuators.feetech.ports.find_servo_port", return_value=candidates),
    ):
        if len(candidates) == 1:
            assert detect_servo_port() == candidates[0]
        else:
            with pytest.raises(PortNotFoundError, match="Specify --port"):
                detect_servo_port()


def test_two_matching_adapters_require_explicit_port():
    ports = [
        SimpleNamespace(device=device, vid=0x1A86, pid=0x55D3)
        for device in ["/dev/one", "/dev/two"]
    ]
    with (
        patch("humanoid.hardware.actuators.feetech.ports.list_ports.comports", return_value=ports),
        pytest.raises(PortNotFoundError, match="Specify --port"),
    ):
        detect_servo_port()
