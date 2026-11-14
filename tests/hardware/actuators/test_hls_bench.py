# ruff: noqa: PLR2004 - literal protocol values are intentional test expectations
from unittest.mock import MagicMock

import pytest

from humanoid.hardware.actuators.feetech.scripts import hls_test
from humanoid.hardware.actuators.feetech.scripts.hls_test import (
    GOAL_SPEED,
    GOAL_TORQUE,
    TORQUE_ENABLE,
    checked,
    cleanup,
    set_mode,
    signed_word,
)


def test_signed_torque_encoding():
    assert signed_word(-50) == 0x8032
    assert signed_word(50) == 50
    assert signed_word(0) == 0


@pytest.mark.parametrize("result", [(0, 1), (-1, 0), (42, 0, 1)])
def test_errors_are_not_silently_accepted(result):
    with pytest.raises(RuntimeError):
        checked(result, "test")


def test_cleanup_attempts_zero_commands_even_if_disable_fails():
    handler = MagicMock()
    handler.write1ByteTxRx.return_value = (-1, 0)
    handler.write2ByteTxRx.return_value = (0, 0)
    handler.read1ByteTxRx.return_value = (1, 0, 0)
    with pytest.raises(RuntimeError, match="Cleanup incomplete"):
        cleanup(handler, 7, 0, 500)
    handler.write1ByteTxRx.assert_called_once_with(7, TORQUE_ENABLE, 0)
    handler.write2ByteTxRx.assert_any_call(7, GOAL_TORQUE, 0)
    handler.write2ByteTxRx.assert_any_call(7, GOAL_SPEED, 0)


def test_mode_change_relocks_after_failed_write():
    handler = MagicMock()
    handler.read1ByteTxRx.return_value = (0, 0, 0)
    handler.write1ByteTxRx.side_effect = [(0, 0), (-1, 0), (0, 0)]
    with pytest.raises(RuntimeError, match="write mode"):
        set_mode(handler, 7, 2)
    assert handler.write1ByteTxRx.call_args.args == (7, 55, 1)


def test_unchanged_mode_does_not_write_eeprom():
    handler = MagicMock()
    handler.read1ByteTxRx.return_value = (2, 0, 0)
    set_mode(handler, 7, 2)
    handler.write1ByteTxRx.assert_not_called()


@pytest.mark.parametrize("mode", ["position", "velocity", "torque"])
@pytest.mark.parametrize("interrupted", [False, True])
def test_bench_restores_settings_and_closes_port(monkeypatch, mode, interrupted):
    handler = MagicMock()
    port = MagicMock()
    registers = {33: 0, 40: 0, 48: 500, 63: 25, 69: 0}

    def write(servo_id, address, value):
        registers[address] = value
        # Reproduce the HLS cleanup failure observed on the physical motor.
        if address in (GOAL_TORQUE, GOAL_SPEED):
            registers[TORQUE_ENABLE] = 1
        return (0, 0)

    handler.write1ByteTxRx.side_effect = write
    handler.write2ByteTxRx.side_effect = write
    handler.read1ByteTxRx.side_effect = lambda servo_id, address: (registers[address], 0, 0)
    handler.read2ByteTxRx.side_effect = lambda servo_id, address: (registers[address], 0, 0)
    handler.ping.return_value = (123, 0, 0)
    handler.ReadPos.return_value = (2048, 0, 0)
    handler.ReadSpeed.return_value = (0, 0, 0)
    handler.WritePosEx.return_value = (0, 0)
    handler.WriteSpec.return_value = (0, 0)
    handler.scs_tohost.return_value = 0
    if interrupted:
        handler.ReadSpeed.side_effect = KeyboardInterrupt
    monkeypatch.setattr(hls_test.scs, "PortHandler", lambda _: port)
    monkeypatch.setattr(hls_test.scs, "hls", lambda _: handler)
    monkeypatch.setattr(
        "sys.argv",
        ["hls_test", mode, "--port", "/dev/fake", "--actuator-id", "7", "--duration", "0.01"],
    )
    hls_test.main()
    assert registers[40] == 0
    assert registers[33] == 0
    assert registers[48] == 500
    assert registers[44] == registers[46] == 0
    port.closePort.assert_called_once()
