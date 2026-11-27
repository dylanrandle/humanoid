"""Passive LCM rate accounting independent of UI polling and payload decoding."""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from humanoid.telemetry.lcm import LCMRateMonitor
from humanoid.types.telemetry import LCMChannelStatus


@pytest.fixture
def monitor() -> Iterator[tuple[LCMRateMonitor, MagicMock, MagicMock]]:
    clock = MagicMock(return_value=0.0)
    with patch("humanoid.telemetry.lcm.lcm.LCM") as factory:
        bus = factory.return_value
        with patch("humanoid.telemetry.lcm.threading.Thread"):
            instance = LCMRateMonitor(url="memq://", clock=clock)
        factory.assert_called_once_with("memq://")
        yield instance, clock, bus
        instance.close()
        bus.unsubscribe.assert_called_once_with(bus.subscribe.return_value)


def test_rates_use_rolling_window_and_do_not_depend_on_polling(monitor):
    instance, clock, bus = monitor
    pattern, receive = bus.subscribe.call_args.args
    assert pattern == ".*"
    assert instance.snapshot() == []
    for index in range(1, 201):
        clock.return_value = index / 100
        receive("ROBOT/STATE", b"not decoded")
    expected = [LCMChannelStatus("ROBOT/STATE", 100.0, 0.0, True)]
    assert instance.snapshot() == expected
    assert instance.snapshot() == expected
    clock.return_value = 3.0
    assert instance.snapshot() == [LCMChannelStatus("ROBOT/STATE", 50.0, 1.0, True)]
    clock.return_value = 4.0
    assert instance.snapshot() == [LCMChannelStatus("ROBOT/STATE", 0.0, 2.0, False)]
    receive("ROBOT/STATE", b"")
    assert instance.snapshot() == [LCMChannelStatus("ROBOT/STATE", 0.5, 0.0, True)]


def test_custom_channels_startup_and_reset(monitor):
    instance, clock, bus = monitor
    receive = bus.subscribe.call_args.args[1]
    receive("CUSTOM/EVENT", b"")
    assert instance.snapshot()[0].publish_rate_hz == 0.0
    clock.return_value = 0.5
    receive("CAMERA/custom/RGB", b"")
    assert instance.snapshot() == [
        LCMChannelStatus("CAMERA/custom/RGB", 2.0, 0.0, True),
        LCMChannelStatus("CUSTOM/EVENT", 2.0, 0.5, True),
    ]
    instance.reset()
    assert instance.snapshot() == []
    clock.return_value = 1.0
    receive("CUSTOM/EVENT", b"")
    assert instance.snapshot() == [LCMChannelStatus("CUSTOM/EVENT", 2.0, 0.0, True)]


@pytest.mark.parametrize("window", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_window_is_rejected(window):
    with pytest.raises(ValueError, match="window_seconds"):
        LCMRateMonitor(window_seconds=window)
