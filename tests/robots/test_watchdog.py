"""Tests for shared velocity command-loss watchdog behavior."""

from unittest.mock import MagicMock

import pytest

from humanoid.robots.watchdog import VelocityCommandWatchdog


def test_stops_an_active_stale_command_only_once():
    now = 0.0
    stop = MagicMock()
    watchdog = VelocityCommandWatchdog(stop, timeout_seconds=0.1, clock=lambda: now)

    watchdog.observe_command(velocity_active=True)
    now = 0.05
    assert watchdog.stop_if_stale() is False
    now = 0.11
    assert watchdog.stop_if_stale() is True
    now = 0.2
    assert watchdog.stop_if_stale() is False

    stop.assert_called_once_with()


def test_inactive_commands_do_not_arm_the_watchdog_and_shutdown_always_stops():
    now = 0.0
    stop = MagicMock()
    watchdog = VelocityCommandWatchdog(stop, timeout_seconds=0.1, clock=lambda: now)

    watchdog.observe_command(velocity_active=False)
    now = 1.0
    assert watchdog.stop_if_stale() is False
    watchdog.stop()

    stop.assert_called_once_with()


@pytest.mark.parametrize("timeout_seconds", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_non_positive_or_non_finite_timeout(timeout_seconds):
    with pytest.raises(ValueError, match="timeout must be positive and finite"):
        VelocityCommandWatchdog(lambda: None, timeout_seconds=timeout_seconds)
