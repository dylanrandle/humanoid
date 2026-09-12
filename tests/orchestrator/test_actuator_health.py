from unittest.mock import MagicMock

from humanoid.constants import Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.orchestrator.monitor.actuator import ActuatorHealthMonitor
from humanoid.types.actuator import ActuatorHealth, ActuatorHealthReport

EXPECTED_TEMPERATURE_CELSIUS = 31.0
EXPECTED_QUEUED_REPORT_AGE_SECONDS = 9.0


def _report(
    *,
    timestamp: float = 1.0,
    healthy: bool = True,
    error: str | None = None,
) -> ActuatorHealthReport:
    return ActuatorHealthReport(
        timestamp=timestamp,
        actuators=(
            ActuatorHealth(
                joint_name="arm_1",
                controller="main",
                actuator_id=1,
                healthy=healthy,
                temperature_celsius=31.0,
                issue=None if healthy else "No feedback returned.",
            ),
        ),
        error=error,
    )


def test_reports_fresh_healthy_actuator_telemetry():
    now = 10.0
    subscriber = MagicMock(spec=Subscriber)
    subscriber.receive.side_effect = [_report(timestamp=now), None]
    monitor = ActuatorHealthMonitor(subscriber=subscriber, clock=lambda: now)

    status = monitor.snapshot()

    assert status.connected is True
    assert status.healthy is True
    assert status.age_seconds == 0.0
    assert status.actuators[0].temperature_celsius == EXPECTED_TEMPERATURE_CELSIUS


def test_retains_last_error_after_driver_telemetry_becomes_stale():
    now = 10.0
    subscriber = MagicMock(spec=Subscriber)
    subscriber.receive.side_effect = [
        _report(
            timestamp=now,
            healthy=False,
            error="Feetech feedback sync read failed.",
        ),
        None,
        None,
    ]
    monitor = ActuatorHealthMonitor(
        subscriber=subscriber,
        max_age_seconds=2.0,
        clock=lambda: now,
    )

    fresh = monitor.snapshot()
    now = 13.0
    stale = monitor.snapshot()

    assert fresh.connected is True
    assert fresh.healthy is False
    assert stale.connected is False
    assert stale.healthy is False
    assert stale.error == "Feetech feedback sync read failed."
    assert stale.actuators == fresh.actuators


def test_queued_report_age_uses_publish_time_instead_of_poll_time():
    subscriber = MagicMock(spec=Subscriber)
    subscriber.receive.side_effect = [_report(timestamp=1.0), None]
    monitor = ActuatorHealthMonitor(
        subscriber=subscriber,
        max_age_seconds=2.0,
        clock=lambda: 10.0,
    )

    status = monitor.snapshot()

    assert status.connected is False
    assert status.healthy is False
    assert status.age_seconds == EXPECTED_QUEUED_REPORT_AGE_SECONDS


def test_reset_clears_retained_actuator_failure():
    subscriber = MagicMock(spec=Subscriber)
    subscriber.receive.side_effect = [_report(error="failure"), None, None, None]
    monitor = ActuatorHealthMonitor(subscriber=subscriber, clock=lambda: 1.0)
    assert monitor.snapshot().error == "failure"

    monitor.reset()

    status = monitor.snapshot()
    assert status.connected is False
    assert status.actuators == ()
    assert status.error is None
    subscriber.receive.assert_any_call(Topic.ACTUATOR_HEALTH)
