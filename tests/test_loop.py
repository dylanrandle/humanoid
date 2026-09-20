import pytest

from humanoid.utils.loop import _next_deadline, _sleep_until


def test_next_deadline_advances_one_period_when_on_schedule():
    assert _next_deadline(1.0, 0.1, 1.05) == pytest.approx(1.1)


def test_next_deadline_skips_missed_periods_without_catch_up():
    assert _next_deadline(1.0, 0.1, 1.36) == pytest.approx(1.4)


def test_next_deadline_is_strictly_after_current_time():
    assert _next_deadline(1.0, 0.1, 1.1) == pytest.approx(1.2)


def test_sleep_until_rechecks_time_after_coalesced_wakeups(monkeypatch):
    now = 1.0
    deadline = now + 1 / 30
    sleeps = []

    def sleep(duration):
        nonlocal now
        sleeps.append(duration)
        # Model proportional timer coalescing, rather than ideal sleeps.
        now += duration * 1.3

    monkeypatch.setattr("humanoid.utils.loop.time.perf_counter", lambda: now)
    monkeypatch.setattr("humanoid.utils.loop.time.sleep", sleep)
    _sleep_until(deadline)

    tolerance = 0.001
    assert deadline <= now <= deadline + tolerance
    assert len(sleeps) > 1
    assert all(duration > 0 for duration in sleeps)


def test_sleep_until_does_not_wait_on_a_missed_deadline(monkeypatch):
    monkeypatch.setattr("humanoid.utils.loop.time.perf_counter", lambda: 2.0)

    def unexpected_sleep(_):
        pytest.fail("A past deadline must not cause another wait")

    monkeypatch.setattr("humanoid.utils.loop.time.sleep", unexpected_sleep)
    _sleep_until(1.0)
