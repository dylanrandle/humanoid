import pytest

from humanoid.utils.loop import _next_deadline


def test_next_deadline_advances_one_period_when_on_schedule():
    assert _next_deadline(1.0, 0.1, 1.05) == pytest.approx(1.1)


def test_next_deadline_skips_missed_periods_without_catch_up():
    assert _next_deadline(1.0, 0.1, 1.36) == pytest.approx(1.4)


def test_next_deadline_is_strictly_after_current_time():
    assert _next_deadline(1.0, 0.1, 1.1) == pytest.approx(1.2)
