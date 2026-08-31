"""Unit tests for dashboard topic-rate sampling without a ROS installation."""

from pathlib import Path
from typing import Any, cast

import pytest

RATE_MODULE = (
    Path(__file__).resolve().parents[2]
    / "ros_ws"
    / "src"
    / "triskel_operator"
    / "triskel_operator"
    / "rates.py"
)


def _topic_rate_type() -> type[Any]:
    namespace: dict[str, object] = {}
    exec(compile(RATE_MODULE.read_text(), str(RATE_MODULE), "exec"), namespace)
    return cast(type[Any], namespace["TopicRate"])


def test_topic_rate_reports_frequency_and_freshness() -> None:
    rate = _topic_rate_type()(window_seconds=2.0)
    for timestamp in (10.0, 10.1, 10.2, 10.3):
        rate.observe(timestamp)

    frequency, age = rate.sample(10.35)

    assert frequency == pytest.approx(10.0)
    assert age == pytest.approx(0.05)


def test_topic_rate_expires_samples_outside_window() -> None:
    rate = _topic_rate_type()(window_seconds=1.0)
    rate.observe(5.0)

    assert rate.sample(6.01) == (0.0, None)


def test_topic_rate_rejects_non_positive_window() -> None:
    topic_rate = _topic_rate_type()

    with pytest.raises(ValueError, match="positive"):
        topic_rate(window_seconds=0.0)
