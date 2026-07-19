"""Tests for shared node loop-rate reporting."""

from unittest.mock import MagicMock, call

import pytest

from humanoid.constants import Topic
from humanoid.middleware.publisher import Publisher
from humanoid.nodes.rate import NodeRateReporter
from humanoid.types.node import NodeRateSample


def test_reports_target_then_measured_rate_over_a_window():
    publisher = MagicMock(spec=Publisher)
    clock = MagicMock(side_effect=[10.0, 10.5, 11.0])
    wall_clock = MagicMock(side_effect=[100.0, 101.0])
    reporter = NodeRateReporter(
        "ExampleNode",
        2.0,
        publisher=publisher,
        clock=clock,
        wall_clock=wall_clock,
        pid=123,
    )

    reporter.start()
    reporter.observe_iteration()
    reporter.observe_iteration()
    reporter.observe_iteration()

    assert publisher.publish.call_args_list == [
        call(
            NodeRateSample(
                timestamp=100.0,
                node_name="ExampleNode",
                pid=123,
                target_rate_hz=2.0,
                measured_rate_hz=0.0,
            ),
            topic=Topic.NODE_RATE,
        ),
        call(
            NodeRateSample(
                timestamp=101.0,
                node_name="ExampleNode",
                pid=123,
                target_rate_hz=2.0,
                measured_rate_hz=2.0,
            ),
            topic=Topic.NODE_RATE,
        ),
    ]


@pytest.mark.parametrize(
    ("target_rate_hz", "report_interval_seconds", "expected"),
    [
        (0.0, 1.0, "target_rate_hz"),
        (-1.0, 1.0, "target_rate_hz"),
        (float("nan"), 1.0, "target_rate_hz"),
        (float("inf"), 1.0, "target_rate_hz"),
        (1.0, 0.0, "report_interval_seconds"),
        (1.0, -1.0, "report_interval_seconds"),
        (1.0, float("nan"), "report_interval_seconds"),
        (1.0, float("inf"), "report_interval_seconds"),
    ],
)
def test_rejects_non_positive_or_non_finite_configuration(
    target_rate_hz,
    report_interval_seconds,
    expected,
):
    with pytest.raises(ValueError, match=expected):
        NodeRateReporter(
            "ExampleNode",
            target_rate_hz,
            report_interval_seconds=report_interval_seconds,
            publisher=MagicMock(spec=Publisher),
        )
