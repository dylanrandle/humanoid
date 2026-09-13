"""Tests for controller-tracking comparison configuration types."""

from dataclasses import replace

import pytest

from humanoid.robots.utils.controller_tracking.endpoints import (
    TRISKEL_CONTROLLER_TRACKING_COMPARISON,
)
from humanoid.types.controller_tracking import ControllerTrackingComparisonConfig


def test_triskel_comparison_endpoints_share_the_tool_command_frame():
    comparison = TRISKEL_CONTROLLER_TRACKING_COMPARISON

    assert comparison.start.task_frame == "root_joint"
    assert comparison.end.task_frame == comparison.start.task_frame
    assert comparison.start.joint_positions_rad.keys() == comparison.end.joint_positions_rad.keys()


def test_endpoint_rejects_non_normalized_quaternion():
    with pytest.raises(ValueError, match="normalized"):
        replace(
            TRISKEL_CONTROLLER_TRACKING_COMPARISON.start,
            task_quaternion_wxyz=(1.0, 1.0, 0.0, 0.0),
        )


def test_comparison_rejects_mismatched_task_frames():
    with pytest.raises(ValueError, match="same task frame"):
        ControllerTrackingComparisonConfig(
            start=TRISKEL_CONTROLLER_TRACKING_COMPARISON.start,
            end=replace(
                TRISKEL_CONTROLLER_TRACKING_COMPARISON.end,
                task_frame="other",
            ),
        )
