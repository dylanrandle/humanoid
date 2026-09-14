"""Controller tracking diagnostic package."""

from humanoid.robots.utils.controller_tracking.cli import main
from humanoid.robots.utils.controller_tracking.models import (
    DEFAULT_COMMAND_RATE_HZ,
    DEFAULT_DERIVATIVE_SMOOTHING_SECONDS,
    DEFAULT_GRIPPER_LIMIT_MARGIN_FRACTION,
    DEFAULT_GRIPPER_PERIOD_SECONDS,
    DEFAULT_JOINT_CYCLES,
    DEFAULT_SHAKE_CUTOFF_HZ,
    ControllerTrackingSettings,
    Segment,
    TrackingRun,
    TrackingSample,
)
from humanoid.robots.utils.controller_tracking.report import (
    TrackingPlotPaths,
    write_controller_timing_csv,
    write_native_joint_telemetry_csv,
    write_run_metrics_json,
    write_tracking_csv,
    write_tracking_plots,
)
from humanoid.robots.utils.controller_tracking.runtime import run_controller_tracking
from humanoid.robots.utils.controller_tracking.sampling import tracking_statistics
from humanoid.robots.utils.controller_tracking.smoothness import analyze_smoothness
from humanoid.robots.utils.controller_tracking.timing import controller_publication_statistics
from humanoid.robots.utils.controller_tracking.trajectory import (
    figure_eight_offset,
    figure_eight_velocity,
    gripper_sinusoid,
    interpolated_cartesian_comparison_pose,
    interpolated_cartesian_comparison_velocity,
    joint_space_targets,
    resolve_tracking_comparison,
)
from humanoid.types.controller_tracking import (
    ControllerCommandTiming,
    ControllerPublicationStatistics,
    NativeJointSample,
)

__all__ = [
    "DEFAULT_COMMAND_RATE_HZ",
    "DEFAULT_DERIVATIVE_SMOOTHING_SECONDS",
    "DEFAULT_GRIPPER_LIMIT_MARGIN_FRACTION",
    "DEFAULT_GRIPPER_PERIOD_SECONDS",
    "DEFAULT_JOINT_CYCLES",
    "DEFAULT_SHAKE_CUTOFF_HZ",
    "ControllerCommandTiming",
    "ControllerPublicationStatistics",
    "ControllerTrackingSettings",
    "NativeJointSample",
    "Segment",
    "TrackingPlotPaths",
    "TrackingRun",
    "TrackingSample",
    "analyze_smoothness",
    "controller_publication_statistics",
    "figure_eight_offset",
    "figure_eight_velocity",
    "gripper_sinusoid",
    "interpolated_cartesian_comparison_pose",
    "interpolated_cartesian_comparison_velocity",
    "joint_space_targets",
    "main",
    "resolve_tracking_comparison",
    "run_controller_tracking",
    "tracking_statistics",
    "write_controller_timing_csv",
    "write_native_joint_telemetry_csv",
    "write_run_metrics_json",
    "write_tracking_csv",
    "write_tracking_plots",
]
