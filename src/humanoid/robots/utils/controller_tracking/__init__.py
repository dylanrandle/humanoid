"""Controller tracking diagnostic package."""

from humanoid.robots.utils.controller_tracking.cli import (
    controller_tracking_settings_from_arguments,
    main,
    run_controller_tracking_diagnostic,
)
from humanoid.robots.utils.controller_tracking.models import (
    DEFAULT_COMMAND_RATE_HZ,
    DEFAULT_DERIVATIVE_SMOOTHING_SECONDS,
    DEFAULT_GRIPPER_LIMIT_MARGIN_FRACTION,
    DEFAULT_GRIPPER_PERIOD_SECONDS,
    DEFAULT_ORIENTATION_BIAS_RADIANS,
    DEFAULT_SHAKE_CUTOFF_HZ,
    FIGURE_EIGHT_SETTINGS,
    ControllerTrackingSettings,
    FigureEightSetting,
    Segment,
    TrackingRun,
    TrackingSample,
)
from humanoid.robots.utils.controller_tracking.report import (
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
    figure_eight_pose,
    figure_eight_velocity,
    gripper_sinusoid,
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
    "DEFAULT_ORIENTATION_BIAS_RADIANS",
    "DEFAULT_SHAKE_CUTOFF_HZ",
    "FIGURE_EIGHT_SETTINGS",
    "ControllerCommandTiming",
    "ControllerPublicationStatistics",
    "ControllerTrackingSettings",
    "FigureEightSetting",
    "NativeJointSample",
    "Segment",
    "TrackingRun",
    "TrackingSample",
    "analyze_smoothness",
    "controller_publication_statistics",
    "controller_tracking_settings_from_arguments",
    "figure_eight_offset",
    "figure_eight_pose",
    "figure_eight_velocity",
    "gripper_sinusoid",
    "main",
    "run_controller_tracking",
    "run_controller_tracking_diagnostic",
    "tracking_statistics",
    "write_controller_timing_csv",
    "write_native_joint_telemetry_csv",
    "write_run_metrics_json",
    "write_tracking_csv",
    "write_tracking_plots",
]
