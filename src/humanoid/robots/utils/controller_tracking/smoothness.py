"""Offline native-rate motion smoothness analysis."""

import math
from collections import defaultdict
from collections.abc import Collection

import numpy as np
from numpy.typing import NDArray

from humanoid.robots.utils.controller_tracking.models import (
    DEFAULT_MAXIMUM_SPECTRUM_HZ,
    ControllerTrackingSettings,
    Segment,
)
from humanoid.types.controller_tracking import (
    JointSmoothnessStatistics,
    JointTelemetryStream,
    MotionTrace,
    NativeJointSample,
    SmoothnessAnalysis,
)

MINIMUM_TRACE_SAMPLES = 7
MINIMUM_STATISTIC_SAMPLES = 2
SETTLE_DISCARD_SECONDS = 1.0
ACCELERATION_LIMIT_THRESHOLD_FRACTION = 0.9


def analyze_smoothness(
    samples: list[NativeJointSample],
    *,
    motion_segments: Collection[Segment],
    settle_segments: Collection[Segment] = (),
    settings: ControllerTrackingSettings,
    acceleration_limits_rad_s2: dict[str, float] | None = None,
) -> SmoothnessAnalysis | None:
    """Analyze motion without bridging independent test windows or quiet setup moves."""
    state_traces = _build_traces(
        samples,
        stream="state",
        segments=(*motion_segments, *settle_segments),
        settings=settings,
    )
    motion_state_traces = tuple(trace for trace in state_traces if trace.segment in motion_segments)
    if not motion_state_traces:
        return None

    command_traces = _build_traces(
        samples,
        stream="controller",
        segments=motion_segments,
        settings=settings,
    )
    joint_names = motion_state_traces[0].joint_names
    if any(trace.joint_names != joint_names for trace in (*state_traces, *command_traces)):
        raise ValueError("Native joint names must remain consistent within a tracking phase")

    settle_traces = tuple(trace for trace in state_traces if trace.segment in settle_segments)
    limits = acceleration_limits_rad_s2 or {}
    statistics = tuple(
        _joint_statistics(
            joint_index,
            joint_name,
            motion_state_traces,
            settle_traces,
            command_traces,
            limits.get(joint_name),
            settings.shake_cutoff_hz,
        )
        for joint_index, joint_name in enumerate(joint_names)
    )

    tool_residuals = np.vstack(
        [trace.tool_high_frequency_position_m for trace in motion_state_traces]
    )
    tool_norms = np.linalg.norm(tool_residuals, axis=1)
    tool_peak_to_peak = max(
        float(np.linalg.norm(np.ptp(trace.tool_high_frequency_position_m, axis=0)))
        for trace in motion_state_traces
    )
    return SmoothnessAnalysis(
        joint_names=joint_names,
        statistics=statistics,
        state_traces=state_traces,
        command_traces=command_traces,
        shake_cutoff_hz=settings.shake_cutoff_hz,
        tool_high_frequency_rms_m=float(np.sqrt(np.mean(np.square(tool_norms)))),
        tool_high_frequency_peak_to_peak_m=tool_peak_to_peak,
    )


def smoothness_metrics(analysis: SmoothnessAnalysis) -> dict[str, object]:
    """Return a JSON-ready representation for reproducible run comparison."""
    joints = {}
    for name, stats in zip(analysis.joint_names, analysis.statistics, strict=True):
        joints[name] = {
            "high_frequency_rms_rad": stats.high_frequency_rms_rad,
            "high_frequency_peak_to_peak_rad": stats.high_frequency_peak_to_peak_rad,
            "acceleration_p95_rad_s2": stats.acceleration_p95_rad_s2,
            "jerk_rms_rad_s3": stats.jerk_rms_rad_s3,
            "command_acceleration_rms_rad_s2": stats.command_acceleration_rms_rad_s2,
            "command_acceleration_p95_rad_s2": stats.command_acceleration_p95_rad_s2,
            "acceleration_limit_fraction": stats.acceleration_limit_fraction,
            "settle_peak_to_peak_rad": stats.settle_peak_to_peak_rad,
            "dominant_frequency_hz": stats.dominant_frequency_hz,
        }
    return {
        "joints": joints,
        "shake_cutoff_hz": analysis.shake_cutoff_hz,
        "tool_high_frequency_rms_m": analysis.tool_high_frequency_rms_m,
        "tool_high_frequency_peak_to_peak_m": analysis.tool_high_frequency_peak_to_peak_m,
    }


def _build_traces(
    samples: list[NativeJointSample],
    *,
    stream: JointTelemetryStream,
    segments: Collection[Segment],
    settings: ControllerTrackingSettings,
) -> tuple[MotionTrace, ...]:
    grouped: dict[int, list[NativeJointSample]] = defaultdict(list)
    segment_set = frozenset(segments)
    for sample in samples:
        if sample.stream == stream and sample.segment in segment_set:
            grouped[sample.window_index].append(sample)
    traces = []
    for window_samples in grouped.values():
        trace = _build_trace(window_samples, settings)
        if trace is not None:
            traces.append(trace)
    return tuple(sorted(traces, key=lambda trace: trace.times_s[0]))


def _build_trace(
    samples: list[NativeJointSample], settings: ControllerTrackingSettings
) -> MotionTrace | None:
    samples = sorted(samples, key=lambda sample: sample.source_timestamp_s)
    unique_samples = []
    for sample in samples:
        if unique_samples and sample.source_timestamp_s <= unique_samples[-1].source_timestamp_s:
            if sample.source_timestamp_s == unique_samples[-1].source_timestamp_s:
                unique_samples[-1] = sample
                continue
            raise ValueError("Native telemetry timestamps must be non-decreasing")
        unique_samples.append(sample)
    if len(unique_samples) < MINIMUM_TRACE_SAMPLES:
        return None

    names = unique_samples[0].joint_names
    expected_shape = (len(names),)
    if any(
        sample.joint_names != names or sample.joint_positions_rad.shape != expected_shape
        for sample in unique_samples
    ):
        raise ValueError("Native telemetry values must match their joint names")

    source_times = np.array([sample.source_timestamp_s for sample in unique_samples])
    periods = np.diff(source_times)
    median_period_s = float(np.median(periods))
    if not math.isfinite(median_period_s) or median_period_s <= 0.0:
        return None
    uniform_times = np.arange(
        source_times[0],
        source_times[-1] + median_period_s * 0.25,
        median_period_s,
    )
    if len(uniform_times) < MINIMUM_TRACE_SAMPLES:
        return None

    positions = np.unwrap(
        np.vstack([sample.joint_positions_rad for sample in unique_samples]),
        axis=0,
    )
    uniform_positions = _interpolate_columns(source_times, positions, uniform_times)
    tool_positions = _interpolate_columns(
        source_times,
        np.vstack([sample.tool_position_m for sample in unique_samples]),
        uniform_times,
    )
    sample_rate_hz = 1.0 / median_period_s
    derivative_window = _odd_window_length(
        settings.derivative_smoothing_s * sample_rate_hz,
        len(uniform_times),
    )
    trend_window = _odd_window_length(
        sample_rate_hz / settings.shake_cutoff_hz,
        len(uniform_times),
    )
    filtered_positions = _local_polynomial_smooth(uniform_positions, derivative_window)
    position_trend = _local_polynomial_smooth(uniform_positions, trend_window)
    high_frequency_positions = uniform_positions - position_trend
    tool_trend = _local_polynomial_smooth(tool_positions, trend_window)

    # Position is the actual setpoint sent to a position-controlled actuator. Build
    # derivatives from that trajectory for both controller and measured streams.
    # The controller's published velocity is retained as a raw trace, but it is a
    # solver result/speed hint and can diverge from dq/dt when control and publication
    # periods differ. Differentiating it produced misleading, non-repeatable command
    # acceleration and acceleration-limit occupancy.
    velocities = np.gradient(filtered_positions, median_period_s, axis=0)
    supplied_velocities = [sample.joint_velocities_rad_s for sample in unique_samples]
    if all(velocity is not None for velocity in supplied_velocities):
        source_velocities = np.vstack(
            [velocity for velocity in supplied_velocities if velocity is not None]
        )
        raw_velocities = _interpolate_columns(source_times, source_velocities, uniform_times)
    else:
        source_velocities = None
        raw_velocities = velocities
    accelerations = np.gradient(velocities, median_period_s, axis=0)
    jerks = np.gradient(accelerations, median_period_s, axis=0)
    frequencies, psd = _position_psd(high_frequency_positions, sample_rate_hz)
    receipt_offset_s = unique_samples[0].received_timestamp_s - unique_samples[0].source_timestamp_s
    return MotionTrace(
        segment=unique_samples[0].segment,
        setting=unique_samples[0].setting,
        window_index=unique_samples[0].window_index,
        stream=unique_samples[0].stream,
        times_s=uniform_times + receipt_offset_s,
        source_times_s=source_times,
        joint_names=names,
        positions_rad=uniform_positions,
        filtered_positions_rad=filtered_positions,
        high_frequency_positions_rad=high_frequency_positions,
        raw_velocities_rad_s=raw_velocities,
        source_velocities_rad_s=source_velocities,
        velocities_rad_s=velocities,
        accelerations_rad_s2=accelerations,
        jerks_rad_s3=jerks,
        frequencies_hz=frequencies,
        position_psd_rad2_hz=psd,
        tool_high_frequency_position_m=tool_positions - tool_trend,
    )


def _joint_statistics(  # noqa: PLR0913 - combines distinct measured/commanded signals
    joint_index: int,
    joint_name: str,
    motion_traces: tuple[MotionTrace, ...],
    settle_traces: tuple[MotionTrace, ...],
    command_traces: tuple[MotionTrace, ...],
    acceleration_limit_rad_s2: float | None,
    shake_cutoff_hz: float,
) -> JointSmoothnessStatistics:
    residuals = np.concatenate(
        [trace.high_frequency_positions_rad[:, joint_index] for trace in motion_traces]
    )
    accelerations = np.concatenate(
        [trace.accelerations_rad_s2[:, joint_index] for trace in motion_traces]
    )
    jerks = np.concatenate([trace.jerks_rad_s3[:, joint_index] for trace in motion_traces])
    high_frequency_peak_to_peak = max(
        float(np.ptp(trace.high_frequency_positions_rad[:, joint_index])) for trace in motion_traces
    )
    settle_peak_to_peak = _settle_peak_to_peak(joint_index, settle_traces)
    dominant_frequency = _dominant_frequency(joint_index, motion_traces, shake_cutoff_hz)

    command_acceleration_rms = None
    command_acceleration_p95 = None
    limit_fraction = None
    if command_traces:
        command_accelerations = np.concatenate(
            [trace.accelerations_rad_s2[:, joint_index] for trace in command_traces]
        )
        command_acceleration_rms = float(np.sqrt(np.mean(np.square(command_accelerations))))
        command_acceleration_p95 = float(np.percentile(np.abs(command_accelerations), 95))
        if acceleration_limit_rad_s2 is not None:
            limit_fraction = float(
                np.mean(
                    np.abs(command_accelerations)
                    >= ACCELERATION_LIMIT_THRESHOLD_FRACTION * acceleration_limit_rad_s2
                )
            )
    return JointSmoothnessStatistics(
        high_frequency_rms_rad=float(np.sqrt(np.mean(np.square(residuals)))),
        high_frequency_peak_to_peak_rad=high_frequency_peak_to_peak,
        acceleration_p95_rad_s2=float(np.percentile(np.abs(accelerations), 95)),
        jerk_rms_rad_s3=float(np.sqrt(np.mean(np.square(jerks)))),
        command_acceleration_rms_rad_s2=command_acceleration_rms,
        command_acceleration_p95_rad_s2=command_acceleration_p95,
        acceleration_limit_fraction=limit_fraction,
        settle_peak_to_peak_rad=settle_peak_to_peak,
        dominant_frequency_hz=dominant_frequency,
    )


def _settle_peak_to_peak(joint_index: int, settle_traces: tuple[MotionTrace, ...]) -> float | None:
    values = []
    for trace in settle_traces:
        elapsed = trace.times_s - trace.times_s[0]
        retained = trace.positions_rad[elapsed >= SETTLE_DISCARD_SECONDS, joint_index]
        if retained.size < MINIMUM_STATISTIC_SAMPLES:
            retained = trace.positions_rad[len(trace.positions_rad) // 2 :, joint_index]
        if retained.size >= MINIMUM_STATISTIC_SAMPLES:
            values.append(float(np.ptp(retained)))
    return max(values) if values else None


def _dominant_frequency(
    joint_index: int,
    traces: tuple[MotionTrace, ...],
    shake_cutoff_hz: float,
) -> float | None:
    strongest_power = -math.inf
    dominant = None
    for trace in traces:
        frequency_mask = (trace.frequencies_hz >= shake_cutoff_hz) & (
            trace.frequencies_hz <= DEFAULT_MAXIMUM_SPECTRUM_HZ
        )
        if not np.any(frequency_mask):
            continue
        frequencies = trace.frequencies_hz[frequency_mask]
        powers = trace.position_psd_rad2_hz[frequency_mask, joint_index]
        peak_index = int(np.argmax(powers))
        if powers[peak_index] > strongest_power:
            strongest_power = float(powers[peak_index])
            dominant = float(frequencies[peak_index])
    return dominant


def _odd_window_length(requested: float, sample_count: int) -> int:
    length = max(3, round(requested))
    if length % 2 == 0:
        length += 1
    maximum = sample_count if sample_count % 2 == 1 else sample_count - 1
    return min(length, maximum)


def _local_polynomial_smooth(
    values: NDArray[np.float64], window_length: int
) -> NDArray[np.float64]:
    half_window = window_length // 2
    coordinates = np.arange(-half_window, half_window + 1, dtype=float)
    polynomial_order = min(3, window_length - 1)
    design = np.vander(coordinates, polynomial_order + 1, increasing=True)
    coefficients = np.linalg.pinv(design)[0]
    return _apply_centered_filter(values, coefficients)


def _apply_centered_filter(
    values: NDArray[np.float64], coefficients: NDArray[np.float64]
) -> NDArray[np.float64]:
    half_window = len(coefficients) // 2
    left_steps = np.arange(half_window, 0, -1, dtype=float)[:, None]
    right_steps = np.arange(1, half_window + 1, dtype=float)[:, None]
    left_slope = values[1] - values[0]
    right_slope = values[-1] - values[-2]
    padded = np.vstack(
        (
            values[0] - left_steps * left_slope,
            values,
            values[-1] + right_steps * right_slope,
        )
    )
    return np.column_stack(
        [
            np.convolve(padded[:, index], coefficients, mode="valid")
            for index in range(values.shape[1])
        ]
    )


def _interpolate_columns(
    source_times: NDArray[np.float64],
    values: NDArray[np.float64],
    target_times: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.column_stack(
        [
            np.interp(target_times, source_times, values[:, index])
            for index in range(values.shape[1])
        ]
    )


def _position_psd(
    values: NDArray[np.float64], sample_rate_hz: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    window = np.hanning(len(values))
    centered = values - np.mean(values, axis=0)
    spectrum = np.fft.rfft(centered * window[:, None], axis=0)
    scale = sample_rate_hz * float(np.sum(np.square(window)))
    psd = np.square(np.abs(spectrum)) / scale
    if len(psd) > MINIMUM_STATISTIC_SAMPLES:
        psd[1:-1] *= 2.0
    frequencies = np.asarray(
        np.fft.rfftfreq(len(values), d=1.0 / sample_rate_hz),
        dtype=np.float64,
    )
    return frequencies, np.asarray(psd, dtype=np.float64)
