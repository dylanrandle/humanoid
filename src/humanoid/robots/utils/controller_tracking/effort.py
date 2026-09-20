"""Assemble native actuator effort feedback for diagnostic reports."""

from collections import defaultdict
from itertools import pairwise

import numpy as np

from humanoid.robots.base import Robot
from humanoid.types.actuator import ActuatorEffortLimits, ActuatorEffortSource, ActuatorEffortUnit
from humanoid.types.controller_tracking import ActuatorEffortTrace, NativeJointSample
from humanoid.types.robot import RobotConfig


def actuator_effort_limits(robot_config: RobotConfig) -> dict[str, ActuatorEffortLimits]:
    """Return explicitly configured output ratings, without inferring continuous limits."""
    if robot_config.hardware is None or robot_config.hardware.actuators is None:
        return {}
    return {
        name: actuator.effort_limits
        for name, actuator in robot_config.hardware.actuators.joints.items()
        if actuator.effort_limits is not None
    }


def build_effort_traces(
    samples: list[NativeJointSample], robot_config: RobotConfig
) -> dict[str, ActuatorEffortTrace]:
    """Preserve every state sample, including missing effort and settle intervals."""
    groups: dict[str, list[NativeJointSample]] = defaultdict(list)
    for sample in samples:
        if sample.stream == "state":
            groups[sample.setting].append(sample)
    if not groups:
        return {}
    robot = Robot(robot_config)
    units: dict[str, ActuatorEffortUnit] = {
        name: (
            "N"
            if robot.model.joints[robot.model.getJointId(name)]
            .shortname()
            .startswith("JointModelP")
            else "N·m"
        )
        for name in robot_config.actuator_control_modes
    }
    traces = {}
    for setting, group in groups.items():
        ordered = sorted(group, key=lambda sample: sample.received_timestamp_s)
        names = ordered[0].joint_names
        efforts = []
        for sample in ordered:
            if sample.joint_names != names:
                raise ValueError("Effort joint names must remain consistent within a setting.")
            values = sample.joint_efforts
            if values is not None and values.shape != (len(names),):
                raise ValueError("Effort feedback must match the sample's joint names.")
            efforts.append(np.full(len(names), np.nan) if values is None else values)
        sources = {sample.effort_source for sample in ordered if sample.effort_source is not None}
        labels = []
        if ActuatorEffortSource.CURRENT_ESTIMATE in sources:
            labels.append("Torque estimated from current feedback and configured motor constant")
        if ActuatorEffortSource.SIMULATION in sources:
            labels.append("MuJoCo actuator effort")
        traces[setting] = ActuatorEffortTrace(
            joint_names=names,
            times_s=np.array([sample.received_timestamp_s for sample in ordered]),
            efforts=np.vstack(efforts),
            units=tuple(units[name] for name in names),
            source="; ".join(labels) or "Actuator effort feedback",
            segment_boundaries_s=tuple(
                sample.received_timestamp_s
                for previous, sample in pairwise(ordered)
                if sample.segment != previous.segment
            ),
        )
    return traces
