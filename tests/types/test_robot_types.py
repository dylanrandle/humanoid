import os
import subprocess
import sys

import numpy as np
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.constants import DEFAULT_HUMANOID_ROBOT, ROBOT_ENVIRONMENT_VARIABLE
from humanoid.types.homing import HomingPreset
from humanoid.types.robot import (
    CartesianVelocityLimits,
    RobotBaseConfig,
    RobotConfig,
    RobotName,
    RobotToolConfig,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, DEFAULT_HUMANOID_ROBOT),
        ("", DEFAULT_HUMANOID_ROBOT),
        (" panda ", RobotName.PANDA),
        ("SO101", RobotName.SO101),
    ],
)
def test_robot_environment_values(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(ROBOT_ENVIRONMENT_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(ROBOT_ENVIRONMENT_VARIABLE, value)

    assert RobotName.from_environment() is expected


def test_invalid_robot_environment_is_rejected(monkeypatch):
    monkeypatch.setenv(ROBOT_ENVIRONMENT_VARIABLE, "typo")

    with pytest.raises(ValueError, match="typo"):
        RobotName.from_environment()


def test_invalid_robot_environment_breaks_fresh_config_import():
    environment = {**os.environ, ROBOT_ENVIRONMENT_VARIABLE: "typo"}

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from humanoid.config import ROBOT_NAME; print(ROBOT_NAME.value)",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode != 0
    assert "'typo' is not a valid RobotName" in result.stderr


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_cartesian_velocity_limits_must_be_positive_and_finite(value):
    with pytest.raises(ValueError, match="linear velocity"):
        CartesianVelocityLimits(linear=value, angular=1.0)

    with pytest.raises(ValueError, match="angular velocity"):
        CartesianVelocityLimits(linear=1.0, angular=value)


def test_robot_base_config_rejects_empty_frame():
    with pytest.raises(ValueError, match="frame must not be empty"):
        RobotBaseConfig(
            frame=" ",
            velocity_limits=CartesianVelocityLimits(linear=0.2, angular=1.0),
        )


def test_robot_tool_config_rejects_empty_frame():
    with pytest.raises(ValueError, match="frame must not be empty"):
        RobotToolConfig(
            frame=" ",
            velocity_limits=CartesianVelocityLimits(linear=0.5, angular=1.0),
        )


def _robot_config_with_presets(
    presets: dict[HomingPreset, np.ndarray],
) -> RobotConfig:
    return RobotConfig(
        name=RobotName.PANDA,
        tool=RobotToolConfig(frame="tool"),
        homing_presets=presets,
        actuator_control_modes={},
    )


def test_tool_and_base_configs_use_default_velocity_limits():
    tool = RobotToolConfig(frame="tool")
    base = RobotBaseConfig(frame="base")

    assert tool.velocity_limits.linear == pytest.approx(1.0)
    assert tool.velocity_limits.angular == pytest.approx(np.pi)
    assert base.velocity_limits == tool.velocity_limits


def test_homing_presets_require_every_named_preset():
    with pytest.raises(ValueError, match="define every HomingPreset"):
        _robot_config_with_presets({HomingPreset.HOME: np.zeros(1)})


def test_homing_presets_require_matching_one_dimensional_shapes():
    with pytest.raises(ValueError, match="one-dimensional"):
        _robot_config_with_presets(
            {
                HomingPreset.HOME: np.zeros((1, 1)),
                HomingPreset.REST: np.zeros((1, 1)),
            }
        )

    with pytest.raises(ValueError, match="matching shapes"):
        _robot_config_with_presets(
            {
                HomingPreset.HOME: np.zeros(1),
                HomingPreset.REST: np.zeros(2),
            }
        )


def test_homing_presets_require_finite_values():
    with pytest.raises(ValueError, match="finite"):
        _robot_config_with_presets(
            {
                HomingPreset.HOME: np.array([np.nan]),
                HomingPreset.REST: np.zeros(1),
            }
        )


def test_mobile_robot_combines_base_frame_and_velocity_limits():
    base = ROBOT_CONFIGS[RobotName.ELROBOT_MOBILE].base

    assert base is not None
    assert base.frame == "root_joint"
    assert base.velocity_limits.linear == pytest.approx(0.2)
    assert base.velocity_limits.angular == pytest.approx(1.0)


def test_robot_combines_tool_frame_limits_and_homing_presets():
    config = ROBOT_CONFIGS[RobotName.PANDA]

    assert config.tool.frame == "panda_hand_tcp"
    assert config.tool.velocity_limits.linear == pytest.approx(1.0)
    assert (
        config.homing_presets[HomingPreset.HOME].shape
        == config.homing_presets[HomingPreset.REST].shape
    )
