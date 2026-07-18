import os
import subprocess
import sys

import pytest

from humanoid.constants import DEFAULT_HUMANOID_ROBOT, ROBOT_ENVIRONMENT_VARIABLE
from humanoid.types.robot import RobotName


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
