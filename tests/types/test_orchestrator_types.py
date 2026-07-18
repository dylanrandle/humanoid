"""Tests for orchestrator-domain API types."""

import json

import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.types.homing import HomingPreset
from humanoid.types.orchestrator import (
    Mode,
    OrchestratorParameter,
    OrchestratorRequest,
)
from humanoid.types.process import ProcessName, Runtime
from humanoid.types.robot import RobotName


def test_ui_enums_preserve_the_json_api_contract():
    payload = {
        "runtime": Runtime.REAL,
        "robot": RobotName.ELROBOT_MOBILE,
        "processes": {ProcessName.STACK: {"running": True}},
        "mode": Mode.HOMING,
        "preset": HomingPreset.REST,
    }

    assert json.loads(json.dumps(payload)) == {
        "runtime": "real",
        "robot": "elrobot_mobile",
        "processes": {"stack": {"running": True}},
        "mode": "homing",
        "preset": "rest",
    }


def test_orchestrator_request_parameterizes_homing_mode():
    request = OrchestratorRequest(
        mode=Mode.HOMING,
        parameters={OrchestratorParameter.PRESET: HomingPreset.HOME},
    )

    assert request.mode is Mode.HOMING
    assert request.parameters[OrchestratorParameter.PRESET] is HomingPreset.HOME


def test_orchestrator_request_normalizes_parameter_names_and_values():
    request = OrchestratorRequest(mode=Mode.HOMING, parameters={"preset": "rest"})

    assert request.parameters == {OrchestratorParameter.PRESET: HomingPreset.REST}


@pytest.mark.parametrize(
    ("mode", "parameters"),
    [
        (Mode.HOMING, {}),
        (Mode.IDLE, {OrchestratorParameter.PRESET: HomingPreset.REST}),
        (Mode.HOMING, {OrchestratorParameter.PRESET: "unknown"}),
        (Mode.HOMING, {"unknown": "value"}),
    ],
)
def test_orchestrator_request_rejects_invalid_mode_parameters(mode, parameters):
    with pytest.raises(ValueError):
        OrchestratorRequest(mode=mode, parameters=parameters)


def test_robot_name_enum_matches_configured_robots():
    assert set(ROBOT_CONFIGS) == set(RobotName)
