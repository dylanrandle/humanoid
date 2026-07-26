from http import HTTPStatus

import pytest

from humanoid.types.homing import HomingPreset
from humanoid.types.orchestrator import (
    Mode,
    OrchestratorParameter,
    OrchestratorRequest,
    SafetyContext,
)
from humanoid.types.process import ProcessAction, ProcessName, Runtime
from humanoid.types.robot import RobotName
from humanoid.types.simulation import MujocoScene
from humanoid.ui.errors import ApiError
from humanoid.ui.validation import (
    parse_mujoco_scene,
    parse_orchestrator_request,
    parse_process_action,
    parse_process_name,
    parse_robot_name,
    parse_runtime,
    parse_safety_context,
)


@pytest.mark.parametrize(
    ("parser", "value", "expected"),
    [
        (parse_runtime, Runtime.SIM.value, Runtime.SIM),
        (parse_robot_name, RobotName.PANDA.value, RobotName.PANDA),
        (parse_mujoco_scene, MujocoScene.FLOOR_AND_CUBE.value, MujocoScene.FLOOR_AND_CUBE),
        (parse_process_name, ProcessName.STACK.value, ProcessName.STACK),
        (parse_process_action, ProcessAction.START.value, ProcessAction.START),
    ],
)
def test_parsers_return_typed_enum_members(parser, value, expected):
    assert parser(value) is expected


@pytest.mark.parametrize(
    ("mode", "parameters", "expected"),
    [
        (Mode.IDLE.value, None, OrchestratorRequest(mode=Mode.IDLE)),
        (
            Mode.HOMING.value,
            {"preset": HomingPreset.HOME.value},
            OrchestratorRequest(
                mode=Mode.HOMING,
                parameters={OrchestratorParameter.PRESET: HomingPreset.HOME},
            ),
        ),
    ],
)
def test_parse_orchestrator_request_parameterizes_modes(mode, parameters, expected):
    assert parse_orchestrator_request(mode, parameters) == expected


@pytest.mark.parametrize(
    ("parser", "value", "status"),
    [
        (parse_runtime, "hardware", HTTPStatus.BAD_REQUEST),
        (parse_robot_name, "unknown", HTTPStatus.BAD_REQUEST),
        (parse_mujoco_scene, "warehouse", HTTPStatus.BAD_REQUEST),
        (parse_process_name, "unknown", HTTPStatus.NOT_FOUND),
        (parse_process_action, "restart", HTTPStatus.NOT_FOUND),
    ],
)
def test_parsers_raise_api_errors(parser, value, status):
    with pytest.raises(ApiError) as error:
        parser(value)

    assert error.value.status is status


def test_replay_process_is_only_available_through_replay_api():
    with pytest.raises(ApiError) as error:
        parse_process_name(ProcessName.REPLAY)

    assert error.value.status is HTTPStatus.NOT_FOUND


@pytest.mark.parametrize(
    ("mode", "parameters"),
    [
        ("dance", None),
        (Mode.HOMING.value, None),
        (Mode.HOMING.value, {"preset": "dance"}),
        (Mode.IDLE.value, {"preset": HomingPreset.HOME.value}),
        (Mode.HOMING.value, []),
        (Mode.HOMING.value, {"unknown": "value"}),
    ],
)
def test_parse_orchestrator_request_rejects_invalid_parameters(mode, parameters):
    with pytest.raises(ApiError) as error:
        parse_orchestrator_request(mode, parameters)

    assert error.value.status is HTTPStatus.BAD_REQUEST


def test_parse_safety_context_returns_typed_values():
    assert parse_safety_context("sim", "panda", "empty", True) == SafetyContext(
        expected_runtime=Runtime.SIM,
        expected_robot=RobotName.PANDA,
        expected_scene=MujocoScene.EMPTY,
        real_hardware_acknowledged=True,
    )


@pytest.mark.parametrize("acknowledged", [None, "true", 1])
def test_parse_safety_context_requires_boolean_acknowledgement(acknowledged):
    with pytest.raises(ApiError) as error:
        parse_safety_context("sim", "panda", "empty", acknowledged)

    assert error.value.status is HTTPStatus.BAD_REQUEST
