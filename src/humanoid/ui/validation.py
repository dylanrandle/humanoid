"""Validation for operator-console request values."""

from enum import StrEnum
from http import HTTPStatus

from humanoid.orchestrator.constants import CONTROLLED_PROCESS_NAMES
from humanoid.types.orchestrator import Mode, OrchestratorRequest, SafetyContext
from humanoid.types.process import ProcessAction, ProcessName, Runtime
from humanoid.types.robot import RobotName
from humanoid.types.simulation import MujocoScene
from humanoid.ui.errors import ApiError


def parse_process_name(name: str) -> ProcessName:
    process_name = _parse_enum(name, ProcessName, "Unknown process.", HTTPStatus.NOT_FOUND)
    if process_name not in CONTROLLED_PROCESS_NAMES:
        raise ApiError("Unknown process.", HTTPStatus.NOT_FOUND)
    return process_name


def parse_runtime(value: str) -> Runtime:
    return _parse_enum(
        value,
        Runtime,
        "Runtime must be either 'sim' or 'real'.",
        HTTPStatus.BAD_REQUEST,
    )


def parse_robot_name(value: str) -> RobotName:
    return _parse_enum(
        value,
        RobotName,
        "Unknown robot.",
        HTTPStatus.BAD_REQUEST,
    )


def parse_mujoco_scene(value: str) -> MujocoScene:
    return _parse_enum(
        value,
        MujocoScene,
        "Unknown MuJoCo scene.",
        HTTPStatus.BAD_REQUEST,
    )


def parse_safety_context(
    expected_runtime: object,
    expected_robot: object,
    expected_scene: object,
    real_hardware_acknowledged: object,
) -> SafetyContext:
    if not isinstance(real_hardware_acknowledged, bool):
        raise ApiError(
            "Real hardware acknowledgement must be a boolean.",
            HTTPStatus.BAD_REQUEST,
        )
    return SafetyContext(
        expected_runtime=parse_runtime(str(expected_runtime or "")),
        expected_robot=parse_robot_name(str(expected_robot or "")),
        expected_scene=parse_mujoco_scene(str(expected_scene or "")),
        real_hardware_acknowledged=real_hardware_acknowledged,
    )


def parse_orchestrator_request(mode_value: str, parameters: object) -> OrchestratorRequest:
    mode = _parse_enum(
        mode_value,
        Mode,
        "Unknown orchestrator mode.",
        HTTPStatus.BAD_REQUEST,
    )
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        raise ApiError("Orchestrator parameters must be a JSON object.", HTTPStatus.BAD_REQUEST)
    parameter_values: dict[str, object] = {}
    for name, value in parameters.items():
        if not isinstance(name, str):
            raise ApiError("Orchestrator parameter names must be strings.", HTTPStatus.BAD_REQUEST)
        parameter_values[name] = value
    try:
        return OrchestratorRequest(mode=mode, parameters=parameter_values)
    except ValueError as exc:
        raise ApiError(str(exc), HTTPStatus.BAD_REQUEST) from exc


def parse_process_action(value: str) -> ProcessAction:
    return _parse_enum(
        value,
        ProcessAction,
        "Unknown process action.",
        HTTPStatus.NOT_FOUND,
    )


def _parse_enum[EnumType: StrEnum](
    value: str,
    enum_type: type[EnumType],
    message: str,
    status: HTTPStatus,
) -> EnumType:
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ApiError(message, status) from exc
