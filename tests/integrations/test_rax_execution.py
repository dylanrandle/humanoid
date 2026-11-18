from unittest.mock import Mock

import pytest

from humanoid.integrations.rax.execution import RaxExecution
from humanoid.types.rax import RaxOperation
from humanoid.types.remote_api import ControlOperation, RemoteControlEndpoint


@pytest.fixture
def execution(monkeypatch):
    monkeypatch.setattr("humanoid.integrations.rax.execution.RemoteControlClient", Mock())
    return RaxExecution(Mock(), RemoteControlEndpoint("http://localhost:8767", "test-token"))


@pytest.mark.parametrize("operation", [RaxOperation.MOVE_BASE, "move_base", "gripper"])
def test_rax_tools_use_the_general_robot_api(execution, operation):
    result = {
        "reached": False,
        "position_error_m": 0.04,
        "orientation_error_rad": 0.0,
        "gripper_error_rad": 0.0,
        "elapsed_seconds": 1.0,
        "step_id": 3,
    }
    execution.client.execute.return_value = result
    outcome = execution.execute(operation, {"duration": 0.1})
    expected = (
        ControlOperation.SET_GRIPPER if operation == "gripper" else ControlOperation.MOVE_BASE
    )
    execution.client.execute.assert_called_once_with(
        expected, {"duration": 0.1}, timeout=execution.config.max_duration + 3.0
    )
    assert outcome == {**result, "operation": RaxOperation(operation)}
    assert execution.step_id == result["step_id"]


@pytest.mark.parametrize("operation", ["unknown", "cancel"])
def test_invalid_operation_is_not_sent(execution, operation):
    with pytest.raises(ValueError):
        execution.execute(operation, {})
    execution.client.execute.assert_not_called()


def test_observations_and_cancellation_use_remote_service(execution):
    assert execution.state() is execution.client.state.return_value
    assert execution.observe() is execution.client.observe.return_value
    execution.cancel()
    execution.client.cancel.assert_called_once()
