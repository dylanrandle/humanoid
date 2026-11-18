import threading
from unittest.mock import Mock

import pytest

from humanoid.orchestrator.autonomy.manager import AutonomyManager
from humanoid.types.orchestrator import Mode, OrchestratorError, SafetyContext
from humanoid.types.process import ProcessName, Runtime
from humanoid.types.robot import RobotName


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.delenv("HUMANOID_RAX_COMMAND", raising=False)
    monkeypatch.setattr("humanoid.orchestrator.autonomy.manager.default_command", list)
    service = Mock()
    service._lock = threading.RLock()
    service.node_manager.runtime = Runtime.SIM
    service.node_manager.robot = RobotName.TRISKEL
    service._run_manager_action.side_effect = lambda action: action()
    return AutonomyManager(service)


def test_task_selection_configures_normal_simulation_stack(manager):
    safety = SafetyContext(expected_runtime=Runtime.SIM, expected_robot=RobotName.TRISKEL)
    result = manager.select("toy_car_cleanup", safety)
    selection = manager.service.node_manager.set_simulation_task.call_args.args[0]
    assert selection.name == "toy_car_cleanup"
    assert not result.running and not result.ready
    manager.service.node_manager.start.assert_not_called()


def test_missing_runner_and_unqualified_hardware_cannot_start(manager):
    safety = SafetyContext(expected_runtime=Runtime.SIM, expected_robot=RobotName.TRISKEL)
    manager.task = "toy_car_cleanup"
    with pytest.raises(OrchestratorError, match="runner"):
        manager.start(safety)
    manager.service.node_manager.runtime = Runtime.REAL
    with pytest.raises(OrchestratorError, match="Hardware"):
        manager.start(safety)


def test_stop_revokes_token_cancels_motion_and_requests_idle(manager):
    manager._token = "old-token"
    session = Mock()
    manager._session = session
    manager.stop()
    session.execution.cancel.assert_called_once()
    manager.service.orchestrator_client.request_idle.assert_called_once()
    with pytest.raises(PermissionError):
        manager.rpc("Bearer old-token", {"method": "start", "arguments": {}})


def test_episode_activates_autonomy_mode_and_managed_policy(manager):
    manager.task = "toy_car_cleanup"
    manager._process = Mock()
    manager.start_episode(formal=False, seed=0)
    manager.service.orchestrator_client.request_autonomy.assert_called_once()
    manager.service.orchestrator_client.request_system.assert_not_called()
    manager.service._wait_for_mode.assert_called_once_with(Mode.AUTONOMY)
    manager.service.node_manager.start.assert_any_call(ProcessName.AUTONOMY)
