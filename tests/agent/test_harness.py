from unittest.mock import Mock

import pytest

from humanoid.agent.harness import model, worker
from humanoid.types.agent import AgentTask
from humanoid.types.orchestrator import Mode


@pytest.fixture
def setup_worker(monkeypatch):
    monkeypatch.setattr(worker, "start_thread", Mock())
    tools = Mock()
    tools.observe.return_value = ({"rgb_frame": 1}, "data:image/jpeg;base64,test")
    monkeypatch.setattr(worker, "TriskelTools", lambda _: tools)
    client = Mock(session_id="session")
    client.observation.return_value.mode = Mode.AUTONOMY
    client.execute.return_value = {"reached": True, "position_error_m": 0.001}
    decision = Mock()
    monkeypatch.setattr(worker, "decide", decision)
    return client, decision


def test_model_account_and_catalog():
    codex = Mock()
    codex.account.return_value.account = None
    with pytest.raises(RuntimeError, match="Connect"):
        model.image_models(codex)
    codex.account.return_value.account = Mock()
    codex.account.return_value.account.root.type = "apiKey"
    with pytest.raises(RuntimeError, match="Connect"):
        model.image_models(codex)
    codex.models.assert_not_called()
    codex.account.return_value.account.root.type = "chatgpt"
    codex.models.return_value.data = [
        Mock(model="text", input_modalities=[Mock(value="text")]),
        Mock(model="vision", input_modalities=[Mock(value="image")]),
    ]
    assert model.connect(codex, Mock()) == ("vision",)
    codex.login_chatgpt_device_code.assert_not_called()


@pytest.mark.parametrize("mode,live_mode", [(Mode.IDLE, Mode.AUTONOMY), (Mode.AUTONOMY, Mode.IDLE)])
def test_motion_needs_both_submitted_and_current_autonomy(setup_worker, mode, live_mode):
    client, decision = setup_worker
    client.observation.return_value.mode = live_mode
    decision.return_value = {"message": "moving", "command": {"operation": "hold", "arguments": {}}}
    with pytest.raises(RuntimeError, match="Autonomy"):
        worker.run_task(Mock(), client, AgentTask("task", "vision", mode, "session"), Mock())
    client.execute.assert_not_called()


def test_motion_result_reaches_next_model_turn(setup_worker):
    client, decision = setup_worker
    decision.side_effect = [
        {"message": "Holding", "command": {"operation": "hold", "arguments": {"duration": 1}}},
        {"message": "Held", "command": None},
    ]
    assert (
        worker.run_task(
            Mock(), client, AgentTask("hold", "vision", Mode.AUTONOMY, "session"), Mock()
        )
        == "Held"
    )
    client.execute.assert_called_once_with("hold", {"duration": 1})
    assert decision.call_args.args[1]["last_result"] == client.execute.return_value


def test_worker_refuses_new_robot_session(setup_worker):
    client, decision = setup_worker
    with pytest.raises(RuntimeError, match="restarted"):
        worker.run_task(Mock(), client, AgentTask("task", "vision", Mode.AUTONOMY, "old"), Mock())
    decision.assert_not_called()
