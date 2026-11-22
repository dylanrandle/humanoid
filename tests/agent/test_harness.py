import json
from typing import cast
from unittest.mock import Mock

import pytest
from PIL import Image
from pydantic import ValidationError

from humanoid.agent.harness import model, worker
from humanoid.agent.harness.service import AgentConsole
from humanoid.config.agent import get_agent_policy_config
from humanoid.types.agent import (
    AgentPanelStatus,
    AgentPhase,
    AgentTask,
    Decision,
    DecisionContext,
    MotionCall,
    ObservationStatus,
    PerceptionCall,
    PoseRecord,
)
from humanoid.types.camera import ObservationId
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import RobotName
from humanoid.types.segmentation import MaskId, MaskProposal, SegmentationResult
from humanoid.types.tools import CompleteDepth, DepthCompletionResult, SegmentAtPoint


def decision_reply(message, operation=None, arguments=None):
    command = {"operation": operation, "arguments": arguments} if operation else None
    return Decision[MotionCall | PerceptionCall].model_validate_json(
        json.dumps({"message": message, "command": command})
    )


def observation_status() -> ObservationStatus:
    pose: PoseRecord = {"position_m": [0.0, 0.0, 0.0], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}
    return ObservationStatus(
        cameras={},
        mode=Mode.IDLE,
        base_pose_world=pose,
        tool_frame="tool",
        tool_pose_reference_frame="base",
        tool_pose=pose,
        gripper_units=[],
        gripper_positions=[],
        gripper_limits=[],
        rgb_frame=1,
        observation_id=ObservationId("view"),
        rgb_size_xy=[8, 8],
        registered_depth_available=True,
    )


def test_completed_connection_can_immediately_start_task(monkeypatch):
    console = AgentConsole()
    context = Mock()
    monkeypatch.setattr(console, "_context", context)
    incoming = Mock()
    context.Pipe.return_value = incoming, Mock()
    process = context.Process.return_value
    console.start()
    incoming.poll.side_effect = [True, False, False]
    incoming.recv.return_value = AgentPanelStatus(phase=AgentPhase.COMPLETED, models=("vision",))
    process.is_alive.side_effect = [True, False]
    assert console.status().phase is AgentPhase.CONNECTING
    assert console.status().phase is AgentPhase.COMPLETED
    task = AgentTask("inspect scene", "vision", Mode.IDLE, "session", RobotName.TRISKEL)
    assert console.start(task).phase is AgentPhase.RUNNING


@pytest.fixture
def setup_worker(monkeypatch):
    monkeypatch.setattr(worker, "start_thread", Mock())
    tools = Mock()
    tools.observe.return_value = (
        observation_status(),
        {"primary RGB": "data:image/jpeg;base64,test"},
    )
    monkeypatch.setattr(worker, "RobotTools", Mock(return_value=tools))
    client = Mock(session_id="session")
    tools.client = client
    client.observation.return_value.mode = Mode.AUTONOMY
    client.execute.return_value = {"reached": True, "position_error_m": 0.001}
    decision = Mock()
    monkeypatch.setattr(worker, "decide", decision)
    return client, decision, tools


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


def test_model_receives_rgb_from_both_cameras():
    thread = Mock()
    thread.run.return_value.status.value = "completed"
    thread.run.return_value.final_response = '{"message":"Observed","command":null}'
    context = DecisionContext(
        instruction="Observe",
        observation=observation_status(),
        last_result=None,
        commands_remaining=0,
        perception_remaining=1,
        limits=get_agent_policy_config(),
    )
    views = {"primary RGB": "rgb-image", "wrist RGB": "wrist-image"}
    model.decide(thread, context, views, allow_motion=False)
    images = [i.url for i in thread.run.call_args.args[0] if isinstance(i, model.ImageInput)]
    assert images == list(views.values())


@pytest.mark.parametrize("mode,live_mode", [(Mode.IDLE, Mode.AUTONOMY), (Mode.AUTONOMY, Mode.IDLE)])
def test_motion_needs_both_submitted_and_current_autonomy(setup_worker, mode, live_mode):
    client, decision, _ = setup_worker
    client.observation.return_value.mode = live_mode
    decision.return_value = decision_reply("moving", "hold", {"duration": 1})
    with pytest.raises(RuntimeError, match="Autonomy"):
        worker.run_task(
            Mock(), client, AgentTask("task", "vision", mode, "session", RobotName.TRISKEL), Mock()
        )
    client.execute.assert_not_called()


def test_motion_result_reaches_next_model_turn(setup_worker):
    client, decision, _ = setup_worker
    decision.side_effect = [
        decision_reply("Holding", "hold", {"duration": 1}),
        decision_reply("Held"),
    ]
    assert (
        worker.run_task(
            Mock(),
            client,
            AgentTask("hold", "vision", Mode.AUTONOMY, "session", RobotName.PANDA),
            Mock(),
        )
        == "Held"
    )
    assert cast(Mock, worker.RobotTools).call_args.args[1].config.name is RobotName.PANDA
    client.execute.assert_called_once_with("hold", {"duration": 1})
    assert decision.call_args.args[1].last_result == client.execute.return_value


def test_worker_refuses_new_robot_session(setup_worker):
    client, decision, _ = setup_worker
    with pytest.raises(RuntimeError, match="restarted"):
        worker.run_task(
            Mock(),
            client,
            AgentTask("task", "vision", Mode.AUTONOMY, "old", RobotName.TRISKEL),
            Mock(),
        )
    decision.assert_not_called()


def test_idle_segmentation_passes_numbered_mask_to_next_turn(setup_worker):
    client, decision, tools = setup_worker
    client.observation.return_value.mode = Mode.IDLE
    masks = SegmentationResult(
        observation_id=ObservationId("view"),
        masks=(
            MaskProposal(
                number=1,
                mask_id=MaskId("choice"),
                box_xyxy=(2, 3, 5, 6),
                pixels=12,
                predicted_iou=0.95,
            ),
        ),
    )
    tools.segmentation.segment.return_value = masks, Image.new("RGB", (8, 8))
    args = {"observation_id": "view", "point_xy": [3, 4], "return_alternatives": True}
    decision.side_effect = [
        decision_reply("Select the cube", "segment_at_point", args),
        decision_reply("Selected choice"),
    ]
    assert (
        worker.run_task(
            Mock(),
            client,
            AgentTask("select", "vision", Mode.IDLE, "session", RobotName.TRISKEL),
            Mock(),
        )
        == "Selected choice"
    )
    tools.segmentation.segment.assert_called_once_with(
        SegmentAtPoint(
            observation_id=ObservationId("view"), point_xy=(3, 4), return_alternatives=True
        )
    )
    tools.observe.assert_called_once()
    assert decision.call_args.args[1].last_result == masks
    assert decision.call_args.args[2]["Perception result for view"].startswith("data:image/jpeg;")
    assert not decision.call_args.kwargs["allow_motion"]
    client.execute.assert_not_called()


@pytest.mark.parametrize("mode", [Mode.IDLE, Mode.AUTONOMY])
def test_complete_depth_is_an_explicit_perception_command(setup_worker, mode):
    client, decision, tools = setup_worker
    client.observation.return_value.mode = mode
    result = DepthCompletionResult(
        observation_id=ObservationId("view"), valid_pixels=128, depths_m=(0.5,)
    )
    tools.depth.complete_depth.return_value = result
    decision.side_effect = [
        decision_reply(
            "Check depth", "complete_depth", {"observation_id": "view", "points_xy": [[4, 3]]}
        ),
        decision_reply("Depth ready"),
    ]
    assert (
        worker.run_task(
            Mock(), client, AgentTask("depth", "vision", mode, "session", RobotName.TRISKEL), Mock()
        )
        == "Depth ready"
    )
    tools.depth.complete_depth.assert_called_once_with(
        CompleteDepth(observation_id=ObservationId("view"), points_xy=((4, 3),))
    )
    tools.segmentation.segment.assert_not_called()
    assert decision.call_args.args[1].last_result == result
    assert set(decision.call_args.args[2]) == {"primary RGB"}
    client.execute.assert_not_called()


@pytest.mark.parametrize(
    "operation,arguments",
    [
        (
            "segment_at_point",
            {"observation_id": "view", "point_xy": [True, 3], "return_alternatives": False},
        ),
        (
            "segment_in_box",
            {"observation_id": "view", "point_xy": [3, 4], "return_alternatives": False},
        ),
        ("complete_depth", {"mask_id": "mask"}),
        ("observe_again", {"unexpected": True}),
        ("hold", {"duration": 1}),
    ],
)
def test_model_contract_rejects_invalid_or_unavailable_commands(operation, arguments):
    contract, _ = model._decision_contract(allow_motion=False, allow_perception=True)
    with pytest.raises(ValidationError):
        contract.model_validate_json(
            json.dumps(
                {"message": "act", "command": {"operation": operation, "arguments": arguments}}
            )
        )


def test_perception_budget_restricts_schema_and_parser():
    contract, schema = model._decision_contract(allow_motion=True, allow_perception=False)
    assert "complete_depth" not in json.dumps(schema)
    with pytest.raises(ValidationError):
        contract.model_validate_json(
            '{"message":"act","command":{"operation":"observe_again","arguments":{}}}'
        )
