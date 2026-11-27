import base64
import json
from threading import Event
from typing import cast
from unittest.mock import Mock

import pytest
from PIL import Image
from pydantic import ValidationError

from humanoid.agent.harness import model, worker
from humanoid.agent.harness.console import AgentConsole
from humanoid.agent.harness.review import ExplorationReview
from humanoid.config.agent import CONSOLE_TASK_SECONDS
from humanoid.config.motion import DEFAULT_MOTION_LIMITS, DEFAULT_MOTION_TIMEOUT_SECONDS
from humanoid.types.agent import (
    AgentExecution,
    AgentPanelStatus,
    AgentPhase,
    AgentTask,
    Decision,
    DecisionContext,
    EpisodeTimeExpired,
    TimeBudget,
)
from humanoid.types.camera import CameraSummary, ObservationId, PoseRecord
from humanoid.types.observation import ObservationSummary
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import RobotName
from humanoid.types.segmentation import MaskId, MaskProposal, SegmentationResult
from humanoid.types.tools import (
    CompleteDepth,
    DepthCompletionResult,
    MotionCall,
    PerceptionCall,
    SegmentAtPoint,
)
from humanoid.types.trial import TaskScore


def decision_reply(message, operation=None, arguments=None):
    command = {"operation": operation, "arguments": arguments} if operation else None
    return Decision[MotionCall | PerceptionCall].model_validate_json(
        json.dumps({"message": message, "command": command})
    )


def observation_status() -> ObservationSummary:
    pose: PoseRecord = {"position_m": [0.0, 0.0, 0.0], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}
    return ObservationSummary(
        cameras={
            "base": CameraSummary(
                optical_frame="base_camera",
                camera_pose=pose,
                intrinsics={
                    "width": 8,
                    "height": 8,
                    "fx": 4.0,
                    "fy": 4.0,
                    "cx": 4.0,
                    "cy": 4.0,
                    "distortion_model": "none",
                    "distortion_coefficients": (),
                },
                model=None,
                observation_id=ObservationId("view"),
                frame_number=1,
                device_timestamp=0.0,
                timestamp_domain="simulation",
                raw_depth=None,
                registered_depth_available=True,
            )
        },
        mode=Mode.IDLE,
        base_pose_world=pose,
        tool_frame="tool",
        tool_pose_reference_frame="base",
        tool_pose=pose,
        gripper_units=[],
        gripper_positions=[],
        gripper_limits=[],
        primary_camera="base",
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
    execution = AgentExecution(
        AgentTask("inspect scene"),
        "vision",
        Mode.IDLE,
        "session",
        RobotName.TRISKEL,
        motion_session_id=None,
    )
    assert console.start(execution).phase is AgentPhase.RUNNING
    assert context.Process.call_args.kwargs["args"][1] is execution


@pytest.fixture
def setup_worker(monkeypatch):
    monkeypatch.setattr(worker, "start_thread", Mock())
    tools = Mock()
    tools.observe.return_value = (
        observation_status(),
        {"primary RGB": "data:image/jpeg;base64,test"},
    )
    monkeypatch.setattr(worker, "RobotTools", Mock(return_value=tools))
    client = Mock(runtime_id="session", motion_session_id="motion")
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
    result = thread.turn.return_value.run.return_value
    result.status.value = "completed"
    result.final_response = '{"message":"Observed","command":null}'
    context = DecisionContext(
        instruction="Observe",
        observation=observation_status(),
        last_result=None,
        seconds_remaining=10,
        limits=DEFAULT_MOTION_LIMITS,
    )
    views = {"primary RGB": "rgb-image", "wrist RGB": "wrist-image"}
    model.decide(thread, context, views, allow_motion=False, timer=TimeBudget(10))
    images = [i.url for i in thread.turn.call_args.args[0] if isinstance(i, model.ImageInput)]
    assert images == list(views.values())


@pytest.mark.parametrize("mode,live_mode", [(Mode.IDLE, Mode.AUTONOMY), (Mode.AUTONOMY, Mode.IDLE)])
def test_motion_needs_both_submitted_and_current_autonomy(setup_worker, mode, live_mode):
    client, decision, _ = setup_worker
    client.observation.return_value.mode = live_mode
    decision.return_value = decision_reply("moving", "hold", {"duration": 1})
    with pytest.raises(RuntimeError, match="API"):
        worker.run_execution(
            Mock(),
            client,
            AgentExecution(
                AgentTask("task"),
                "vision",
                mode,
                "session",
                RobotName.TRISKEL,
                motion_session_id="motion" if mode is Mode.AUTONOMY else None,
            ),
            Mock(),
        )
    client.execute.assert_not_called()


def test_motion_result_reaches_next_model_turn(setup_worker):
    client, decision, _ = setup_worker
    decision.side_effect = [
        decision_reply("Holding", "hold", {"duration": 1}),
        decision_reply("Held"),
    ]
    assert (
        worker.run_execution(
            Mock(),
            client,
            AgentExecution(
                AgentTask("hold"),
                "vision",
                Mode.AUTONOMY,
                "session",
                RobotName.PANDA,
                motion_session_id="motion",
            ),
            Mock(),
        )
        == "Held"
    )
    assert cast(Mock, worker.RobotTools).call_args.args[1].config.name is RobotName.PANDA
    client.execute.assert_called_once_with(
        "hold", {"duration": 1}, timeout=DEFAULT_MOTION_TIMEOUT_SECONDS
    )
    assert decision.call_args.args[1].last_result == client.execute.return_value


def test_worker_refuses_new_robot_session(setup_worker):
    client, decision, _ = setup_worker
    with pytest.raises(RuntimeError, match="restarted"):
        worker.run_execution(
            Mock(),
            client,
            AgentExecution(
                AgentTask("task"),
                "vision",
                Mode.AUTONOMY,
                "old",
                RobotName.TRISKEL,
                motion_session_id="motion",
            ),
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
        worker.run_execution(
            Mock(),
            client,
            AgentExecution(
                AgentTask("select"),
                "vision",
                Mode.IDLE,
                "session",
                RobotName.TRISKEL,
                motion_session_id=None,
            ),
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


def test_exploration_review_pairs_inputs_decisions_and_results(tmp_path, setup_worker):
    client, decide, tools = setup_worker
    jpeg = worker.image_url(Image.new("RGB", (8, 8), "blue"))
    inputs = {"base RGB": jpeg}
    tools.observe.return_value = observation_status(), inputs
    tools.segmentation.segment.return_value = (
        SegmentationResult(observation_id=ObservationId("view"), masks=()),
        Image.new("RGB", (8, 8), "green"),
    )
    decide.side_effect = [
        decision_reply(
            "Inspect </script>",
            "segment_at_point",
            {
                "observation_id": "view",
                "point_xy": [1, 2],
                "return_alternatives": False,
            },
        ),
        decision_reply("Hold", "hold", {"duration": 1}),
        decision_reply("Finished"),
    ]
    seed = 7
    folder = tmp_path / f"explore-{seed}"
    with ExplorationReview(folder, seed) as review:
        worker.run_steps(
            Mock(),
            tools,
            AgentTask("Explore"),
            Mock(),
            mode=Mode.AUTONOMY,
            budget=TimeBudget(30),
            review=review,
        )
        review.finish("agent_finished", TaskScore(False, {}))
    page = (folder / "index.html").read_text()
    data = json.loads(page.split('type="application/json">')[1].split("</script>")[0])
    first, second, final = data["turns"]
    assert data["status"] == "agent_finished" and data["seed"] == seed
    assert first["decision"]["message"] == "Inspect </script>"
    assert first["decision"]["command"]["arguments"]["point_xy"] == [1, 2]
    assert list(first["images"]) == ["base RGB"]
    assert (folder / first["images"]["base RGB"]).read_bytes() == base64.b64decode(
        jpeg.split(",")[1]
    )
    assert first["result_image"] == second["images"]["Perception result for view"]
    assert second["context"]["last_result"] == first["result"]
    assert second["result"] == client.execute.return_value == final["context"]["last_result"]
    assert final["decision"]["command"] is None


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
        worker.run_execution(
            Mock(),
            client,
            AgentExecution(
                AgentTask("depth"),
                "vision",
                mode,
                "session",
                RobotName.TRISKEL,
                motion_session_id="motion" if mode is Mode.AUTONOMY else None,
            ),
            Mock(),
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
    contract, _ = model._decision_contract(allow_motion=False)
    with pytest.raises(ValidationError):
        contract.model_validate_json(
            json.dumps(
                {"message": "act", "command": {"operation": operation, "arguments": arguments}}
            )
        )


def test_console_task_can_repeat_perception_and_motion_within_its_time_budget(setup_worker):
    client, decide, _ = setup_worker
    attempts = 50
    decide.side_effect = [
        *[decision_reply("Look", "observe_again", {})] * attempts,
        *[decision_reply("Probe", "hold", {"duration": 1})] * attempts,
        decision_reply("Finished"),
    ]
    worker.run_execution(
        Mock(),
        client,
        AgentExecution(
            AgentTask("Explore cleanup"),
            "vision",
            Mode.AUTONOMY,
            "session",
            RobotName.TRISKEL,
            motion_session_id="motion",
        ),
        Mock(),
    )
    context = decide.call_args.args[1]
    assert client.execute.call_count == attempts
    assert 0 < context.seconds_remaining <= CONSOLE_TASK_SECONDS


def test_console_motion_uses_remaining_time_and_reports_expiry(setup_worker, monkeypatch):
    client, decide, tools = setup_worker
    elapsed = CONSOLE_TASK_SECONDS - 0.25
    monkeypatch.setattr(TimeBudget, "elapsed_seconds", property(lambda self: elapsed))
    decide.return_value = decision_reply("Holding", "hold", {"duration": 1})

    def timeout(*args, **kwargs):
        nonlocal elapsed
        elapsed = CONSOLE_TASK_SECONDS
        raise RuntimeError("Timed out waiting for the motion command to finish.")

    client.execute.side_effect = timeout
    with pytest.raises(EpisodeTimeExpired, match="Task time budget"):
        worker.run_execution(
            Mock(),
            client,
            AgentExecution(
                AgentTask("Hold"),
                "vision",
                Mode.AUTONOMY,
                "session",
                RobotName.TRISKEL,
                motion_session_id="motion",
            ),
            Mock(),
        )
    client.execute.assert_called_once_with("hold", {"duration": 1}, timeout=0.25)
    tools.observe.assert_called_once()


def test_timed_decision_interrupts_an_in_flight_model_turn():
    thread = Mock()
    turn = thread.turn.return_value
    interrupted = Event()
    turn.interrupt.side_effect = interrupted.set
    turn.run.side_effect = lambda: interrupted.wait(timeout=2)
    context = DecisionContext(
        instruction="Explore",
        observation=observation_status(),
        last_result=None,
        seconds_remaining=0.1,
        limits=DEFAULT_MOTION_LIMITS,
    )
    with pytest.raises(EpisodeTimeExpired):
        model.decide(
            thread, context, {"primary RGB": "rgb"}, allow_motion=True, timer=TimeBudget(0.1)
        )
    turn.interrupt.assert_called_once_with()


@pytest.mark.parametrize("mode", [Mode.IDLE, Mode.AUTONOMY])
def test_execution_motion_identity_is_required_only_for_motion(setup_worker, mode):
    client, decision, _ = setup_worker
    client.observation.return_value.mode = mode
    decision.return_value = decision_reply("Observed")
    task = AgentTask("Inspect the scene")
    execution = AgentExecution(
        task=task,
        model="vision",
        mode=mode,
        runtime_id="session",
        robot=RobotName.TRISKEL,
        motion_session_id="previous-motion" if mode is Mode.AUTONOMY else None,
    )
    if mode is Mode.AUTONOMY:
        with pytest.raises(RuntimeError, match="Motion executor restarted"):
            worker.run_execution(Mock(), client, execution, Mock())
        decision.assert_not_called()
    else:
        assert worker.run_execution(Mock(), client, execution, Mock()) == "Observed"
        assert decision.call_args.args[1].instruction == task.instruction
        assert decision.call_args.kwargs["allow_motion"] is False
    client.execute.assert_not_called()


def test_model_schema_requires_fields_that_api_callers_can_omit():
    _, schema = model._decision_contract(allow_motion=True)
    for name in ("Hold", "MoveBase", "MoveTool", "SetGripper", "CompleteDepth"):
        definition = schema["$defs"][name]
        assert set(definition["required"]) == set(definition["properties"])
        assert all("default" not in field for field in definition["properties"].values())


def test_agent_motion_failure_is_json_feedback(setup_worker):
    client, decide, tools = setup_worker
    client.execute.side_effect = ValueError("duration exceeds limit")
    decide.side_effect = [
        decision_reply("Hold", "hold", {"duration": 10}),
        decision_reply("Stopped"),
    ]
    worker.run_steps(
        Mock(), tools, AgentTask("Hold"), Mock(), mode=Mode.AUTONOMY, budget=TimeBudget(10)
    )
    feedback = decide.call_args.args[1].last_result
    assert json.loads(json.dumps(feedback)) == {
        "operation": "hold",
        "error": "duration exceeds limit",
    }
