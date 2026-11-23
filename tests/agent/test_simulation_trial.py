import json
import time
from contextlib import closing, nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import lcm
import mujoco
import pytest

from humanoid.agent.harness import trial
from humanoid.config.agent import get_agent_endpoint
from humanoid.config.robot import PANDA_CONFIG, SO101_CONFIG, TRISKEL_CONFIG
from humanoid.config.simulation import DEFAULT_MUJOCO_SIMULATION_CONFIG
from humanoid.constants import Topic
from humanoid.recording import RecordingCatalog
from humanoid.simulation.tasks.toy_cleanup import TOYS
from humanoid.simulation.trial_stack import TrialStack
from humanoid.types.agent import AgentTask
from humanoid.types.orchestrator import Mode
from humanoid.types.process import Runtime
from humanoid.types.robot import RobotName
from humanoid.types.simulation import SimulationScene
from humanoid.types.trial import SimulationTask, TaskScore, TrialStep


class WaitTask(SimulationTask):
    target_seconds = 0.0
    scene = SimulationScene.EMPTY
    goal = "Hold still in the empty scene."

    def score(self, data: mujoco.MjData) -> TaskScore:
        return TaskScore(data.time >= self.target_seconds, {"elapsed": data.time})


def test_registered_task_explores_generates_and_evaluates_in_an_empty_scene(tmp_path, monkeypatch):
    task = WaitTask()
    monkeypatch.setitem(trial.TASKS, "wait", lambda parser: lambda args: task)
    thread = Mock()
    thread.run.return_value.status.value = "completed"
    thread.run.return_value.final_response = '{"source":"test controller"}'
    monkeypatch.setattr(trial, "open_codex", nullcontext)
    monkeypatch.setattr(trial, "start_thread", lambda *_: thread)

    def practice(thread, tools, agent_task, record, **kwargs):
        assert tools.robot.config.name == TRISKEL_CONFIG.name
        assert task.goal in agent_task.instruction
        first = tools.client.capture()["base"][1].rgb
        time.sleep(0.2)
        second = tools.client.capture()["base"][1].rgb
        assert second.frame_number > first.frame_number
        assert second.device_timestamp > first.device_timestamp
        return "Explored"

    monkeypatch.setattr(trial, "run_steps", practice)
    step = TrialStep.model_validate(
        {"memory": {}, "command": {"operation": "hold", "arguments": {"duration": 0.2}}}
    )

    def controller(source, obs, memory, timer):
        expected = {"base", "wrist"}
        assert obs["cameras"].keys() == obs["images"].keys() == expected
        for name in expected:
            image = obs["images"][name]
            assert image["registered_depth_m"] and image["raw_depth_m"] and image["rgb"]
        base, wrist = obs["cameras"]["base"], obs["cameras"]["wrist"]
        assert (base["model"], wrist["model"]) == ("d435", "d405")
        assert base["camera_pose"] != wrist["camera_pose"]
        assert base["observation_id"] != wrist["observation_id"]
        assert obs["images"]["base"]["rgb"] != obs["images"]["wrist"]["rgb"]
        return step

    monkeypatch.setattr(trial, "program_step", controller)
    trial.main(
        [
            "--record",
            "--robot",
            "triskel",
            "--explore",
            "1",
            "--test",
            "1",
            "--output",
            str(tmp_path),
            "wait",
        ]
    )

    (folder,) = tmp_path.iterdir()
    assert task.goal in thread.run.call_args.args[0]
    assert json.loads((folder / "settings.json").read_text())["task"] == "wait"
    practice_result = json.loads((folder / "explore-0.json").read_text())
    assert practice_result["success"]
    result = json.loads((folder / "test-1.json").read_text())
    assert result["success"]
    assert result["elapsed"] > 0
    assert result["ended_by"] == "task_complete"
    assert len(result["trace"]) == 1
    assert practice_result["lcm_recording"] != result["lcm_recording"]
    recordings = RecordingCatalog(folder / "recordings")
    for outcome in (practice_result, result):
        path = Path(outcome["lcm_recording"])
        bundle = recordings.get(path.parent.name)
        assert bundle.log_path == path and bundle.robot == RobotName.TRISKEL
        with closing(lcm.EventLog(str(path), "r")) as log:
            channels = {event.channel for event in iter(log.read_next_event, None)}
        assert {
            Topic.ROBOT_STATE.value,
            Topic.SIMULATION_STATE.value,
            Topic.ORCHESTRATOR_MODE.value,
        } <= channels
        for name in ("BASE", "WRIST"):
            assert {
                f"CAMERA/{name}/{kind}" for kind in ("RGB", "DEPTH", "REGISTERED_DEPTH")
            } <= channels
        if outcome is result:
            assert Topic.ROBOT_JOINT_COMMAND.value in channels


def test_episode_requires_a_camera():
    with pytest.raises(ValueError, match="RGB-D camera"):
        TrialStack(
            WaitTask(),
            0,
            robot_config=TRISKEL_CONFIG,
            simulation_config=replace(DEFAULT_MUJOCO_SIMULATION_CONFIG, enable_cameras=False),
        )


def test_rejected_motion_returns_feedback_and_replay_continues(tmp_path, monkeypatch):
    replay = tmp_path / "controller.py"
    replay.write_text("# Controller decisions supplied by the test.")

    def controller(source, obs, memory, timer):
        last = obs["last_result"]
        if last is None:
            command = {"operation": "hold", "arguments": {"duration": 0}}
        elif "error" in last:
            assert "duration" in last["error"]
            command = {"operation": "hold", "arguments": {"duration": 0.1}}
        else:
            assert "elapsed_seconds" in last
            command = None
        return TrialStep.model_validate({"command": command, "memory": {}})

    monkeypatch.setattr(trial, "program_step", controller)
    output = tmp_path / "results"
    trial.main(
        [
            "--robot",
            "triskel",
            "--replay",
            str(replay),
            "--test",
            "2",
            "--output",
            str(output),
            "toy_cleanup",
            "--items",
            str(len(TOYS)),
        ]
    )
    (folder,) = output.iterdir()
    assert not (folder / "recordings").exists()
    for seed in range(2):
        result = json.loads((folder / f"test-{seed}.json").read_text())
        assert result["total"] == len(TOYS)
        assert "lcm_recording" not in result
        assert result["ended_by"] == "controller_finished"
        rejected, recovered = result["trace"]
        assert "duration" in rejected["result"]["error"]
        assert (
            recovered["result"]["elapsed_seconds"] >= recovered["command"]["arguments"]["duration"]
        )


@pytest.mark.parametrize("robot", [PANDA_CONFIG, SO101_CONFIG], ids=lambda config: config.name)
@pytest.mark.parametrize("backend", [Runtime.SIM, Runtime.REAL])
def test_vision_trial_rejects_robots_without_cameras(robot, backend):
    with (
        pytest.raises(ValueError, match="RGB-D camera"),
        trial.open_episode(
            0,
            task=WaitTask(),
            robot_config=robot,
            backend=backend,
            endpoint=get_agent_endpoint(),
        ),
    ):
        pass


def test_time_budget_ends_both_phases_without_dispatching_a_decision(tmp_path, monkeypatch):
    task = WaitTask()
    agent = AgentTask(task.goal, "test", Mode.AUTONOMY, "simulation", RobotName.TRISKEL)
    thread = Mock()
    output = tmp_path / "explore.json"
    episodes = partial(
        trial.open_episode,
        task=task,
        backend=Runtime.SIM,
        endpoint=get_agent_endpoint(),
        robot_config=TRISKEL_CONFIG,
    )
    trial.explore(thread, agent, 0, output, 1e-12, episodes=episodes)
    thread.turn.assert_not_called()
    assert json.loads(output.read_text())["ended_by"] == "time_budget"
    step = Mock()
    monkeypatch.setattr(trial, "program_step", step)
    result = trial.evaluate(
        "", 1, 1e-12, episodes=episodes, completion_reason=task.completion_reason
    )
    step.assert_not_called()
    assert result.ended_by == "time_budget"
    assert result.error is None


def test_trial_routes_renamed_cameras_from_robot_config():
    config = replace(
        TRISKEL_CONFIG,
        cameras={
            "chassis": TRISKEL_CONFIG.cameras["base"],
            "inspection_2": TRISKEL_CONFIG.cameras["wrist"],
        },
    )
    with trial.open_episode(
        0,
        task=WaitTask(),
        robot_config=config,
        backend=Runtime.SIM,
        endpoint=get_agent_endpoint(),
    ) as (_, tools):
        status, images = tools.observe(Mode.AUTONOMY)
        assert {name: view["model"] for name, view in status["cameras"].items()} == {
            "chassis": "d435",
            "inspection_2": "d405",
        }
        assert set(images) == {"chassis RGB", "inspection_2 RGB"}
