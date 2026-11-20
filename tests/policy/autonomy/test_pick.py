"""Pickup obeys the Policy interface while perception runs asynchronously."""

import threading
import weakref
from concurrent.futures import Future
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pinocchio as pin
import pytest

from humanoid.config.pick import GRASPGENX_CONFIG, PICK_CONFIG
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.policy.autonomy.pick import PickPolicy
from humanoid.types.camera import CameraPreview
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode
from humanoid.types.pick import GraspProposal, PickPhase, PickPlan, PickPreview, PickWaypoint
from humanoid.types.robot import RobotState


@pytest.fixture
def runtime(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("humanoid.policy.autonomy.pick.time.monotonic", lambda: clock[0])
    policy = PickPolicy(TRISKEL_CONFIG, PICK_CONFIG, GRASPGENX_CONFIG, MagicMock())
    q = pin.neutral(policy.robot.model)
    observation = Observation(
        robot_state=RobotState(
            timestamp=10,
            joint_positions=q,
            joint_velocities=np.zeros(policy.robot.model.nv),
            actuator_temperatures=np.zeros(8),
        ),
        mode=Mode.SYSTEM,
        mode_timestamp=10,
        robot_joint_command=MagicMock(timestamp=10),
        camera_frames=MagicMock(rgb=MagicMock(timestamp=10)),
    )
    return policy, observation, clock


def test_pending_perception_does_not_block_policy_or_takeover(runtime, monkeypatch):
    policy, observation, _ = runtime
    # This test controls inference timing; mesh loading is covered separately.
    policy.perception_robot = MagicMock()
    entered, release = threading.Event(), threading.Event()

    def perceive(*args):
        entered.set()
        release.wait(3)
        raise RuntimeError("No object")

    monkeypatch.setattr("humanoid.policy.autonomy.pick.perceive", perceive)
    try:
        policy.start("toy car", observation)
        assert entered.wait(1)
        action = policy(observation)
        assert action.tool_pose is not None
        assert action.base_pose is None
        assert policy.phase is PickPhase.PLANNING
        empty = policy(replace(observation, mode=Mode.KEYBOARD))
        assert empty.tool_pose is None
        assert policy.phase is PickPhase.CANCELLED
        release.set()
        with pytest.raises(RuntimeError, match="No object"):
            policy._future.result(timeout=1)
        assert policy(observation).tool_pose is None
        assert policy.phase is PickPhase.CANCELLED
    finally:
        release.set()


def test_stale_feedback_aborts_without_new_commands(runtime, monkeypatch):
    policy, observation, clock = runtime
    monkeypatch.setattr("humanoid.policy.autonomy.pick.threading.Thread.start", lambda _: None)
    policy.start("car", observation)
    assert policy(observation).tool_pose is not None
    clock[0] += PICK_CONFIG.feedback_timeout + 0.01
    assert policy(observation).tool_pose is None
    assert policy.phase is PickPhase.FAILED
    assert "feedback stopped" in policy.message


def test_successful_plan_executes_waypoints_through_actions(runtime, monkeypatch):
    policy, observation, clock = runtime
    monkeypatch.setattr("humanoid.policy.autonomy.pick.threading.Thread.start", lambda _: None)
    monkeypatch.setattr("humanoid.policy.autonomy.pick.confirm_scene", lambda *args: None)
    policy.start("car", observation)
    pose = policy.robot.get_tool_command_pose(observation.robot_state.joint_positions).copy()
    plan = PickPlan(
        grasp=GraspProposal(pose=pose, width=0.025, score=0.9, depth=0, height=0.04),
        object_points=np.zeros((30, 3)),
        waypoints=tuple(
            PickWaypoint(name=name, pose=pose, gripper=0.0)
            for name in ["approach", "grasp", "squeeze", "lift"]
        ),
    )
    future = Future()
    future.set_result((plan, MagicMock()))
    policy._future = future
    assert policy(observation).tool_pose is not None
    for _ in plan.waypoints:
        clock[0] += 1 / PICK_CONFIG.rate_hz
        action = policy(observation)
        np.testing.assert_allclose(action.tool_pose.homogeneous, pose.homogeneous)
    assert policy.phase is PickPhase.COMPLETED
    assert not policy.busy
    assert policy(observation).tool_pose is None


def test_missing_camera_or_non_system_start_is_rejected(runtime):
    policy, observation, _ = runtime
    with pytest.raises(RuntimeError, match="SYSTEM"):
        policy.start("car", replace(observation, mode=Mode.IDLE))
    with pytest.raises(RuntimeError, match="RGB"):
        policy.start("car", replace(observation, camera_frames=None))


def test_failed_requests_reuse_perception_robot_and_release_failed_frames(runtime, monkeypatch):
    policy, observation, _ = runtime
    robots, arrays = [], []

    def perceive(_observation, _description, robot, *_args):
        robots.append(robot)
        cloud = np.zeros((100, 3))
        arrays.append(weakref.ref(cloud))
        raise RuntimeError("No object")

    monkeypatch.setattr("humanoid.policy.autonomy.pick.perceive", perceive)
    for _ in range(3):
        policy.start("car", observation)
        future = policy._future
        assert str(future.exception(timeout=3)) == "No object"
        # No GC needed: the stored error does not retain the failed plan's locals.
        assert arrays[-1]() is None
        policy(observation)
        assert policy.phase is PickPhase.FAILED
        assert policy._future is None
    assert all(robot is policy.perception_robot for robot in robots)
    assert policy.perception_robot is not policy.robot


def test_cancel_during_preview_render_cannot_restore_old_result(runtime, monkeypatch):
    policy, _, _ = runtime
    entered, release = threading.Event(), threading.Event()

    def render(_preview):
        entered.set()
        release.wait(2)
        return MagicMock()

    monkeypatch.setattr("humanoid.policy.autonomy.pick.render_preview", render)
    generation = policy._invalidate_preview(clear=True)
    thread = threading.Thread(target=policy._update_preview, args=(MagicMock(), generation))
    thread.start()
    try:
        assert entered.wait(1)
        policy.cancel()
        release.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert policy.preview(3) is None
    finally:
        release.set()
        thread.join(timeout=2)


def test_failed_preview_is_retained_but_next_request_clears_it(runtime, monkeypatch):
    policy, observation, _ = runtime
    monkeypatch.setattr("humanoid.policy.autonomy.pick.threading.Thread.start", lambda _: None)
    image = CameraPreview(connected=True, image="inference", frame_number=1)
    rendered = PickPreview(
        timestamp=10,
        rgb=image,
        depths={3.0: image},
        object_description="car",
        mask_count=1,
        grasp_count=2,
        selected=False,
    )
    monkeypatch.setattr("humanoid.policy.autonomy.pick.render_preview", lambda _: rendered)
    policy._update_preview(MagicMock(), policy._preview_generation)
    policy.fail("No reachable grasp")
    assert policy.preview(3).rgb.image == "inference"
    assert policy.preview(3).object_description == "car"
    policy.start("new object", observation)
    assert policy.preview(3) is None
