"""Physical cleanup task and privileged expert regressions."""

import os

import mujoco
import numpy as np
import pytest

from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.simulation.tasks.cleanup.navigation import wrap_angle
from humanoid.simulation.tasks.cleanup.task import CleanupTask

MIN_MOBILE_TRAVEL_METERS = 3.0
MAX_PLANNED_TRAVEL_METERS = 4.5
MAX_TRACKING_TRAVEL_METERS = 1.0
MIN_BASE_YAW_RANGE = 2.0


def test_scene_objects_settle_on_floor_and_reset_without_changing_robot_binding():
    task = CleanupTask()
    engine = task.create_engine(0)
    initial = engine.data.qpos.copy()
    engine.step(400)
    for item in task.scene.objects:
        assert engine.model.joint(f"{item.name}_free").type == mujoco.mjtJoint.mjJNT_FREE
        assert engine.data.body(item.name).xpos[2] > task.scene.floor_height
    assert not any(warning.number for warning in engine.data.warning)
    assert len(engine.binding.joints) == len(TRISKEL_CONFIG.actuator_control_modes)
    engine.reset()
    assert engine.data.qpos == pytest.approx(initial)


def test_bin_has_five_collision_surfaces_and_no_roof():
    engine = CleanupTask().create_engine(0)
    surfaces = [engine.model.geom(index).name for index in range(engine.model.ngeom)]
    assert sum(name.startswith("bin_") for name in surfaces) == 5  # noqa: PLR2004
    assert engine.model.geom("floor").contype == 2  # noqa: PLR2004 - external collision bit


@pytest.mark.skipif(
    os.environ.get("HUMANOID_RUN_SLOW_TESTS") != "1",
    reason="Full real-time cleanup episodes; set HUMANOID_RUN_SLOW_TESTS=1 to run.",
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_privileged_expert_physically_fetches_and_deposits_all_items(seed, stack_runtime):
    task = CleanupTask()
    engine = task.create_engine(seed)
    initial_object_joints = engine.data.qpos.copy()
    runtime = stack_runtime(task, seed)
    engine = runtime.engine
    yaws = []

    def observe(state, command, index):
        if index % 5 or task.motion is None:
            return
        motion = task.motion
        # Open-loop commands can diverge from measured state. Check task
        # execution against the measured scene, without a servo-error bound.
        assert task.observation_errors(runtime) == ()
        yaws.append(motion.base_pose()[2])
        if runtime.phase in {"grasp", "place"}:
            target = (
                np.asarray(task.scene.bin_position)
                if runtime.phase == "place"
                else task._item(engine, runtime.target).pose.translation[:2]
            )
            assert (
                abs(wrap_angle(motion.base_pose()[2] - motion.facing_yaw(target)))
                < task.mobile_config.yaw_tolerance
            )

    runtime.observe = observe
    result = task.run(runtime)
    assert result.success, (result.reason, task.state(runtime))
    assert runtime.action_count > 0
    assert task.success(engine)
    assert task.motion is not None
    assert task.motion.base_distance > MIN_MOBILE_TRAVEL_METERS
    assert task.motion.base_distance < task.motion.planned_distance + MAX_TRACKING_TRAVEL_METERS
    assert np.ptp(np.unwrap(yaws)) > MIN_BASE_YAW_RANGE
    # Geometric route optimality is covered in test_navigation. Actual travel
    # also includes small stationary corrections under the unfiltered base
    # controller. Bound that extra travel separately from planned-route length.
    assert MIN_MOBILE_TRAVEL_METERS < task.motion.planned_distance < MAX_PLANNED_TRAVEL_METERS
    assert task.attempts == len(task.scene.objects)
    assert engine.data.qpos != pytest.approx(initial_object_joints)
    assert not any(warning.number for warning in engine.data.warning)
    assert engine.binding.root is not None
    runtime.poll()
    assert runtime.mode.value == "idle"


def test_cleanup_reset_is_seeded_and_clears_task_progress():
    task = CleanupTask()
    first = task.create_engine(13).data.qpos.copy()
    task.attempts = 5
    second = task.create_engine(13).data.qpos.copy()
    assert first == pytest.approx(second)
    assert task.attempts == 0
    different = task.create_engine(14).data.qpos.copy()
    assert first != pytest.approx(different)


def test_bin_success_requires_full_obb_and_settled_velocity():
    task = CleanupTask()
    engine = task.create_engine(0)
    item = task.scene.objects[0]
    joint = engine.model.joint(f"{item.name}_free")
    address = int(joint.qposadr[0])
    qpos = engine.data.qpos[address : address + 7]
    qpos[:] = [
        *task.scene.bin_position,
        task.scene.floor_height + task.scene.wall_thickness + item.size[2] / 2,
        1,
        0,
        0,
        0,
    ]
    mujoco.mj_forward(engine.model, engine.data)
    assert task.in_bin(task.objects(engine)[0])
    assert not task.success(engine)  # Other objects still outside.
    qpos[0] += task.scene.bin_half_size[0] - task.scene.wall_thickness - 0.001
    mujoco.mj_forward(engine.model, engine.data)
    assert not task.in_bin(task.objects(engine)[0])  # Center inside, box on wall.
    qpos[0] = task.scene.bin_position[0]
    engine.data.qvel[int(joint.dofadr[0])] = 0.1
    mujoco.mj_forward(engine.model, engine.data)
    assert not task.in_bin(task.objects(engine)[0])


def test_car_obb_includes_wheels():
    task = CleanupTask()
    engine = task.create_engine(0)
    car = task.objects(engine)[-1]
    assert car.bbox_half_size is not None
    assert car.bbox_half_size[1] > task.scene.objects[-1].size[1] / 2
