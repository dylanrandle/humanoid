import numpy as np
import pink
import pinocchio as pin
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.controllers.manipulability import ManipulabilityTask
from humanoid.robots.base import Robot
from humanoid.types.homing import HomingPreset


@pytest.fixture(scope="module", params=["panda", "triskel", "so101"])
def robot(request):
    return Robot(ROBOT_CONFIGS[request.param])


@pytest.fixture
def task(robot):
    return ManipulabilityTask(
        model=robot.model,
        frame=robot.config.tool.frame,
        controlled_v_indices=np.asarray(
            robot.get_joint_velocity_indices(robot.get_arm_joint_indices())
        ),
        cost=1e-3,
        regularization=1e-6,
    )


@pytest.fixture
def configuration(robot):
    return pink.Configuration(
        robot.model, robot.model.createData(), robot.config.homing_presets[HomingPreset.HOME]
    )


@pytest.mark.parametrize("perturbed", [False, True])
def test_gradient_matches_finite_differences(robot, task, configuration, perturbed):
    if perturbed:
        displacement = np.zeros(robot.model.nv)
        arm_indices = robot.get_joint_velocity_indices(robot.get_arm_joint_indices())
        displacement[arm_indices] = np.linspace(-0.2, 0.2, len(arm_indices))
        configuration.integrate_inplace(displacement, 1.0)
    q = configuration.q.copy()
    gradient = task.compute_jacobian(configuration)[0]
    numerical_gradient = np.zeros(robot.model.nv)
    step = 1e-4
    for index, displacement in enumerate(step * np.eye(robot.model.nv)):
        configuration.update(pin.integrate(robot.model, q, displacement))
        plus = task.compute_log_manipulability(configuration)
        configuration.update(pin.integrate(robot.model, q, -displacement))
        minus = task.compute_log_manipulability(configuration)
        numerical_gradient[index] = (plus - minus) / (2.0 * step)

    np.testing.assert_allclose(gradient, numerical_gradient, rtol=1e-5, atol=2e-6)


def test_objective_step_increases_manipulability(task, configuration):
    before = task.compute_log_manipulability(configuration)
    hessian, linear = task.compute_qp_objective(configuration)
    # Add joint damping, as in the controller, to make the step unique.
    displacement = np.linalg.solve(hessian + 0.01 * np.eye(configuration.model.nv), -linear)

    configuration.integrate_inplace(displacement, 1.0)

    assert task.compute_log_manipulability(configuration) > before


def test_objective_excludes_base_wheels_and_gripper(robot, task, configuration):
    arm_indices = robot.get_joint_velocity_indices(robot.get_arm_joint_indices())
    locked_indices = np.setdiff1d(np.arange(robot.model.nv), arm_indices)
    jacobian = configuration.get_frame_jacobian(robot.config.tool.frame)[:, arm_indices]
    expected = 0.5 * np.linalg.slogdet(jacobian @ jacobian.T + task.regularization * np.eye(6))[1]
    hessian, linear = task.compute_qp_objective(configuration)
    before = task.compute_log_manipulability(configuration)

    assert before == pytest.approx(expected)
    np.testing.assert_array_equal(hessian[locked_indices], 0.0)
    np.testing.assert_array_equal(hessian[:, locked_indices], 0.0)
    np.testing.assert_array_equal(linear[locked_indices], 0.0)

    displacement = np.zeros(robot.model.nv)
    displacement[locked_indices] = 0.5
    configuration.integrate_inplace(displacement, 1.0)

    assert task.compute_log_manipulability(configuration) == pytest.approx(before, abs=1e-9)


def test_gradient_does_not_mutate_configuration(task, configuration):
    q_before = configuration.q.copy()
    jacobian_before = configuration.get_frame_jacobian(task.frame).copy()
    pose_before = configuration.get_transform_frame_to_world(task.frame).copy()

    task.compute_qp_objective(configuration)

    np.testing.assert_array_equal(configuration.q, q_before)
    np.testing.assert_array_equal(configuration.get_frame_jacobian(task.frame), jacobian_before)
    np.testing.assert_array_equal(
        configuration.get_transform_frame_to_world(task.frame).homogeneous,
        pose_before.homogeneous,
    )


def test_singular_configuration_produces_finite_objective(robot, task, configuration):
    eigenvalue_tolerance = 1e-12
    configuration.update(pin.neutral(robot.model))

    hessian, linear = task.compute_qp_objective(configuration)

    assert np.isfinite(task.compute_log_manipulability(configuration))
    assert np.isfinite(hessian).all()
    assert np.isfinite(linear).all()
    assert np.linalg.eigvalsh(hessian).min() >= -eigenvalue_tolerance
