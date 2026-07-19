import numpy as np
import pytest

from humanoid.state_estimation.root.base import RootState
from humanoid.state_estimation.root.dead_reckoning import DeadReckoningIntegrator

FORWARD_SPEED = 1.0
ELAPSED_SECONDS = 2.0
EXPECTED_DISTANCE = FORWARD_SPEED * ELAPSED_SECONDS
QUARTER_TURN = np.pi / 2


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _state(
    *,
    position: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
) -> RootState:
    return RootState(
        position=(position.copy() if position is not None else np.array([0.0, 0.0, 1.0, 0.0])),
        velocity=velocity.copy() if velocity is not None else np.zeros(3),
    )


def test_integrates_body_velocity_into_planar_position():
    clock = FakeClock()
    integrator = DeadReckoningIntegrator(_state(), clock=clock)
    integrator.update_velocity(np.array([FORWARD_SPEED, 0.0, 0.0]))

    clock.advance(ELAPSED_SECONDS)
    state = integrator.read_state()

    assert state.position[0] == pytest.approx(EXPECTED_DISTANCE)
    assert state.position[1] == pytest.approx(0.0)
    np.testing.assert_allclose(state.velocity, [FORWARD_SPEED, 0.0, 0.0])


def test_transforms_body_velocity_using_current_heading():
    clock = FakeClock()
    initial_position = np.array([0.0, 0.0, np.cos(QUARTER_TURN), np.sin(QUARTER_TURN)])
    integrator = DeadReckoningIntegrator(_state(position=initial_position), clock=clock)
    integrator.update_velocity(np.array([FORWARD_SPEED, 0.0, 0.0]))

    clock.advance(ELAPSED_SECONDS)
    state = integrator.read_state()

    assert state.position[0] == pytest.approx(0.0, abs=1e-12)
    assert state.position[1] == pytest.approx(EXPECTED_DISTANCE)


def test_integrates_yaw_rate_as_normalized_cosine_and_sine():
    clock = FakeClock()
    integrator = DeadReckoningIntegrator(_state(), clock=clock)
    integrator.update_velocity(np.array([0.0, 0.0, QUARTER_TURN]))

    clock.advance(1.0)
    state = integrator.read_state()

    assert state.position[2] == pytest.approx(0.0, abs=1e-12)
    assert state.position[3] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "velocity",
    [np.zeros(2), np.zeros(4), np.array([0.0, np.nan, 0.0])],
)
def test_rejects_invalid_velocity_measurements(velocity):
    clock = FakeClock()
    integrator = DeadReckoningIntegrator(_state(), clock=clock)

    with pytest.raises(ValueError, match="Root velocity"):
        integrator.update_velocity(velocity)
