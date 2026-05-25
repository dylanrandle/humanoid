import numpy as np

from humanoid.policy.base import Policy
from humanoid.types.action import Action
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode


def _smooth_step(t: float) -> float:
    return 3 * t**2 - 2 * t**3


def _generate_trajectory(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    speed: float,
    dt: float,
    min_duration: float = 0.1,
) -> list[np.ndarray]:
    max_displacement = float(np.max(np.abs(q_goal - q_start)))
    duration = max(max_displacement / speed, min_duration)
    num_steps = max(int(duration / dt), 1)
    return [
        (1 - _smooth_step(i / num_steps)) * q_start + _smooth_step(i / num_steps) * q_goal
        for i in range(num_steps + 1)
    ]


class HomingPolicy(Policy):
    """Moves the robot to a fixed target joint configuration along a smooth trajectory.

    The trajectory is generated lazily on the first call using q_start from the
    observation, so reset() allows re-homing from a new starting pose.
    """

    mode = Mode.HOMING

    def __init__(
        self,
        target_position: np.ndarray,
        speed: float = 1.0,
        dt: float = 0.01,
    ):
        """Args:
        target_position: Target joint configuration (q) to move to
        speed: Maximum joint speed in rad/s used to size the trajectory duration
        dt: Time step matching the execution rate (1 / rate_hz)
        """
        self.target_position = target_position
        self.speed = speed
        self.dt = dt
        self._trajectory: list[np.ndarray] = []
        self._step = 0

    @property
    def is_done(self) -> bool:
        return len(self._trajectory) > 0 and self._step >= len(self._trajectory)

    def reset(self) -> None:
        self._trajectory = []
        self._step = 0

    def step(self, observation: Observation) -> Action:
        if not self._trajectory:
            q_start = observation.robot_state.joint_positions
            self._trajectory = _generate_trajectory(
                q_start, self.target_position, self.speed, self.dt
            )
            self._step = 0

        idx = min(self._step, len(self._trajectory) - 1)
        self._step += 1
        return Action(joint_positions=self._trajectory[idx])
