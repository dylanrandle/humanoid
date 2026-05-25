import numpy as np

from humanoid.config import ROBOT_CONFIG
from humanoid.policy.base import Policy
from humanoid.robots.base import Robot
from humanoid.types.action import Action
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import RobotConfig
from humanoid.types.servo import ServoControlMode


class HomingPolicy(Policy):
    """Moves the robot to a target joint configuration along a smooth trajectory.

    The target is provided dynamically via :meth:`set_target`; the trajectory is
    then generated lazily on the next call using q_start from the observation.
    Without a target, the policy is idle and returns an empty Action.
    """

    mode = Mode.HOMING

    def __init__(
        self,
        speed: float = 1.0,
        dt: float = 0.01,
        robot_config: RobotConfig = ROBOT_CONFIG,
    ):
        """Args:
        speed: Maximum joint speed in rad/s used to size the trajectory duration
        dt: Time step matching the execution rate (1 / rate_hz)
        robot_config: Robot configuration used to identify which joints are
            position-controlled (the only ones homing should move).
        """
        self._target_position: np.ndarray | None = None
        self.speed = speed
        self.dt = dt
        self._trajectory: list[np.ndarray] = []
        self._step = 0
        self._position_controlled_indices = _position_controlled_position_indices(robot_config)

    def set_target(self, target_position: np.ndarray) -> None:
        """Set or replace the homing target; regenerates the trajectory on next call."""
        self._target_position = target_position.copy()
        self._trajectory = []
        self._step = 0

    @property
    def is_done(self) -> bool:
        return len(self._trajectory) > 0 and self._step >= len(self._trajectory)

    def reset(self) -> None:
        # Only clear the trajectory: keep the target so reactivation continues
        # toward the same goal from a fresh q_start.
        self._trajectory = []
        self._step = 0

    def step(self, observation: Observation) -> Action:
        if self._target_position is None or self.is_done:
            return Action()

        if not self._trajectory:
            if observation.robot_joint_command:
                q_start = observation.robot_joint_command.joint_positions
            else:
                q_start = observation.robot_state.joint_positions

            # Hold every joint at q_start except for position-controlled ones,
            # which move to the configured home target.
            q_end = q_start.copy()
            q_end[self._position_controlled_indices] = self._target_position[
                self._position_controlled_indices
            ]

            self._trajectory = _generate_trajectory(q_start, q_end, self.speed, self.dt)
            self._step = 0

        idx = min(self._step, len(self._trajectory) - 1)
        self._step += 1
        return Action(joint_positions=self._trajectory[idx])


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


def _position_controlled_position_indices(robot_config: RobotConfig) -> list[int]:
    """Return position-vector indices for joints that are position-controlled."""
    robot = Robot(robot_config)
    indices: list[int] = []
    for joint_idx, servo_id in robot_config.joint_idx_to_servo_id.items():
        if robot_config.servo_control_modes[servo_id] is not ServoControlMode.POSITION:
            continue
        indices.append(robot.joint_idx_to_position_idx(joint_idx))
    return indices
