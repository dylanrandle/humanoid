"""Adapt RAX tools to the agent-independent robot control API."""

from collections.abc import Mapping

from humanoid.config.remote_control import get_remote_control_policy_config
from humanoid.remote_control.client import RemoteControlClient
from humanoid.robots.base import Robot
from humanoid.types.camera import CameraFrames
from humanoid.types.rax import RaxCommandResult, RaxOperation
from humanoid.types.remote_api import ControlOperation, RemoteControlEndpoint
from humanoid.types.remote_control import MotionResult, RemoteControlPolicyConfig
from humanoid.types.robot import RobotState


class RaxExecution:
    def __init__(
        self,
        robot: Robot,
        endpoint: RemoteControlEndpoint,
        *,
        config: RemoteControlPolicyConfig | None = None,
    ) -> None:
        self.robot = robot
        self.config = config or get_remote_control_policy_config()
        self.client = RemoteControlClient(endpoint)
        self.step_id = 0

    def wait_until_ready(self, timeout: float = 30.0) -> None:
        self.client.wait_until_ready(timeout)

    def state(self) -> RobotState:
        return self.client.state()

    def observe(self) -> CameraFrames:
        return self.client.observe()

    def execute(
        self, operation: RaxOperation | str, arguments: Mapping[str, object]
    ) -> RaxCommandResult:
        operation = RaxOperation(operation)
        if operation is RaxOperation.CANCEL:
            raise ValueError("Use cancel() to interrupt autonomy commands.")
        remote_operation = (
            ControlOperation.SET_GRIPPER
            if operation is RaxOperation.GRIPPER
            else ControlOperation(operation.value)
        )
        result: MotionResult = self.client.execute(
            remote_operation, arguments, timeout=self.config.max_duration + 3.0
        )
        self.step_id = result["step_id"]
        return {
            "operation": operation,
            "reached": result["reached"],
            "position_error_m": result["position_error_m"],
            "orientation_error_rad": result["orientation_error_rad"],
            "gripper_error_rad": result["gripper_error_rad"],
            "elapsed_seconds": result["elapsed_seconds"],
            "step_id": result["step_id"],
        }

    def hold(self) -> None:
        self.client.hold()

    def cancel(self) -> None:
        self.client.cancel()

    def close(self) -> None:
        self.client.close()
