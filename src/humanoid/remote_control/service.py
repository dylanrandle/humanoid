"""Command bookkeeping and snapshots; no HTTP or controller writes."""

import copy
import json
import math
import re
import threading
import time
import uuid
from concurrent.futures import CancelledError

from humanoid.policy.remote_control import RemoteControlPolicy
from humanoid.remote_control.commands import motion_command
from humanoid.types.observation import Observation
from humanoid.types.remote_api import (
    CommandRecord,
    CommandRequest,
    CommandState,
    CommandStatus,
    ControlOperation,
    RemoteControlServerConfig,
)
from humanoid.types.remote_control import CommandCancelledError, CommandExpiredError


class RemoteControlService:
    def __init__(self, policy: RemoteControlPolicy, config: RemoteControlServerConfig) -> None:
        self.session_id = uuid.uuid4().hex
        self.policy = policy
        self.config = config
        self._lock = threading.Lock()
        self._commands: dict[str, CommandRecord] = {}
        self._observation: Observation | None = None
        self._closed = False

    def observe(self, observation: Observation) -> None:
        with self._lock:
            self._observation = observation

    def observation(self, *, camera: bool = False) -> Observation:
        with self._lock:
            observation = self._observation
            if self._closed or observation is None:
                raise RuntimeError("Robot control service is not ready.")
        age = time.perf_counter() - observation.robot_state.timestamp
        if not 0 <= age <= self.policy.config.feedback_timeout:
            raise RuntimeError("Robot feedback is stale.")
        if camera:
            frames = observation.camera_frames
            if frames is None or any(
                not 0 <= time.perf_counter() - frame.timestamp <= self.config.camera_timeout_seconds
                for frame in (frames.rgb, frames.depth)
            ):
                raise RuntimeError("No fresh paired RGB/depth observation is available.")
        return observation

    def submit(self, request: CommandRequest) -> CommandStatus:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request.command_id):
            raise ValueError("command_id must be 1-64 letters, digits, underscores, or hyphens.")
        expiry = request.expires_in_seconds
        if (
            type(expiry) not in (int, float)
            or not math.isfinite(expiry)
            or not 0 < expiry <= self.config.max_start_delay_seconds
        ):
            raise ValueError("Invalid command start deadline.")
        arguments = copy.deepcopy(dict(request.arguments))
        fingerprint = json.dumps(
            [request.operation, arguments, expiry], sort_keys=True, allow_nan=False
        )
        command = motion_command(request.operation, arguments)
        if not 0 < command.duration <= self.policy.config.max_duration:
            raise ValueError("Command duration exceeds policy limits.")
        with self._lock:
            if self._closed or self._observation is None:
                raise RuntimeError("Robot control service is not ready.")
            previous = self._commands.get(request.command_id)
            if previous is not None:
                if previous.fingerprint != fingerprint:
                    raise ValueError("command_id was already used for a different command.")
                return self._status(request.command_id, previous)
            # Retain IDs for the entire service session: retries cannot replay motion.
            if len(self._commands) >= self.config.max_commands:
                raise RuntimeError("Command history is full; start a new control service session.")
            if request.operation is ControlOperation.HOLD:
                self.policy.cancel()
            future = self.policy.submit(command, deadline=time.perf_counter() + expiry)
            record = CommandRecord(fingerprint, future)
            self._commands[request.command_id] = record
            return self._status(request.command_id, record)

    def status(self, command_id: str) -> CommandStatus:
        with self._lock:
            return self._status(command_id, self._commands[command_id])

    def cancel(self, command_id: str) -> CommandStatus:
        with self._lock:
            record = self._commands[command_id]
            if not record.result.done():
                self.policy.cancel()
            return self._status(command_id, record)

    @staticmethod
    def _status(command_id: str, record: CommandRecord) -> CommandStatus:
        future = record.result
        if not future.done():
            state = CommandState.RUNNING if future.running() else CommandState.QUEUED
            return CommandStatus(command_id, state)
        try:
            return CommandStatus(command_id, CommandState.COMPLETED, result=future.result())
        except CommandExpiredError as error:
            return CommandStatus(command_id, CommandState.EXPIRED, error=str(error))
        except (CancelledError, CommandCancelledError) as error:
            return CommandStatus(command_id, CommandState.CANCELLED, error=str(error))
        except ValueError as error:
            return CommandStatus(command_id, CommandState.REJECTED, error=str(error))
        except RuntimeError as error:
            return CommandStatus(command_id, CommandState.FAILED, error=str(error))

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self.policy.reset()
