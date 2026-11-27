"""Command bookkeeping and snapshots; no HTTP or controller writes."""

import copy
import json
import math
import re
import threading
import time
import uuid
from concurrent.futures import CancelledError

from humanoid.motion.validation import motion_target
from humanoid.policy.motion import MotionPolicy
from humanoid.types.motion import (
    ControlOperation,
    MotionCancelledError,
    MotionExpiredError,
    MotionRecord,
    MotionRequest,
    MotionServiceConfig,
    MotionState,
    MotionStatus,
)
from humanoid.types.observation import Observation


class MotionExecutor:
    def __init__(self, policy: MotionPolicy, config: MotionServiceConfig) -> None:
        self.motion_session_id = uuid.uuid4().hex
        self.policy = policy
        self.config = config
        self._lock = threading.Lock()
        self._motions: dict[str, MotionRecord] = {}
        self._observation: Observation | None = None
        self._closed = False

    def observe(self, observation: Observation) -> None:
        with self._lock:
            self._observation = observation

    @property
    def ready(self) -> bool:
        with self._lock:
            if self._closed or self._observation is None:
                return False
            try:
                self.policy.validate_observation(self._observation)
            except (ValueError, RuntimeError):
                return False
            return True

    def submit(self, request: MotionRequest, *, deadline: float | None = None) -> MotionStatus:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request.motion_id):
            raise ValueError("motion_id must be 1-64 letters, digits, underscores, or hyphens.")
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
        command = motion_target(request.operation, arguments)
        if not 0 < command.duration <= self.policy.config.max_duration:
            raise ValueError("Command duration exceeds policy limits.")
        with self._lock:
            if self._closed or self._observation is None:
                raise RuntimeError("Robot control service is not ready.")
            previous = self._motions.get(request.motion_id)
            if previous is not None:
                if previous.fingerprint != fingerprint:
                    raise ValueError("motion_id was already used for a different command.")
                return self._status(request.motion_id, previous)
            # Retain IDs for the entire service session: retries cannot replay motion.
            if len(self._motions) >= self.config.max_motions:
                raise RuntimeError("Command history is full; start a new control service session.")
            # Check admission here as well as on the execution tick. A command
            # submitted in Idle must never become eligible on the next mode switch.
            self.policy.validate_observation(self._observation)
            if request.operation is ControlOperation.HOLD:
                self.policy.cancel()
            future = self.policy.submit(
                command,
                deadline=time.perf_counter() + expiry if deadline is None else deadline,
            )
            record = MotionRecord(fingerprint, future)
            self._motions[request.motion_id] = record
            return self._status(request.motion_id, record)

    def status(self, motion_id: str) -> MotionStatus:
        with self._lock:
            return self._status(motion_id, self._motions[motion_id])

    def cancel(self, motion_id: str) -> MotionStatus:
        with self._lock:
            record = self._motions[motion_id]
            if not record.result.done():
                self.policy.cancel()
            return self._status(motion_id, record)

    @staticmethod
    def _status(motion_id: str, record: MotionRecord) -> MotionStatus:
        future = record.result
        if not future.done():
            state = MotionState.RUNNING if future.running() else MotionState.QUEUED
            return MotionStatus(motion_id, state)
        try:
            return MotionStatus(motion_id, MotionState.COMPLETED, result=future.result())
        except MotionExpiredError as error:
            return MotionStatus(motion_id, MotionState.EXPIRED, error=str(error))
        except (CancelledError, MotionCancelledError) as error:
            return MotionStatus(motion_id, MotionState.CANCELLED, error=str(error))
        except ValueError as error:
            return MotionStatus(motion_id, MotionState.REJECTED, error=str(error))
        except RuntimeError as error:
            return MotionStatus(motion_id, MotionState.FAILED, error=str(error))

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self.policy.reset()
