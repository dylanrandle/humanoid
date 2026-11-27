"""Managed motion policy and its private pipe adapter."""

import logging
import queue
import threading
import time
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue

from humanoid.config import ROBOT_CONFIG
from humanoid.config.motion import (
    MOTION_PIPE_CONFIG,
    MOTION_PROTOCOL_CONFIG,
    get_motion_policy_config,
    get_motion_service_config,
)
from humanoid.constants import DEFAULT_LCM_URL, Topic
from humanoid.environment.realtime import ActionTopics, RealtimeEnvironment
from humanoid.logger import setup_queue_logging
from humanoid.middleware.pipe import Pipe
from humanoid.motion.executor import MotionExecutor
from humanoid.nodes.base import Node
from humanoid.orchestrator.client import OrchestratorClient
from humanoid.policy.motion import MotionPolicy
from humanoid.robots.base import Robot
from humanoid.types.motion import (
    MotionEnvelope,
    MotionErrorKind,
    MotionMessageKind,
    MotionPolicyConfig,
)
from humanoid.types.robot import RobotConfig


class MotionPolicyNode(Node):
    """Run bounded motions; pipe communication stays off the control loop."""

    def __init__(
        self,
        connection: Connection,
        motion_session_id: str,
        robot_config: RobotConfig = ROBOT_CONFIG,
        *,
        config: MotionPolicyConfig | None = None,
        lcm_url: str = DEFAULT_LCM_URL,
    ) -> None:
        config = config or get_motion_policy_config(robot_config)
        policy = MotionPolicy(Robot(robot_config), config=config)
        self.policy = policy
        self.rate_hz = 1.0 / config.dt
        self._ready = False
        self.env = RealtimeEnvironment(
            ActionTopics(tool=Topic.AUTONOMY_TOOL_COMMAND, base=Topic.AUTONOMY_BASE_COMMAND),
            lcm_url=lcm_url,
            subscriptions=config.subscriptions,
        )
        self.executor = MotionExecutor(policy, get_motion_service_config())
        self.executor.motion_session_id = motion_session_id
        self.pipe: Pipe[MotionEnvelope] = Pipe(connection, MotionEnvelope, MOTION_PIPE_CONFIG)
        self._disconnected = False
        self._worker = threading.Thread(target=self._serve, daemon=True, name="motion-executor")
        self._worker.start()

    @classmethod
    def run_connected(
        cls,
        log_queue: Queue[logging.LogRecord] | None,
        connection: Connection,
        motion_session_id: str,
    ) -> None:
        """Spawn the motion node with its private connection and session identity."""
        if log_queue is not None:
            setup_queue_logging(log_queue)
        cls.main(connection, motion_session_id)

    def _serve(self) -> None:
        try:
            while not self.pipe.closed.is_set():
                try:
                    message = self.pipe.inbox.get(timeout=MOTION_PIPE_CONFIG.poll_seconds)
                except queue.Empty:
                    continue
                reply = MotionEnvelope(
                    kind=MotionMessageKind.REPLY,
                    motion_session_id=self.executor.motion_session_id,
                    correlation_id=message.correlation_id,
                    ready=self.executor.ready,
                )
                try:
                    if message.motion_session_id != self.executor.motion_session_id:
                        raise RuntimeError("Control session has expired.")
                    if message.kind is MotionMessageKind.SUBMIT and message.request is not None:
                        if message.deadline is None:
                            raise ValueError("A motion start deadline is required.")
                        reply.status = self.executor.submit(
                            message.request, deadline=message.deadline
                        )
                    elif message.kind is MotionMessageKind.STATUS:
                        reply.status = self.executor.status(message.motion_id)
                    elif message.kind is MotionMessageKind.CANCEL:
                        reply.status = self.executor.cancel(message.motion_id)
                    elif message.kind is not MotionMessageKind.PING:
                        raise ValueError("Invalid executor request.")
                except (ValueError, KeyError, RuntimeError) as exc:
                    reply.error = str(exc)
                    reply.error_kind = (
                        MotionErrorKind.INVALID
                        if isinstance(exc, ValueError)
                        else MotionErrorKind.MISSING
                        if isinstance(exc, KeyError)
                        else MotionErrorKind.CONFLICT
                    )
                self.pipe.send(reply)
        except (RuntimeError, ValueError):
            self.pipe.closed.set()

    def setup(self) -> None:
        self.env.reset()
        self._ready = True
        self.executor.observe(self.env.observe())

    def step(self) -> None:
        observation = self.env.observe()
        self.executor.observe(observation)
        if (
            self.pipe.closed.is_set()
            or time.monotonic() - self.pipe.last_received
            > MOTION_PROTOCOL_CONFIG.contact_timeout_seconds
        ):
            self.policy.reset()
            self.env.step(self.policy(observation))
            OrchestratorClient(self.env.publisher).release_autonomy()
            self._disconnected = True
            return
        self.env.step(self.policy(observation))

    def stop_condition(self) -> bool:
        return self._disconnected

    def on_close(self) -> None:
        self.pipe.close()
        self._worker.join(timeout=MOTION_PIPE_CONFIG.join_seconds)
        self.executor.close()
        self.policy.reset()
        try:
            if self._ready:
                self.env.step(self.policy(self.env.observe()))
        finally:
            self.env.close()
