"""Thin policy/environment runner with an operator request mailbox."""

import threading
import time
from dataclasses import asdict, replace

import numpy as np

from humanoid.config import ROBOT_CONFIG
from humanoid.config.pick import GRASPGENX_CONFIG, PICK_CONFIG, PICK_CONTROL_SOCKET
from humanoid.config.serving import SERVING_CONFIG
from humanoid.constants import Topic
from humanoid.environment.realtime import ActionTopics, RealtimeEnvironment
from humanoid.nodes.base import Node
from humanoid.orchestrator.client import OrchestratorClient
from humanoid.policy.autonomy.pick import PickPolicy
from humanoid.serving.client import ModelClient
from humanoid.serving.server import ModelServer
from humanoid.serving.transport import LocalServer, json_arrays, read_json
from humanoid.types.action import Action
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode
from humanoid.types.pick import PickPhase, PickStatus
from humanoid.types.robot import RobotName
from humanoid.types.serving import ServingState, ServingStatus


class ModelServerNode(Node):
    rate_hz = 5.0

    def __init__(self):
        self.server = ModelServer(SERVING_CONFIG)

    def setup(self) -> None:
        self.server.start()

    def step(self) -> None:
        self.server.monitor()
        if self.server.status.state is ServingState.FAILED:
            raise RuntimeError(self.server.status.message)

    def on_close(self) -> None:
        self.server.close()


class PickNode(Node):
    rate_hz = PICK_CONFIG.rate_hz

    def __init__(self):
        if ROBOT_CONFIG.name is not RobotName.TRISKEL:
            raise RuntimeError("Pickup is configured for Triskel")
        self.models = ModelClient(SERVING_CONFIG)
        self.policy = PickPolicy(ROBOT_CONFIG, PICK_CONFIG, GRASPGENX_CONFIG, self.models)
        self.env = RealtimeEnvironment(ActionTopics(tool=Topic.SYSTEM_TOOL_COMMAND))
        self.orchestrator = OrchestratorClient(self.env.publisher)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._models = ServingStatus(state=ServingState.LOADING, message="Loading models")
        self._snapshot = PickStatus(phase=PickPhase.READY, message="Loading models")
        self._pending: str | None = None
        self._waiting: str | None = None
        self._mode_deadline = 0.0
        self._cancel = False
        self._owned = False
        self._control: LocalServer | None = None
        self._poller: threading.Thread | None = None

    def setup(self) -> None:
        self.observation = self.env.reset()
        self._control = LocalServer(PICK_CONTROL_SOCKET, self._handle)
        self._poller = threading.Thread(target=self._monitor_models, daemon=True)
        self._poller.start()

    def _monitor_models(self) -> None:
        while not self._stop.is_set():
            self._models = self.models.status()
            self._stop.wait(0.5)

    def _handle(self, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        payload = read_json(arrays)
        if payload["operation"] == "preview":
            preview = self.policy.preview(float(payload["depth_far"]))
            return json_arrays(asdict(preview) if preview is not None else None)
        with self._lock:
            operation = payload["operation"]
            if operation == "run":
                description = payload.get("object", "")
                if not isinstance(description, str) or not any(c.isalnum() for c in description):
                    raise ValueError("Enter an object name or description")
                if not self._snapshot.models_ready:
                    raise RuntimeError("Wait for the models to finish loading")
                if self._snapshot.busy:
                    raise RuntimeError("A pickup request is already in progress")
                self._pending = description.strip()
                self._snapshot = replace(self._snapshot, busy=True, message="Starting pickup")
            elif operation == "cancel":
                self._cancel = True
            elif operation != "status":
                raise ValueError("Unknown pickup operation")
            return json_arrays(asdict(self._snapshot))

    def step(self) -> None:
        observation = self.env.observe()
        with self._lock:
            pending, cancel = self._pending, self._cancel
            self._pending = None
            self._cancel = False
        try:
            if cancel:
                pending = None
                self._waiting = None
                self.policy.cancel()
            if pending is not None:
                self._request_run(pending, observation)
            self._activate(observation)
            if self.policy.active and self._models.state is not ServingState.READY:
                raise RuntimeError("Model server stopped; pickup aborted")
            action = self.policy(observation)
            self.env.step(action)
        except (ValueError, RuntimeError) as error:
            self._waiting = None
            self.policy.fail(str(error))
        if self._owned and self._waiting is None and not self.policy.active:
            if observation.mode is Mode.SYSTEM:
                self.orchestrator.request_idle()
            self._owned = False
        self.observation = observation
        ready = self._models.state is ServingState.READY
        snapshot = self.policy.status(models_ready=ready)
        if self._waiting is not None:
            snapshot = replace(snapshot, busy=True, message="Activating pickup")
        elif not ready:
            snapshot = replace(snapshot, message=self._models.message)
        with self._lock:
            # A request can arrive during this tick; don't lose its reserved busy state.
            self._snapshot = replace(snapshot, busy=snapshot.busy or self._pending is not None)

    def _request_run(self, description: str, observation: Observation) -> None:
        if observation.mode is not Mode.IDLE:
            raise RuntimeError("Select Idle before running pickup")
        if self._models.state is not ServingState.READY:
            raise RuntimeError(self._models.message)
        self.policy.require_camera(observation)
        q = observation.robot_state.joint_positions
        self.env.step(
            Action(
                tool_pose=self.policy.robot.get_tool_command_pose(q).copy(),
                gripper_positions=q[self.policy.robot.get_gripper_position_indices()].copy(),
            )
        )
        self._waiting = description
        self._mode_deadline = time.monotonic() + 2.0
        self._owned = True
        self.orchestrator.request_system()

    def _activate(self, observation: Observation) -> None:
        if self._waiting is None:
            return
        if observation.mode is Mode.SYSTEM:
            self.policy.start(self._waiting, observation)
            self._waiting = None
        elif observation.mode is not Mode.IDLE or time.monotonic() > self._mode_deadline:
            raise RuntimeError("Pickup mode activation interrupted or timed out")

    def on_close(self) -> None:
        self._stop.set()
        self.policy.cancel("Pick policy stopped")
        if self._owned and self.observation.mode is Mode.SYSTEM:
            self.orchestrator.request_idle()
        if self._control is not None:
            self._control.close()
        if self._poller is not None:
            self._poller.join(timeout=1.0)
        self.env.close()
