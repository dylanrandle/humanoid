"""Mode requests and observed acknowledgements; no process or recording ownership."""

import time
from collections.abc import Callable

from humanoid.config.backend import (
    MODE_TRANSITION_POLL_INTERVAL_SECONDS,
    MODE_TRANSITION_TIMEOUT_SECONDS,
    PARAMETERIZED_REQUEST_TIMEOUT_SECONDS,
)
from humanoid.orchestrator.client import OrchestratorClient
from humanoid.orchestrator.monitor.mode import ModeMonitor
from humanoid.types.homing import HomingPreset
from humanoid.types.orchestrator import (
    Mode,
    OrchestratorParameter,
    OrchestratorRequest,
    OrchestratorState,
)
from humanoid.types.robot import RobotConfig


class ModeService:
    def __init__(self, client: OrchestratorClient, monitor: ModeMonitor) -> None:
        self.orchestrator_client = client
        self.mode_monitor = monitor
        self._parameterized_request: OrchestratorRequest | None = None
        self._parameterized_request_at: float | None = None
        self._parameterized_mode_seen = False

    def request(self, request: OrchestratorRequest, robot: RobotConfig) -> None:
        if request.mode is Mode.HOMING:
            preset = request.parameters[OrchestratorParameter.PRESET]
            assert isinstance(preset, HomingPreset)
            self.orchestrator_client.request_homing(robot.homing_presets[preset])
            self._select_parameterized_request(request)
            return
        methods: dict[Mode, Callable[[], None]] = {
            Mode.IDLE: self.orchestrator_client.request_idle,
            Mode.KEYBOARD: self.orchestrator_client.request_keyboard,
            Mode.OCULUS: self.orchestrator_client.request_oculus,
            Mode.SYSTEM: self.orchestrator_client.request_system,
            Mode.AUTONOMY: self.orchestrator_client.request_autonomy,
        }
        methods[request.mode]()
        self.clear()

    def state(self) -> OrchestratorState:
        mode_status = self.mode_monitor.snapshot()
        mode = mode_status.mode

        request = self._parameterized_request
        if request is not None:
            if mode is request.mode:
                self._parameterized_mode_seen = True
            elif self._parameterized_mode_seen or self._parameterized_request_expired():
                self.clear()
        request = self._parameterized_request

        return OrchestratorState(
            mode=mode_status.mode,
            connected=mode_status.connected,
            age_seconds=mode_status.age_seconds,
            parameters=dict(request.parameters) if request is not None else {},
        )

    def _select_parameterized_request(self, request: OrchestratorRequest) -> None:
        self._parameterized_request = request
        self._parameterized_request_at = time.monotonic()
        self._parameterized_mode_seen = False

    def clear(self) -> None:
        self._parameterized_request = None
        self._parameterized_request_at = None
        self._parameterized_mode_seen = False

    def _parameterized_request_expired(self) -> bool:
        return (
            self._parameterized_request_at is not None
            and time.monotonic() - self._parameterized_request_at
            > PARAMETERIZED_REQUEST_TIMEOUT_SECONDS
        )

    def wait_for_mode(self, mode: Mode) -> bool:
        deadline = time.monotonic() + MODE_TRANSITION_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.mode_monitor.snapshot().mode is mode:
                return True
            time.sleep(MODE_TRANSITION_POLL_INTERVAL_SECONDS)
        return False
