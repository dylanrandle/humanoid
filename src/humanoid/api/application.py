"""Application service coordinating processes and orchestrator requests."""

import threading
import uuid
from functools import partial

from humanoid.api.services.agent_tasks import AgentTaskService
from humanoid.api.services.cameras import CameraFrameService
from humanoid.api.services.logs import LogService
from humanoid.api.services.modes import ModeService
from humanoid.api.services.motion import MotionService
from humanoid.api.services.observations import ObservationService
from humanoid.api.services.recording import RecordingService
from humanoid.api.services.replay import ReplayManager
from humanoid.api.services.runtime import RuntimeService
from humanoid.api.services.streaming import StreamingService
from humanoid.config import ROBOT_CONFIGS
from humanoid.config.backend import (
    INPUT_PROCESSES,
)
from humanoid.config.logs import get_log_path
from humanoid.config.quest import get_quest_config
from humanoid.constants import DEFAULT_LCM_URL, Topic
from humanoid.hardware.teleop.quest.input import QuestInputState
from humanoid.hardware.teleop.quest.server import QuestServer
from humanoid.logger import get_logger
from humanoid.middleware.publisher import Publisher
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.groups import process_display_name
from humanoid.nodes.manager import NodeManager
from humanoid.orchestrator.client import OrchestratorClient
from humanoid.orchestrator.monitor.mode import ModeMonitor
from humanoid.recording import RecordingCatalog
from humanoid.telemetry.actuator import ActuatorHealthMonitor
from humanoid.telemetry.host import HostResourceMonitor
from humanoid.telemetry.lcm import LCMRateMonitor
from humanoid.telemetry.node import NODE_RATE_SUBSCRIBER_QUEUE_SIZE, NodeRateMonitor
from humanoid.telemetry.streaming import StreamTelemetry
from humanoid.types.agent import AgentPanelStatus
from humanoid.types.api import BackendError, BackendErrorMessage, BackendStatus, SafetyContext
from humanoid.types.logs import LogSnapshot
from humanoid.types.observation import CameraFrameCaptures, Observation
from humanoid.types.orchestrator import (
    Mode,
    OrchestratorRequest,
)
from humanoid.types.process import ProcessAction, ProcessName, Runtime
from humanoid.types.quest import QuestCameraSources, QuestVideoServices
from humanoid.types.recording import RecordingOwner, RecordingState, RecordingStatus
from humanoid.types.robot import RobotConfig, RobotName
from humanoid.types.simulation import SimulationScene

logger = get_logger(__name__)


class BackendApplication:
    """Coordinates process lifecycle and orchestrator requests."""

    def __init__(
        self,
        runtime: Runtime | None = None,
        robot: RobotName | None = None,
        lcm_url: str = DEFAULT_LCM_URL,
        *,
        serve_quest: bool = True,
    ) -> None:
        # Runtime state and supervision.
        self.lcm_url = lcm_url
        self.runtime_id = uuid.uuid4().hex
        self.camera_generation = 0
        quest_config = get_quest_config()
        self.quest_input = QuestInputState(quest_config.input_timeout)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._closed = False
        self._supervisor: threading.Thread | None = None

        # Services and process management.
        self.log_service = LogService(get_log_path())
        self.motion_service = MotionService()
        self.recording_service = RecordingService(lcm_url=lcm_url)
        self.replay_manager = ReplayManager()
        self.recording_catalog = RecordingCatalog()
        self.node_manager = NodeManager(
            runtime=runtime,
            robot=robot,
            log_queue=self.log_service.queue,
            motion=self.motion_service,
        )
        self.orchestrator_client = OrchestratorClient(Publisher(lcm_url))

        # Services initialized on first use.
        self._observation_service: ObservationService | None = None
        self._streaming_service: StreamingService | None = None
        self._camera_frame_services: dict[str, CameraFrameService] = {}

        # Monitors and telemetry.
        self.mode_monitor = ModeMonitor(Subscriber(url=lcm_url, topics=[Topic.ORCHESTRATOR_MODE]))
        self.node_rate_monitor = NodeRateMonitor(
            Subscriber(
                url=lcm_url,
                topics=[Topic.NODE_RATE],
                queue_size=NODE_RATE_SUBSCRIBER_QUEUE_SIZE,
            )
        )
        self.actuator_health_monitor = ActuatorHealthMonitor(
            Subscriber(url=lcm_url, topics=[Topic.ACTUATOR_HEALTH])
        )
        self.host_resource_monitor = HostResourceMonitor()
        self.lcm_rate_monitor = LCMRateMonitor(url=lcm_url)
        self.stream_telemetry = StreamTelemetry()

        # Coordinating services depend on the components above.
        self.mode_service = ModeService(self.orchestrator_client, self.mode_monitor)
        self.runtime_service = RuntimeService(
            self.node_manager,
            self.mode_service,
            self.recording_service,
            self.replay_manager,
            self.recording_catalog,
            self.status,
            self._reset_runtime,
            lambda: self.agent_task_service.stop(tasks_only=True),
            self._lock,
            self.motion_service.detach,
            lambda: self.agent_task_service.stop_motion_task(),  # noqa: PLW0108 - late-bound collaborator
        )
        self.agent_task_service = AgentTaskService(
            status=self.status,
            require_configuration=self.runtime_service.require_expected_configuration,
            observation=lambda: self.observation_service.observation(),  # noqa: PLW0108 - lazy sensors
            runtime_id=lambda: self.runtime_id,
            motion_session_id=lambda: self.motion_service.motion_session_id,
            release_motion=self.orchestrator_client.release_autonomy,
            lock=self._lock,
        )

        self.quest_server = (
            QuestServer(
                quest_config,
                self.quest_input,
                QuestVideoServices(
                    cameras=self._quest_camera_sources,
                    offer=lambda offer, source: self.streaming_service.offer(offer, source),  # noqa: PLW0108 - lazy service
                    remove=lambda stream_id: self.streaming_service.remove(stream_id),  # noqa: PLW0108 - lazy service
                    metrics=self.stream_telemetry.snapshot,
                ),
            )
            if serve_quest
            else None
        )

    def _quest_camera_sources(self) -> QuestCameraSources:
        """Snapshot readers and generation together; retain lazy sensor initialization."""
        with self._lock:
            return QuestCameraSources(
                generation=self.camera_generation,
                sources={
                    name: partial(camera.read_frame, "rgb")
                    for name, camera in self.camera_frame_services.items()
                },
            )

    def status(self) -> BackendStatus:
        with self._lock:
            processes, replay = self.runtime_service.reconcile()
            self.recording_service.reconcile_runtime(self.runtime_id)
            stack = processes[ProcessName.STACK]
            orchestrator = self.mode_service.state()
            stack_running = stack.running
            replay_running = processes[ProcessName.REPLAY].running or replay.running
            if orchestrator.connected and not (stack_running or replay_running):
                raise BackendError(BackendErrorMessage.EXTERNAL_STACK)
            return BackendStatus(
                runtime=self.node_manager.runtime,
                robot=self.node_manager.robot,
                robots=list(RobotName),
                scene=self.node_manager.scene,
                scenes=list(SimulationScene),
                processes=processes,
                node_rates=self.node_rate_monitor.snapshot(self.node_manager.active_nodes()),
                lcm_channels=self.lcm_rate_monitor.snapshot(),
                host_resources=self.host_resource_monitor.snapshot(),
                actuator_health=self.actuator_health_monitor.snapshot(),
                recording=self.recording_service.snapshot(),
                recordings=self.recording_catalog.list(),
                replay=replay,
                orchestrator=orchestrator,
            )

    @property
    def observation_service(self) -> ObservationService:
        with self._lock:
            if self._observation_service is None:
                return self.create_observation_service(ROBOT_CONFIGS[self.node_manager.robot])
            return self._observation_service

    @property
    def camera_frame_services(self) -> dict[str, CameraFrameService]:
        with self._lock:
            for name, mount in ROBOT_CONFIGS[self.node_manager.robot].cameras.items():
                source = (
                    mount.simulation if self.node_manager.runtime is Runtime.SIM else mount.hardware
                )
                if source is not None and name not in self._camera_frame_services:
                    self._camera_frame_services[name] = CameraFrameService(
                        camera_name=name, lcm_url=self.lcm_url
                    )
            return self._camera_frame_services.copy()

    @property
    def streaming_service(self) -> StreamingService:
        with self._lock:
            if self._streaming_service is None:
                self._streaming_service = StreamingService(self.stream_telemetry)
            return self._streaming_service

    def create_observation_service(self, robot_config: RobotConfig) -> ObservationService:
        """Create the cached observation service, closing any previous instance."""
        with self._lock:
            if self._observation_service is not None:
                self._observation_service.close()
                self._observation_service = None
            self._observation_service = ObservationService(robot_config, lcm_url=self.lcm_url)
            return self._observation_service

    def observe(self) -> tuple[str, Observation]:
        with self._lock:
            runtime_id = self.runtime_id
            return runtime_id, self.observation_service.observation()

    def capture(self) -> tuple[str, CameraFrameCaptures]:
        with self._lock:
            runtime_id = self.runtime_id
            return runtime_id, self.observation_service.capture()

    def logs(self, after: int | None = None) -> LogSnapshot:
        return self.log_service.snapshot(after)

    def _reset_sensor_services(self) -> None:
        self.camera_generation += 1
        if self._streaming_service is not None:
            self._streaming_service.reset()
        if self._observation_service is not None:
            self._observation_service.close()
            self._observation_service = None
        for monitor in self._camera_frame_services.values():
            monitor.close()
        self._camera_frame_services.clear()

    def request_mode(self, request: OrchestratorRequest) -> BackendStatus:
        with self._lock:
            current = self.status()
            self.runtime_service.require_ready_stack(current)
            if current.orchestrator.mode is not request.mode:
                self.agent_task_service.stop(tasks_only=True)

            input_process = INPUT_PROCESSES.get(request.mode)
            if input_process is not None and not current.processes[input_process].running:
                raise BackendError(
                    f"Start {process_display_name(input_process)} before activating its mode."
                )

            self.mode_service.request(request, ROBOT_CONFIGS[self.node_manager.robot])
            return self.status()

    def set_recording(self, action: ProcessAction) -> BackendStatus:
        with self._lock:
            current = self.status()
            self.runtime_service.require_ready_stack(current)
            self._request_recording(action)
            return self.status()

    def recording_status(self) -> RecordingStatus:
        return self.recording_service.snapshot()

    def toggle_recording(self) -> RecordingStatus:
        with self._lock:
            self.runtime_service.require_ready_stack(self.status())
            state = self.recording_service.snapshot().state
            action = ProcessAction.STOP if state is RecordingState.RUNNING else ProcessAction.START
            self._request_recording(action)
            return self.recording_service.snapshot()

    def _request_recording(self, action: ProcessAction) -> None:
        if action is ProcessAction.START:
            self.recording_service.configure(ROBOT_CONFIGS[self.node_manager.robot])
            status = self.recording_service.start()
        else:
            status = self.recording_service.stop()
        if status.state is RecordingState.FAILED:
            raise BackendError(status.error or "Recording failed.")

    def set_episode_recording(
        self, action: ProcessAction, episode_id: str, runtime_id: str
    ) -> RecordingStatus:
        with self._lock:
            owner = RecordingOwner(runtime_id, episode_id)
            if action is ProcessAction.STOP:
                status = self.recording_service.stop(owner=owner)
            else:
                current = self.status()
                if runtime_id != self.runtime_id:
                    raise BackendError("Robot runtime has expired.")
                if self.node_manager.runtime is not Runtime.REAL:
                    raise BackendError("Hardware recording requires the real stack.")
                self.runtime_service.require_ready_stack(current)
                self.recording_service.configure(ROBOT_CONFIGS[self.node_manager.robot])
                status = self.recording_service.start(owner=owner)
            if status.state is RecordingState.FAILED:
                raise BackendError(status.error or "Recording failed.")
            return status

    def start(self) -> None:
        """Start Quest transport and supervise health independently of browser polling."""
        if self._closed:
            raise RuntimeError("Backend is closed")
        if self._supervisor is None:
            try:
                if self.quest_server is not None:
                    self.quest_server.start()
            except BaseException:
                self.close()
                raise
            self._supervisor = threading.Thread(
                target=self._supervise, daemon=True, name="backend-supervisor"
            )
            self._supervisor.start()

    def _supervise(self) -> None:
        while not self._stop.wait(0.1):
            try:
                self.status()
            except BackendError:
                continue
            except Exception:
                logger.exception("Backend supervision failed")

    def _reset_runtime(self) -> None:
        self.quest_input.clear()
        self.runtime_id = uuid.uuid4().hex
        self.recording_service.reconcile_runtime(self.runtime_id)
        self.actuator_health_monitor.reset()
        self.lcm_rate_monitor.reset()
        self._reset_sensor_services()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._supervisor is not None:
            self._supervisor.join(timeout=2)
        try:
            if self.quest_server is not None:
                self.quest_server.close()
        except Exception:
            logger.exception("Failed to close Quest server cleanly")
        try:
            self.agent_task_service.stop()
            self.runtime_service.stop_stack()
        except Exception:
            logger.exception("Failed to stop the main stack cleanly")
            self.node_manager.close()
        finally:
            self.motion_service.close()
            self.mode_monitor.close()
            self.recording_service.close()
            self.node_rate_monitor.close()
            self.lcm_rate_monitor.close()
            self.actuator_health_monitor.close()
            with self._lock:
                self._reset_sensor_services()
            self.replay_manager.close()
            self.log_service.close()
            if self._streaming_service is not None:
                self._streaming_service.close()

    def set_runtime(self, runtime: Runtime, safety: SafetyContext) -> BackendStatus:
        return self.runtime_service.set_runtime(runtime, safety)

    def set_scene(self, scene: SimulationScene, safety: SafetyContext) -> BackendStatus:
        return self.runtime_service.set_scene(scene, safety)

    def set_robot(self, robot: RobotName, safety: SafetyContext) -> BackendStatus:
        return self.runtime_service.set_robot(robot, safety)

    def start_process(self, name: ProcessName, safety: SafetyContext) -> BackendStatus:
        return self.runtime_service.start_process(name, safety)

    def stop_process(self, name: ProcessName) -> BackendStatus:
        return self.runtime_service.stop_process(name)

    def start_replay(self, recording_id: str, safety: SafetyContext) -> BackendStatus:
        return self.runtime_service.start_replay(recording_id, safety)

    def stop_replay(self) -> BackendStatus:
        return self.runtime_service.stop_replay()

    def run_agent(
        self, instruction: str, model: str, mode: Mode, safety: SafetyContext
    ) -> AgentPanelStatus:
        return self.agent_task_service.run(instruction, model, mode, safety)

    def stop_agent(self) -> AgentPanelStatus:
        return self.agent_task_service.stop()

    def release_autonomy(self, motion_session_id: str) -> None:
        with self._lock:
            self.motion_service.require_session(motion_session_id)
            self.orchestrator_client.release_autonomy()
