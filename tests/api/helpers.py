"""Real unified HTTP routes and pipe transport with explicitly supplied feedback."""

import threading
from collections.abc import Callable
from multiprocessing import get_context
from unittest.mock import MagicMock

from werkzeug.serving import make_server

from humanoid.api.application import BackendApplication
from humanoid.api.server import create_app
from humanoid.api.services.motion import MotionService
from humanoid.api.services.observations import ObservationService
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.constants import private_lcm_url
from humanoid.motion.executor import MotionExecutor
from humanoid.nodes.policy.motion import MotionPolicyNode


class ApiTestServer:
    def __init__(
        self,
        service: MotionExecutor | ObservationService,
        release: Callable[[], None] | None = None,
    ) -> None:
        self.backend = MagicMock(spec=BackendApplication)
        self.backend.motion_service = MotionService()
        self.backend.runtime_id = "test-runtime"
        self.node: MotionPolicyNode | None = None
        if isinstance(service, MotionExecutor):
            observations = ObservationService(TRISKEL_CONFIG, collect=False)
            assert service._observation is not None
            observations.observe(service._observation)
            parent, child = get_context("spawn").Pipe()
            self.node = MotionPolicyNode(
                child, service.motion_session_id, TRISKEL_CONFIG, lcm_url=private_lcm_url()
            )
            self.node.executor = service
            self.node.policy = service.policy
            self.backend.motion_service.attach(parent, service.motion_session_id)
        else:
            observations = service
        self.backend.observation_service = observations
        self.backend.orchestrator_client = MagicMock()
        self.backend.release_autonomy.side_effect = lambda session: (
            BackendApplication.release_autonomy(self.backend, session)
        )
        self.backend._lock = threading.RLock()
        self.backend.observe.side_effect = lambda: BackendApplication.observe(self.backend)
        self.backend.capture.side_effect = lambda: BackendApplication.capture(self.backend)
        if release is not None:
            self.backend.orchestrator_client.release_autonomy.side_effect = release
        self.server = make_server("127.0.0.1", 0, create_app(self.backend), threaded=True)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        if self.thread.is_alive():
            self.server.shutdown()
            self.thread.join(timeout=2)
        self.server.server_close()
        self.backend.motion_service.close()
        self.backend.observation_service.close()
        if self.node is not None:
            self.node.close()
