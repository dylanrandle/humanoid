"""Trial-owned digital twin, controller, orchestrator and HTTP robot API."""

import time
import uuid
from contextlib import ExitStack, closing
from dataclasses import replace
from multiprocessing import get_context
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Event
from pathlib import Path

from werkzeug.serving import make_server

from humanoid.api.application import BackendApplication
from humanoid.api.clients.robot import RobotClient
from humanoid.api.server import create_app
from humanoid.api.services.recording import RecordingService
from humanoid.config.selection import ROBOT_CONFIG
from humanoid.config.simulation import DEFAULT_MUJOCO_SIMULATION_CONFIG
from humanoid.constants import Host, Port, private_lcm_url
from humanoid.middleware.publisher import Publisher
from humanoid.nodes.base import Node
from humanoid.nodes.orchestrator import OrchestratorNode
from humanoid.nodes.policy.motion import MotionPolicyNode
from humanoid.nodes.robot.controller import RobotControllerNode
from humanoid.nodes.robot.simulation import SimulationNode
from humanoid.orchestrator.client import OrchestratorClient
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.types.api import ApiEndpoint
from humanoid.types.orchestrator import Mode
from humanoid.types.process import Runtime
from humanoid.types.robot import RobotConfig
from humanoid.types.simulation import MujocoSimulationConfig
from humanoid.types.trial import SimulationTask, TaskScore
from humanoid.utils.loop import loop_at_rate


def _run_node(node: Node, stop: Event) -> None:
    with closing(node):
        node.setup()
        loop_at_rate(
            node.step, node.rate_hz, stop_condition=lambda: stop.is_set() or node.stop_condition()
        )


def _run_controller(robot: RobotConfig, url: str, stop: Event) -> None:
    _run_node(RobotControllerNode(robot, lcm_url=url), stop)


def _run_orchestrator(url: str, stop: Event) -> None:
    _run_node(OrchestratorNode(lcm_url=url), stop)


def _run_api(  # noqa: PLR0913 - isolated trial startup arguments
    robot: RobotConfig,
    url: str,
    stop: Event,
    ready: Connection,
    motion_connection: Connection,
    motion_session_id: str,
) -> None:
    backend = BackendApplication(
        runtime=Runtime.SIM, robot=robot.name, lcm_url=url, serve_quest=False
    )
    backend.create_observation_service(robot)
    backend.motion_service.attach(motion_connection, motion_session_id)
    try:
        with make_server(Host.LOOPBACK, Port.DYNAMIC, create_app(backend), threaded=True) as server:
            server.timeout = 0.1
            with ready:
                ready.send(server.server_port)
            while not stop.is_set():
                server.handle_request()
    finally:
        backend.close()


def _run_motion(
    robot: RobotConfig,
    url: str,
    stop: Event,
    connection: Connection,
    motion_session_id: str,
) -> None:
    _run_node(MotionPolicyNode(connection, motion_session_id, robot, lcm_url=url), stop)


def _run_twin(  # noqa: PLR0913 - process startup arguments
    robot: RobotConfig,
    config: MujocoSimulationConfig,
    task: SimulationTask,
    seed: int,
    url: str,
    stop: Event,
    scoring: Connection,
) -> None:
    with ExitStack() as resources:
        resources.enter_context(scoring)
        engine = NativeMujocoEngine(
            robot, config, configure_scene=lambda s: task.configure_scene(s, seed)
        )
        node = resources.enter_context(
            closing(SimulationNode(robot, config, engine=engine, lcm_url=url))
        )
        if not engine.cameras:
            raise ValueError("The trial requires a configured robot or scene RGB-D camera.")
        node.setup()
        score_at = None

        def tick() -> None:
            nonlocal score_at
            node.step()
            if scoring.poll():
                score_at = engine.data.time + scoring.recv()
            if score_at is not None and engine.data.time >= score_at:
                scoring.send(task.score(engine.data))
                score_at = None

        loop_at_rate(tick, node.rate_hz, stop_condition=stop.is_set)


class TrialStack:
    def __init__(
        self,
        task: SimulationTask,
        seed: int,
        *,
        recording_root: Path | None = None,
        robot_config: RobotConfig = ROBOT_CONFIG,
        simulation_config: MujocoSimulationConfig | None = None,
    ) -> None:
        config = simulation_config or DEFAULT_MUJOCO_SIMULATION_CONFIG
        config = replace(config, scene=task.scene)
        names = tuple(name for name, mount in robot_config.cameras.items() if mount.simulation)
        if not config.enable_cameras or not names:
            raise ValueError("The trial requires a configured RGB-D camera.")
        # All nodes in this run share a private multicast group. Never connect
        # trial control traffic to the operator's hardware or another trial.
        url = self.lcm_url = private_lcm_url()
        ctx = get_context("spawn")
        self._stop = ctx.Event()
        self._children: list[BaseProcess] = []
        self._scoring, sim_end = ctx.Pipe()
        port, api_end = ctx.Pipe(duplex=False)
        motion_parent, motion_child = ctx.Pipe()
        motion_session_id = uuid.uuid4().hex
        self.orchestrator = OrchestratorClient(Publisher(url))
        self._recording = ExitStack()
        self.lcm_recording: str | None = None
        try:
            with sim_end, api_end, port, motion_parent, motion_child:
                if recording_root is not None:
                    logger = self._recording.enter_context(
                        closing(
                            RecordingService(
                                log_dir=recording_root.resolve(),
                                lcm_url=url,
                                robot_config=robot_config,
                            )
                        )
                    )
                    self.lcm_recording = self._recording.enter_context(logger.recording())
                for name, target, args in (
                    (
                        "trial-twin",
                        _run_twin,
                        (robot_config, config, task, seed, url, self._stop, sim_end),
                    ),
                    ("trial-controller", _run_controller, (robot_config, url, self._stop)),
                    ("trial-orchestrator", _run_orchestrator, (url, self._stop)),
                    (
                        "trial-motion",
                        _run_motion,
                        (robot_config, url, self._stop, motion_child, motion_session_id),
                    ),
                    (
                        "trial-api",
                        _run_api,
                        (robot_config, url, self._stop, api_end, motion_parent, motion_session_id),
                    ),
                ):
                    process = ctx.Process(name=name, target=target, args=args, daemon=True)
                    process.start()
                    self._children.append(process)
                if not port.poll(30):
                    raise RuntimeError("Timed out starting the trial robot API.")
                self.client = RobotClient(ApiEndpoint(f"http://{Host.LOOPBACK}:{port.recv()}"))
                self._wait_until_ready(names)
        except BaseException:
            self.close()
            raise

    def _check_nodes(self) -> None:
        for node in self._children:
            if node.exitcode is not None:
                raise RuntimeError(f"{node.name} stopped with exit code {node.exitcode}.")

    def _wait_until_ready(self, names: tuple[str, ...]) -> None:
        deadline = time.monotonic() + 30
        while True:
            self._check_nodes()
            self.orchestrator.request_autonomy()
            try:
                views = self.client.capture()
                if (
                    views.keys() == set(names)
                    and all(
                        obs.mode is Mode.AUTONOMY and frames.registered is not None
                        for obs, frames in views.values()
                    )
                    and self.client.motion_ready()
                ):
                    return
            except RuntimeError as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Trial state and RGB-D capture did not become ready."
                    ) from exc
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out awaiting API and registered RGB-D.")
            time.sleep(0.1)

    def score(self, *, settle_seconds: float = 0) -> TaskScore:
        self._check_nodes()
        self._scoring.send(settle_seconds)
        if not self._scoring.poll(10 + settle_seconds):
            self._check_nodes()
            raise RuntimeError("Timed out waiting for the simulator score.")
        return self._scoring.recv()

    def close(self) -> None:
        try:
            if hasattr(self, "client"):
                self.client.close()
        finally:
            with self._recording:
                self._stop.set()
                for process in self._children:
                    process.join(timeout=5)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=2)
                    if process.is_alive():
                        process.kill()
                        process.join()
                self._scoring.close()
        self._check_exit_codes()

    def _check_exit_codes(self) -> None:
        failures = [p.name for p in self._children if p.exitcode]
        if failures:
            raise RuntimeError(f"Trial nodes exited unsuccessfully: {', '.join(failures)}.")
