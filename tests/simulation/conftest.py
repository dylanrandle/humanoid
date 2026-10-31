"""Integration fixtures launch the same independent processes as the operator stack."""

import multiprocessing
import os
import signal
import time
import uuid
from types import SimpleNamespace

import pytest

from humanoid.config.collection import DEFAULT_COLLECTION_CONFIG
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.nodes.orchestrator import OrchestratorNode
from humanoid.nodes.robot.controller import RobotControllerNode
from humanoid.nodes.robot.logger import RobotLoggerNode
from humanoid.nodes.robot.simulation import MujocoSimulationNode
from humanoid.simulation.client import SimulationClient
from humanoid.simulation.collection.collector import EpisodeCollector


def _run_node(node_type, url, ready):
    node = (
        node_type(lcm_url=url)
        if node_type is OrchestratorNode
        else node_type(robot_config=TRISKEL_CONFIG, lcm_url=url)
    )
    ready.set()
    node.run()


@pytest.fixture
def lcm_url():
    if configured := os.environ.get("HUMANOID_TEST_LCM_URL"):
        return configured
    identifier = uuid.uuid4().int
    group = f"239.254.{(identifier >> 8) & 255}.{identifier & 255}"
    return f"udpm://{group}:{20000 + identifier % 30000}?ttl=1"


@pytest.fixture
def running_stack(lcm_url):
    context = multiprocessing.get_context("spawn")
    processes = []
    ready_events = []
    try:
        for node_type in (
            MujocoSimulationNode,
            RobotControllerNode,
            OrchestratorNode,
            RobotLoggerNode,
        ):
            ready = context.Event()
            process = context.Process(
                target=_run_node, args=(node_type, lcm_url, ready), name=node_type.__name__
            )
            process.start()
            processes.append(process)
            ready_events.append(ready)
        for ready in ready_events:
            assert ready.wait(20), "Node initialization timed out."
        client = SimulationClient(TRISKEL_CONFIG, lcm_url)
        try:
            client.require_idle()
            yield SimpleNamespace(client=client, processes=processes, url=lcm_url)
        finally:
            client.close()
    finally:
        for process in processes:
            if process.is_alive():
                os.kill(process.pid, signal.SIGINT)
        deadline = time.monotonic() + 8
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
            if process.is_alive():
                process.terminate()
                process.join(2)
            if process.is_alive():
                process.kill()
                process.join(2)


@pytest.fixture
def stack_runtime(running_stack):
    def create(task, seed=0):
        client = running_stack.client
        client.reset(task, seed, DEFAULT_COLLECTION_CONFIG)
        client.activate()
        return client

    return create


@pytest.fixture
def episode_collector(running_stack, tmp_path):
    def create(task, config=DEFAULT_COLLECTION_CONFIG):
        return EpisodeCollector(task, running_stack.client, tmp_path, config)

    return create
