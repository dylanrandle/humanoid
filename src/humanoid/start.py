import signal
import sys
import time
from multiprocessing import Process, set_start_method

from humanoid.constants import Topic
from humanoid.logger import get_logger
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.base import Node
from humanoid.nodes.orchestrator import OrchestratorNode
from humanoid.nodes.policy.homing import HomingNode
from humanoid.nodes.robot_controller import RobotController
from humanoid.nodes.robot_driver import RobotDriver
from humanoid.nodes.robot_visualizer import RobotVisualizer

logger = get_logger(__name__)

TIMEOUT_SECONDS = 10
TIMEOUT_INTERVAL = 0.1


class NodeManager:
    def __init__(self, state_timeout_seconds: float = 10, state_timeout_interval: float = 0.1):
        self.processes: list[Process] = []
        self.running = False
        self.state_timeout_seconds = state_timeout_seconds
        self.state_timeout_interval = state_timeout_interval

    def start(self):
        logger.info("Starting node manager")
        self.running = True

        # Ensure consistent start behavior (spawn) across platforms (e.g. Linux + Darwin)
        set_start_method("spawn")

        self._start_nodes([RobotDriver, RobotController, RobotVisualizer, OrchestratorNode])

        # Wait for first RobotState messages before spawning remaining nodes
        self._wait_for_robot_state()

        self._start_nodes([HomingNode])

    def stop(self, timeout=5):
        logger.info("Stopping node manager")
        self.running = False
        for process in self.processes:
            process.join(timeout=timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout=timeout)
                if process.is_alive():
                    process.kill()
        self.processes.clear()

    def wait(self):
        for process in self.processes:
            process.join()

    def _wait_for_robot_state(self):
        sub = Subscriber(topics=[Topic.ROBOT_STATE])
        elapsed = 0
        while not sub.receive(Topic.ROBOT_STATE):
            logger.debug("Waiting for robot state")
            time.sleep(TIMEOUT_INTERVAL)
            elapsed += TIMEOUT_INTERVAL
            if elapsed > TIMEOUT_SECONDS:
                raise RuntimeError("Timed out waiting for robot state")

        sub.close()

    def _start_nodes(self, nodes: list[type[Node]]):
        for node in nodes:
            p = Process(target=node.main, name=node.__name__)
            p.start()
            self.processes.append(p)


def main():
    manager = NodeManager()

    def signal_handler(sig, frame):
        manager.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    manager.start()
    manager.wait()


if __name__ == "__main__":
    main()
