import signal
import sys
from multiprocessing import Process

from humanoid.logger import get_logger
from humanoid.nodes.robot_controller import RobotController
from humanoid.nodes.robot_driver import RobotDriver
from humanoid.nodes.robot_visualizer import RobotVisualizer

logger = get_logger(__name__)


def _start_robot_driver():
    """Standalone function to start robot driver in subprocess."""
    driver = RobotDriver()
    driver.run()


def _start_robot_controller():
    """Standalone function to start robot controller in subprocess."""
    controller = RobotController()
    controller.run()


def _start_robot_visualizer():
    """Standalone function to start robot visualizer in subprocess."""
    visualizer = RobotVisualizer()
    visualizer.run()


class NodeManager:
    def __init__(self):
        self.processes: list[Process] = []
        self.running = False

    def start(self):
        logger.info("Starting node manager")
        self.running = True

        driver_process = Process(target=_start_robot_driver, name="RobotDriver")
        driver_process.start()
        self.processes.append(driver_process)

        controller_process = Process(target=_start_robot_controller, name="RobotController")
        controller_process.start()
        self.processes.append(controller_process)

        visualizer_process = Process(target=_start_robot_visualizer, name="RobotVisualizer")
        visualizer_process.start()
        self.processes.append(visualizer_process)

    def stop(self, timeout=5):
        logger.info("Stopping node manager")
        self.running = False
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=timeout)
                if process.is_alive():
                    process.kill()
        self.processes.clear()

    def wait(self):
        for process in self.processes:
            process.join()


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
