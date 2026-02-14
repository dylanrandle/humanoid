import signal
import sys
from multiprocessing import Process

from humanoid.logger import get_logger
from humanoid.nodes.robot_driver import RobotDriver

logger = get_logger(__name__)


class NodeManager:
    def __init__(self):
        self.processes: list[Process] = []
        self.running = False

    def start_robot_driver(self):
        driver = RobotDriver()
        driver.run()

    def start(self):
        logger.info("Starting node manager")
        self.running = True

        driver_process = Process(target=self.start_robot_driver, name="RobotDriver")
        driver_process.start()

        self.processes.append(driver_process)

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
