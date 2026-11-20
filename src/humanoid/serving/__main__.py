"""Run the persistent model server independently of robot motion."""

import time

from humanoid.config.serving import SERVING_CONFIG
from humanoid.serving.server import ModelServer


def main() -> None:
    server = ModelServer(SERVING_CONFIG)
    try:
        server.start()
        while True:
            server.monitor()
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()
