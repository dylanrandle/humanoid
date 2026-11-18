"""Run the robot API against an existing simulation or hardware stack."""

import argparse
import os
from dataclasses import replace

from humanoid.config.remote_api import get_remote_control_server_config
from humanoid.remote_control.server import RemoteControlServerNode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    token = os.environ.get("HUMANOID_CONTROL_TOKEN", "")
    if not token:
        parser.error("Set HUMANOID_CONTROL_TOKEN before starting the service.")
    config = replace(get_remote_control_server_config(token), host=args.host, port=args.port)
    RemoteControlServerNode.main(server_config=config)


if __name__ == "__main__":
    main()
