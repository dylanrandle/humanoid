"""CLI for publishing events to the orchestrator.

Examples::

    uv run fsm home --target home   # request HOMING with ROBOT_CONFIG.home_position
    uv run fsm home --target rest   # request HOMING with ROBOT_CONFIG.rest_position
    uv run fsm oculus               # switch to OCULUS
    uv run fsm keyboard             # switch to KEYBOARD
    uv run fsm idle                 # switch to IDLE
"""

import argparse

from humanoid.config import ROBOT_CONFIG
from humanoid.orchestrator.client import OrchestratorClient

_HOMING_TARGETS = {
    "home": ROBOT_CONFIG.home_position,
    "rest": ROBOT_CONFIG.rest_position,
}


def main():
    parser = argparse.ArgumentParser(description="Publish an event to the orchestrator")
    sub = parser.add_subparsers(dest="event", required=True)

    home = sub.add_parser("home", help="Request HOMING mode with a target")
    home.add_argument(
        "--target",
        choices=sorted(_HOMING_TARGETS.keys()),
        default="home",
        help="Named target position from ROBOT_CONFIG",
    )

    sub.add_parser("oculus", help="Request OCULUS mode")
    sub.add_parser("keyboard", help="Request KEYBOARD mode")
    sub.add_parser("idle", help="Request IDLE mode")

    args = parser.parse_args()

    client = OrchestratorClient()
    if args.event == "home":
        client.request_homing(_HOMING_TARGETS[args.target])
    elif args.event == "oculus":
        client.request_oculus()
    elif args.event == "keyboard":
        client.request_keyboard()
    elif args.event == "idle":
        client.request_idle()


if __name__ == "__main__":
    main()
