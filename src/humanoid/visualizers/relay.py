"""Read-only SSH helper, run with the robot installation's existing Python and LCM."""

import base64
import select
import sys

import lcm

from humanoid.constants import DEFAULT_LCM_URL


def main(channels: list[str]) -> None:
    bus = lcm.LCM(DEFAULT_LCM_URL)

    def forward(channel: str, packet: bytes) -> None:
        sys.stdout.buffer.write(channel.encode("ascii") + b" " + base64.b64encode(packet) + b"\n")
        sys.stdout.buffer.flush()

    for channel in channels:
        bus.subscribe(channel, forward).set_queue_capacity(1)
    while True:
        ready, _, _ = select.select([sys.stdin, bus.fileno()], [], [])
        if sys.stdin in ready:
            return
        bus.handle()


if __name__ == "__main__":
    main(sys.argv[1:])
