"""Subscribe to the stack's original state and command packets locally or over SSH."""

import base64
import os
import select
import shlex
import struct
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import lcm

from humanoid.constants import DEFAULT_LCM_URL, DEFAULT_ROBOT_DIR, EnvironmentVariable, Topic
from humanoid.types.lcm import (
    robot_base_command_t,
    robot_joint_command_t,
    robot_state_t,
    robot_tool_command_t,
    simulation_state_t,
)
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.robot import RobotBaseCommand, RobotJointCommand, RobotState, RobotToolCommand
from humanoid.types.simulation import SimulationState

_CHANNELS = (
    Topic.SIMULATION_STATE,
    Topic.ROBOT_STATE,
    Topic.ROBOT_JOINT_COMMAND,
    Topic.ROBOT_TOOL_COMMAND,
    Topic.ROBOT_BASE_COMMAND,
)
_MAX_LINE_BYTES = 8 * 1024 * 1024
type Telemetry = (
    SimulationState | RobotState | RobotJointCommand | RobotToolCommand | RobotBaseCommand
)
type Receive = Callable[[int | None], list[Telemetry]]


def _decode(channel: str, packet: bytes) -> Telemetry:
    match Topic(channel):
        case Topic.SIMULATION_STATE:
            return LCMConverter.simulation_state_from_lcm(simulation_state_t.decode(packet))
        case Topic.ROBOT_STATE:
            return LCMConverter.robot_state_from_lcm(robot_state_t.decode(packet))
        case Topic.ROBOT_JOINT_COMMAND:
            return LCMConverter.robot_joint_command_from_lcm(robot_joint_command_t.decode(packet))
        case Topic.ROBOT_TOOL_COMMAND:
            return LCMConverter.robot_tool_command_from_lcm(robot_tool_command_t.decode(packet))
        case Topic.ROBOT_BASE_COMMAND:
            return LCMConverter.robot_base_command_from_lcm(robot_base_command_t.decode(packet))
        case _:
            raise ValueError(f"Unexpected viewer channel {channel!r}.")


@contextmanager
def local_frames() -> Iterator[Receive]:
    bus = lcm.LCM(DEFAULT_LCM_URL)
    packets: list[Telemetry] = []

    def collect(channel: str, packet: bytes) -> None:
        packets.append(_decode(channel, packet))

    subscriptions = [bus.subscribe(topic.value, collect) for topic in _CHANNELS]
    for subscription in subscriptions:
        subscription.set_queue_capacity(1)

    def receive(timeout_ms: int | None) -> list[Telemetry]:
        packets.clear()
        timeout = None if timeout_ms is None else timeout_ms / 1000
        if select.select([bus.fileno()], [], [], timeout)[0]:
            bus.handle()
            for _ in subscriptions:
                if not bus.handle_timeout(0):
                    break
        return packets.copy()

    try:
        yield receive
    finally:
        for subscription in subscriptions:
            bus.unsubscribe(subscription)


@contextmanager
def ssh_frames(host: str) -> Iterator[Receive]:
    root = PurePosixPath(os.environ.get(EnvironmentVariable.ROBOT_DIR, DEFAULT_ROBOT_DIR))
    relay = Path(__file__).with_name("relay.py").read_text()
    remote = shlex.join(
        [str(root / ".venv/bin/python"), "-u", "-c", relay, *(t.value for t in _CHANNELS)]
    )
    with subprocess.Popen(
        [
            "ssh",
            "-T",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "--",
            host,
            "exec " + remote,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    ) as process:
        assert process.stdout is not None and process.stdin is not None
        stream = process.stdout
        pending = bytearray()

        def receive(timeout_ms: int | None) -> list[Telemetry]:
            timeout = None if timeout_ms is None else timeout_ms / 1000
            if not select.select([stream], [], [], timeout)[0]:
                return []
            chunk = os.read(stream.fileno(), 65536)
            if not chunk:
                raise RuntimeError(
                    f"Viewer stream from {host} closed. "
                    f"Check SSH and {EnvironmentVariable.ROBOT_DIR}."
                )
            pending.extend(chunk)
            lines = pending.split(b"\n")
            pending[:] = lines.pop()
            if len(pending) > _MAX_LINE_BYTES or any(len(line) > _MAX_LINE_BYTES for line in lines):
                raise ValueError(f"Oversized LCM packet from {host}.")
            packets = []
            try:
                for line in lines:
                    channel, encoded = line.split(b" ", 1)
                    packets.append(
                        _decode(channel.decode("ascii"), base64.b64decode(encoded, validate=True))
                    )
            except (ValueError, struct.error) as exc:
                raise ValueError(f"Invalid LCM packet from {host}: {exc}") from exc
            return packets

        try:
            yield receive
        finally:
            process.stdin.close()
            stream.close()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
