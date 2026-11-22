"""Inspect the running simulation from a free camera in MuJoCo's native viewer."""

import argparse
import base64
import os
import select
import shlex
import struct
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path, PurePosixPath

import lcm

from humanoid.constants import DEFAULT_LCM_URL, Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.types.lcm import simulation_state_t
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.simulation import SimulationState

_MAX_LINE_BYTES = 8 * 1024 * 1024


def _relay_frames() -> None:
    """Write LCM frames to SSH stdout until the client closes stdin."""
    bus = lcm.LCM(DEFAULT_LCM_URL)

    def send(_channel: str, packet: bytes) -> None:
        sys.stdout.buffer.write(base64.b64encode(packet) + b"\n")
        sys.stdout.buffer.flush()

    subscription = bus.subscribe(Topic.SIMULATION_STATE.value, send)
    subscription.set_queue_capacity(1)
    while True:
        ready, _, _ = select.select([sys.stdin, bus.fileno()], [], [])
        if sys.stdin in ready:
            return
        bus.handle()


@contextmanager
def ssh_frames(host: str) -> Iterator[Callable[[int | None], SimulationState | None]]:
    root = PurePosixPath(os.environ.get("HUMANOID_ROBOT_DIR", "/opt/humanoid"))
    remote = shlex.join([str(root / ".venv/bin/sim-viewer"), "--relay"])
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

        def receive(timeout_ms: int | None) -> SimulationState | None:
            timeout = None if timeout_ms is None else timeout_ms / 1000
            if not select.select([stream], [], [], timeout)[0]:
                return None
            chunk = os.read(stream.fileno(), 65536)
            if not chunk:
                raise RuntimeError(
                    f"Simulation stream from {host} closed. Check SSH and the remote installation."
                )
            pending.extend(chunk)
            if len(pending) > _MAX_LINE_BYTES:
                raise ValueError(f"Oversized simulation frame from {host}.")
            end = pending.rfind(b"\n")
            if end < 0:
                return None
            # During model loading several frames may accumulate. Display the latest.
            start = pending.rfind(b"\n", 0, end) + 1
            line = pending[start:end]
            del pending[: end + 1]
            try:
                packet = base64.b64decode(line, validate=True)
                return LCMConverter.simulation_state_from_lcm(simulation_state_t.decode(packet))
            except (ValueError, struct.error) as exc:
                raise ValueError(f"Invalid simulation frame from {host}: {exc}") from exc

        try:
            yield receive
        finally:
            process.stdin.close()
            stream.close()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


@contextmanager
def _local_frames() -> Iterator[Callable[[int | None], SimulationState | None]]:
    with closing(Subscriber([Topic.SIMULATION_STATE])) as subscriber:
        yield lambda timeout: subscriber.receive(Topic.SIMULATION_STATE, timeout=timeout)


def _view_world(
    first: SimulationState, receive: Callable[[int | None], SimulationState | None]
) -> SimulationState | None:
    # Only the graphical client needs the renderer and robot model.
    import mujoco.viewer  # noqa: PLC0415

    from humanoid.simulation.mirror import SimulationMirror  # noqa: PLC0415

    mirror = SimulationMirror(first)
    with mujoco.viewer.launch_passive(mirror.model, mirror.data) as viewer:
        while viewer.is_running():
            viewer.sync()
            frame = receive(100)
            if frame is not None:
                if not mirror.matches(frame):
                    return frame
                mirror.apply(frame)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--relay", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--host", help="Read simulation state over SSH, e.g. triskel or user@host.")
    args = parser.parse_args()
    if args.relay:
        _relay_frames()
        return
    if sys.platform == "darwin" and not os.environ.get("MJPYTHON_BIN"):
        # Cocoa needs the native main thread. mjpython supplies that event loop.
        launcher = Path(sys.executable).with_name("mjpython")
        os.execv(launcher, [str(launcher), "-m", "humanoid.simulation.viewer", *sys.argv[1:]])

    location = f" on {args.host}" if args.host else ""
    print(f"Waiting for the simulation stack{location}…", flush=True)
    try:
        with ssh_frames(args.host) if args.host else _local_frames() as receive:
            frame = receive(None)
            while frame is None:
                frame = receive(None)
            while frame is not None:
                frame = _view_world(frame, receive)
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
