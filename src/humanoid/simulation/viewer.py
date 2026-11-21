"""Inspect the running simulation from a free camera in MuJoCo's native viewer."""

import os
import sys
from pathlib import Path

import mujoco
import mujoco.viewer

from humanoid.constants import Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.simulation.mirror import SimulationMirror
from humanoid.types.simulation import SimulationState


def _view_world(first: SimulationState, subscriber: Subscriber) -> SimulationState | None:
    mirror = SimulationMirror(first)
    with mujoco.viewer.launch_passive(mirror.model, mirror.data) as viewer:
        while viewer.is_running():
            viewer.sync()
            frame = subscriber.receive(Topic.SIMULATION_STATE, timeout=100)
            if frame is not None:
                if not mirror.matches(frame):
                    return frame
                mirror.apply(frame)
    return None


def main() -> None:
    if sys.platform == "darwin" and not os.environ.get("MJPYTHON_BIN"):
        # Cocoa needs the native main thread. mjpython supplies that event loop.
        launcher = Path(sys.executable).with_name("mjpython")
        os.execv(launcher, [str(launcher), "-m", "humanoid.simulation.viewer"])

    print("Waiting for the simulation stack…", flush=True)
    subscriber = Subscriber([Topic.SIMULATION_STATE])
    try:
        frame = subscriber.receive(Topic.SIMULATION_STATE, timeout=None)
        while frame is not None:
            frame = _view_world(frame, subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.close()


if __name__ == "__main__":
    main()
