"""Render and publish the latest scheduled physics snapshot on a dedicated thread."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager
from threading import Condition

import mujoco

from humanoid.middleware.publisher import Publisher
from humanoid.simulation.camera import FrameClock, MujocoCamera
from humanoid.types.camera import camera_topics
from humanoid.types.simulation import RenderFrame, SimulationState


class SimulationRenderStream:
    def __init__(
        self,
        name: str,
        model: mujoco.MjModel,
        fps: int,
        renderer: AbstractContextManager[RenderFrame],
    ):
        self.clock = FrameClock(fps)
        self._condition = Condition()
        self._pending: tuple[SimulationState, float] | None = None
        self._stopped = False
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"render-{name}")
        self._worker = self._pool.submit(self._run, model, renderer)

    def submit(self, state: SimulationState, timestamp: float) -> None:
        if self._worker.done():
            self._worker.result()
        with self._condition:
            self._pending = (state, timestamp)
            self._condition.notify()

    def _run(self, model: mujoco.MjModel, renderer: AbstractContextManager[RenderFrame]) -> None:
        data = mujoco.MjData(model)
        with renderer as render:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stopped or self._pending is not None)
                    if self._pending is None:
                        return
                    state, timestamp = self._pending
                    self._pending = None
                mujoco.mj_setState(model, data, state.state, mujoco.mjtState.mjSTATE_INTEGRATION)
                mujoco.mj_forward(model, data)
                render(data, timestamp)

    def close(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify()
        self._pool.shutdown(wait=True)
        self._worker.result()


@contextmanager
def publish_camera(name: str, camera: MujocoCamera, url: str) -> Iterator[RenderFrame]:
    publisher = Publisher(url=url)
    topics = camera_topics(name)

    def publish(data: mujoco.MjData, timestamp: float) -> None:
        publisher.publish_camera(camera.render(data, timestamp), topics)

    try:
        yield publish
    finally:
        camera.close()
