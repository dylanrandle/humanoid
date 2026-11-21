"""The current camera capture shared by the agent's perception tools."""

from uuid import uuid4

from humanoid.perception.masks import MaskStore
from humanoid.types.camera import CameraView, ObservationId


class PinnedObservation:
    def __init__(self, store: MaskStore):
        self.store = store
        self.frames: CameraView | None = None
        self.id = ObservationId("")
        self._saved = False

    def pin(self, frames: CameraView) -> ObservationId:
        self.frames = frames
        self.id = ObservationId(uuid4().hex)
        self._saved = False
        return self.id

    def save(self) -> None:
        if self.frames is None:
            raise ValueError("No camera observation is pinned.")
        if not self._saved:
            self.store.save_capture(self.id, self.frames)
            self._saved = True
