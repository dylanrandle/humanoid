"""The current camera capture shared by the agent's perception tools."""

from uuid import uuid4

from humanoid.perception.masks import MaskStore
from humanoid.types.camera import CameraView, ObservationId


class PinnedObservation:
    def __init__(self, store: MaskStore):
        self.store = store
        self.frames: CameraView | None = None
        self.id = ObservationId("")
        self.views: dict[ObservationId, CameraView] = {}
        self._saved: set[ObservationId] = set()

    def pin(self, frames: CameraView, *, additional: bool = False) -> ObservationId:
        self.frames = frames
        self.id = ObservationId(uuid4().hex)
        if not additional:
            self.views.clear()
            self._saved.clear()
        self.views[self.id] = frames
        return self.id

    def select(self, observation_id: ObservationId) -> None:
        if observation_id not in self.views:
            raise ValueError("Unknown observation ID or mask from a different observation.")
        self.id = observation_id
        self.frames = self.views[observation_id]

    def save(self) -> None:
        if self.frames is None:
            raise ValueError("No camera observation is pinned.")
        if self.id not in self._saved:
            self.store.save_capture(self.id, self.frames)
            self._saved.add(self.id)
