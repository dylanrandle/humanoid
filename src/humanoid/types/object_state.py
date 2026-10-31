"""Tracked object state from simulation ground truth or real pose estimation."""

from dataclasses import dataclass
from itertools import product

import numpy as np
import pinocchio as pin


@dataclass(frozen=True)
class ObjectState:
    """Object pose in frame_id, with optional geometry and velocity.

    timestamp is measurement time in seconds on the collection clock. The episode
    runtime determines whether these are simulation labels or real estimates.
    Bounding-box center/half-size are in object-local meters;
    velocities are expressed in frame_id. None means unavailable, never zero.
    """

    timestamp: float
    name: str
    frame_id: str
    pose: pin.SE3
    bbox_center: np.ndarray | None = None
    bbox_half_size: np.ndarray | None = None
    linear_velocity: np.ndarray | None = None
    angular_velocity: np.ndarray | None = None

    def __post_init__(self) -> None:
        if (self.bbox_center is None) != (self.bbox_half_size is None):
            raise ValueError("Bounding-box center and half-size must be supplied together.")
        if (self.linear_velocity is None) != (self.angular_velocity is None):
            raise ValueError("Linear and angular velocity must be supplied together.")

    @property
    def corners(self) -> np.ndarray:
        if self.bbox_center is None or self.bbox_half_size is None:
            raise ValueError(f"No bounding box is available for {self.name}.")
        local = (
            self.bbox_center + np.asarray(list(product((-1, 1), repeat=3))) * self.bbox_half_size
        )
        return local @ self.pose.rotation.T + self.pose.translation
