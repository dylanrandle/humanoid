"""Native RGB pixel prompts and persisted segmentation results."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from humanoid.types.camera import CameraIntrinsics


class SegmentOperation(StrEnum):
    EVERYTHING = "segment_everything"
    POINT = "segment_at_point"
    BOX = "segment_in_box"
    REFINE = "refine_mask"


type Pixel = tuple[int, int]
type Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class MaskCandidate:
    score: float
    logits: np.ndarray
    points: tuple[Pixel, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class MaskSelection:
    mask_id: str
    observation_id: str
    rgb: np.ndarray
    completed_depth_m: np.ndarray | None
    mask: np.ndarray
    intrinsics: CameraIntrinsics
    capture: dict
