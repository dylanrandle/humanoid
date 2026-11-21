"""Native RGB pixel prompts and persisted segmentation results."""

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType

import numpy as np

from humanoid.types.camera import CameraIntrinsics, CaptureMetadata, ObservationId

MAX_TOKENS = 16


class SegmentOperation(StrEnum):
    EVERYTHING = "segment_everything"
    POINT = "segment_at_point"
    BOX = "segment_in_box"
    REFINE = "refine_mask"


MaskId = NewType("MaskId", str)

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
    mask_id: MaskId
    observation_id: ObservationId
    rgb: np.ndarray
    completed_depth_m: np.ndarray | None
    mask: np.ndarray
    intrinsics: CameraIntrinsics
    capture: CaptureMetadata


@dataclass(frozen=True, kw_only=True)
class MaskProposal:
    number: int
    mask_id: MaskId
    box_xyxy: Box
    pixels: int
    predicted_iou: float


@dataclass(frozen=True, kw_only=True)
class SegmentationResult:
    observation_id: ObservationId
    masks: tuple[MaskProposal, ...]
