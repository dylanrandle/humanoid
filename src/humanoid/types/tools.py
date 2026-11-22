"""Typed arguments and results for the agent's perception tools."""

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from humanoid.types.camera import ObservationId
from humanoid.types.segmentation import MAX_TOKENS, Box, MaskId, Pixel


class ToolModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)


class SegmentEverything(ToolModel):
    observation_id: ObservationId
    region_xyxy: Box | None


class SegmentAtPoint(ToolModel):
    observation_id: ObservationId
    point_xy: Pixel
    return_alternatives: bool


class SegmentInBox(ToolModel):
    observation_id: ObservationId
    box_xyxy: Box
    return_alternatives: bool


class RefineMask(ToolModel):
    mask_id: MaskId
    include_points: Annotated[tuple[Pixel, ...], Field(max_length=MAX_TOKENS)]
    exclude_points: Annotated[tuple[Pixel, ...], Field(max_length=MAX_TOKENS)]


type SegmentRequest = SegmentEverything | SegmentAtPoint | SegmentInBox | RefineMask


class CompleteDepth(ToolModel):
    # Structured model replies must name every field. An empty list requests only the map.
    model_config = ConfigDict(json_schema_extra={"required": ["observation_id", "points_xy"]})

    observation_id: ObservationId
    points_xy: tuple[Pixel, ...] = Field(default_factory=tuple)


@dataclass(frozen=True, kw_only=True)
class DepthCompletionResult:
    observation_id: ObservationId
    valid_pixels: int
    depths_m: tuple[float | None, ...]
    operation: Literal["complete_depth"] = "complete_depth"
