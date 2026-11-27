"""WebRTC boundary contracts and stream metrics."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class StreamingConfig:
    fps: int
    max_width: int
    max_sessions: int
    negotiation_timeout: float
    idle_timeout: float


class StreamOffer(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    camera: str
    stream: Literal["rgb", "depth"]
    depth_far: float
    sdp: str = Field(min_length=1, max_length=65536)
    type: Literal["offer"]


@dataclass(frozen=True)
class StreamAnswer:
    session_id: str
    sdp: str
    type: str


@dataclass(frozen=True)
class StreamMetrics:
    camera: str
    stream: str
    connection_state: str
    frames_sent: int = 0
    frame_age_seconds: float | None = None
