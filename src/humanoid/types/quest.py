"""Validated Quest/Vuer wire contracts and local hosting configuration."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal, TypedDict

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from humanoid.types.camera import CameraImage
from humanoid.types.streaming import StreamAnswer, StreamMetrics, StreamOffer, VideoSource
from humanoid.types.teleop import QuestInputSnapshot

Axis = Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]


class QuestRoute(StrEnum):
    """Headset application paths, including aiohttp route placeholders."""

    ROOT = "/"
    CONNECT = "/connect"
    WEBSOCKET = "/ws"
    STATUS = "/status"
    VIDEO_OFFER = "/offer/{owner}/{camera}"
    VUER_ASSETS = "/assets"
    QUEST_ASSETS = "/quest-assets"


@dataclass(frozen=True)
class QuestCameraSources:
    """A consistent generation and RGB readers in the robot's configured camera order."""

    generation: int
    sources: Mapping[str, Callable[[], CameraImage | None]]


@dataclass(frozen=True)
class QuestVideoServices:
    """Host-provided camera and streaming capabilities; no application lifecycle access.

    Cameras and metrics are read on the Quest event loop. Offer and remove run on
    worker threads. Callbacks must tolerate camera-runtime changes during negotiation.
    """

    cameras: Callable[[], QuestCameraSources]
    offer: Callable[[StreamOffer, VideoSource], StreamAnswer]
    remove: Callable[[str], None]
    metrics: Callable[[], Mapping[str, StreamMetrics]]


@dataclass(frozen=True)
class QuestConfig:
    public_host: str | None
    port: int
    input_timeout: float
    poll_seconds: float
    request_timeout: float
    base_height: float
    base_down_degrees: float
    wrist_fraction: float
    wrist_margin: float


class ControllerState(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    squeeze: StrictBool
    aButton: StrictBool
    bButton: StrictBool
    thumbstickValue: tuple[Axis, Axis]


class ControllerFrame(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    left: list[float] = Field(min_length=16, max_length=16)
    right: list[float] = Field(min_length=16, max_length=16)
    leftState: ControllerState
    rightState: ControllerState

    @field_validator("left", "right")
    @classmethod
    def rigid_pose(cls, values: list[float]) -> list[float]:
        pose = np.array(values).reshape(4, 4, order="F")
        rotation = pose[:3, :3]
        if not (
            np.allclose(pose[3], [0, 0, 0, 1], atol=1e-4)
            and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
            and np.isclose(np.linalg.det(rotation), 1, atol=1e-3)
        ):
            raise ValueError("Controller pose must be a rigid transform")
        return values

    @property
    def engaged(self) -> bool:
        return self.leftState.squeeze or self.rightState.squeeze


class QuestSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    frame: ControllerFrame | None = None
    captured_monotonic: float | None = None
    last_disengaged_monotonic: float | None = None

    def policy_snapshot(self) -> QuestInputSnapshot:
        frame = self.frame
        return QuestInputSnapshot(
            transforms={}
            if frame is None
            else {
                "l": np.array(frame.left).reshape(4, 4, order="F"),
                "r": np.array(frame.right).reshape(4, 4, order="F"),
            },
            buttons={}
            if frame is None
            else {
                "LG": frame.leftState.squeeze,
                "RG": frame.rightState.squeeze,
                "A": frame.rightState.aButton,
                "B": frame.rightState.bButton,
                "X": frame.leftState.aButton,
                "Y": frame.leftState.bButton,
                "leftJS": (frame.leftState.thumbstickValue[0], -frame.leftState.thumbstickValue[1]),
                "rightJS": (
                    frame.rightState.thumbstickValue[0],
                    -frame.rightState.thumbstickValue[1],
                ),
            },
            captured_monotonic=self.captured_monotonic,
            last_disengaged_monotonic=self.last_disengaged_monotonic,
        )


class VuerControllerEvent(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    etype: Literal["CONTROLLER_MOVE"]
    ts: float = Field(gt=0)
    value: ControllerFrame


class QuestGuard(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    kind: Literal["clock", "frame"]
    client_ms: float = Field(gt=0)
    nonce: str | None = None
    frame_ms: float | None = None


class QuestVideoOffer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sdp: str = Field(min_length=1, max_length=65536)
    type: Literal["offer"]


class QuestStatus(TypedDict):
    connected: bool
    tracking: bool
    engaged: bool
    cameras: list[str]


class QuestClockChallenge(TypedDict):
    kind: Literal["clock"]
    nonce: str


class QuestViewerStatus(TypedDict):
    kind: Literal["status"]
    tracking: bool
    engaged: bool
    video: int


class QuestVideoAnswer(TypedDict):
    sdp: str
    type: str
