from dataclasses import dataclass
from enum import StrEnum


class WheelType(StrEnum):
    REGULAR = "regular"
    OMNI = "omni"


@dataclass
class WheelConfig:
    frame: str
    floor_frame: str
    radius: float
    type: WheelType
