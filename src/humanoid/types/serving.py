"""Configuration and status for resident inference processes."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

type Arrays = dict[str, np.ndarray]
type Handler = Callable[[Arrays], Arrays]


class ModelKind(StrEnum):
    SEGMENTATION = "segmentation"
    GRASP = "grasp"
    ANYGRASP = "anygrasp"


class ModelPipeline(StrEnum):
    PROMPTED = "prompted"
    ANYGRASP = "anygrasp"


@dataclass(frozen=True, kw_only=True)
class AnyGraspConfig:
    python: str
    sdk_directory: str
    checkpoint: str
    max_gripper_width: float
    gripper_height: float
    minimum_score: float
    maximum_grasps: int
    minimum_points: int
    maximum_points: int
    warmup_points: int


class ServingState(StrEnum):
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    STOPPED = "stopped"


class GraspRuntime(StrEnum):
    PYTORCH = "pytorch"
    NATIVE = "native"


@dataclass(frozen=True, kw_only=True)
class ServingConfig:
    pipeline: ModelPipeline
    anygrasp: AnyGraspConfig
    socket_path: str
    model_directory: str
    segmentation_python: str
    grasp_python: str
    startup_timeout: float
    request_timeout: float
    minimum_host_memory_bytes: int
    segmentation_model: str
    segmentation_revision: str
    image_size: int
    confidence: float
    segmentation_artifacts: str
    segmentation_threads: int
    segmentation_iou: float
    segmentation_max_instances: int
    prompt_cache_size: int
    grasp_warmup_points: int
    grasp_warmup_conditioning: tuple[float, ...]
    grasp_count: int
    grasp_threshold: float
    grasp_runtime: GraspRuntime
    grasp_library: str
    grasp_artifacts: str


@dataclass(frozen=True, kw_only=True)
class ServingStatus:
    state: ServingState
    message: str
    segmentation_pid: int | None = None
    grasp_pid: int | None = None
