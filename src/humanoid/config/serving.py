"""Jetson defaults for resident segmentation and grasp inference."""

from pathlib import Path

from humanoid.config.pick import GRASPGENX_CONFIG, PICK_CONFIG
from humanoid.types.serving import AnyGraspConfig, GraspRuntime, ModelPipeline, ServingConfig

MODEL_DIRECTORY = Path.home() / "humanoid-models"
SERVING_CONFIG = ServingConfig(
    pipeline=ModelPipeline.PROMPTED,
    anygrasp=AnyGraspConfig(
        python=str(MODEL_DIRECTORY / ".venv/bin/python"),
        sdk_directory=str(MODEL_DIRECTORY / "anygrasp_sdk/grasp_detection"),
        checkpoint=str(
            MODEL_DIRECTORY / "anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar"
        ),
        max_gripper_width=PICK_CONFIG.open_gap,
        gripper_height=PICK_CONFIG.gripper_height,
        minimum_score=0.3,
        maximum_grasps=100,
        minimum_points=100,
        maximum_points=640 * 480,
        warmup_points=20000,
    ),
    socket_path=str(Path.home() / ".cache/humanoid/models.sock"),
    model_directory=str(MODEL_DIRECTORY),
    segmentation_python=str(MODEL_DIRECTORY / ".venv/bin/python"),
    grasp_python=str(MODEL_DIRECTORY / ".venv/bin/python"),
    startup_timeout=240.0,
    request_timeout=120.0,
    minimum_host_memory_bytes=1024**3,
    segmentation_model=PICK_CONFIG.segmenter_model,
    segmentation_revision=PICK_CONFIG.segmenter_revision,
    image_size=PICK_CONFIG.segmentation_image_size,
    confidence=PICK_CONFIG.detection_threshold,
    segmentation_artifacts=str(MODEL_DIRECTORY / "exports/yoloe26-s"),
    segmentation_threads=4,
    segmentation_iou=0.7,
    segmentation_max_instances=16,
    prompt_cache_size=32,
    grasp_warmup_points=3500,
    # Warmup only, never used to produce robot commands. Actual requests carry
    # conditioning computed from the current robot's URDF.
    grasp_warmup_conditioning=(
        0.05,
        0.04,
        0.07,
        0.0,
        0.0,
        0.10,
        0.025,
        0.04,
        0.07,
        0.0,
        0.0,
        0.10,
    ),
    grasp_count=GRASPGENX_CONFIG.num_grasps,
    grasp_threshold=GRASPGENX_CONFIG.minimum_score,
    grasp_runtime=GraspRuntime.NATIVE,
    grasp_library=str(
        MODEL_DIRECTORY / "GraspGenX/graspgenx/fast_inference/lib/libgraspgenx_runtime.so"
    ),
    grasp_artifacts=str(
        MODEL_DIRECTORY / "GraspGenX/cpp_runtime/artifacts/ddpm20-p3500-2048-1024-512-512"
    ),
)
