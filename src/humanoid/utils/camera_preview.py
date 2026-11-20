"""Image encoding shared by live camera and model-result previews."""

import base64
import io

import numpy as np
from PIL import Image

from humanoid.types.camera import CameraImage, CameraPreview, CameraPreviewConfig


def colorize_depth(frame: CameraImage, near: float, far: float) -> np.ndarray:
    meters = frame.data.astype(np.float32) * frame.depth_scale
    distance = np.clip((meters - near) / (far - near), 0, 1)
    # Amber (near) -> teal -> blue (far). CSS legend uses the same stops.
    palette = np.array([[248, 196, 66], [49, 184, 186], [51, 71, 169]])
    colors = np.stack(
        [np.interp(distance, [0, 0.5, 1], channel) for channel in palette.T], axis=-1
    ).astype(np.uint8)
    colors[frame.data == 0] = 0
    return colors


def encode_preview(
    frame: CameraImage, pixels: np.ndarray, config: CameraPreviewConfig
) -> CameraPreview:
    image = Image.fromarray(pixels)
    image.thumbnail((config.max_width, config.max_width))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=config.jpeg_quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return CameraPreview(
        connected=True,
        image=f"data:image/jpeg;base64,{encoded}",
        width=frame.intrinsics.width,
        height=frame.intrinsics.height,
        frame_id=frame.frame_id,
        frame_number=frame.frame_number,
    )
