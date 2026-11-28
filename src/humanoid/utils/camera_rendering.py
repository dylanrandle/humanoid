"""Render camera previews without modifying source images or calibration."""

import numpy as np
from PIL import Image, ImageDraw

from humanoid.types.camera import CameraImage
from humanoid.types.streaming import DisplayFrame


def compose_camera_views(  # noqa: PLR0913 - source images and independent composition options
    primary: CameraImage | None,
    secondary: CameraImage | None,
    *,
    inset_fraction: float,
    inset_margin: float,
    labels: tuple[str, str],
    include_inset: bool,
) -> DisplayFrame | None:
    """Preserve source pixels and use the oldest available image's timestamp.

    The inset fraction scales both primary dimensions; margin is a fraction of
    its height. Labels identify the primary and secondary cameras when unavailable.
    Without an inset, pass through the primary frame. Return None if no image is available.
    """
    if not include_inset:
        return (
            DisplayFrame(data=primary.data, timestamp=primary.timestamp)
            if primary is not None
            else None
        )
    anchor = primary if primary is not None else secondary
    if anchor is None:
        return None
    image = (
        Image.fromarray(primary.data)
        if primary is not None
        else Image.new("RGB", (anchor.intrinsics.width, anchor.intrinsics.height))
    )
    if primary is None:
        ImageDraw.Draw(image).text((20, 20), f"{labels[0]} UNAVAILABLE", fill="white")
    width, height = image.size
    inset_size = (
        max(1, round(width * inset_fraction)),
        max(1, round(height * inset_fraction)),
    )
    inset = Image.new("RGB", inset_size)
    if secondary is None:
        ImageDraw.Draw(inset).text((8, 8), f"{labels[1]} UNAVAILABLE", fill="white")
    else:
        secondary_image = Image.fromarray(secondary.data)
        secondary_image.thumbnail(inset_size)
        inset.paste(
            secondary_image,
            (
                (inset.width - secondary_image.width) // 2,
                (inset.height - secondary_image.height) // 2,
            ),
        )
    margin = max(1, round(height * inset_margin))
    origin = (max(0, width - inset.width - margin), margin)
    image.paste(inset, origin)
    ImageDraw.Draw(image).rectangle(
        (origin[0], origin[1], origin[0] + inset.width - 1, origin[1] + inset.height - 1),
        outline="white",
        width=2,
    )
    return DisplayFrame(
        data=np.asarray(image),
        timestamp=min(frame.timestamp for frame in (primary, secondary) if frame is not None),
    )
