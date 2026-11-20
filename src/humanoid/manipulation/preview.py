"""Project model outputs into the exact RGB and native-depth inference frames."""

import numpy as np
import pinocchio as pin
from PIL import Image, ImageDraw

from humanoid.config.camera import CAMERA_PREVIEW_CONFIG
from humanoid.manipulation.pick_geometry import distort
from humanoid.types.camera import CameraIntrinsics, CameraPreviewConfig
from humanoid.types.pick import GraspProposal, PickPerceptionPreview, PickPreview
from humanoid.utils.camera_preview import colorize_depth, encode_preview

# These are display glyph dimensions, not generated or executable grasp geometry.
GLYPH_FINGER_LENGTH = 0.03
GLYPH_STEM_LENGTH = 0.015
MASK_COLOR = (40, 210, 240)
CANDIDATE_COLOR = (255, 190, 45)
SELECTED_COLOR = (70, 255, 110)
MIN_PROJECTION_DEPTH = 0.001
MAX_PIXEL_COORDINATE = 1e6


def project(points: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Return pixels (u, v); nonfinite/behind-camera points remain NaN."""
    pixels = np.full((len(points), 2), np.nan)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > MIN_PROJECTION_DEPTH)
    xy = distort(points[valid, :2] / points[valid, 2, None], intrinsics, inverse=False)
    pixels[valid] = xy * [intrinsics.fx, intrinsics.fy] + [intrinsics.cx, intrinsics.cy]
    return pixels


def segmentation_masks(preview: PickPerceptionPreview) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.zeros(preview.frames.rgb.data.shape[:2], dtype=bool)
    for mask in preview.masks:
        rgb |= mask.mask
    depth = np.zeros(preview.frames.depth.data.shape, dtype=bool)
    # Use the same occlusion-aware RGB association as grasp inference. Do not
    # stretch an RGB mask over a different optical viewpoint or label missing depth.
    association = preview.cloud.rgb_pixels
    visible = (association >= 0).all(axis=1)
    indices = np.flatnonzero(visible)
    indices = indices[rgb[association[indices, 0], association[indices, 1]]]
    pixels = project(preview.cloud.points[indices], preview.frames.depth.intrinsics)
    finite = np.isfinite(pixels).all(axis=1)
    u, v = np.rint(pixels[finite]).astype(int).T
    inside = (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
    depth[v[inside], u[inside]] = True
    return rgb, depth


def grasp_segments(grasp: GraspProposal) -> np.ndarray:
    tip = grasp.depth
    heel = tip - GLYPH_FINGER_LENGTH
    if grasp.width is None:
        # A pose with unresolved width gets only an approach indicator.
        points = np.array([[heel, 0, 0], [tip, 0, 0]])
        edges = [(0, 1)]
    else:
        half = grasp.width / 2
        points = np.array(
            [
                [tip, -half, 0],
                [heel, -half, 0],
                [heel, half, 0],
                [tip, half, 0],
                [heel - GLYPH_STEM_LENGTH, 0, 0],
                [heel, 0, 0],
            ]
        )
        edges = [(0, 1), (1, 2), (2, 3), (4, 5)]
    points = points @ grasp.pose.rotation.T + grasp.pose.translation
    return points[np.asarray(edges)]


def draw_overlay(
    image: Image.Image,
    preview: PickPerceptionPreview,
    intrinsics: CameraIntrinsics,
    view_from_depth: pin.SE3,
) -> None:
    draw = ImageDraw.Draw(image)
    grasps = [(grasp, CANDIDATE_COLOR, 2) for grasp in preview.grasps]
    if preview.selected is not None:
        grasps.append((preview.selected, SELECTED_COLOR, 4))
    for grasp, color, width in grasps:
        segments = grasp_segments(grasp)
        points = segments.reshape(-1, 3) @ view_from_depth.rotation.T + view_from_depth.translation
        pixels = project(points, intrinsics).reshape(-1, 2, 2)
        for line in pixels:
            if np.isfinite(line).all() and (np.abs(line) < MAX_PIXEL_COORDINATE).all():
                draw.line([tuple(point) for point in line], fill=color, width=width)


def render_preview(
    preview: PickPerceptionPreview, config: CameraPreviewConfig = CAMERA_PREVIEW_CONFIG
) -> PickPreview:
    rgb_mask, depth_mask = segmentation_masks(preview)
    rgb = preview.frames.rgb
    depth = preview.frames.depth

    def annotate(
        pixels: np.ndarray,
        mask: np.ndarray,
        intrinsics: CameraIntrinsics,
        transform: pin.SE3,
    ) -> np.ndarray:
        pixels = pixels.copy()
        pixels[mask] = (pixels[mask] * 0.6 + np.asarray(MASK_COLOR) * 0.4).astype(np.uint8)
        image = Image.fromarray(pixels)
        draw_overlay(image, preview, intrinsics, transform)
        return np.asarray(image)

    rgb_pixels = annotate(rgb.data, rgb_mask, rgb.intrinsics, preview.rgb_from_depth)
    depths = {}
    for far in config.depth_ranges_meters:
        pixels = colorize_depth(depth, config.depth_near_meters, far)
        depths[far] = encode_preview(
            depth, annotate(pixels, depth_mask, depth.intrinsics, pin.SE3.Identity()), config
        )
    return PickPreview(
        timestamp=rgb.timestamp,
        rgb=encode_preview(rgb, rgb_pixels, config),
        depths=depths,
        object_description=preview.object_description,
        mask_count=len(preview.masks),
        grasp_count=len(preview.grasps),
        selected=preview.selected is not None,
    )
