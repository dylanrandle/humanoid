"""Inference overlays preserve source images and respect native optical calibration."""

# ruff: noqa: PLR2004 -- explicit synthetic camera geometry
import base64
import io
from dataclasses import replace

import numpy as np
import pinocchio as pin
import pytest
from PIL import Image

from humanoid.manipulation.preview import (
    grasp_segments,
    project,
    render_preview,
    segmentation_masks,
)
from humanoid.types.camera import CameraEncoding, CameraFrames, CameraImage, CameraIntrinsics
from humanoid.types.pick import CameraPointCloud, GraspProposal, ObjectMask, PickPerceptionPreview


@pytest.fixture
def preview():
    intrinsics = CameraIntrinsics(
        width=160,
        height=120,
        fx=100,
        fy=100,
        cx=80,
        cy=60,
        distortion_model="none",
        distortion_coefficients=(),
    )
    rgb = CameraImage(
        timestamp=10,
        device_timestamp=9,
        timestamp_domain="device",
        frame_number=12,
        frame_id="rgb",
        serial_number="test",
        encoding=CameraEncoding.RGB8,
        data=np.full((120, 160, 3), 80, dtype=np.uint8),
        intrinsics=intrinsics,
    )
    depth_pixels = np.zeros((120, 160), dtype=np.uint16)
    depth_pixels[60, 80] = 500
    depth = replace(
        rgb,
        frame_id="depth",
        encoding=CameraEncoding.DEPTH16,
        data=depth_pixels,
        depth_scale=0.001,
        frame_number=13,
    )
    mask = np.zeros((120, 160), dtype=bool)
    mask[55:66, 85:96] = True
    cloud = CameraPointCloud(
        points=np.array([[0, 0, 0.5], [0.01, 0, 0.5]], dtype=np.float32),
        rgb_pixels=np.array([[60, 90], [-1, -1]]),
        base_from_camera=pin.SE3.Identity(),
    )
    grasp = GraspProposal(
        pose=pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.5])),
        width=0.04,
        height=0.04,
        depth=0,
        score=0.9,
    )
    return PickPerceptionPreview(
        frames=CameraFrames(rgb=rgb, depth=depth),
        cloud=cloud,
        rgb_from_depth=pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0])),
        object_description="toy car",
        masks=(ObjectMask(mask=mask, score=0.9, box=(85, 55, 96, 66)),),
        grasps=(grasp,),
        selected=grasp,
    )


def pixels(frame):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(frame.image.split(",")[1]))))


def test_native_depth_mask_uses_registration_and_excludes_occluded_samples(preview):
    rgb, depth = segmentation_masks(preview)
    assert rgb[60, 90] and not rgb[60, 80]
    assert depth[60, 80] and not depth[60, 90]
    assert np.count_nonzero(depth) == 1
    points = preview.cloud.points[:1]
    np.testing.assert_allclose(project(points, preview.frames.depth.intrinsics), [[80, 60]])
    transformed = points @ preview.rgb_from_depth.rotation.T + preview.rgb_from_depth.translation
    np.testing.assert_allclose(project(transformed, preview.frames.rgb.intrinsics), [[90, 60]])


def test_projection_handles_distortion_and_rejects_behind_camera(preview):
    intr = replace(
        preview.frames.rgb.intrinsics,
        distortion_model="brown_conrady",
        distortion_coefficients=(0.1, 0, 0, 0, 0),
    )
    result = project(np.array([[0.5, 0, 1], [0, 0, -1], [0, 0, 0], [np.nan, 0, 1]]), intr)
    np.testing.assert_allclose(result[0], [131.25, 60])
    assert np.isnan(result[1:]).all()


def test_grasp_glyph_uses_predicted_gap_and_insertion_depth(preview):
    grasp = replace(preview.grasps[0], depth=0.02)
    segments = grasp_segments(grasp)
    np.testing.assert_allclose(segments[0, 0], [0.02, -0.02, 0.5])
    np.testing.assert_allclose(segments[2, 1], [0.02, 0.02, 0.5])
    assert grasp_segments(replace(grasp, width=None)).shape == (1, 2, 3)


def test_encoded_overlays_preserve_source_and_native_metadata(preview):
    original_rgb = preview.frames.rgb.data.copy()
    original_depth = preview.frames.depth.data.copy()
    result = render_preview(preview)
    assert result.rgb.frame_number == 12
    assert result.depths[3.0].frame_number == 13
    assert result.rgb.frame_id == "rgb" and result.depths[3.0].frame_id == "depth"
    assert result.timestamp == 10
    assert result.selected and result.mask_count == result.grasp_count == 1
    assert set(result.depths) == {1.0, 3.0, 5.0, 10.0}
    np.testing.assert_array_equal(preview.frames.rgb.data, original_rgb)
    np.testing.assert_array_equal(preview.frames.depth.data, original_depth)
    for frame in (result.rgb, result.depths[3.0]):
        rgb = pixels(frame).astype(int)
        assert ((rgb[:, :, 1] - rgb[:, :, 0] > 70) & (rgb[:, :, 1] > 150)).any()
    no_detections = render_preview(replace(preview, masks=(), grasps=(), selected=None))
    assert not no_detections.selected and no_detections.grasp_count == 0
    assert no_detections.rgb.image != result.rgb.image
