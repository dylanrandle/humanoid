"""The native boundary refuses invalid/overflowed geometry and filters aligned results."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from humanoid.config.serving import SERVING_CONFIG
from humanoid.serving.native_grasp import NativeGraspBackend


@pytest.fixture
def backend():
    result = NativeGraspBackend.__new__(NativeGraspBackend)
    result.config = SERVING_CONFIG
    result.memory = MagicMock()
    result.encoder = MagicMock()
    result.encoder.run.return_value = [np.ones((1, 512)), np.ones((1, 512))]
    result.library = MagicMock()
    result.library.graspgenx_runtime_set_gripper_device_async.return_value = 0
    result.library.graspgenx_runtime_infer_device_async.return_value = 0
    result.library.graspgenx_runtime_copy_valid_async.return_value = 0
    result.handle = result.poses = result.scores = result.valid = result.overflow = None
    result.conditioning = None
    result.seed = result.sequence = 0
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 0, 3] = [0.1, 0.2, 0.3]
    outputs = {
        "poses": poses,
        "scores": np.array([0.95, 0.85, 0.5]),
        "overflow": np.zeros(4, dtype=np.uint8),
        "valid": np.array([0, 1, 1], dtype=np.uint8),
    }
    result.memory.download.side_effect = outputs.__getitem__
    return result, outputs


def test_validity_and_score_masks_preserve_pose_alignment(backend):
    model, _ = backend
    points = np.zeros((100, 3))
    conditioning = np.asarray(SERVING_CONFIG.grasp_warmup_conditioning)
    for _ in range(2):
        found = model.predict(points, conditioning)
        np.testing.assert_allclose(found["poses"][:, 0, 3], [0.2])
        np.testing.assert_allclose(found["scores"], [0.85])
    model.encoder.run.assert_called_once()  # Same gripper reuses its learned embeddings.
    assert model.library.graspgenx_runtime_infer_device_async.call_count == 2  # noqa: PLR2004
    assert model.sequence > 0


def test_overflow_invalidates_even_high_confidence_native_grasps(backend):
    model, outputs = backend
    outputs["overflow"][0] = 1
    with pytest.raises(RuntimeError, match="overflow"):
        model.predict(np.zeros((100, 3)), np.asarray(SERVING_CONFIG.grasp_warmup_conditioning))


@pytest.mark.parametrize("bad", ["small_cloud", "nonfinite", "negative_gripper"])
def test_invalid_request_is_rejected_before_gpu_call(backend, bad):
    model, _ = backend
    points = np.zeros((100, 3))
    conditioning = np.asarray(SERVING_CONFIG.grasp_warmup_conditioning)
    if bad == "small_cloud":
        points = points[:99]
    elif bad == "nonfinite":
        points[0, 0] = np.nan
    else:
        conditioning[0] = -1
    with pytest.raises(ValueError, match="Invalid grasp"):
        model.predict(points, conditioning)
    model.library.graspgenx_runtime_infer_device_async.assert_not_called()
