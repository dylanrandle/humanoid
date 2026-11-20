"""ONNX masks preserve RGB/depth registration and instance correspondence."""

from dataclasses import replace

import numpy as np
import pytest

from humanoid.config.serving import SERVING_CONFIG
from humanoid.serving.client import ModelClient
from humanoid.serving.segmentation import decode_instances, prepare_image


def test_letterboxing_preserves_rgb_channels_and_centers_non_square_image():
    rgb = np.zeros((24, 40, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    tensor = prepare_image(rgb, 40)
    assert tensor.shape == (1, 3, 40, 40)
    np.testing.assert_array_equal(tensor[0, 0, 8:32], 1)
    np.testing.assert_array_equal(tensor[0, 1:, 8:32], 0)
    np.testing.assert_allclose(tensor[0, :, :8], 114 / 255)


def test_decoded_masks_restore_native_rgb_coordinates_and_suppress_duplicates():
    config = replace(SERVING_CONFIG, image_size=40)
    # Model input is 40x40 with eight rows of padding above and below 24x40 RGB.
    predictions = np.array(
        [
            [6, 13, 6, 6, 0.9, 1],
            [6, 13, 6, 6, 0.7, 1],  # same box must not select the same object twice
            [30, 24, 10, 8, 0.8, 1],
            [5, 10, 4, 4, 0.1, 1],  # below threshold
        ],
        dtype=np.float32,
    ).T[None]
    result = decode_instances(predictions, np.ones((1, 1, 10, 10)), (24, 40), config)
    expected = np.zeros((2, 24, 40), dtype=bool)
    expected[0, 2:8, 3:9] = True
    expected[1, 12:20, 25:35] = True
    np.testing.assert_array_equal(result["masks"], expected)
    np.testing.assert_allclose(result["scores"], [0.9, 0.8])
    np.testing.assert_allclose(result["boxes"], [[3, 2, 9, 8], [25, 12, 35, 20]])


def test_no_detection_returns_native_sized_empty_masks():
    result = decode_instances(
        np.zeros((1, 6, 10)), np.ones((1, 1, 160, 160)), (480, 640), SERVING_CONFIG
    )
    assert result["masks"].shape == (0, 480, 640)
    assert result["boxes"].shape == (0, 4)


@pytest.mark.parametrize("bad", ["resized_masks", "nonfinite", "counts"])
def test_malformed_model_output_cannot_select_wrong_depth_pixels(monkeypatch, bad):
    result = {
        "scores": np.array([0.9]),
        "boxes": np.array([[0, 0, 5, 4]]),
        "masks": np.zeros((1, 4, 5)),
    }
    if bad == "resized_masks":
        result["masks"] = np.zeros((1, 640, 640))
    elif bad == "nonfinite":
        result["scores"][0] = np.nan
    else:
        result["boxes"] = np.zeros((2, 4))
    monkeypatch.setattr("humanoid.serving.client.request", lambda *_: result)
    with pytest.raises(ValueError, match="segmentation output"):
        ModelClient(SERVING_CONFIG).segment(np.zeros((4, 5, 3), dtype=np.uint8), "car")
