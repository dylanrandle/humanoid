from unittest.mock import Mock, call

import numpy as np
import pytest

from humanoid.perception.segmentation import (
    ENCODER_SIZE,
    MASK_SIZE,
    MAX_TOKENS,
    Segmenter,
    resize_mask,
)


@pytest.fixture
def segmenter():
    # Exercise the real image and prompt pipeline with CPU stand-ins for the GPU engines.
    model = Segmenter.__new__(Segmenter)
    model.enc = Mock()
    model.dec = Mock()
    model.size = None
    model.closed = False
    return model


def test_encoder_normalizes_rgb_and_pads_below_landscape_image(segmenter):
    rgb = np.full((16, 32, 3), [255, 128, 0], np.uint8)
    segmenter.set_image(rgb)
    name, batch = segmenter.enc.upload.call_args.args
    assert name == "image" and batch.shape == (1, 3, ENCODER_SIZE, ENCODER_SIZE)
    assert batch.dtype == np.float32
    edge = ENCODER_SIZE // 2
    expected = [(255 - 123.675) / 58.395, (128 - 116.28) / 57.12, -103.53 / 57.375]
    np.testing.assert_allclose(batch[0, :, edge - 1, -1], expected, rtol=1e-5)
    assert not batch[:, :, edge:, :].any()
    assert segmenter.size == (32, 16)


def test_point_then_refinement_reuses_image_and_passes_raw_logits(segmenter):
    segmenter.set_image(np.zeros((32, 64, 3), np.uint8))
    decoder = segmenter.dec
    scores = np.array([0.8, 0.9], np.float32)
    logits = np.full((2, MASK_SIZE, MASK_SIZE), -0.75, np.float32)
    decoder.download.side_effect = {"iou_predictions": scores, "low_res_masks": logits}.__getitem__
    returned_scores, returned_masks = segmenter.predict(((16, 8),), (1,))
    inputs = dict(c.args for c in decoder.upload.call_args_list)
    np.testing.assert_array_equal(inputs["point_coords"], [[256, 128], [0, 0]])
    np.testing.assert_array_equal(inputs["point_labels"], [1, -1])
    np.testing.assert_array_equal(inputs["has_mask_input"], [0])
    np.testing.assert_array_equal(inputs["mask_input"], np.zeros((MASK_SIZE, MASK_SIZE)))
    np.testing.assert_array_equal(returned_scores, scores)
    np.testing.assert_array_equal(returned_masks, logits)

    decoder.reset_mock()
    segmenter.predict(((16, 8), (32, 12), (40, 20)), (1, 1, 0), prior=returned_masks[1])
    assert decoder.context.set_input_shape.call_args_list == [
        call("point_coords", (1, 3, 2)),
        call("point_labels", (1, 3)),
    ]
    inputs = dict(c.args for c in decoder.upload.call_args_list)
    np.testing.assert_array_equal(inputs["point_coords"], [[256, 128], [512, 192], [640, 320]])
    np.testing.assert_array_equal(inputs["point_labels"], [1, 1, 0])
    np.testing.assert_array_equal(inputs["has_mask_input"], [1])
    np.testing.assert_array_equal(inputs["mask_input"], logits[1])
    segmenter.enc.run.assert_called_once()


def test_invalid_prompts_do_not_run_decoder(segmenter):
    with pytest.raises(RuntimeError, match="encoded image"):
        segmenter.predict(((1, 2),), (1,))
    segmenter.set_image(np.zeros((8, 8, 3), np.uint8))
    for count, labels in [(0, ()), (1, (1, 0)), (MAX_TOKENS + 1, (1,) * (MAX_TOKENS + 1))]:
        with pytest.raises(ValueError, match="prompt coordinates"):
            segmenter.predict(((1, 2),) * count, labels)
    with pytest.raises(ValueError, match="prior logit shape"):
        segmenter.predict(((1, 2),), (1,), prior=np.zeros((8, 8)))
    segmenter.dec.run.assert_not_called()


@pytest.mark.parametrize("size", [(64, 32), (32, 64)])
def test_resize_mask_discards_padding_and_preserves_rgb_coordinates(size):
    w, h = size
    logits = np.ones((MASK_SIZE, MASK_SIZE), np.float32)
    valid = logits[: MASK_SIZE * h // max(size), : MASK_SIZE * w // max(size)]
    valid[:] = -1
    valid[: valid.shape[0] // 2, valid.shape[1] // 2 :] = 1
    expected = np.zeros((h, w), bool)
    expected[: h // 2, w // 2 :] = True
    mask = resize_mask(logits, size)
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, expected)
