"""Promptable YOLOE inference with ONNX Runtime; no PyTorch imports."""

import html
import json
import math
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from humanoid.types.serving import Arrays, ServingConfig

RGB_DIMENSIONS = 3
RGB_CHANNELS = 3
BOX_AND_SCORE_CHANNELS = 5


def letterbox_span(size: tuple[int, int], original: tuple[int, int]) -> tuple[int, int, int, int]:
    gain = min(size[0] / original[0], size[1] / original[1])
    height, width = (max(round(value * gain), 1) for value in original)
    return round((size[0] - height) / 2 - 0.1), round((size[1] - width) / 2 - 0.1), height, width


def prepare_image(rgb: np.ndarray, size: int) -> np.ndarray:
    if rgb.ndim != RGB_DIMENSIONS or rgb.shape[-1] != RGB_CHANNELS or rgb.dtype != np.uint8:
        raise ValueError("Expected an H x W x 3 uint8 RGB image")
    top, left, height, width = letterbox_span((size, size), rgb.shape[:2])
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + height, left : left + width] = cv2.resize(rgb, (width, height))
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255


def suppress_overlaps(boxes: np.ndarray, scores: np.ndarray, iou: float, limit: int) -> list[int]:
    order = scores.argsort()[::-1]
    keep = []
    areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    while len(order) and len(keep) < limit:
        index = int(order[0])
        keep.append(index)
        remaining = order[1:]
        intersect = np.maximum(
            np.minimum(boxes[index, 2:], boxes[remaining, 2:])
            - np.maximum(boxes[index, :2], boxes[remaining, :2]),
            0,
        ).prod(axis=1)
        union = areas[index] + areas[remaining] - intersect
        order = remaining[intersect / np.maximum(union, 1e-7) <= iou]
    return keep


def decode_instances(
    detections: np.ndarray,
    prototypes: np.ndarray,
    shape: tuple[int, int],
    config: ServingConfig,
) -> Arrays:
    """Decode masks in original RGB pixels, preserving native-depth registration."""
    if (
        prototypes.ndim != 4  # noqa: PLR2004 -- BCHW
        or prototypes.shape[0] != 1
        or detections.ndim != RGB_DIMENSIONS
        or detections.shape[:2] != (1, BOX_AND_SCORE_CHANNELS + prototypes.shape[1])
        or not np.isfinite(detections).all()
        or not np.isfinite(prototypes).all()
    ):
        raise ValueError("Malformed YOLOE segmentation output")
    rows = detections[0].T
    rows = rows[rows[:, 4] > config.confidence]
    boxes = np.concatenate((rows[:, :2] - rows[:, 2:4] / 2, rows[:, :2] + rows[:, 2:4] / 2), axis=1)
    keep = suppress_overlaps(
        boxes, rows[:, 4], config.segmentation_iou, config.segmentation_max_instances
    )
    rows, boxes = rows[keep], boxes[keep]
    height, width = shape
    top, left, _, _ = letterbox_span((config.image_size, config.image_size), shape)
    gain = config.image_size / max(shape)
    boxes = (boxes - [left, top, left, top]) / gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    proto = prototypes[0]
    _, mh, mw = proto.shape
    mask_gain = min(4 * mh / height, 4 * mw / width)
    pad = (4 * mh - max(round(height * mask_gain), 1)) | (4 * mw - max(round(width * mask_gain), 1))
    factor = 4 // math.gcd(4, pad | pad // 2)
    # Decode one mask at a time to avoid a large N*H*W float intermediate.
    masks = np.zeros((len(rows), height, width), dtype=bool)
    for index, row in enumerate(rows):
        logits = (row[5:] @ proto.reshape(len(proto), -1)).reshape(mh, mw)
        if factor > 1:
            logits = cv2.resize(logits, (mw * factor, mh * factor))
        mt, ml, ch, cw = letterbox_span(logits.shape, shape)
        mask = cv2.resize(logits[mt : mt + ch, ml : ml + cw], (width, height)) > 0
        x1, y1, x2, y2 = np.ceil(boxes[index]).astype(int)
        masks[index, y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    nonempty = masks.any(axis=(1, 2))
    return {"masks": masks[nonempty], "scores": rows[nonempty, 4], "boxes": boxes[nonempty]}


class SegmentationBackend:
    def __init__(self, config: ServingConfig):
        import onnxruntime as ort  # noqa: PLC0415 -- isolated serving dependency
        from tokenizers import Tokenizer  # noqa: PLC0415

        self.config = config
        root = Path(config.segmentation_artifacts).expanduser()
        manifest = json.loads((root / "manifest.json").read_text())
        expected = {
            "model": config.segmentation_model,
            "revision": config.segmentation_revision,
            "image_size": config.image_size,
            "format": "promptable-yoloe-v1",
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("YOLOE artifacts do not match serving configuration; re-export them")
        options = ort.SessionOptions()  # ty: ignore[possibly-missing-attribute] -- pybind export
        options.intra_op_num_threads = config.segmentation_threads
        options.inter_op_num_threads = 1
        self.text = ort.InferenceSession(
            str(root / "text.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.vision = ort.InferenceSession(
            str(root / "vision.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self._embeddings: OrderedDict[str, np.ndarray] = OrderedDict()
        self.predict(np.zeros((480, 640, 3), dtype=np.uint8), "object")

    def embedding(self, prompt: str) -> np.ndarray:
        import ftfy  # noqa: PLC0415 -- isolated serving dependency

        prompt = " ".join(html.unescape(html.unescape(ftfy.fix_text(prompt))).split()).lower()
        if not prompt or not any(character.isalnum() for character in prompt):
            raise ValueError("Enter an object name or description")
        embedding = self._embeddings.pop(prompt, None)
        if embedding is None:
            tokens = np.asarray([self.tokenizer.encode(prompt).ids], dtype=np.int32)
            embedding = np.asarray(self.text.run(None, {"tokens": tokens})[0]).reshape(1, 1, 512)
        self._embeddings[prompt] = embedding
        while len(self._embeddings) > self.config.prompt_cache_size:
            self._embeddings.popitem(last=False)
        return embedding

    def predict(self, rgb: np.ndarray, prompt: str) -> Arrays:
        inputs = {
            "images": prepare_image(rgb, self.config.image_size),
            "text": self.embedding(prompt),
        }
        detections, prototypes = self.vision.run(None, inputs)
        return decode_instances(
            np.asarray(detections), np.asarray(prototypes), rgb.shape[:2], self.config
        )
