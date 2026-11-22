"""Session-local prompt execution on a pinned RGB-D observation."""

from contextlib import ExitStack, closing

import cv2
import numpy as np
from PIL import Image, ImageDraw

from humanoid.agent.harness.observation import PinnedObservation
from humanoid.perception.segmentation import MASK_SIZE, MAX_TOKENS, Segmenter, resize_mask
from humanoid.types.camera import CameraView, ObservationId
from humanoid.types.segmentation import Box, MaskCandidate, MaskProposal, Pixel, SegmentationResult
from humanoid.types.tools import (
    RefineMask,
    SegmentAtPoint,
    SegmentEverything,
    SegmentInBox,
    SegmentRequest,
)

_GRID_ROWS = 12
_GRID_COLS = 16
_STABILITY = 0.90
_MIN_SCORE = 0.82
_MIN_AREA_FRACTION = 0.0005
_MAX_AREA_FRACTION = 0.60
_DUPLICATE_IOU = 0.70


def _pixel(value: Pixel, size: tuple[int, int]) -> Pixel:
    x, y = value
    if not (0 <= x < size[0] and 0 <= y < size[1]):
        raise ValueError(f"Pixel lies outside the {size[0]} by {size[1]} observation.")
    return x, y


def _box(value: Box, size: tuple[int, int]) -> Box:
    x, y = _pixel(value[:2], size)
    r, b = _pixel(value[2:], size)
    if r <= x or b <= y:
        raise ValueError("Box corners must be ordered and enclose an area.")
    return x, y, r, b


def _distinct(ranked: list[MaskCandidate], size: tuple[int, int]) -> list[MaskCandidate]:
    kept = []
    masks = []
    w, h = size
    for c in sorted(ranked, key=lambda c: -c.score):
        visible = c.logits[: int(MASK_SIZE * h / max(size)), : int(MASK_SIZE * w / max(size))]
        mask = visible > 0
        area = int(mask.sum())
        if not area or not np.isfinite(c.score):
            continue
        for other, n in masks:
            if min(n, area) / max(n, area) > _DUPLICATE_IOU:
                overlap = np.count_nonzero(mask & other)
                if overlap / (n + area - overlap) > _DUPLICATE_IOU:
                    break
        else:
            kept.append(c)
            masks.append((mask, area))
    return kept


class SegmentationTools:
    def __init__(self, observation: PinnedObservation):
        self.observation = observation
        self.resources = ExitStack()
        self.segmenter: Segmenter | None = None
        self.encoded_observation_id: str | None = None

    def close(self):
        self.resources.close()

    def _prepare(self, frames: CameraView) -> Segmenter:
        if self.segmenter is None:
            self.segmenter = self.resources.enter_context(closing(Segmenter()))
        if self.encoded_observation_id != self.observation.id:
            self.observation.save()
            self.segmenter.set_image(frames.rgb.data)
            self.encoded_observation_id = self.observation.id
        return self.segmenter

    def _grid(
        self, segmenter: Segmenter, region: Box, size: tuple[int, int]
    ) -> list[MaskCandidate]:
        x, y, r, b = region
        ranked = []
        for cy in (np.arange(_GRID_ROWS) + 0.5) * (b - y + 1) / _GRID_ROWS + y:
            for cx in (np.arange(_GRID_COLS) + 0.5) * (r - x + 1) / _GRID_COLS + x:
                points = ((int(cx), int(cy)),)
                scores, masks = segmenter.predict(points, (1,))
                for score, low in zip(scores[1:], masks[1:], strict=True):
                    if score < _MIN_SCORE:
                        continue
                    w, h = size
                    visible = low[
                        : int(MASK_SIZE * h / max(size)), : int(MASK_SIZE * w / max(size))
                    ]
                    area = np.count_nonzero(visible > 0) / visible.size
                    stability = np.count_nonzero(visible > 1) / max(
                        1, np.count_nonzero(visible > -1)
                    )
                    if _MIN_AREA_FRACTION <= area <= _MAX_AREA_FRACTION and stability >= _STABILITY:
                        ranked.append(MaskCandidate(float(score), low.copy(), points, (1,)))
        return _distinct(ranked, size)

    def _prompt(self, request: SegmentAtPoint | SegmentInBox | RefineMask, size: tuple[int, int]):
        prior = None
        if isinstance(request, SegmentInBox):
            x, y, r, b = _box(request.box_xyxy, size)
            points, labels = ((x, y), (r, b)), (2, 3)
        elif isinstance(request, SegmentAtPoint):
            points, labels = (_pixel(request.point_xy, size),), (1,)
        else:
            previous = self.observation.store.prior(request.mask_id, self.observation.id)
            include = tuple(_pixel(p, size) for p in request.include_points)
            exclude = tuple(_pixel(p, size) for p in request.exclude_points)
            if not include and not exclude:
                raise ValueError("Supply at least one include or exclude point.")
            points = previous.points + include + exclude
            labels = previous.labels + (1,) * len(include) + (0,) * len(exclude)
            prior = previous.logits
        if len(points) > MAX_TOKENS:
            raise ValueError(f"Prompt history exceeds {MAX_TOKENS} coordinates; start a new mask.")
        return points, labels, prior

    def segment(self, request: SegmentRequest) -> tuple[SegmentationResult, Image.Image]:
        observation_id = (
            ObservationId(request.mask_id.partition(".")[0])
            if isinstance(request, RefineMask)
            else request.observation_id
        )
        self.observation.select(observation_id)
        frames = self.observation.frames
        assert frames is not None
        h, w = frames.rgb.data.shape[:2]
        size = (w, h)
        if isinstance(request, SegmentEverything):
            region = request.region_xyxy
            bounds = (0, 0, w - 1, h - 1) if region is None else _box(region, size)
            candidates = self._grid(self._prepare(frames), bounds, size)
        else:
            points, labels, prior = self._prompt(request, size)
            scores, masks = self._prepare(frames).predict(points, labels, prior)
            candidates = [
                MaskCandidate(float(scores[i]), masks[i], points, labels)
                for i in np.argsort(-scores)
            ]
            candidates = _distinct(candidates, size)
            if isinstance(request, RefineMask) or not request.return_alternatives:
                candidates = candidates[:1]
        return self._result(frames, candidates)

    def _result(
        self, frames: CameraView, candidates: list[MaskCandidate]
    ) -> tuple[SegmentationResult, Image.Image]:
        canvas = frames.rgb.data.copy()
        h, w = canvas.shape[:2]
        rows = []
        markers = []
        for c in candidates:
            mask = resize_mask(c.logits, (w, h))
            if not mask.any():
                continue
            number = len(rows) + 1
            color = tuple(
                int(v)
                for v in cv2.cvtColor(
                    np.array([[[(number * 41) % 180, 205, 250]]], dtype=np.uint8), cv2.COLOR_HSV2RGB
                )[0, 0]
            )
            ys, xs = np.nonzero(mask)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, color, 2)
            # Place the number inside its mask, including for concave objects.
            interior = mask.astype(np.uint8)
            for (px, py), _, _ in markers:
                cv2.circle(interior, (px, py), 12, 0, -1)
            if not interior.any():
                interior = mask.astype(np.uint8)
            pos = cv2.minMaxLoc(cv2.distanceTransform(interior, cv2.DIST_L2, 3))[3]
            markers.append((pos, number, color))
            rows.append(
                MaskProposal(
                    number=number,
                    mask_id=self.observation.store.save_mask(self.observation.id, c, mask),
                    box_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                    pixels=len(xs),
                    predicted_iou=c.score,
                )
            )
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        for (x, y), n, color in markers:
            draw.text((x, y), str(n), fill=color, stroke_width=2, stroke_fill="black", anchor="mm")
        return SegmentationResult(observation_id=self.observation.id, masks=tuple(rows)), image
