"""Immutable, frame-specific masks and RGB-aligned depth for downstream perception."""

import json
import re
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
from PIL import Image
from pydantic import TypeAdapter

from humanoid.types.camera import CameraView, CaptureMetadata, ObservationId
from humanoid.types.segmentation import MaskCandidate, MaskId, MaskSelection
from humanoid.utils.paths import find_data_root

_CAPTURE_METADATA = TypeAdapter(CaptureMetadata)


def _read_capture(folder: Path) -> CaptureMetadata:
    capture = _CAPTURE_METADATA.validate_json((folder / "capture.json").read_bytes(), strict=True)
    if capture.observation_id != folder.name:
        raise ValueError("Capture metadata belongs to a different observation.")
    return capture


class MaskStore:
    def __init__(self, root: Path | None = None):
        self.root = root if root is not None else find_data_root() / "logs" / "segmentation"

    def _observation(self, observation_id: ObservationId) -> Path:
        if not re.fullmatch("[a-f0-9]{32}", observation_id):
            raise ValueError("Invalid observation ID.")
        return self.root / observation_id

    def save_capture(
        self, observation_id: ObservationId, frames: CameraView, depth_m: np.ndarray | None = None
    ):
        dest = self._observation(observation_id)
        reg = frames.registered
        if depth_m is not None and (
            reg is None or depth_m.shape != frames.rgb.data.shape[:2] or depth_m.dtype != np.float32
        ):
            raise ValueError("Expected registered float32 depth matching the RGB capture.")
        rgb, depth = frames.rgb, frames.depth
        info = CaptureMetadata(
            observation_id=observation_id,
            camera_model=reg.camera_model if reg else None,
            serial_number=rgb.serial_number,
            rgb_frame_id=rgb.frame_id,
            depth_frame_id=depth.frame_id if depth else None,
            rgb_frame_number=rgb.frame_number,
            depth_frame_number=depth.frame_number if depth else None,
            depth_scale_m_per_unit=depth.depth_scale if depth else None,
            rgb_device_timestamp=rgb.device_timestamp,
            depth_device_timestamp=depth.device_timestamp if depth else None,
            rgb_timestamp_domain=rgb.timestamp_domain,
            depth_timestamp_domain=depth.timestamp_domain if depth else None,
            intrinsics=rgb.intrinsics,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.root) as tmp:
            stage = Path(tmp) / "capture"
            stage.mkdir()
            Image.fromarray(rgb.data).save(stage / "rgb.png")
            if depth_m is not None:
                np.save(stage / "completed_depth_m.npy", depth_m, allow_pickle=False)
            (stage / "capture.json").write_text(json.dumps(asdict(info), allow_nan=False))
            stage.rename(dest)

    def save_depth(self, observation_id: ObservationId, depth_m: np.ndarray) -> None:
        folder = self._observation(observation_id)
        capture = _read_capture(folder)
        intr = capture.intrinsics
        if (
            capture.camera_model is None
            or depth_m.shape != (intr.height, intr.width)
            or depth_m.dtype != np.float32
        ):
            raise ValueError("Expected registered float32 depth matching the RGB capture.")
        with TemporaryDirectory(dir=folder) as tmp:
            stage = Path(tmp) / "depth.npy"
            np.save(stage, depth_m, allow_pickle=False)
            (folder / "completed_depth_m.npy").hardlink_to(stage)

    def resolve_depth(self, observation_id: ObservationId) -> np.ndarray | None:
        """Read the full metric depth map by capture ID, including captures without masks."""
        folder = self._observation(observation_id)
        if not folder.is_dir():
            raise ValueError("Unknown saved observation ID.")
        path = folder / "completed_depth_m.npy"
        return np.load(path, mmap_mode="r", allow_pickle=False) if path.is_file() else None

    def save_mask(
        self, observation_id: ObservationId, candidate: MaskCandidate, mask: np.ndarray
    ) -> MaskId:
        folder = self._observation(observation_id)
        key = uuid4().hex
        with TemporaryDirectory(dir=folder) as tmp:
            stage = Path(tmp) / "mask.npz"
            np.savez_compressed(
                stage,
                mask=mask,
                predicted_iou=candidate.score,
                logits=candidate.logits,
                points=candidate.points,
                labels=candidate.labels,
            )
            stage.rename(folder / f"{key}.npz")
        return MaskId(f"{observation_id}.{key}")

    def _path(self, mask_id: MaskId) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}\.[a-f0-9]{32}", mask_id):
            raise ValueError("Invalid mask ID.")
        obs, key = mask_id.split(".")
        path = self._observation(ObservationId(obs)) / f"{key}.npz"
        if not path.is_file():
            raise ValueError("Unknown mask ID.")
        return path

    def prior(self, mask_id: MaskId, observation_id: ObservationId) -> MaskCandidate:
        path = self._path(mask_id)
        if path.parent.name != observation_id:
            raise ValueError("The mask belongs to a different observation.")
        with np.load(path, allow_pickle=False) as data:
            return MaskCandidate(
                float(data["predicted_iou"]),
                data["logits"],
                tuple(map(tuple, data["points"].tolist())),
                tuple(data["labels"].tolist()),
            )

    def resolve(self, mask_id: MaskId) -> MaskSelection:
        """Load the mask and original full scene in calibrated RGB coordinates."""
        path = self._path(mask_id)
        folder = path.parent
        capture = _read_capture(folder)
        with np.load(path, allow_pickle=False) as data:
            mask = data["mask"]
        with Image.open(folder / "rgb.png") as image:
            rgb = np.array(image)
        depth = self.resolve_depth(capture.observation_id)
        if rgb.shape[:2] != mask.shape or (depth is not None and depth.shape != mask.shape):
            raise ValueError("Inconsistent saved observation.")
        return MaskSelection(
            mask_id, capture.observation_id, rgb, depth, mask, capture.intrinsics, capture
        )
