"""Compare a CDM TensorRT engine against saved full-frame PyTorch predictions."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from humanoid.perception.depth import DepthCompleter
from humanoid.types.depth import DepthCamera

MIN_VALID_AGREEMENT = 0.999


def compare(output: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict:
    if reference.shape != output.shape or mask.shape != output.shape:
        raise ValueError("Predictions, reference and object mask must share the same shape")
    valid = np.isfinite(output) & (output > 0)
    expected = np.isfinite(reference) & (reference > 0)
    joint = valid & expected
    if not np.any(joint & mask):
        raise ValueError("No comparable pixels on the object")
    difference = np.abs(output - reference)
    relative = difference[joint] / reference[joint]
    object_mm = difference[joint & mask] * 1000
    return {
        "valid_agreement": float(np.mean(valid == expected)),
        "mean_relative": float(relative.mean()),
        "p99_relative": float(np.quantile(relative, 0.99)),
        "object_median_mm": float(np.median(object_mm)),
        "object_p95_mm": float(np.quantile(object_mm, 0.95)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("camera", choices=list(DepthCamera))
    for name in ("models", "captures", "references", "masks"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    scenes = sorted(args.captures.glob("scene-*"))
    if not scenes:
        p.error("Capture directory contains no scenes")
    passed = True
    with DepthCompleter(args.camera, args.models) as model:
        model.warmup()
        for scene in scenes:
            meta = json.loads((scene / "metadata.json").read_text())
            # The original D435 session predates the camera_model metadata field.
            recorded_camera = meta.get("camera_model")
            if recorded_camera and not recorded_camera.lower().endswith(model.camera):
                raise ValueError(f"Camera mismatch in {scene}")
            rgb = np.array(Image.open(scene / "rgb.png").convert("RGB"))
            raw = np.array(Image.open(scene / "depth_in_rgb.png"))
            start = perf_counter()
            output = model.infer(rgb, raw, metres_per_unit=meta["depth_scale_m_per_unit"])
            seconds = perf_counter() - start
            ref = np.load(args.references / f"{scene.name}-cdm.npy", allow_pickle=False)
            mask = np.asarray(Image.open(args.masks / f"{scene.name}-mask.png")).astype(bool)
            metrics = compare(output, ref, mask)
            limits = {
                "mean_relative": 0.01,
                "p99_relative": 0.03,
                "object_median_mm": 2,
                "object_p95_mm": 5,
            }
            ok = metrics["valid_agreement"] >= MIN_VALID_AGREEMENT and all(
                metrics[k] <= v for k, v in limits.items()
            )
            passed &= ok
            print(
                json.dumps({"scene": scene.name, "seconds": seconds, **metrics, "passed": ok}),
                flush=True,
            )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
