"""One resident model per dependency environment; no robot commands."""

import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from humanoid.serving.backends import GraspBackend
from humanoid.serving.native_grasp import NativeGraspBackend
from humanoid.serving.transport import LocalServer, json_arrays
from humanoid.types.serving import (
    AnyGraspConfig,
    Arrays,
    GraspRuntime,
    ModelKind,
    ModelPipeline,
    ServingConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", type=ModelKind, required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--owner", type=int, required=True)
    args = parser.parse_args()
    values = json.loads(args.config.read_text())
    values["anygrasp"] = AnyGraspConfig(**values["anygrasp"])
    values["pipeline"] = ModelPipeline(values["pipeline"])
    config = ServingConfig(**values)
    if args.kind is ModelKind.SEGMENTATION:
        from humanoid.serving.segmentation import SegmentationBackend  # noqa: PLC0415

        segmenter = SegmentationBackend(config)

        def predict(arrays: Arrays) -> Arrays:
            return segmenter.predict(arrays["rgb"], str(arrays["prompt"]))
    elif args.kind is ModelKind.ANYGRASP:
        from humanoid.serving.anygrasp import AnyGraspBackend  # noqa: PLC0415

        detector = AnyGraspBackend(config.anygrasp)

        def predict(arrays: Arrays) -> Arrays:
            return detector.predict(arrays["points"])
    else:
        grasp = (
            NativeGraspBackend(config)
            if GraspRuntime(config.grasp_runtime) is GraspRuntime.NATIVE
            else GraspBackend(config)
        )

        def predict(arrays: Arrays) -> Arrays:
            return grasp.predict(arrays["points"], arrays["conditioning"])

    lock = threading.Lock()

    def handle(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if str(arrays.get("operation", "")) == "status":
            return json_arrays({"ready": True, "pid": os.getpid()})
        with lock:
            return predict(arrays)

    server = LocalServer(args.socket, handle, config.request_timeout)
    try:
        while True:
            try:
                os.kill(args.owner, 0)
            except ProcessLookupError:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        if args.kind is ModelKind.GRASP and isinstance(grasp, NativeGraspBackend):
            grasp.close()


if __name__ == "__main__":
    main()
