"""Depth completion for aligned 640 x 480 RealSense frames on Triskel."""

import mmap
from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
from threading import Lock
from types import TracebackType
from typing import Self

import cv2
import numpy as np

from humanoid.types.depth import DepthCamera
from humanoid.utils.paths import find_data_root

_FRAME = (480, 640)
_NETWORK = (518, 686)
_INPUT = (1, 4, *_NETWORK)
_OUTPUT = (1, *_NETWORK)
_MAX_DEPTH_M = 25


def _prepare(rgb: np.ndarray, depth: np.ndarray, scale: float) -> np.ndarray:
    if rgb.shape != (*_FRAME, 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected a 480 x 640 uint8 RGB image")
    if depth.shape != _FRAME or depth.dtype.kind not in "uif":
        raise ValueError("Expected a 480 x 640 numeric depth image aligned to RGB")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Depth scale must be positive and finite")
    metres = depth.astype(np.float32) * scale
    inverse = np.divide(
        1, metres, out=np.zeros_like(metres), where=(metres > 0) & (metres <= _MAX_DEPTH_M)
    )
    # Preserve the channel reversal in CDM's upstream RGB file inference path.
    color = cv2.resize(rgb[:, :, ::-1] / 255.0, _NETWORK[::-1], interpolation=cv2.INTER_CUBIC)
    color = (color - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
    tensor = np.empty(_INPUT, np.float16)
    tensor[0, :3] = color.astype(np.float32).transpose(2, 0, 1)
    tensor[0, 3] = cv2.resize(inverse, _NETWORK[::-1], interpolation=cv2.INTER_NEAREST)
    return tensor


def _metres(output: np.ndarray) -> np.ndarray:
    inverse = cv2.resize(
        output[0].astype(np.float32), _FRAME[::-1], interpolation=cv2.INTER_NEAREST
    )
    return np.divide(
        1, inverse, out=np.full_like(inverse, np.nan), where=np.isfinite(inverse) & (inverse > 0)
    )


def _cuda(result: tuple):
    error, *values = result
    if int(error):
        raise RuntimeError(f"CUDA operation failed with {error}")
    return values[0] if values else None


class DepthCompleter:
    """One camera-specific engine. Use as a context manager or call close()."""

    def __init__(self, camera: DepthCamera | str, model_dir: Path | None = None):
        self.camera = DepthCamera(camera)
        directory = find_data_root() / "models" / "cdm" if model_dir is None else Path(model_dir)
        path = directory / f"{self.camera}.plan"
        self._lock = Lock()
        self._closed = False
        with path.open("rb") as file:
            if file.read(42).startswith(b"version https://git-lfs"):
                raise ValueError(f"{path} is a Git LFS pointer. Run git lfs pull.")
            trt = import_module("tensorrt")
            self._cuda = import_module("cuda.bindings.runtime")
            _cuda(self._cuda.cudaSetDevice(0))
            self._logger = trt.Logger(trt.Logger.WARNING)
            self._runtime = trt.Runtime(self._logger)
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
                self._engine = self._runtime.deserialize_cuda_engine(data)
        if self._engine is None:
            raise RuntimeError(
                f"TensorRT could not load {path}. Check the GPU and TensorRT version."
            )
        for name, shape, mode in (
            ("rgbd", _INPUT, trt.TensorIOMode.INPUT),
            ("inverse_depth", _OUTPUT, trt.TensorIOMode.OUTPUT),
        ):
            if (
                self._engine.get_tensor_mode(name) != mode
                or tuple(self._engine.get_tensor_shape(name)) != shape
                or self._engine.get_tensor_dtype(name) != trt.float16
            ):
                raise ValueError(f"Unexpected CDM engine binding {name} in {path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("TensorRT could not create a CDM execution context")
        self._output = np.empty(_OUTPUT, np.float16)
        with ExitStack() as resources:
            self._stream = _cuda(self._cuda.cudaStreamCreate())
            resources.callback(lambda: _cuda(self._cuda.cudaStreamDestroy(self._stream)))
            self._input_device = self._allocate(
                "rgbd", int(np.prod(_INPUT)) * np.dtype(np.float16).itemsize, resources
            )
            self._output_device = self._allocate("inverse_depth", self._output.nbytes, resources)
            self._resources = resources.pop_all()

    def _allocate(self, name: str, size: int, resources: ExitStack) -> int:
        address = _cuda(self._cuda.cudaMalloc(size))
        resources.callback(lambda: _cuda(self._cuda.cudaFree(address)))
        if not self._context.set_tensor_address(name, int(address)):
            raise RuntimeError(f"TensorRT could not bind {name}")
        return address

    def infer(
        self, rgb: np.ndarray, aligned_depth: np.ndarray, *, metres_per_unit: float
    ) -> np.ndarray:
        """Return float32 axial depth in metres. Invalid predictions are NaN."""
        tensor = _prepare(rgb, aligned_depth, metres_per_unit)
        with self._lock:
            if self._closed:
                raise RuntimeError("DepthCompleter is closed")
            cuda = self._cuda
            _cuda(cuda.cudaSetDevice(0))
            _cuda(
                cuda.cudaMemcpy(
                    self._input_device,
                    tensor.ctypes.data,
                    tensor.nbytes,
                    cuda.cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
            )
            if not self._context.execute_async_v3(int(self._stream)):
                raise RuntimeError("TensorRT depth inference failed")
            _cuda(cuda.cudaStreamSynchronize(self._stream))
            _cuda(
                cuda.cudaMemcpy(
                    self._output.ctypes.data,
                    self._output_device,
                    self._output.nbytes,
                    cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )
            )
            return _metres(self._output)

    def warmup(self, runs: int = 1) -> None:
        """Run the full inference path with a synthetic one-metre frame."""
        if runs < 1:
            raise ValueError("Warmup requires at least one run")
        rgb = np.zeros((*_FRAME, 3), np.uint8)
        depth = np.ones(_FRAME, np.float32)
        for _ in range(runs):
            self.infer(rgb, depth, metres_per_unit=1.0)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                _cuda(self._cuda.cudaSetDevice(0))
                self._resources.close()
                del self._context, self._engine, self._runtime
                self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
