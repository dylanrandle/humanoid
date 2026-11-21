"""Point- and box-prompted instance segmentation on Triskel."""

from contextlib import ExitStack
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from humanoid.perception.depth import _cuda
from humanoid.types.segmentation import Pixel
from humanoid.utils.paths import find_data_root

ENCODER_SIZE = 1024
MASK_SIZE = 256
MAX_TOKENS = 16


class _Engine:
    """One execution context with reusable device buffers owned by the caller's stack."""

    def __init__(self, path: Path, runtime, cuda, stream, stack: ExitStack):
        raw = path.read_bytes()
        if raw.startswith(b"version https://git-lfs"):
            raise RuntimeError(f"Fetch the Git LFS model at {path}.")
        self.engine = runtime.deserialize_cuda_engine(raw)
        if self.engine is None:
            raise RuntimeError(
                f"TensorRT could not load {path}. Check the GPU and JetPack version."
            )
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create a TensorRT context for {path}.")
        self.cuda = cuda
        self.stream = stream
        self.buffers = {}
        self.shapes = {}
        self.stack = stack

    def bind(self, name, shape, *, pointer=None):
        if pointer is None:
            pointer = _cuda(
                self.cuda.cudaMalloc(int(np.prod(shape)) * np.dtype("float32").itemsize)
            )
            self.stack.callback(lambda: _cuda(self.cuda.cudaFree(pointer)))
        if not self.context.set_tensor_address(name, pointer):
            raise RuntimeError(f"Cannot bind NanoSAM tensor {name!r}.")
        self.buffers[name] = pointer
        self.shapes[name] = shape
        return pointer

    def upload(self, name, array):
        data = np.ascontiguousarray(array, dtype=np.float32)
        if data.size != np.prod(self.context.get_tensor_shape(name)):
            raise ValueError(f"Unexpected NanoSAM input size for {name}.")
        _cuda(
            self.cuda.cudaMemcpy(
                self.buffers[name],
                data.ctypes.data,
                data.nbytes,
                self.cuda.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
        )

    def download(self, name):
        array = np.empty(tuple(self.context.get_tensor_shape(name)), dtype=np.float32)
        _cuda(
            self.cuda.cudaMemcpy(
                array.ctypes.data,
                self.buffers[name],
                array.nbytes,
                self.cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )
        )
        return array[0]

    def run(self):
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError("NanoSAM TensorRT execution failed.")
        _cuda(self.cuda.cudaStreamSynchronize(self.stream))


class Segmenter:
    """Encode RGB once; predict new masks or refine a prior mask on that image."""

    def __init__(self, model_dir: Path | None = None):
        directory = model_dir or find_data_root() / "models" / "nanosam"
        trt = import_module("tensorrt")
        cuda = import_module("cuda.bindings.runtime")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.size: tuple[int, int] | None = None
        self.closed = False
        with ExitStack() as stack:
            _cuda(cuda.cudaSetDevice(0))
            stream = _cuda(cuda.cudaStreamCreate())
            stack.callback(lambda: _cuda(cuda.cudaStreamDestroy(stream)))
            self.enc = _Engine(directory / "encoder.plan", self.runtime, cuda, stream, stack)
            self.dec = _Engine(directory / "decoder.plan", self.runtime, cuda, stream, stack)
            for engine in (self.enc, self.dec):
                for i in range(engine.engine.num_io_tensors):
                    name = engine.engine.get_tensor_name(i)
                    if engine.engine.get_tensor_dtype(name) != trt.float32:
                        raise ValueError(f"Unexpected NanoSAM binding dtype for {name}.")
                    shape = tuple(engine.engine.get_tensor_shape(name))
                    if name in ("point_coords", "point_labels"):
                        _, _, shape = engine.engine.get_tensor_profile_shape(name, 0)
                        if shape[1] != MAX_TOKENS:
                            raise ValueError("NanoSAM decoder needs the multi-point profile.")
                        engine.context.set_input_shape(name, tuple(shape))
            for engine in (self.enc, self.dec):
                for i in range(engine.engine.num_io_tensors):
                    name = engine.engine.get_tensor_name(i)
                    shape = tuple(engine.context.get_tensor_shape(name))
                    if min(shape) <= 0:
                        raise ValueError(f"Unresolved NanoSAM binding {name} {shape}.")
                    shared = self.enc.buffers.get(name) if engine is self.dec else None
                    if shared is not None and shape != self.enc.shapes[name]:
                        raise ValueError(f"NanoSAM engines disagree on the shape of {name}.")
                    engine.bind(name, shape, pointer=shared)
            self.resources = stack.pop_all()

    def set_image(self, rgb: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("NanoSAM is closed.")
        h, w = rgb.shape[:2]
        if rgb.shape != (h, w, 3) or rgb.dtype != np.uint8 or min(h, w) <= 0:
            raise ValueError("NanoSAM requires uint8 RGB pixels.")
        scale = ENCODER_SIZE / max(w, h)
        rw, rh = int(w * scale), int(h * scale)
        pixels = np.asarray(Image.fromarray(rgb).resize((rw, rh)), dtype=np.float32)
        pixels -= np.array([123.675, 116.28, 103.53], np.float32)
        pixels /= np.array([58.395, 57.12, 57.375], np.float32)
        batch = np.zeros((1, 3, ENCODER_SIZE, ENCODER_SIZE), np.float32)
        batch[0, :, :rh, :rw] = pixels.transpose(2, 0, 1)
        self.enc.upload("image", batch)
        self.enc.run()
        self.size = (w, h)

    def predict(
        self, points: tuple[Pixel, ...], labels: tuple[int, ...], prior: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.size is None:
            raise RuntimeError("NanoSAM needs an encoded image.")
        if not points or len(points) != len(labels) or len(points) > MAX_TOKENS:
            raise ValueError(f"Expected 1 to {MAX_TOKENS} prompt coordinates.")
        coords = np.asarray(points, np.float32) * ENCODER_SIZE / max(self.size)
        if len(points) == 1:
            coords = np.vstack([coords, np.zeros((1, 2), np.float32)])
            labels = (*labels, -1)
        n = len(labels)
        if not self.dec.context.set_input_shape(
            "point_coords", (1, n, 2)
        ) or not self.dec.context.set_input_shape("point_labels", (1, n)):
            raise RuntimeError("NanoSAM decoder rejected the prompt length.")
        self.dec.upload("point_coords", coords)
        self.dec.upload("point_labels", labels)
        if prior is not None and prior.shape != (MASK_SIZE, MASK_SIZE):
            raise ValueError("Invalid NanoSAM prior logit shape.")
        self.dec.upload(
            "mask_input", np.zeros((MASK_SIZE, MASK_SIZE), np.float32) if prior is None else prior
        )
        self.dec.upload("has_mask_input", [float(prior is not None)])
        self.dec.run()
        return self.dec.download("iou_predictions"), self.dec.download("low_res_masks")

    def close(self):
        if not self.closed:
            self.resources.close()
            del self.enc, self.dec, self.runtime
            self.size = None
            self.closed = True


def resize_mask(logits: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    valid = logits[: int(MASK_SIZE * h / max(size)), : int(MASK_SIZE * w / max(size))]
    return cv2.resize(valid, size, interpolation=cv2.INTER_LINEAR) > 0
