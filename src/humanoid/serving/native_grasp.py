"""NumPy/CUDA C ABI client for the official PyTorch-free GraspGenX runtime."""

import ctypes as ct
import ctypes.util
import secrets
from pathlib import Path

import numpy as np

from humanoid.types.serving import Arrays, ServingConfig

MIN_POINTS = 100
EMBEDDING_SIZE = 512


class _RuntimeConfig(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_size_t),
        ("artifact_directory", ct.c_char_p),
        ("initial_gripper_depth", ct.c_float),
        ("denoiser_type", ct.c_int),
        ("diffusion_grasps", ct.c_int),
        ("obb_grasps", ct.c_int),
    ]


class CudaMemory:
    """Own small reusable host/device buffers without importing a GPU framework."""

    def __init__(self):
        self.library = ct.CDLL(
            ct.util.find_library("cudart") or "/usr/local/cuda/lib64/libcudart.so"
        )
        self.library.cudaMalloc.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t]
        self.library.cudaFree.argtypes = [ct.c_void_p]
        self.library.cudaMemcpy.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int]
        self.library.cudaGetErrorString.argtypes = [ct.c_int]
        self.library.cudaGetErrorString.restype = ct.c_char_p
        self.buffers: dict[str, tuple[np.ndarray, ct.c_void_p]] = {}
        self.check(self.library.cudaSetDevice(0))

    def check(self, error: int) -> None:
        if error:
            raise RuntimeError(self.library.cudaGetErrorString(error).decode())

    def allocate(self, name: str, shape: tuple[int, ...], dtype: np.dtype | type) -> ct.c_void_p:
        existing = self.buffers.get(name)
        if existing is not None and existing[0].shape == shape and existing[0].dtype == dtype:
            return existing[1]
        if existing is not None:
            self.check(self.library.cudaFree(existing[1]))
            del self.buffers[name]
        host = np.empty(shape, dtype=dtype)
        device = ct.c_void_p()
        self.check(self.library.cudaMalloc(ct.byref(device), host.nbytes))
        self.buffers[name] = host, device
        return device

    def upload(self, name: str, array: np.ndarray) -> ct.c_void_p:
        array = np.ascontiguousarray(array)
        device = self.allocate(name, array.shape, array.dtype)
        self.check(self.library.cudaMemcpy(device, array.ctypes.data, array.nbytes, 1))
        return device

    def download(self, name: str) -> np.ndarray:
        host, device = self.buffers[name]
        self.check(self.library.cudaMemcpy(host.ctypes.data, device, host.nbytes, 2))
        return host.copy()

    def synchronize(self) -> None:
        self.check(self.library.cudaDeviceSynchronize())

    def close(self) -> None:
        self.synchronize()
        for _, device in self.buffers.values():
            self.check(self.library.cudaFree(device))
        self.buffers.clear()


def bind_runtime(path: str) -> ct.CDLL:
    library = ct.CDLL(str(Path(path).expanduser()), mode=ct.RTLD_GLOBAL)
    pointer = ct.c_void_p
    signatures = {
        "create_v2": [ct.POINTER(_RuntimeConfig), ct.POINTER(pointer)],
        "destroy": [pointer],
        "set_gripper_device_async": [pointer, pointer, pointer, ct.c_float, pointer],
        "infer_device_async": [
            pointer,
            pointer,
            ct.c_int64,
            ct.c_uint64,
            ct.c_uint64,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
        ],
        "copy_valid_async": [pointer, pointer, pointer],
    }
    for name, signature in signatures.items():
        function = getattr(library, f"graspgenx_runtime_{name}")
        function.argtypes = signature
        function.restype = None if name == "destroy" else ct.c_int
    library.graspgenx_runtime_last_error.restype = ct.c_char_p
    return library


class NativeGraspBackend:
    def __init__(self, config: ServingConfig):
        import onnxruntime as ort  # noqa: PLC0415 -- isolated serving dependency

        self.config = config
        self.memory = CudaMemory()
        self.library = bind_runtime(config.grasp_library)
        self.handle = ct.c_void_p()
        self.seed = secrets.randbits(64)
        self.sequence = 0
        options = ort.SessionOptions()  # ty: ignore[possibly-missing-attribute] -- pybind export
        options.intra_op_num_threads = 1
        self.encoder = ort.InferenceSession(
            str(Path(config.grasp_artifacts).expanduser() / "gripper.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        native = _RuntimeConfig(
            ct.sizeof(_RuntimeConfig),
            str(Path(config.grasp_artifacts).expanduser()).encode(),
            0.1034,
            0,
            config.grasp_count,
            0,
        )
        try:
            self.check(
                self.library.graspgenx_runtime_create_v2(ct.byref(native), ct.byref(self.handle))
            )
            count = config.grasp_count
            self.poses = self.memory.allocate("poses", (count, 4, 4), np.float32)
            self.scores = self.memory.allocate("scores", (count,), np.float32)
            self.overflow = self.memory.allocate("overflow", (4,), np.uint8)
            self.valid = self.memory.allocate("valid", (count,), np.uint8)
            self.conditioning: np.ndarray | None = None
            points = np.random.default_rng(0).uniform(
                -0.025, 0.025, (config.grasp_warmup_points, 3)
            )
            self.predict(points.astype(np.float32), np.asarray(config.grasp_warmup_conditioning))
        except Exception:
            self.close()
            raise

    def check(self, error: int) -> None:
        if error:
            detail = self.library.graspgenx_runtime_last_error()
            raise RuntimeError(detail.decode() if detail else "Native GraspGenX error")

    def predict(self, points: np.ndarray, conditioning: np.ndarray) -> Arrays:
        if (
            points.ndim != 2  # noqa: PLR2004 -- Nx3 point cloud
            or points.shape[1:] != (3,)
            or len(points) < MIN_POINTS
            or conditioning.shape != (12,)
            or not np.isfinite(points).all()
            or not np.isfinite(conditioning).all()
            or np.any(conditioning[[0, 1, 2, 6, 7, 8]] <= 0)
        ):
            raise ValueError("Invalid grasp cloud or gripper conditioning")
        self.memory.check(self.memory.library.cudaSetDevice(0))
        if self.conditioning is None or not np.array_equal(self.conditioning, conditioning):
            generator, discriminator = self.encoder.run(
                None, {"conditioning": conditioning.astype(np.float32)[None]}
            )
            self.generator = self.memory.upload(
                "generator", np.asarray(generator, dtype=np.float16).reshape(512)
            )
            self.discriminator = self.memory.upload(
                "discriminator", np.asarray(discriminator, dtype=np.float32).reshape(512)
            )
            self.conditioning = conditioning.copy()
        depth = float(conditioning[5] + conditioning[2] / 2)
        self.check(
            self.library.graspgenx_runtime_set_gripper_device_async(
                self.handle,
                self.generator,
                self.discriminator,
                depth,
                None,
            )
        )
        device_points = self.memory.upload("points", points.astype(np.float32))
        self.check(
            self.library.graspgenx_runtime_infer_device_async(
                self.handle,
                device_points,
                len(points),
                self.seed,
                self.sequence,
                self.poses,
                self.scores,
                None,
                None,
                None,
                None,
                self.overflow,
                None,
            )
        )
        self.check(self.library.graspgenx_runtime_copy_valid_async(self.handle, self.valid, None))
        self.memory.synchronize()
        self.sequence = (self.sequence + 20 * self.config.grasp_count * 3) % (1 << 64)
        if self.memory.download("overflow").any():
            raise RuntimeError(
                "Native grasp encoder capacity/spatial overflow; refusing the result"
            )
        poses, scores = self.memory.download("poses"), self.memory.download("scores")
        if not np.isfinite(poses).all() or not np.isfinite(scores).all():
            raise RuntimeError("Native grasp model returned non-finite predictions")
        keep = (scores >= self.config.grasp_threshold) & self.memory.download("valid").astype(bool)
        return {"poses": poses[keep], "scores": scores[keep]}

    def close(self) -> None:
        if self.handle.value:
            self.memory.synchronize()
            self.library.graspgenx_runtime_destroy(self.handle)
            self.handle.value = None
        self.memory.close()
