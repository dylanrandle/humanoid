"""Resident model supervisor with serialized inference and independent health RPC."""

import contextlib
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import psutil

from humanoid.serving.transport import LocalServer, json_arrays, request
from humanoid.types.serving import (
    ModelKind,
    ModelPipeline,
    ServingConfig,
    ServingState,
    ServingStatus,
)


class ModelServer:
    def __init__(self, config: ServingConfig):
        self.config = config
        self._status = ServingStatus(state=ServingState.LOADING, message="Loading models")
        self._stop = threading.Event()
        self._inference_lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._processes: dict[ModelKind, subprocess.Popen] = {}
        self._paths: dict[ModelKind, str] = {}
        self._temporary = TemporaryDirectory(prefix="hm-model-", dir="/tmp")
        self._server: LocalServer | None = None
        self._loader: threading.Thread | None = None

    @property
    def status(self) -> ServingStatus:
        return self._status

    def start(self) -> None:
        self._server = LocalServer(
            self.config.socket_path, self._handle, self.config.request_timeout
        )
        self._loader = threading.Thread(target=self._load, name="model-loader", daemon=True)
        self._loader.start()

    def workers(self) -> tuple[tuple[ModelKind, str], ...]:
        if self.config.pipeline is ModelPipeline.ANYGRASP:
            return ((ModelKind.ANYGRASP, self.config.anygrasp.python),)
        return (
            (ModelKind.SEGMENTATION, self.config.segmentation_python),
            (ModelKind.GRASP, self.config.grasp_python),
        )

    def _load(self) -> None:
        try:
            root = Path(self._temporary.name)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(asdict(self.config)))
            directory = Path(self.config.model_directory).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            pids = {}
            for kind, python in self.workers():
                self._status = ServingStatus(state=ServingState.LOADING, message=f"Loading {kind}")
                path = str(root / f"{kind}.sock")
                self._paths[kind] = path
                environment = dict(os.environ, YOLO_AUTOINSTALL="false")
                with self._process_lock:
                    if self._stop.is_set():
                        return
                    process = subprocess.Popen(
                        [
                            shutil.which("uv") or str(Path.home() / ".local/bin/uv"),
                            "run",
                            "--no-project",
                            "--python",
                            str(Path(python).expanduser()),
                            "python",
                            "-m",
                            "humanoid.serving.worker",
                            "--kind",
                            kind,
                            "--socket",
                            path,
                            "--config",
                            str(config_path),
                            "--owner",
                            str(os.getpid()),
                        ],
                        cwd=directory,
                        env=environment,
                        start_new_session=True,
                    )
                    self._processes[kind] = process
                deadline = time.monotonic() + self.config.startup_timeout
                while not self._stop.wait(0.1):
                    self._check_resources()
                    if process.poll() is not None:
                        raise RuntimeError(f"{kind} worker exited with code {process.returncode}")
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"Timed out loading {kind}")
                    try:
                        from humanoid.serving.transport import read_json  # noqa: PLC0415

                        status = read_json(request(path, {"operation": np.asarray("status")}, 1.0))
                        pids[kind] = status["pid"]
                        break
                    except (OSError, ConnectionError):
                        continue
            if not self._stop.is_set():
                self._status = ServingStatus(
                    state=ServingState.READY,
                    message="Models ready",
                    segmentation_pid=pids.get(ModelKind.SEGMENTATION),
                    grasp_pid=pids.get(ModelKind.ANYGRASP, pids.get(ModelKind.GRASP)),
                )
        except Exception as error:
            self.fail(str(error))

    def _handle(self, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        operation = str(arrays.get("operation", ""))
        if operation == "status":
            return json_arrays(asdict(self.status))
        kind = ModelKind(operation)
        if self.status.state is not ServingState.READY:
            raise RuntimeError(self.status.message)
        if kind not in self._paths:
            raise ValueError(f"{kind} is not loaded in the {self.config.pipeline} pipeline")
        if not self._inference_lock.acquire(blocking=False):
            raise RuntimeError("Model server is busy")
        try:
            self._check_resources()
            return request(self._paths[kind], arrays, self.config.request_timeout)
        finally:
            self._inference_lock.release()

    def _check_resources(self) -> None:
        if psutil.virtual_memory().available < self.config.minimum_host_memory_bytes:
            raise MemoryError("Model service stopped to preserve host memory")

    def monitor(self) -> None:
        if self.status.state not in {ServingState.LOADING, ServingState.READY}:
            return
        try:
            self._check_resources()
            with self._process_lock:
                for kind, process in self._processes.items():
                    if process.poll() is not None:
                        raise RuntimeError(f"{kind} worker exited with code {process.returncode}")
        except (MemoryError, RuntimeError) as error:
            self.fail(str(error))

    def fail(self, message: str) -> None:
        self._status = ServingStatus(state=ServingState.FAILED, message=message)
        self._stop.set()
        self._stop_workers()

    def _stop_workers(self) -> None:
        with self._process_lock:
            # uv and Python share a private process group. Kill the entire group
            # even if uv has already exited, then reap every launcher separately.
            for process in self._processes.values():
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            for process in self._processes.values():
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=2.0)

    def close(self) -> None:
        self._stop.set()
        self._stop_workers()
        if self._loader is not None:
            self._loader.join(timeout=3.0)
        if self._server is not None:
            self._server.close()
        self._temporary.cleanup()
        self._status = ServingStatus(state=ServingState.STOPPED, message="Models stopped")
