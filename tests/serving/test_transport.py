"""Private IPC preserves numeric data and supports repeated resident requests."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from humanoid.serving.transport import LocalServer, encode, request


@pytest.fixture
def socket_directory():
    # macOS has a short AF_UNIX path limit; pytest's default temp path exceeds it.
    with TemporaryDirectory(prefix="hm-test-", dir="/tmp") as directory:
        yield Path(directory)


def test_repeated_requests_preserve_arrays_and_do_not_restart_handler(socket_directory):
    path = str(socket_directory / "model.sock")
    calls = []

    def infer(arrays):
        calls.append(arrays["points"])
        return {"points": arrays["points"] * len(calls)}

    server = LocalServer(path, infer)
    try:
        points = np.arange(12, dtype=np.float32).reshape(4, 3)
        for multiplier in (1, 2):
            response = request(path, {"points": points}, 1)
            np.testing.assert_array_equal(response["points"], points * multiplier)
        with pytest.raises(RuntimeError, match="already running"):
            LocalServer(path, infer)
    finally:
        server.close()
    assert not (socket_directory / "model.sock").exists()


def test_model_error_is_returned_without_killing_service(socket_directory):
    path = str(socket_directory / "model.sock")

    def infer(arrays):
        if not len(arrays["points"]):
            raise ValueError("Empty target cloud")
        return arrays

    server = LocalServer(path, infer)
    try:
        with pytest.raises(RuntimeError, match="Empty target"):
            request(path, {"points": np.empty(0)}, 1)
        assert len(request(path, {"points": np.ones(1)}, 1)["points"]) == 1
    finally:
        server.close()


def test_transport_never_serializes_python_objects():
    with pytest.raises(ValueError, match="Object arrays"):
        encode({"unsafe": np.array([{}], dtype=object)})
