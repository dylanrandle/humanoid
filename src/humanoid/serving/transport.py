"""Bounded numeric-array RPC over private local Unix sockets; no pickle."""

import contextlib
import io
import json
import socket
import socketserver
import struct
import threading
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from humanoid.types.serving import Arrays, Handler

MAX_MESSAGE_BYTES = 32 * 1024 * 1024
HEADER = struct.Struct("!I")


def encode(arrays: Arrays) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, array in arrays.items():
            with archive.open(f"{name}.npy", "w") as entry:
                np.lib.format.write_array(entry, array, allow_pickle=False)
    data = stream.getvalue()
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("Model request exceeds the message limit")
    return data


def decode(data: bytes) -> Arrays:
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def json_arrays(value: object) -> Arrays:
    return {"json": np.asarray(json.dumps(value))}


def read_json(arrays: Arrays) -> Any:
    return json.loads(str(arrays["json"]))


def _read(connection: socket.socket, length: int) -> bytes:
    pieces = []
    while length:
        piece = connection.recv(min(length, 65536))
        if not piece:
            raise ConnectionError("Model service closed the connection")
        pieces.append(piece)
        length -= len(piece)
    return b"".join(pieces)


def receive(connection: socket.socket) -> Arrays:
    length = HEADER.unpack(_read(connection, HEADER.size))[0]
    if length > MAX_MESSAGE_BYTES:
        raise ValueError("Model response exceeds the message limit")
    return decode(_read(connection, length))


def send(connection: socket.socket, arrays: Arrays) -> None:
    data = encode(arrays)
    connection.sendall(HEADER.pack(len(data)) + data)


def request(path: str, arrays: Arrays, timeout: float) -> Arrays:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(Path(path).expanduser()))
        send(connection, arrays)
        result = receive(connection)
    if "error" in result:
        raise RuntimeError(str(result["error"]))
    return result


class LocalServer:
    """Private socket with concurrent status requests and bounded connections."""

    def __init__(self, path: str, handler: Handler, timeout: float = 120.0):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Never unlink a live server's socket. A stale socket may be removed.
        if self.path.exists():
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                try:
                    probe.connect(str(self.path))
                except ConnectionRefusedError:
                    self.path.unlink()
                else:
                    raise RuntimeError(f"Service already running at {self.path}")

        class Connection(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                self.request.settimeout(timeout)
                try:
                    result = handler(receive(self.request))
                except Exception as error:
                    result = {"error": np.asarray(str(error))}
                # The caller may cancel while inference is in progress.
                with contextlib.suppress(OSError):
                    send(self.request, result)

        self.server = socketserver.ThreadingUnixStreamServer(str(self.path), Connection)
        self.server.daemon_threads = True
        self.path.chmod(0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
        self.path.unlink(missing_ok=True)
