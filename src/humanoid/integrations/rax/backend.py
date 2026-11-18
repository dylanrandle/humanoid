"""Stdlib-only RAX backend: the host never imports robot or simulator packages.

Implements the lifecycle of RAX dexjoco/native at revision
4b577aa2ba0039ad01707ede236be9913274180e. Endpoint/token are operator environment
settings, excluded from the generated agent workspace. Only simulation is served.
"""

import json
import os
import urllib.error
import urllib.request
from typing import override
from urllib.parse import urlsplit

from humanoid.integrations.rax.tools import tool_definitions, validate_request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    @override
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "RPC redirects are forbidden", headers, fp)


class HumanoidBackend:
    mutating = frozenset(
        {"humanoid_move_tool", "humanoid_move_base", "humanoid_gripper", "humanoid_hold"}
    )

    def __init__(self, config):
        self.config = config
        self.endpoint = os.environ.get(
            "HUMANOID_RAX_ENDPOINT", "http://127.0.0.1:8765/api/autonomy/rpc"
        )
        self.token = os.environ.get("HUMANOID_RAX_TOKEN", "")
        if not self.token:
            raise ValueError("Operator must set HUMANOID_RAX_TOKEN.")
        self._active = False

    def tools(self) -> list[dict]:
        return tool_definitions()

    @staticmethod
    def validate(name: str, arguments: dict) -> None:
        validate_request(name, arguments)

    def _request(self, method: str, **arguments):
        endpoint = urlsplit(self.endpoint)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"}
        if endpoint.hostname == "host.docker.internal":
            headers["Host"] = f"localhost:{endpoint.port or 8765}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps({"method": method, "arguments": arguments}, allow_nan=False).encode(),
            headers=headers,
            method="POST",
        )
        try:
            opener = urllib.request.build_opener(NoRedirect, urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=120) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Humanoid service rejected request: HTTP {error.code}") from error
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["result"]

    def start(self, *, formal: bool, seed: int):
        result = self._request("start", task=self.config.task, formal=formal, seed=seed)
        self._active = True
        return result["value"], result["images"]

    def call(self, name: str, arguments: dict):
        self.validate(name, arguments)
        result = self._request("call", name=name, arguments=arguments)
        return result["value"], result["images"]

    def status(self) -> dict:
        return self.call("humanoid_status", {})[0]

    def evaluate(self, agent_assessment=None, confidence=None) -> dict:
        return self._request("evaluate")

    def close(self) -> None:
        if self._active:
            self._request("close")
            self._active = False
