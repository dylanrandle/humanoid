"""Native RAX observation trial. The only robot request is a camera/state GET."""

import base64
import io
import json
import os
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from PIL import Image, ImageDraw

from .backend import NoRedirect
from .tools import CONTACT_PREVIEW_TASK, contact_preview_tools

CONTACT_COUNT = 2
PIXEL_DIMENSIONS = 2


class ContactPreviewBackend:
    mutating = frozenset()

    def __init__(self, config):
        self.root = Path(config.root) / "contact-previews"
        self.endpoint = os.environ.get(
            "HUMANOID_OBSERVATION_ENDPOINT",
            "http://host.docker.internal:8765/api/hardware/observation",
        )
        self.snapshot: dict = {}
        self.close()

    def tools(self):
        return contact_preview_tools()

    @staticmethod
    def validate(name, arguments):
        schemas = {t["name"]: t["inputSchema"] for t in contact_preview_tools()}
        if (
            name not in schemas
            or not isinstance(arguments, dict)
            or set(arguments) != set(schemas[name]["required"])
        ):
            raise ValueError("Unsupported read-only trial call or arguments.")

    def _capture(self):
        self.observation_id = ""
        url = urlsplit(self.endpoint)
        headers = (
            {"Host": f"localhost:{url.port or 80}"}
            if url.hostname == "host.docker.internal"
            else {}
        )
        request = Request(self.endpoint, headers=headers, method="GET")
        try:
            with build_opener(NoRedirect, ProxyHandler({})).open(request, timeout=10) as response:
                snapshot = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"Hardware observation endpoint returned HTTP {exc.code}.") from exc
        if snapshot["runtime"] != "real" or snapshot["robot"] != "triskel":
            raise ValueError("Expected a live Triskel hardware observation.")
        self.snapshot = snapshot
        self.observation_id = uuid.uuid4().hex
        self.step_id += 1

    def start(self, *, formal, seed):
        self.close()
        self.episode_id = uuid.uuid4().hex
        try:
            return self.call("humanoid_observe", {})
        except Exception:
            self.close()
            raise

    def status(self):
        return {
            "active": bool(self.episode_id),
            "runtime": "real",
            "task": CONTACT_PREVIEW_TASK,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "observation_id": self.observation_id,
            "state": self.snapshot.get("state", {}),
        }

    def call(self, name, arguments):
        self.validate(name, arguments)
        if not self.episode_id:
            raise RuntimeError("Start a hardware observation episode first.")
        if name == "humanoid_status":
            return self.status(), []
        if name == "humanoid_mark_contacts":
            return self._mark(**arguments), []
        self._capture()
        images, metadata = [], []
        for kind in ("rgb", "depth"):
            frame = dict(self.snapshot["frames"][kind])
            images.append(
                {"type": "image", "mimeType": "image/png", "data": frame.pop("png_base64")}
            )
            metadata.append({**frame, "kind": kind, "content_index": len(images)})
        return {**self.status(), "attachments": metadata}, images

    def _mark(self, observation_id, contacts):
        if not self.observation_id or observation_id != self.observation_id:
            raise ValueError("Contacts must reference the latest observation.")
        if self.reported:
            raise ValueError("A contact preview was already saved for this episode.")
        frames = self.snapshot["frames"]
        rgb = base64.b64decode(frames["rgb"]["png_base64"], validate=True)
        with Image.open(io.BytesIO(rgb)) as source:
            image = source.convert("RGB")
        if (
            not isinstance(contacts, list)
            or len(contacts) != CONTACT_COUNT
            or any(
                not isinstance(point, list)
                or len(point) != PIXEL_DIMENSIONS
                or any(type(v) is not int for v in point)
                or not (0 <= point[0] < image.width and 0 <= point[1] < image.height)
                for point in contacts
            )
            or contacts[0] == contacts[1]
        ):
            raise ValueError("Provide two distinct in-bounds RGB contact pixels.")
        draw = ImageDraw.Draw(image)
        for (u, v), color in zip(contacts, ("cyan", "magenta"), strict=True):
            draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline=color, width=2)
        directory = self.root / self.episode_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "rgb.png").write_bytes(rgb)
        (directory / "depth.png").write_bytes(
            base64.b64decode(frames["depth"]["png_base64"], validate=True)
        )
        metadata = {
            k: {f: v for f, v in frame.items() if f != "png_base64"} for k, frame in frames.items()
        }
        image.save(directory / "contacts.png")
        record = {**self.status(), "frames": metadata, "contacts_rgb_px": contacts}
        (directory / "contacts.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )
        self.reported = True
        return {"saved": True, "observation_id": observation_id, "contacts_rgb_px": contacts}

    def evaluate(self, agent_assessment=None, confidence=None):
        return {
            "task": CONTACT_PREVIEW_TASK,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "status": "completed",
            "task_complete": self.reported,
            "evaluation_kind": "contact_overlay_saved",
            "decided_at_unix_s": time.time(),
        }

    def close(self):
        self.snapshot = {}
        self.episode_id = ""
        self.observation_id = ""
        self.step_id = 0
        self.reported = False
