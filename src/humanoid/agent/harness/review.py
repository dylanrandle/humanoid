"""Save each exploration turn and its exact model-facing images as an offline review."""

import base64
import hashlib
from pathlib import Path
from types import TracebackType

from pydantic_core import to_json

from humanoid.types.agent import AgentDecision, AgentFeedback, DecisionContext
from humanoid.types.trial import ExplorationTurn, TaskScore

REVIEW_INDEX_FILENAME = "index.html"
_TEMP_INDEX_FILENAME = f"{REVIEW_INDEX_FILENAME}.tmp"
_TEMPLATE_PATH = Path(__file__).with_name("exploration.html")
_TEMPLATE_PLACEHOLDER = "<!-- EXPLORATION_DATA -->"
_IMAGE_DIRECTORY = Path("images")
_IMAGE_SUFFIX = ".jpg"
_JPEG_DATA_URL_PREFIX = "data:image/jpeg;base64,"
_EXPLORING_STATUS = "Exploring"


class ExplorationReview:
    def __init__(self, folder: Path, seed: int) -> None:
        self.folder = folder
        self.seed = seed
        self.turns: list[ExplorationTurn] = []
        self.template = _TEMPLATE_PATH.read_text()
        (folder / _IMAGE_DIRECTORY).mkdir(parents=True)
        self._write()

    def _image(self, url: str) -> str:
        if not url.startswith(_JPEG_DATA_URL_PREFIX):
            raise ValueError("Expected a JPEG camera or perception preview.")
        data = base64.b64decode(url.removeprefix(_JPEG_DATA_URL_PREFIX), validate=True)
        relative_path = _IMAGE_DIRECTORY / f"{hashlib.sha256(data).hexdigest()}{_IMAGE_SUFFIX}"
        path = self.folder / relative_path
        if not path.exists():
            path.write_bytes(data)
        return relative_path.as_posix()

    def observe(self, elapsed: float, context: DecisionContext, images: dict[str, str]) -> None:
        paths = {label: self._image(url) for label, url in images.items()}
        self.turns.append(ExplorationTurn(elapsed, context, paths))
        self._write()

    def decided(self, decision: AgentDecision) -> None:
        self.turns[-1].decision = decision
        self._write()

    def outcome(self, result: AgentFeedback | None, preview: str | None = None) -> None:
        turn = self.turns[-1]
        turn.result = result
        if preview is not None:
            turn.result_image = self._image(preview)
        self._write()

    def finish(self, reason: str, score: TaskScore) -> None:
        self._write(reason, score=score)

    def _write(self, status: str = _EXPLORING_STATUS, *, score: TaskScore | None = None) -> None:
        payload = to_json(
            {"seed": self.seed, "status": status, "score": score, "turns": self.turns}
        )
        # Agent text can contain arbitrary markup, including a closing script tag.
        data = payload.decode().replace("<", "\\u003c")
        temporary = self.folder / _TEMP_INDEX_FILENAME
        temporary.write_text(self.template.replace(_TEMPLATE_PLACEHOLDER, data))
        temporary.replace(self.folder / REVIEW_INDEX_FILENAME)

    def __enter__(self):
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if error is not None:
            self._write(f"Interrupted · {type(error).__name__}: {error}")
