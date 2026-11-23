"""Save each exploration turn and its exact model-facing images as an offline review."""

import base64
import hashlib
from pathlib import Path
from types import TracebackType

from pydantic_core import to_json

from humanoid.types.agent import AgentDecision, AgentFeedback, DecisionContext
from humanoid.types.trial import ExplorationTurn, TaskScore


class ExplorationReview:
    def __init__(self, folder: Path, seed: int) -> None:
        self.folder = folder
        self.seed = seed
        self.turns: list[ExplorationTurn] = []
        self.template = Path(__file__).with_name("exploration.html").read_text()
        (folder / "images").mkdir(parents=True)
        self._write("Exploring")

    def _image(self, url: str) -> str:
        prefix = "data:image/jpeg;base64,"
        if not url.startswith(prefix):
            raise ValueError("Expected a JPEG camera or perception preview.")
        data = base64.b64decode(url[len(prefix) :], validate=True)
        name = f"images/{hashlib.sha256(data).hexdigest()}.jpg"
        path = self.folder / name
        if not path.exists():
            path.write_bytes(data)
        return name

    def observe(self, elapsed: float, context: DecisionContext, images: dict[str, str]) -> None:
        paths = {label: self._image(url) for label, url in images.items()}
        self.turns.append(ExplorationTurn(elapsed, context, paths))
        self._write("Exploring")

    def decided(self, decision: AgentDecision) -> None:
        self.turns[-1].decision = decision
        self._write("Exploring")

    def outcome(self, result: AgentFeedback | None, preview: str | None = None) -> None:
        turn = self.turns[-1]
        turn.result = result
        if preview is not None:
            turn.result_image = self._image(preview)
        self._write("Exploring")

    def finish(self, reason: str, score: TaskScore) -> None:
        self._write(reason, score=score)

    def _write(self, status: str, *, score: TaskScore | None = None) -> None:
        payload = to_json(
            {"seed": self.seed, "status": status, "score": score, "turns": self.turns}
        )
        # Agent text can contain arbitrary markup, including a closing script tag.
        data = payload.decode().replace("<", "\\u003c")
        temporary = self.folder / "index.html.tmp"
        temporary.write_text(self.template.replace("<!-- EXPLORATION_DATA -->", data))
        temporary.replace(self.folder / "index.html")

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
