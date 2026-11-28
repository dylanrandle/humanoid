"""Dashboard-facing Quest input and viewer-link endpoints."""

from typing import TYPE_CHECKING

from flask import Flask, jsonify, redirect

from humanoid.config.quest import get_quest_config
from humanoid.constants import Host, Port
from humanoid.types.api import ApiRoute
from humanoid.types.quest import QuestRoute

if TYPE_CHECKING:
    from humanoid.api.application import BackendApplication


def register_quest_routes(app: Flask, backend: "BackendApplication") -> None:
    @app.get(ApiRoute.QUEST_INPUT)
    def quest_input():
        return jsonify(backend.quest_input.snapshot().model_dump())

    @app.get(ApiRoute.QUEST_OPEN)
    def quest_open():
        config = get_quest_config()
        url = (
            f"https://{config.public_host}:{Port.QUEST_GATEWAY}{QuestRoute.CONNECT}"
            if config.public_host
            else f"http://{Host.LOOPBACK}:{config.port}{QuestRoute.CONNECT}"
        )
        return redirect(url)
