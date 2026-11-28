"""Headset web application and threaded loopback listener."""

import asyncio
import threading
from collections.abc import Awaitable, Callable
from html import escape
from pathlib import Path
from urllib.parse import urlsplit

import vuer
from aiohttp import web
from pydantic import ValidationError

from humanoid.config.api import get_api_endpoint
from humanoid.config.quest import QUEST_ASSETS_ROOT
from humanoid.constants import Host, Port
from humanoid.hardware.teleop.quest.input import QuestInputState
from humanoid.hardware.teleop.quest.session import QuestSession
from humanoid.types.api import ServiceConflict
from humanoid.types.quest import QuestConfig, QuestRoute, QuestVideoOffer, QuestVideoServices

VUER_CLIENT = Path(vuer.__file__).parent / "client_build"


def create_app(
    config: QuestConfig, input_state: QuestInputState, video: QuestVideoServices
) -> web.Application:
    """Create the headset application and its session, closed on application shutdown."""
    session = QuestSession(config, input_state, video)

    @web.middleware
    async def protect(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        _check_request_origin(request, config)
        response = await handler(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    async def _landing(request: web.Request) -> web.Response:
        # Read the endpoint after startup, once the dashboard's bound port is known.
        port = urlsplit(get_api_endpoint().url).port or Port.API
        dashboard = (
            f"https://{config.public_host}:{Port.DASHBOARD_GATEWAY}"
            if _is_public_request(request, config)
            else f"http://{request.url.host}:{port}"
        )
        html = (QUEST_ASSETS_ROOT / "index.html").read_text()
        html = html.replace('id="dashboard" href="/"', f'id="dashboard" href="{escape(dashboard)}"')
        return web.Response(text=html, content_type="text/html")

    async def _client(_request: web.Request) -> web.Response:
        html = (VUER_CLIENT / "index.html").read_text()
        html = html.replace(
            "<head>", f'<head><script src="{QuestRoute.QUEST_ASSETS}/guard.js"></script>'
        )
        return web.Response(text=html, content_type="text/html")

    async def _status(_request: web.Request) -> web.Response:
        return web.json_response(session.status())

    async def _offer(request: web.Request) -> web.Response:
        try:
            payload = QuestVideoOffer.model_validate(await request.json())
        except (ValueError, ValidationError) as exc:
            raise web.HTTPBadRequest(text="Invalid video offer") from exc
        try:
            answer = await session.offer(
                request.match_info["owner"],
                request.match_info["camera"],
                request.query.get("generation"),
                payload,
            )
        except ServiceConflict as exc:
            raise web.HTTPConflict(text=str(exc)) from exc
        except LookupError as exc:
            raise web.HTTPNotFound(text=str(exc)) from exc
        return web.json_response(answer)

    async def _websocket(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse(heartbeat=2, max_msg_size=32768)
        try:
            async with session.connection(socket):
                await socket.prepare(request)
                await session.run(socket)
        except ServiceConflict as exc:
            raise web.HTTPConflict(text=str(exc)) from exc
        return socket

    async def _shutdown(_app: web.Application) -> None:
        await session.close()

    app = web.Application(middlewares=[protect], client_max_size=128 * 1024)
    app.router.add_get(QuestRoute.ROOT, _client)
    app.router.add_get(QuestRoute.CONNECT, _landing)
    app.router.add_get(QuestRoute.WEBSOCKET, _websocket)
    app.router.add_get(QuestRoute.STATUS, _status)
    app.router.add_post(QuestRoute.VIDEO_OFFER, _offer)
    app.router.add_static(QuestRoute.VUER_ASSETS, VUER_CLIENT / "assets")
    app.router.add_static(QuestRoute.QUEST_ASSETS, QUEST_ASSETS_ROOT)
    app.on_shutdown.append(_shutdown)
    return app


def _is_public_request(request: web.Request, config: QuestConfig) -> bool:
    return config.public_host is not None and request.host.lower() == (
        f"{config.public_host}:{Port.QUEST_GATEWAY}"
    )


def _check_request_origin(request: web.Request, config: QuestConfig) -> None:
    allowed = {f"{Host.LOOPBACK}:{config.port}", f"{Host.LOCALHOST}:{config.port}"}
    if config.public_host:
        allowed.add(f"{config.public_host}:{Port.QUEST_GATEWAY}")
    if request.host.lower() not in allowed:
        raise web.HTTPForbidden(text="Invalid Quest host")
    origin = request.headers.get("Origin")
    expected = (
        f"https://{request.host}"
        if _is_public_request(request, config)
        else f"http://{request.host}"
    )
    if origin and origin != expected:
        raise web.HTTPForbidden(text="Cross-origin Quest request")
    if request.headers.get("Sec-Fetch-Site") in {"cross-site", "same-site"}:
        raise web.HTTPForbidden(text="Same-origin requests required")


class QuestServer:
    def __init__(
        self, config: QuestConfig, input_state: QuestInputState, video: QuestVideoServices
    ) -> None:
        self.config = config
        self.input_state = input_state
        self.video = video
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="quest-server")

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(10):
            self.close()
            raise RuntimeError("Quest server startup timed out")
        if self._error is not None:
            self._thread.join()
            raise RuntimeError("Could not start Quest server") from self._error

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        runner: web.AppRunner | None = None
        try:
            runner = web.AppRunner(create_app(self.config, self.input_state, self.video))

            async def listen() -> None:
                assert runner is not None
                await runner.setup()
                await web.TCPSite(runner, Host.LOOPBACK, self.config.port).start()

            loop.run_until_complete(listen())
            self._ready.set()
            loop.run_forever()
        except BaseException as exc:
            self._error = exc
            self._ready.set()
        finally:
            try:
                if runner is not None:
                    loop.run_until_complete(runner.cleanup())
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
            finally:
                loop.close()

    def close(self) -> None:
        if self._loop is not None and self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=20)
            if self._thread.is_alive():
                raise RuntimeError("Quest server shutdown timed out")
