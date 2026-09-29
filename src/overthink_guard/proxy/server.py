from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from importlib import resources

import httpx
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from overthink_guard.analysis import Judge, JudgeConfig, ProbeTracker
from overthink_guard.analysis.prober import DEFAULT_PROBE_K
from overthink_guard.backends import OllamaBackend, to_native_chat
from overthink_guard.control import SessionHub
from overthink_guard.proxy.intervene import InterventionStream
from overthink_guard.proxy.passthrough import forward
from overthink_guard.proxy.security import LocalOnlyMiddleware, allowed_hosts
from overthink_guard.storage import ShadowStats
from overthink_guard.templates import template_for_model

_ALL_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]


def create_app(
    backend_url: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8484,
    client: httpx.AsyncClient | None = None,
    heartbeat_s: float = 15.0,
    stats: ShadowStats | None = None,
    judge_config: JudgeConfig | None = None,
    probe: bool = False,
    probe_interval: int = 400,
    probe_converge_k: int = DEFAULT_PROBE_K,
) -> Starlette:
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0))
    backend = OllamaBackend(client, backend_url)
    hub = SessionHub(stats)

    async def passthrough(request: Request) -> Response:
        return await forward(request, client, backend_url, await request.body())

    async def chat_completions(request: Request) -> Response:
        raw = await request.body()
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        native = to_native_chat(body) if isinstance(body, dict) else None
        if native is None:
            return await forward(request, client, backend_url, raw)

        upstream = await backend.open_chat(native)
        if upstream.status_code != 200:
            await upstream.aclose()
            return await forward(request, client, backend_url, raw)

        template = template_for_model(native["model"])
        prompt = next(m["content"] for m in native["messages"] if m["role"] == "user")
        # Pausing to probe reseeds sampling on resume, so a client-fixed seed would no longer reproduce.
        probing = probe and "seed" not in native.get("options", {})
        tracker = ProbeTracker(probe_converge_k) if probing else None
        session = hub.create(native["model"], prompt, Judge(template, judge_config or JudgeConfig()), tracker)
        if probe and not probing:
            session.probe_skipped = "seed"
        stream = InterventionStream(
            hub=hub,
            session=session,
            backend=backend,
            response=upstream,
            native=native,
            template=template,
            include_usage=bool((body.get("stream_options") or {}).get("include_usage")),
            probe_interval=probe_interval,
        )
        return StreamingResponse(
            stream, media_type="text/event-stream", headers={"cache-control": "no-cache", "x-otg-session": session.id}
        )

    async def ui(_: Request) -> Response:
        page = resources.files("overthink_guard").joinpath("ui/static/index.html").read_text(encoding="utf-8")
        return HTMLResponse(page, headers={"cache-control": "no-store"})

    async def events(_: Request) -> Response:
        queue = hub.subscribe()

        async def stream():
            try:
                for event in hub.snapshot():
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=heartbeat_s)
                    except TimeoutError:
                        yield ": ping\n\n"
                        continue
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            finally:
                hub.unsubscribe(queue)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"cache-control": "no-cache"})

    async def stop(request: Request) -> Response:
        if hub.request_stop(request.path_params["session_id"]):
            return JSONResponse({"ok": True})
        return JSONResponse({"ok": False, "error": "session is not thinking"}, status_code=409)

    async def stats_summary(_: Request) -> Response:
        return JSONResponse(hub.stats.summary())

    @asynccontextmanager
    async def lifespan(_: Starlette):
        yield
        await client.aclose()

    app = Starlette(
        routes=[
            Route("/ui", ui),
            Route("/otg/api/events", events),
            Route("/otg/api/stats", stats_summary),
            Route("/otg/api/sessions/{session_id}/stop", stop, methods=["POST"]),
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/{path:path}", passthrough, methods=_ALL_METHODS),
        ],
        middleware=[Middleware(LocalOnlyMiddleware, hosts=allowed_hosts(host, port))],
        lifespan=lifespan,
    )
    app.state.hub = hub
    return app
