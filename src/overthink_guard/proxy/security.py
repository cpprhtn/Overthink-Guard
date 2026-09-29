from __future__ import annotations

from urllib.parse import urlsplit

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def allowed_hosts(host: str, port: int) -> set[str]:
    names = _LOOPBACK | {host}
    return {f"[{n}]:{port}" if ":" in n else f"{n}:{port}" for n in names}


def _local_origin(origin: str) -> bool:
    try:
        return urlsplit(origin).hostname in _LOOPBACK
    except ValueError:
        return False


class LocalOnlyMiddleware:
    """Rejects unexpected Host (DNS rebinding) and non-loopback Origin; a 127.0.0.1 bind alone lets web pages in."""

    def __init__(self, app: ASGIApp, hosts: set[str]) -> None:
        self._app = app
        self._hosts = hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
            origin = headers.get("origin")
            if headers.get("host", "").lower() not in self._hosts:
                await PlainTextResponse("forbidden: unexpected Host header", 403)(scope, receive, send)
                return
            if origin is not None and not _local_origin(origin):
                await PlainTextResponse("forbidden: cross-origin request", 403)(scope, receive, send)
                return
        await self._app(scope, receive, send)
