from __future__ import annotations

import httpx
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import StreamingResponse

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


async def forward(request: Request, client: httpx.AsyncClient, base_url: str, body: bytes) -> StreamingResponse:
    """Relays the request and response bodies byte-for-byte, without decoding either."""
    url = base_url.rstrip("/") + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    headers = [(k, v) for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP | {"host", "content-length"}]
    upstream = await client.send(client.build_request(request.method, url, headers=headers, content=body), stream=True)
    response = StreamingResponse(
        upstream.aiter_raw(), status_code=upstream.status_code, background=BackgroundTask(upstream.aclose)
    )
    response.raw_headers = [
        (k.encode("latin-1"), v.encode("latin-1"))
        for k, v in upstream.headers.multi_items()
        if k.lower() not in _HOP_BY_HOP
    ]
    return response
