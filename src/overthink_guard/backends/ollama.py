from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from overthink_guard.templates import Template

_OPTIONS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "seed": "seed",
    "stop": "stop",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "max_tokens": "num_predict",
    "max_completion_tokens": "num_predict",
}
_KNOWN_KEYS = {"model", "messages", "stream", "stream_options", "n", *_OPTIONS}


def _plain_message(message: object) -> bool:
    return (
        isinstance(message, dict)
        and set(message) <= {"role", "content"}
        and message.get("role") in ("system", "user")
        and isinstance(message.get("content"), str)
    )


def to_native_chat(body: dict) -> dict | None:
    """Ollama /api/chat body for a streamed single-turn text request; None means pass it through (D9)."""
    if body.get("stream") is not True or set(body) - _KNOWN_KEYS or body.get("n", 1) != 1:
        return None
    if set(body.get("stream_options") or {}) - {"include_usage"}:
        return None
    messages = body.get("messages")
    if not isinstance(body.get("model"), str) or not isinstance(messages, list):
        return None
    if not all(_plain_message(m) for m in messages) or sum(m["role"] == "user" for m in messages) != 1:
        return None

    options = {native: body[key] for key, native in _OPTIONS.items() if body.get(key) is not None}
    if isinstance(options.get("stop"), str):
        options["stop"] = [options["stop"]]
    native = {"model": body["model"], "messages": [dict(m) for m in messages], "stream": True}
    if options:
        native["options"] = options
    return native


def answer_prefill(native: dict, thinking: str, template: Template, generated_tokens: int) -> dict:
    """Request that closes the thinking received so far and lets the model answer (spike S1)."""
    prefill = {
        "role": "assistant",
        "content": "",
        "thinking": template.stop_thinking_prefix + thinking.rstrip() + template.stop_injection_text,
    }
    body = {**native, "think": True, "messages": [*native["messages"], prefill]}
    budget = native.get("options", {}).get("num_predict")
    if budget is not None:
        body["options"] = {**native["options"], "num_predict": max(1, budget - generated_tokens)}
    return body


def probe_request(native: dict, thinking: str, template: Template) -> dict:
    """Short greedy completion of an open \\boxed{ after the thinking so far (Tier 2 probe)."""
    prefill = {
        "role": "assistant",
        "content": template.probe_answer_prefix,
        "thinking": template.stop_thinking_prefix + thinking.rstrip() + template.stop_injection_text,
    }
    return {
        "model": native["model"],
        "messages": [*native["messages"], prefill],
        "think": True,
        "stream": False,
        "options": {"num_predict": template.probe_max_tokens, "temperature": 0},
    }


def resume_request(native: dict, thinking: str, template: Template, generated_tokens: int) -> dict:
    """Continues the paused thinking; the continuation streams back in content, tags included."""
    prefill = {"role": "assistant", "content": template.resume_content_prefix + thinking}
    body = {**native, "messages": [*native["messages"], prefill]}
    budget = native.get("options", {}).get("num_predict")
    if budget is not None:
        body["options"] = {**native["options"], "num_predict": max(1, budget - generated_tokens)}
    return body


class OllamaBackend:
    def __init__(self, client: httpx.AsyncClient, base_url: str) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")

    async def open_chat(self, body: dict) -> httpx.Response:
        request = self._client.build_request("POST", f"{self._base_url}/api/chat", json=body)
        return await self._client.send(request, stream=True)

    async def complete(self, body: dict) -> dict:
        response = await self._client.post(f"{self._base_url}/api/chat", json=body)
        response.raise_for_status()
        return response.json()


async def iter_chunks(response: httpx.Response) -> AsyncIterator[dict]:
    async for line in response.aiter_lines():
        if line.strip():
            yield json.loads(line)
