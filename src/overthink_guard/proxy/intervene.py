from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from overthink_guard.backends import OllamaBackend, answer_prefill, iter_chunks
from overthink_guard.control.session import ANSWERING, CANCELLED, DONE, ERROR, THINKING, Session, SessionHub
from overthink_guard.templates import Template


class InterventionStream:
    """Relays one Ollama chat stream as OpenAI SSE and, on a stop request, splices in a prefilled answer."""

    def __init__(
        self,
        *,
        hub: SessionHub,
        session: Session,
        backend: OllamaBackend,
        response: httpx.Response,
        native: dict,
        template: Template,
        include_usage: bool,
    ) -> None:
        self._hub = hub
        self._session = session
        self._backend = backend
        self._response = response
        self._native = native
        self._template = template
        self._include_usage = include_usage
        self._chunks = 0
        self._done: dict | None = None
        self._error: str | None = None

    def _sse(self, delta: dict | None, finish: str | None = None, **extra: object) -> bytes:
        payload = {
            "id": f"chatcmpl-otg-{self._session.id}",
            "object": "chat.completion.chunk",
            "created": int(self._session.created),
            "model": self._session.model,
            "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}],
            **extra,
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()

    async def _relay(self, response: httpx.Response, stoppable: bool) -> AsyncIterator[bytes]:
        """SSE for each upstream chunk; ends early, leaving the upstream open, once a stop is requested."""
        session = self._session
        self._chunks = 0
        self._done = None
        async for chunk in iter_chunks(response):
            if "error" in chunk:
                self._error = str(chunk["error"])
                return
            message = chunk.get("message") or {}
            if thinking := message.get("thinking"):
                # Ollama streams one token per chunk (spike S8: one logprob entry per chunk).
                self._hub.add_thinking(session, thinking, tokens=1)
                yield self._sse({"reasoning": thinking})
            if content := message.get("content"):
                if session.status == THINKING:
                    self._hub.set_status(session, ANSWERING)
                yield self._sse({"content": content})
            if chunk.get("done"):
                self._done = chunk
                return
            self._chunks += 1
            if stoppable and session.status == THINKING and session.stop_requested.is_set():
                return

    async def __aiter__(self) -> AsyncIterator[bytes]:
        session = self._session
        try:
            yield self._sse({"role": "assistant", "content": ""})
            try:
                async for piece in self._relay(self._response, stoppable=True):
                    yield piece
            finally:
                await self._response.aclose()

            otg: dict = {"session": session.id, "intervened": False}
            if self._done is None and self._error is None and session.stop_requested.is_set():
                stopped_after = self._chunks
                async for piece in self._answer_now():
                    yield piece
                answer = self._done or {}
                otg |= {
                    "intervened": True,
                    "stopped_after_tokens": stopped_after,
                    "answer_prompt_tokens": answer.get("prompt_eval_count"),
                    "answer_prompt_cached_tokens": answer.get("prompt_eval_cached_count"),
                }
                prompt_tokens = max(0, answer.get("prompt_eval_count", 0) - stopped_after)
                completion_tokens = stopped_after + answer.get("eval_count", 0)
            else:
                prompt_tokens = (self._done or {}).get("prompt_eval_count", 0)
                completion_tokens = (self._done or {}).get("eval_count", self._chunks)

            finish = "length" if (self._done or {}).get("done_reason") == "length" else "stop"
            if self._error:
                otg["error"] = self._error
            yield self._sse({}, finish, otg=otg)
            if self._include_usage:
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                yield self._sse(None, usage=usage)
            yield b"data: [DONE]\n\n"
            self._hub.set_status(session, ERROR if self._error else DONE)
        except BaseException:
            if session.status in (THINKING, ANSWERING):
                self._hub.set_status(session, CANCELLED)
            raise

    async def _answer_now(self) -> AsyncIterator[bytes]:
        session = self._session
        session.intervened = True
        self._hub.set_status(session, ANSWERING)
        body = answer_prefill(self._native, session.thinking, self._template, self._chunks)
        response = await self._backend.open_chat(body)
        try:
            if response.status_code != 200:
                self._error = f"answer request failed with HTTP {response.status_code}"
                return
            async for piece in self._relay(response, stoppable=False):
                yield piece
        finally:
            await response.aclose()
