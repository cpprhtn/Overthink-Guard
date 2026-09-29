from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import httpx

from overthink_guard.analysis import Probe, read_probe_answer
from overthink_guard.backends import OllamaBackend, answer_prefill, iter_chunks, probe_request, resume_request
from overthink_guard.control.session import ANSWERING, CANCELLED, DONE, ERROR, THINKING, Session, SessionHub
from overthink_guard.stream import ThinkStreamParser
from overthink_guard.templates import Template

_DONE, _STOP, _PROBE, _ERROR = "done", "stop", "probe", "error"


class InterventionStream:
    """Relays Ollama chat as OpenAI SSE; may pause to probe and resume, or splice in an answer on "Answer now"."""

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
        probe_interval: int = 400,
        probe_min_tokens: int = 0,
    ) -> None:
        self._hub = hub
        self._session = session
        self._backend = backend
        self._response = response
        self._native = native
        self._template = template
        self._include_usage = include_usage
        self._probe_interval = probe_interval
        self._probe_min_tokens = probe_min_tokens
        self._last_probe_at = 0
        self._generated = 0
        self._done: dict | None = None
        self._error: str | None = None
        self._prompt_tokens: int | None = None
        self._answer_started = False

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

    def _thinking(self, text: str, tokens: int) -> bytes:
        self._hub.add_thinking(self._session, text, tokens=tokens)
        return self._sse({"reasoning": text})

    def _content(self, text: str) -> bytes | None:
        if not self._answer_started:
            text = text.lstrip()
            if not text:
                return None
            self._answer_started = True
        if self._session.status == THINKING:
            self._hub.set_status(self._session, ANSWERING)
        self._hub.add_answer(self._session, text)
        return self._sse({"content": text})

    def _estimate_prompt(self, follow_up: dict) -> None:
        if self._prompt_tokens is None and "prompt_eval_count" in follow_up:
            self._prompt_tokens = max(0, follow_up["prompt_eval_count"] - self._generated)

    async def _relay(self, response: httpx.Response, parser: ThinkStreamParser | None) -> AsyncIterator[bytes | str]:
        """Yields SSE bytes, then one outcome string. Ollama streams one token per chunk (spike S8)."""
        session = self._session
        async for chunk in iter_chunks(response):
            if "error" in chunk:
                self._error = str(chunk["error"])
                yield _ERROR
                return
            message = chunk.get("message") or {}
            events: list[tuple[str, str]] = []
            if parser is None:
                events = [("thinking", message.get("thinking") or ""), ("answer", message.get("content") or "")]
            else:
                pieces = parser.feed(message.get("content") or "")
                if chunk.get("done"):
                    pieces += parser.finish()
                events = [(e.kind, e.text) for e in pieces]
            tokens = 1
            for kind, text in events:
                if not text:
                    continue
                if kind == "thinking":
                    yield self._thinking(text, tokens)
                    tokens = 0
                elif kind == "answer" and (sse := self._content(text)) is not None:
                    yield sse
            if chunk.get("done"):
                self._done = chunk
                yield _DONE
                return
            self._generated += 1
            if session.status == THINKING and session.stop_requested.is_set():
                yield _STOP
                return
            if (
                session.status == THINKING
                and session.probes is not None
                and session.judge.thinking_tokens >= self._probe_min_tokens
                and session.judge.thinking_tokens - self._last_probe_at >= self._probe_interval
            ):
                yield _PROBE
                return
        self._error = self._error or "upstream closed the stream early"
        yield _ERROR

    async def _probe(self) -> None:
        session = self._session
        started = time.monotonic()
        try:
            result = await self._backend.complete(probe_request(self._native, session.thinking, self._template))
            self._estimate_prompt(result)
            answer = read_probe_answer(result["message"]["content"])
        except (httpx.HTTPError, KeyError, ValueError):
            answer = None
        self._last_probe_at = session.judge.thinking_tokens
        self._hub.add_probe(session, Probe(self._last_probe_at, answer, time.monotonic() - started))

    async def __aiter__(self) -> AsyncIterator[bytes]:
        session = self._session
        try:
            yield self._sse({"role": "assistant", "content": ""})
            response, parser, first = self._response, None, True
            while True:
                outcome = _ERROR
                try:
                    async for item in self._relay(response, parser):
                        if isinstance(item, str):
                            outcome = item
                        else:
                            yield item
                finally:
                    await response.aclose()
                if outcome == _DONE and first:
                    self._prompt_tokens = (self._done or {}).get("prompt_eval_count", 0)
                if outcome != _PROBE:
                    break
                await self._probe()
                if session.stop_requested.is_set():
                    outcome = _STOP
                    break
                response = await self._backend.open_chat(
                    resume_request(self._native, session.thinking, self._template, self._generated)
                )
                if response.status_code != 200:
                    await response.aclose()
                    self._error = f"resume request failed with HTTP {response.status_code}"
                    outcome = _ERROR
                    break
                parser = ThinkStreamParser(
                    self._template.think_start, self._template.think_end, starts_in_thinking=True
                )
                first = False

            otg: dict = {"session": session.id, "intervened": False}
            if session.probes is not None:
                otg["probes"] = len(session.probes.probes)
            if outcome == _STOP:
                stopped_after = self._generated
                async for piece in self._answer_now():
                    yield piece
                answer = self._done or {}
                otg |= {
                    "intervened": True,
                    "stopped_after_tokens": stopped_after,
                    "answer_prompt_tokens": answer.get("prompt_eval_count"),
                    "answer_prompt_cached_tokens": answer.get("prompt_eval_cached_count"),
                }

            finish = "length" if (self._done or {}).get("done_reason") == "length" else "stop"
            if self._error:
                otg["error"] = self._error
            yield self._sse({}, finish, otg=otg)
            if self._include_usage:
                # Exact only when the original stream ran to completion; otherwise counted from chunks and estimated.
                prompt = self._prompt_tokens or 0
                exact = first and outcome == _DONE
                completion = (self._done or {}).get("eval_count", 0) if exact else self._generated
                usage = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
                yield self._sse(None, usage=usage)
            yield b"data: [DONE]\n\n"
            self._hub.set_status(session, ERROR if self._error else DONE)
            if outcome == _DONE and not self._error:
                self._hub.record_shadow(session, time.time() - session.created)
        except BaseException:
            if session.status in (THINKING, ANSWERING):
                self._hub.set_status(session, CANCELLED)
            raise

    async def _answer_now(self) -> AsyncIterator[bytes]:
        session = self._session
        session.intervened = True
        self._hub.set_status(session, ANSWERING)
        body = answer_prefill(self._native, session.thinking, self._template, self._generated)
        response = await self._backend.open_chat(body)
        try:
            if response.status_code != 200:
                self._error = f"answer request failed with HTTP {response.status_code}"
                return
            self._done = None
            async for item in self._relay(response, None):
                if isinstance(item, bytes):
                    yield item
            if self._done is not None:
                self._estimate_prompt(self._done)
        finally:
            await response.aclose()
