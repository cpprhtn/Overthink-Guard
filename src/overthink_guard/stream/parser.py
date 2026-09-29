from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["thinking", "answer", "thinking_end"]


@dataclass(frozen=True)
class StreamEvent:
    kind: Kind
    text: str = ""


def _partial_tag_suffix(buffer: str, tag: str) -> int:
    """Length of the longest buffer suffix that is a proper prefix of tag."""
    for size in range(min(len(tag) - 1, len(buffer)), 0, -1):
        if buffer.endswith(tag[:size]):
            return size
    return 0


class ThinkStreamParser:
    """Splits a streamed completion into thinking and answer text around a single think block."""

    def __init__(self, start: str, end: str, starts_in_thinking: bool = False) -> None:
        self._start = start
        self._end = end
        self._state = "thinking" if starts_in_thinking else "before"
        self._buffer = ""

    @property
    def in_thinking(self) -> bool:
        return self._state == "thinking"

    def feed(self, chunk: str) -> list[StreamEvent]:
        self._buffer += chunk
        events: list[StreamEvent] = []
        while True:
            if self._state == "answer":
                if self._buffer:
                    events.append(StreamEvent("answer", self._buffer))
                    self._buffer = ""
                return events

            tag = self._start if self._state == "before" else self._end
            kind: Kind = "answer" if self._state == "before" else "thinking"
            index = self._buffer.find(tag)
            if index == -1:
                keep = _partial_tag_suffix(self._buffer, tag)
                emit = self._buffer[: len(self._buffer) - keep]
                if emit:
                    events.append(StreamEvent(kind, emit))
                self._buffer = self._buffer[len(emit):]
                return events

            if index:
                events.append(StreamEvent(kind, self._buffer[:index]))
            self._buffer = self._buffer[index + len(tag):]
            if self._state == "before":
                self._state = "thinking"
            else:
                self._state = "answer"
                events.append(StreamEvent("thinking_end"))

    def finish(self) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if self._buffer:
            events.append(StreamEvent("thinking" if self._state == "thinking" else "answer", self._buffer))
            self._buffer = ""
        return events
