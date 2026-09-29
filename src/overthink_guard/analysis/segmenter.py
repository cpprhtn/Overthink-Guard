from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"[.!?。](?=\s)|\n")


class Segmenter:
    """Cuts streamed thinking into paragraph-sized segments at natural boundaries."""

    def __init__(self, max_chars: int = 800) -> None:
        self._max_chars = max_chars
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        segments: list[str] = []
        while True:
            cut = self._buffer.find("\n\n")
            if cut != -1:
                end = cut + 2
            elif len(self._buffer) > self._max_chars:
                ends = [m.end() for m in _SENTENCE_END.finditer(self._buffer, 0, self._max_chars)]
                end = ends[-1] if ends else self._max_chars
            else:
                return segments
            segment, self._buffer = self._buffer[:end], self._buffer[end:]
            if segment.strip():
                segments.append(segment)

    def finish(self) -> list[str]:
        segment, self._buffer = self._buffer, ""
        return [segment] if segment.strip() else []
