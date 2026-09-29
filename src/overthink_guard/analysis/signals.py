from __future__ import annotations

import re
import zlib

from overthink_guard.templates import Template

_BOXED = "\\boxed{"
_ZLIB_WINDOW = 32 * 1024


def extract_boxed(text: str) -> list[tuple[int, str]]:
    """Contents of every \\boxed{...}, matching nested braces."""
    found = []
    start = text.find(_BOXED)
    while start != -1:
        depth = 1
        i = start + len(_BOXED)
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            found.append((start, text[start + len(_BOXED): i - 1]))
        start = text.find(_BOXED, i)
    return found


def normalize_answer(answer: str) -> str:
    answer = answer.strip().strip("*`$").strip()
    answer = re.sub(r"\s+", "", answer).lower()
    return answer.rstrip(".,;:!")


def extract_tentative_answer(text: str, template: Template) -> str | None:
    """The last answer the model states in text, normalized, or None."""
    candidates: list[tuple[int, str]] = []
    if template.extract_boxed:
        candidates.extend(extract_boxed(text))
    for pattern in template.answer_patterns:
        for match in pattern.finditer(text):
            group = "ans" if "ans" in pattern.groupindex else 1
            candidates.append((match.start(), match.group(group)))
    for _, raw in sorted(candidates, key=lambda c: c[0], reverse=True):
        normalized = normalize_answer(raw)
        if normalized:
            return normalized
    return None


def has_revision(text: str, template: Template) -> bool:
    return any(pattern.search(text) for pattern in template.revision_patterns)


def _deflate_size(data: bytes) -> int:
    compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
    return len(compressor.compress(data) + compressor.flush())


def novelty(new: str, context: str) -> float:
    """Share of new that is not predictable from context: ~1 fresh content, ~0 pure repetition."""
    new_bytes = new.encode("utf-8")
    if not new_bytes:
        return 0.0
    context_bytes = context.encode("utf-8")[-max(0, _ZLIB_WINDOW - len(new_bytes)):]
    alone = _deflate_size(new_bytes)
    added = _deflate_size(context_bytes + new_bytes) - _deflate_size(context_bytes)
    return max(0.0, min(1.0, added / alone))


def estimate_tokens(text: str) -> int:
    """Rough token count without a tokenizer: ~4 chars/token for ASCII, ~1 per non-ASCII char."""
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return round(ascii_chars / 4 + (len(text) - ascii_chars))
