from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field

from overthink_guard.analysis import rules
from overthink_guard.analysis.segmenter import Segmenter
from overthink_guard.analysis.signals import estimate_tokens, extract_tentative_answer, has_revision, novelty
from overthink_guard.templates import Template


@dataclass(frozen=True)
class JudgeConfig:
    converge_k: int = 3
    min_thinking_tokens: int = 300
    revision_cooldown_tokens: int = 200
    # [가설] Calibrated only on synthetic text; near-verbatim repeats score ~0.2, paraphrased re-checks ~0.65.
    repetition_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.converge_k < 2:
            raise ValueError("converge_k must be at least 2")


@dataclass(frozen=True)
class Segment:
    index: int
    text: str
    start_char: int
    end_char: int
    start_tokens: int
    end_tokens: int
    phase: str
    answer: str | None
    novelty: float


@dataclass(frozen=True)
class StopDecision:
    segment_index: int
    at_tokens: int
    answer: str
    reasons: tuple[str, ...]
    span_novelty: float


@dataclass
class Judge:
    """Evaluates the auto-stop conditions (concept doc 5.4) on streamed thinking, using Tier 0 signals only."""

    template: Template
    config: JudgeConfig = field(default_factory=JudgeConfig)
    segments: list[Segment] = field(default_factory=list)
    decision: StopDecision | None = None

    def __post_init__(self) -> None:
        self._segmenter = Segmenter()
        self._text = ""
        self._tokens = 0
        self._last_revision_tokens: int | None = None
        self._mark_chars: list[int] = []
        self._mark_tokens: list[int] = []

    @property
    def thinking_tokens(self) -> int:
        return self._mark_tokens[-1] if self._mark_tokens else self._tokens

    def feed(self, thinking: str, tokens: int | None = None) -> StopDecision | None:
        """Returns the decision when stopping first becomes allowed; tokens overrides the text-based estimate."""
        if tokens is not None:
            self._mark_chars.append((self._mark_chars[-1] if self._mark_chars else 0) + len(thinking))
            self._mark_tokens.append(self.thinking_tokens + tokens)
        for segment in self._segmenter.feed(thinking):
            if self._add(segment):
                return self.decision
        return None

    def finish(self) -> None:
        for segment in self._segmenter.finish():
            self._add(segment)

    def _add(self, text: str) -> bool:
        answer = extract_tentative_answer(text, self.template)
        previous = next((s.answer for s in reversed(self.segments) if s.answer is not None), None)
        revised = has_revision(text, self.template) or (answer is not None and previous not in (None, answer))
        seg_novelty = novelty(text, self._text)
        end_char = len(self._text) + len(text)
        if self._mark_chars:
            mark = min(bisect_left(self._mark_chars, end_char), len(self._mark_chars) - 1)
            end_tokens = self._mark_tokens[mark]
        else:
            end_tokens = self._tokens + estimate_tokens(text)
        segment = Segment(
            index=len(self.segments),
            text=text,
            start_char=len(self._text),
            end_char=end_char,
            start_tokens=self._tokens,
            end_tokens=end_tokens,
            phase=rules.classify(
                text,
                index=len(self.segments),
                answer=answer,
                revised=revised,
                novelty=seg_novelty,
                repeat_threshold=self.config.repetition_threshold,
            ),
            answer=answer,
            novelty=seg_novelty,
        )
        self.segments.append(segment)
        self._text += text
        self._tokens = end_tokens
        if revised:
            self._last_revision_tokens = self._tokens
        if self.decision is None:
            self.decision = self._evaluate()
            return self.decision is not None
        return False

    def _evaluate(self) -> StopDecision | None:
        cfg = self.config
        if self._tokens < cfg.min_thinking_tokens:
            return None
        if (
            self._last_revision_tokens is not None
            and self._tokens - self._last_revision_tokens < cfg.revision_cooldown_tokens
        ):
            return None
        answered = [s for s in self.segments if s.answer is not None][-cfg.converge_k :]
        if len(answered) < cfg.converge_k or len({s.answer for s in answered}) != 1:
            return None
        first = answered[0]
        span_novelty = novelty(self._text[first.end_char :], self._text[: first.end_char])
        if span_novelty > cfg.repetition_threshold:
            return None
        return StopDecision(
            segment_index=len(self.segments) - 1,
            at_tokens=self._tokens,
            answer=first.answer,
            reasons=("tentative_answer_converged", "reasoning_repetition"),
            span_novelty=span_novelty,
        )
