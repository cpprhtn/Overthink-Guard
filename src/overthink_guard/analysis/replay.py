from __future__ import annotations

from dataclasses import dataclass

from overthink_guard.analysis.judge import Judge, JudgeConfig, Segment, StopDecision
from overthink_guard.analysis.signals import extract_tentative_answer
from overthink_guard.stream import ThinkStreamParser
from overthink_guard.templates import Template


@dataclass(frozen=True)
class ReplayReport:
    segments: list[Segment]
    thinking_tokens: int
    decision: StopDecision | None
    final_answer: str | None

    @property
    def saved_tokens(self) -> int:
        return self.thinking_tokens - self.decision.at_tokens if self.decision else 0

    @property
    def answer_match(self) -> bool | None:
        """Whether the tentative answer at the stop point equals the final answer (the Shadow proxy metric)."""
        if self.decision is None or self.final_answer is None:
            return None
        return self.decision.answer == self.final_answer


def replay(completion: str, template: Template, config: JudgeConfig, chunk_size: int = 16) -> ReplayReport:
    """Streams a recorded completion through the parser and judge as if it arrived live."""
    starts_in_thinking = template.starts_in_thinking or (
        template.think_end in completion and template.think_start not in completion
    )
    parser = ThinkStreamParser(template.think_start, template.think_end, starts_in_thinking)
    judge = Judge(template, config)
    answer_text = ""
    chunks = [completion[i: i + chunk_size] for i in range(0, len(completion), chunk_size)]
    for events in [*(parser.feed(c) for c in chunks), parser.finish()]:
        for event in events:
            if event.kind == "thinking":
                judge.feed(event.text)
            elif event.kind == "answer":
                answer_text += event.text
    judge.finish()
    return ReplayReport(
        segments=judge.segments,
        thinking_tokens=judge.thinking_tokens,
        decision=judge.decision,
        final_answer=extract_tentative_answer(answer_text, template),
    )
