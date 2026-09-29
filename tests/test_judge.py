from pathlib import Path

import pytest

from overthink_guard.analysis import Judge, JudgeConfig, replay
from overthink_guard.templates import get_template

FIXTURES = Path(__file__).parent / "fixtures"
GENERIC = get_template("generic")


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_stops_once_answer_converges_and_reasoning_repeats():
    report = replay(load("synthetic_overthink.txt"), GENERIC, JudgeConfig())
    assert report.decision is not None
    assert report.decision.answer == "9"
    assert report.decision.reasons == ("tentative_answer_converged", "reasoning_repetition")
    assert report.saved_tokens > 0
    assert report.answer_match is True


def test_min_thinking_tokens_delays_stop():
    early = replay(load("synthetic_overthink.txt"), GENERIC, JudgeConfig(min_thinking_tokens=0))
    late = replay(load("synthetic_overthink.txt"), GENERIC, JudgeConfig(min_thinking_tokens=300))
    assert early.decision.at_tokens < 300 <= late.decision.at_tokens


def test_real_revision_blocks_stop_until_cooldown_and_uses_corrected_answer():
    config = JudgeConfig()
    report = replay(load("synthetic_real_fix.txt"), GENERIC, config)
    revision = next(s for s in report.segments if s.phase == "revision")
    assert report.decision is not None
    assert report.decision.answer == "11"
    assert report.decision.at_tokens - revision.end_tokens >= config.revision_cooldown_tokens
    assert report.answer_match is True


def test_never_stops_without_extractable_answer():
    report = replay(load("synthetic_essay.txt"), GENERIC, JudgeConfig(min_thinking_tokens=0))
    assert report.decision is None
    assert report.saved_tokens == 0


def test_fresh_reasoning_between_same_answers_is_not_repetition():
    judge = Judge(GENERIC, JudgeConfig(min_thinking_tokens=0))
    judge.feed("Counting directly gives 9. So the answer is 9.\n\n")
    judge.feed("Another route: model it as an equation, 3x - 5 + 2 with x = 4 bags, still 9. So the answer is 9.\n\n")
    judge.feed(
        "A third check with a table of each step and a totally different framing confirms 9. So the answer is 9.\n\n"
    )
    assert judge.decision is None


def test_decision_is_reported_once_and_kept():
    judge = Judge(GENERIC, JudgeConfig(min_thinking_tokens=0, converge_k=2))
    first = judge.feed("So the answer is 9.\n\nSo the answer is 9.\n\n")
    assert first is not None
    assert judge.feed("So the answer is 9.\n\n") is None
    assert judge.decision == first


def test_reported_token_counts_replace_the_estimate():
    judge = Judge(GENERIC, JudgeConfig(min_thinking_tokens=0))
    text = "First paragraph of thinking.\n\nSecond one."
    for i in range(0, len(text), 3):
        judge.feed(text[i : i + 3], tokens=1)
    chunks = -(-len(text) // 3)
    assert judge.thinking_tokens == chunks
    judge.finish()
    first, second = judge.segments
    assert first.end_tokens == -(-len("First paragraph of thinking.\n\n") // 3)
    assert second.end_tokens == chunks


def test_converge_k_below_two_is_rejected():
    with pytest.raises(ValueError):
        JudgeConfig(converge_k=1)
