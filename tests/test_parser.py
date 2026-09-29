import pytest

from overthink_guard.stream import StreamEvent, ThinkStreamParser


def run(parser: ThinkStreamParser, text: str, chunk_size: int) -> list[StreamEvent]:
    events = []
    for i in range(0, len(text), chunk_size):
        events.extend(parser.feed(text[i : i + chunk_size]))
    events.extend(parser.finish())
    return events


def collect(events: list[StreamEvent], kind: str) -> str:
    return "".join(e.text for e in events if e.kind == kind)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 1000])
def test_splits_thinking_and_answer_across_any_chunking(chunk_size):
    events = run(ThinkStreamParser("<think>", "</think>"), "<think>a < b </thi</think>answer <think>", chunk_size)
    assert collect(events, "thinking") == "a < b </thi"
    assert collect(events, "answer") == "answer <think>"
    assert sum(e.kind == "thinking_end" for e in events) == 1


def test_starts_in_thinking_when_template_prefills_the_tag():
    events = run(ThinkStreamParser("<think>", "</think>", starts_in_thinking=True), "reasoning</think>done", 4)
    assert collect(events, "thinking") == "reasoning"
    assert collect(events, "answer") == "done"


def test_text_without_tags_is_answer():
    events = run(ThinkStreamParser("<think>", "</think>"), "just an answer", 3)
    assert collect(events, "answer") == "just an answer"
    assert collect(events, "thinking") == ""


def test_unterminated_thinking_is_flushed_on_finish():
    parser = ThinkStreamParser("<think>", "</think>")
    events = run(parser, "<think>still going </th", 5)
    assert collect(events, "thinking") == "still going </th"
    assert parser.in_thinking
