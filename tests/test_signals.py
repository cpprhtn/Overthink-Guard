import pytest

from overthink_guard.analysis.signals import (
    estimate_tokens,
    extract_boxed,
    extract_tentative_answer,
    has_revision,
    normalize_answer,
    novelty,
)
from overthink_guard.templates import get_template

GENERIC = get_template("generic")


def test_boxed_handles_nested_braces():
    assert [a for _, a in extract_boxed(r"so \boxed{\frac{1}{2}} and \boxed{3}")] == [r"\frac{1}{2}", "3"]


def test_unclosed_boxed_is_ignored():
    assert extract_boxed(r"\boxed{\frac{1}{2}") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("So the answer is 9.", "9"),
        ("Therefore, the final answer is 3.5. Done", "3.5"),
        ("Thus x = 3, which fits.", "x=3"),
        (r"First 7, then \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        ("So the answer is 9. Wait, so the answer is 11.", "11"),
        ("The answer is **New York**.", "newyork"),
        ("We still need more work here.", None),
    ],
)
def test_extracts_last_stated_answer(text, expected):
    assert extract_tentative_answer(text, GENERIC) == expected


def test_normalize_ignores_spacing_case_and_trailing_punctuation():
    assert normalize_answer(" X = 3. ") == normalize_answer("x=3")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hmm, that's wrong, I dropped a term.", True),
        ("I made a mistake in the last step.", True),
        ("Wait, let me verify again.", False),
        ("But the total is still 12.", False),
    ],
)
def test_revision_needs_explicit_negation_not_filler_words(text, expected):
    assert has_revision(text, GENERIC) is expected


def test_novelty_separates_verbatim_repeat_from_fresh_text():
    context = "She has 3 bags of 4 apples each, so 3 times 4 is 12 apples. Minus 5 is 7, plus 2 is 9."
    repeat = "Let me verify. She has 3 bags of 4 apples each, so 3 times 4 is 12 apples. Minus 5 is 7, plus 2 is 9."
    fresh = "The triangle with sides 5, 12 and 13 is right-angled, since 25 plus 144 equals 169."
    assert novelty(repeat, context) < 0.35 < novelty(fresh, context)
    assert novelty("", context) == 0.0


def test_estimate_tokens_counts_non_ascii_denser():
    assert estimate_tokens("abcd" * 10) == 10
    assert estimate_tokens("생각") == 2
