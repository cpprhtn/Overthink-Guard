from __future__ import annotations

import re

UNDERSTAND = "understand"
HYPOTHESIS = "hypothesis"
COMPUTE = "compute"
VERIFY = "verify"
CONCLUSION = "conclusion"
REVISION = "revision"
REPEAT = "repeat"
OTHER = "other"

_VERIFY = re.compile(
    r"(?i)\b(?:verify|double[- ]check|check(?:ing)?|confirm|let me (?:re-?)?(?:check|verify)|wait|hmm+|actually)\b"
)
_HYPOTHESIS = re.compile(r"(?i)\b(?:maybe|perhaps|let's try|let me try|one approach|alternatively|what if|suppose)\b")
_UNDERSTAND = re.compile(r"(?i)\b(?:the question asks|we need to|i need to|the problem (?:says|asks|is)|we are given)\b")
_MATH_CHAR = re.compile(r"[\d=+\-*/^×÷]")


def classify(
    text: str,
    *,
    index: int,
    answer: str | None,
    revised: bool,
    novelty: float,
    repeat_threshold: float,
) -> str:
    if revised:
        return REVISION
    if answer is not None:
        return CONCLUSION
    if novelty <= repeat_threshold:
        return REPEAT
    if _VERIFY.search(text):
        return VERIFY
    stripped = text.replace(" ", "")
    if stripped and len(_MATH_CHAR.findall(stripped)) / len(stripped) >= 0.15:
        return COMPUTE
    if _HYPOTHESIS.search(text):
        return HYPOTHESIS
    if index < 2 or _UNDERSTAND.search(text):
        return UNDERSTAND
    return OTHER
