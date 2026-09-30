from __future__ import annotations

import re
from dataclasses import dataclass, field

from overthink_guard.analysis.signals import extract_boxed, normalize_answer

# Probe answers can hold a wrong guess for 5000+ tokens before a late correction, so probing starts late and mainly
# catches runaway thinking. Pre-registered rule C held C1 on 51 fresh completed runs across qwen3 and deepseek-r1
# (docs/spikes/shadow-live-probe.md).
DEFAULT_PROBE_K = 4
DEFAULT_PROBE_MIN_TOKENS = 6000
# [가설] Open-ended questions give no grounded probe answers, so they get a plain thinking budget instead. Completed
# design answers stayed under 6,400 thinking tokens while runaway ones hit the 12k cap.
DEFAULT_OPEN_BUDGET_TOKENS = 10000

CONVERGED = "converged"
BUDGET = "budget"

_DECORATION = re.compile(r"\^\\circ|\\%|\\text\{|\\mathrm\{|[{}$*\\]")
# "1. Use ELK ..." starts a list; its "1" must not count as agreeing with a boxed "1".
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s+(?=[^\W\d_])")


@dataclass(frozen=True)
class Probe:
    at_tokens: int
    answer: str | None
    seconds: float
    grounded: bool = True


def read_probe_answer(continuation: str) -> str | None:
    """Answer inside the \\boxed{ the probe left open, or None if the budget ran out before it closed."""
    found = extract_boxed("\\boxed{" + continuation)
    return normalize_answer(found[0][1]) or None if found else None


def grounded(boxed: str | None, plain: str) -> bool:
    """Whether the one-line answer starts by stating the boxed answer; open-ended prompts box a meaningless constant."""
    if not boxed:
        return False
    core = _DECORATION.sub("", boxed)
    plain = _LIST_MARKER.sub("", plain)
    text = _DECORATION.sub("", re.sub(r"\s+", "", plain.lower()))
    return bool(core) and core in text[: max(len(core) * 3, 30)]


@dataclass
class ProbeTracker:
    """Decides a stop point from probes: k identical grounded answers, or a thinking budget for open-ended prompts."""

    converge_k: int = DEFAULT_PROBE_K
    open_budget: int = DEFAULT_OPEN_BUDGET_TOKENS
    probes: list[Probe] = field(default_factory=list)
    decision: Probe | None = None
    reason: str | None = None

    def add(self, probe: Probe) -> bool:
        """Records a probe; True on the call where a stop point is first reached."""
        self.probes.append(probe)
        if self.decision is not None:
            return False
        recent = [p.answer if p.grounded else None for p in self.probes[-self.converge_k :]]
        if len(recent) == self.converge_k and recent[0] is not None and len(set(recent)) == 1:
            self.decision, self.reason = probe, CONVERGED
        elif self.looks_open and probe.at_tokens >= self.open_budget:
            self.decision, self.reason = probe, BUDGET
        return self.decision is not None

    @property
    def looks_open(self) -> bool:
        """At least 3 of the last 4 probes had no grounded answer."""
        recent = self.probes[-4:]
        return len(recent) == 4 and sum(not p.grounded for p in recent) >= 3

    @property
    def seconds(self) -> float:
        return sum(p.seconds for p in self.probes)
