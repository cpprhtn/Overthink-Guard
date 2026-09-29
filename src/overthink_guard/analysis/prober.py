from __future__ import annotations

from dataclasses import dataclass, field

from overthink_guard.analysis.signals import extract_boxed, normalize_answer

# [가설] Early probe answers can hold a wrong guess for 2000+ tokens; ignoring probes before 3000 thinking tokens
# was the best zero-loss rule on 37 problems, chosen in-sample (docs/spikes/shadow-live-probe.md).
DEFAULT_PROBE_K = 4
DEFAULT_PROBE_MIN_TOKENS = 3000


@dataclass(frozen=True)
class Probe:
    at_tokens: int
    answer: str | None
    seconds: float


def read_probe_answer(continuation: str) -> str | None:
    """Answer inside the \\boxed{ the probe left open, or None if the budget ran out before it closed."""
    found = extract_boxed("\\boxed{" + continuation)
    return normalize_answer(found[0][1]) or None if found else None


@dataclass
class ProbeTracker:
    """Tier 2 convergence: the probe answer is the same, non-empty, k probes in a row."""

    converge_k: int = DEFAULT_PROBE_K
    probes: list[Probe] = field(default_factory=list)
    decision: Probe | None = None

    def add(self, probe: Probe) -> bool:
        """Records a probe; True on the call where convergence is first reached."""
        self.probes.append(probe)
        if self.decision is not None:
            return False
        recent = [p.answer for p in self.probes[-self.converge_k :]]
        if len(recent) == self.converge_k and recent[0] is not None and len(set(recent)) == 1:
            self.decision = probe
            return True
        return False

    @property
    def seconds(self) -> float:
        return sum(p.seconds for p in self.probes)
