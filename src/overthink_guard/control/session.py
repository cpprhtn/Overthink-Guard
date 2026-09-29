from __future__ import annotations

import asyncio
import itertools
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from overthink_guard.analysis import Judge, Probe, ProbeTracker, Segment, StopDecision
from overthink_guard.analysis.signals import extract_tentative_answer
from overthink_guard.storage import ShadowStats

THINKING = "thinking"
ANSWERING = "answering"
DONE = "done"
ERROR = "error"
CANCELLED = "cancelled"


@dataclass
class Session:
    id: str
    model: str
    prompt: str
    judge: Judge
    created: float = field(default_factory=time.time)
    status: str = THINKING
    thinking: str = ""
    answer: str = ""
    intervened: bool = False
    probes: ProbeTracker | None = None
    probe_skipped: str | None = None
    shadow: dict | None = None
    stop_requested: asyncio.Event = field(default_factory=asyncio.Event)


def _segment_event(session: Session, segment: Segment) -> dict:
    return {
        "type": "segment",
        "session": session.id,
        "index": segment.index,
        "phase": segment.phase,
        "start_tokens": segment.start_tokens,
        "end_tokens": segment.end_tokens,
        "answer": segment.answer,
        "text": segment.text,
    }


def _probe_dict(probe: Probe) -> dict:
    return {"at_tokens": probe.at_tokens, "answer": probe.answer, "seconds": round(probe.seconds, 3)}


class SessionHub:
    """In-memory registry of proxied requests and fan-out of their progress to UI subscribers."""

    def __init__(self, stats: ShadowStats | None = None, keep: int = 50) -> None:
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._subscribers: set[asyncio.Queue] = set()
        self._ids = itertools.count(1)
        self._keep = keep
        self.stats = stats or ShadowStats(None)

    def create(self, model: str, prompt: str, judge: Judge, probes: ProbeTracker | None = None) -> Session:
        session = Session(id=str(next(self._ids)), model=model, prompt=prompt[:200], judge=judge, probes=probes)
        self._sessions[session.id] = session
        while len(self._sessions) > self._keep:
            self._sessions.popitem(last=False)
        self._publish(self.summary(session))
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def add_thinking(self, session: Session, text: str, tokens: int | None = None) -> None:
        session.thinking += text
        before = len(session.judge.segments)
        had_decision = session.judge.decision is not None
        session.judge.feed(text, tokens)
        for segment in session.judge.segments[before:]:
            self._publish(_segment_event(session, segment))
        if len(session.judge.segments) > before or (session.judge.decision and not had_decision):
            self._publish(self.summary(session))

    def add_answer(self, session: Session, text: str) -> None:
        session.answer += text

    def add_probe(self, session: Session, probe: Probe) -> None:
        assert session.probes is not None
        converged = session.probes.add(probe)
        self._publish({"type": "probe", "session": session.id, **_probe_dict(probe)})
        if converged:
            self._publish(self.summary(session))

    def record_shadow(self, session: Session, elapsed_seconds: float) -> None:
        """Stores what Tier 0 (and Tier 2, if probing) would have done on a request that thought to completion."""
        final = extract_tentative_answer(session.answer, session.judge.template)

        def outcome(stop: StopDecision | Probe | None) -> dict:
            if stop is None:
                return {"stop_at": None, "match": None}
            return {"stop_at": stop.at_tokens, "match": None if final is None else stop.answer == final}

        record = {
            "ts": round(time.time()),
            "model": session.model,
            "thinking_tokens": session.judge.thinking_tokens,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "tier0": outcome(session.judge.decision),
            "tier2": None,
            # Probe pauses resume with fresh sampling, so this run is not the one the client would get unprobed.
            "perturbed": bool(session.probes and session.probes.probes),
        }
        if session.probes is not None:
            record["tier2"] = {
                **outcome(session.probes.decision),
                "probes": len(session.probes.probes),
                "probe_seconds": round(session.probes.seconds, 3),
            }
        session.shadow = record
        self.stats.add(record)
        self._publish(self.summary(session))
        self._publish({"type": "stats", **self.stats.summary()})

    def set_status(self, session: Session, status: str) -> None:
        if status in (ANSWERING, DONE, ERROR, CANCELLED):
            before = len(session.judge.segments)
            session.judge.finish()
            for segment in session.judge.segments[before:]:
                self._publish(_segment_event(session, segment))
        session.status = status
        self._publish(self.summary(session))

    def request_stop(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if session is None or session.status != THINKING or not session.judge.template.prefill_supported:
            return False
        session.stop_requested.set()
        return True

    def summary(self, session: Session) -> dict:
        decision = session.judge.decision
        return {
            "type": "session",
            "id": session.id,
            "model": session.model,
            "prompt": session.prompt,
            "created": session.created,
            "status": session.status,
            "thinking_tokens": session.judge.thinking_tokens,
            "intervened": session.intervened,
            "can_intervene": session.judge.template.prefill_supported,
            "suggestion": None
            if decision is None
            else {"at_tokens": decision.at_tokens, "answer": decision.answer, "reasons": list(decision.reasons)},
            "probing": session.probes is not None,
            "probe_skipped": session.probe_skipped,
            "probe_converged": None
            if session.probes is None or session.probes.decision is None
            else _probe_dict(session.probes.decision),
            "shadow": session.shadow,
        }

    def snapshot(self) -> list[dict]:
        events: list[dict] = [{"type": "stats", **self.stats.summary()}]
        for session in self._sessions.values():
            events.append(self.summary(session))
            events.extend(_segment_event(session, s) for s in session.judge.segments)
            if session.probes is not None:
                events.extend({"type": "probe", "session": session.id, **_probe_dict(p)} for p in session.probes.probes)
        return events

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def _publish(self, event: dict) -> None:
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
