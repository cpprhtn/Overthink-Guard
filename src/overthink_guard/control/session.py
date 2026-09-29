from __future__ import annotations

import asyncio
import itertools
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from overthink_guard.analysis import Judge, Segment

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
    intervened: bool = False
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


class SessionHub:
    """In-memory registry of proxied requests and fan-out of their progress to UI subscribers."""

    def __init__(self, keep: int = 50) -> None:
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._subscribers: set[asyncio.Queue] = set()
        self._ids = itertools.count(1)
        self._keep = keep

    def create(self, model: str, prompt: str, judge: Judge) -> Session:
        session = Session(id=str(next(self._ids)), model=model, prompt=prompt[:200], judge=judge)
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
        if session is None or session.status != THINKING:
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
            "suggestion": None
            if decision is None
            else {"at_tokens": decision.at_tokens, "answer": decision.answer, "reasons": list(decision.reasons)},
        }

    def snapshot(self) -> list[dict]:
        events = []
        for session in self._sessions.values():
            events.append(self.summary(session))
            events.extend(_segment_event(session, s) for s in session.judge.segments)
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
