"""Read-only observer for Claude Code session logs (~/.claude/projects/*/*.jsonl); never keeps message text."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# User records that do not start a model turn: local slash commands, shell escapes, interruptions.
_INTERRUPTED = "[Request interrupted"
_NO_MODEL_CALL = ("<local-command", "<command-name>", "<command-message>", "<bash-", _INTERRUPTED)


def _claude_dir() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")


def default_projects_dir() -> Path:
    return _claude_dir() / "projects"


def default_sessions_dir() -> Path:
    return _claude_dir() / "sessions"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (OSError, ValueError):
        return False
    return True


def session_states(sessions_dir: Path, pid_alive: Callable[[int], bool] = _pid_alive) -> dict[str, dict]:
    """sessionId -> {status, alive} from Claude Code's <pid>.json state files; never the *.key files beside them."""
    # Interactive sessions report status busy/idle; headless `claude -p` runs leave it empty and delete the file on
    # exit. `kind` is not used: a run started from inside another session inherits "interactive".
    states = {}
    for path in sessions_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("sessionId"), str):
            alive = isinstance(data.get("pid"), int) and pid_alive(data["pid"])
            states[data["sessionId"]] = {"status": data.get("status"), "alive": alive}
    return states


def parse_timestamp(value: str) -> float | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return None


def starts_model_turn(record: dict) -> bool:
    """Whether a user record hands control to the model (a prompt or a tool result)."""
    # A compaction summary is written by Claude Code itself; after an auto-compaction the wait already under way
    # continues, and after a manual /compact nothing follows.
    if record.get("type") != "user" or any(record.get(k) for k in ("isMeta", "isSidechain", "isCompactSummary")):
        return False
    content = (record.get("message") or {}).get("content")
    if isinstance(content, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return True
        texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return bool(texts) and not all(t.lstrip().startswith(_NO_MODEL_CALL) for t in texts)
    return isinstance(content, str) and not content.lstrip().startswith(_NO_MODEL_CALL)


def _blocks(record: dict) -> list[dict]:
    content = (record.get("message") or {}).get("content")
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _texts(record: dict) -> list[str]:
    content = (record.get("message") or {}).get("content")
    if isinstance(content, str):
        return [content]
    return [b.get("text", "") for b in _blocks(record) if b.get("type") == "text"]


def is_interruption(record: dict) -> bool:
    return record.get("type") == "user" and any(t.lstrip().startswith(_INTERRUPTED) for t in _texts(record))


def thinking_tokens(record: dict) -> int:
    usage = (record.get("message") or {}).get("usage") or {}
    return int((usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0)


@dataclass
class _Session:
    session_id: str
    project: str
    waiting_since: float | None = None
    alerted: bool = False
    effort: str | None = None
    pending_tools: set[str] = field(default_factory=set)


def _run_mode(state: dict | None) -> str:
    if state is None:
        return "unknown"
    return "interactive" if state["status"] in ("busy", "idle") else "headless"


def alert_event(session: _Session, now: float, run_mode: str = "unknown") -> dict:
    """Hook event, schema v1 (concept doc 6.2). No text from the session is included."""
    return {
        "schema_version": 1,
        "event": "stop_suggested",
        "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "source": {
            "kind": "subscription",
            "tool": "claude_code",
            "session_id": session.session_id,
            "project": session.project,
            "run_mode": run_mode,
        },
        "signal": {"reasons": ["long_silence"], "confidence": None, "since_last_revision_s": None},
        "stats": {"thinking_elapsed_s": round(now - (session.waiting_since or now)), "summary_chunks": 0},
    }


@dataclass
class ClaudeCodeObserver:
    """Tails session logs and reports turns where the model has been silent longer than after_seconds."""

    projects_dir: Path
    after_seconds: float
    on_alert: Callable[[dict], None]
    clock: Callable[[], float] = time.time
    recent_seconds: float = 900
    sessions_dir: Path | None = None
    pid_alive: Callable[[int], bool] = _pid_alive
    _offsets: dict[Path, int] = field(default_factory=dict)
    _partial: dict[Path, bytes] = field(default_factory=dict)
    _sessions: dict[Path, _Session] = field(default_factory=dict)
    _primed: bool = False
    _had_state: set[str] = field(default_factory=set)

    def poll(self) -> None:
        # Thinking text reaches the log only after the block ends (docs/spikes/s4-claude-code.md), so the live signal
        # is silence: no assistant record since the last prompt or tool result.
        now = self.clock()
        for path in self._recent_logs(now):
            for record in self._new_records(path):
                self._apply(path, record)
        states = session_states(self.sessions_dir, self.pid_alive) if self.sessions_dir else {}
        for session in self._sessions.values():
            waited = now - session.waiting_since if session.waiting_since is not None else 0
            state = states.get(session.session_id)
            if state is not None:
                self._had_state.add(session.session_id)
            if session.waiting_since is not None and not session.alerted and self.after_seconds <= waited:
                # Not thinking: turns already silent when the observer started, idle or exited sessions, and
                # sessions whose state file has disappeared (the process ended mid-turn).
                ended = state is None and session.session_id in self._had_state
                stopped = state is not None and (not state["alive"] or state["status"] == "idle")
                if self._primed and not ended and not stopped:
                    self.on_alert(alert_event(session, now, _run_mode(state)))
                session.alerted = True
        self._primed = True

    def _recent_logs(self, now: float) -> Iterator[Path]:
        for path in self.projects_dir.glob("*/*.jsonl"):
            try:
                if now - path.stat().st_mtime <= self.recent_seconds or path in self._offsets:
                    yield path
            except OSError:
                continue

    def _new_records(self, path: Path) -> Iterator[dict]:
        offset = self._offsets.get(path, 0)
        try:
            with path.open("rb") as fh:
                fh.seek(offset)
                data = self._partial.pop(path, b"") + fh.read()
                self._offsets[path] = fh.tell()
        except OSError:
            return
        *lines, rest = data.split(b"\n")
        if rest:
            self._partial[path] = rest
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                yield record

    def _apply(self, path: Path, record: dict) -> None:
        if record.get("isSidechain"):
            return
        session = self._sessions.get(path)
        if session is None:
            session = self._sessions[path] = _Session(record.get("sessionId") or path.stem, path.parent.name)
        ts = parse_timestamp(record.get("timestamp", ""))
        if ts is None:
            return
        if record.get("type") == "assistant":
            session.waiting_since, session.alerted = None, False
            session.effort = record.get("effort") or session.effort
            session.pending_tools |= {b["id"] for b in _blocks(record) if b.get("type") == "tool_use" and "id" in b}
        elif is_interruption(record) or (record.get("type") == "system" and record.get("subtype") == "api_error"):
            # The model stopped (Esc) or its request failed: whatever follows is not thinking.
            session.waiting_since = None
            session.pending_tools.clear()
        elif starts_model_turn(record):
            results = {b.get("tool_use_id") for b in _blocks(record) if b.get("type") == "tool_result"}
            session.pending_tools -= results
            if not results:
                session.pending_tools.clear()
            # With parallel tool calls the model resumes only after the last result arrives.
            if not session.pending_tools:
                session.waiting_since, session.alerted = ts, False
