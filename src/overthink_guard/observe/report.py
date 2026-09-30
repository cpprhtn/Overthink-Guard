from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from overthink_guard.observe.claude_code import is_interruption, parse_timestamp, starts_model_turn, thinking_tokens


def _quantile(values: list[int | float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))] if ordered else 0


def usage_report(projects_dir: Path, days: float | None = None, now: float | None = None) -> dict:
    """How much of Claude Code's output went to thinking, per effort level, from local session logs (counts only)."""
    now = now if now is not None else time.time()
    since = now - days * 86400 if days else None
    messages: dict[str, dict] = {}
    silences: list[float] = []
    for path in projects_dir.glob("*/*.jsonl"):
        if since is not None and path.stat().st_mtime < since:
            continue
        waiting = None
        with path.open(encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            ts = parse_timestamp(record.get("timestamp", "")) if isinstance(record, dict) else None
            if ts is None or record.get("isSidechain") or (since is not None and ts < since):
                continue
            if starts_model_turn(record):
                waiting = ts
            elif record.get("type") == "assistant":
                if waiting is not None:
                    silences.append(ts - waiting)
                    waiting = None
                message = record.get("message") or {}
                usage = message.get("usage") or {}
                # One API message is logged as several records (one per content block) repeating the same usage.
                key = message.get("id") or record.get("requestId") or record.get("uuid")
                if key and usage.get("output_tokens") is not None:
                    messages[key] = {
                        "effort": record.get("effort") or "unknown",
                        "project": path.parent.name,
                        "thinking": thinking_tokens(record),
                        "output": int(usage.get("output_tokens") or 0),
                    }

    by_effort: dict[str, list[dict]] = defaultdict(list)
    by_project: dict[str, int] = defaultdict(int)
    for m in messages.values():
        by_effort[m["effort"]].append(m)
        by_project[m["project"]] += m["thinking"]
    thinking = sum(m["thinking"] for m in messages.values())
    output = sum(m["output"] for m in messages.values())
    return {
        "responses": len(messages),
        "thinking_tokens": thinking,
        "output_tokens": output,
        "thinking_share": round(thinking / output, 3) if output else 0.0,
        "by_effort": {
            effort: {
                "responses": len(ms),
                "thinking_tokens": sum(m["thinking"] for m in ms),
                "median_thinking": _quantile([m["thinking"] for m in ms], 0.5),
                "p90_thinking": _quantile([m["thinking"] for m in ms], 0.9),
            }
            for effort, ms in sorted(by_effort.items())
        },
        "silence_seconds": {
            "turns": len(silences),
            "p50": round(_quantile(silences, 0.5), 1),
            "p90": round(_quantile(silences, 0.9), 1),
            "over_60s": sum(s > 60 for s in silences),
        },
        "top_projects": sorted(by_project.items(), key=lambda kv: kv[1], reverse=True)[:5],
    }


def _blocks(record: dict) -> list[dict]:
    content = (record.get("message") or {}).get("content")
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def flow_review(projects_dir: Path, flow_log: Path) -> dict:
    """What happened after each recorded flow signal, from the session logs (counts only)."""
    signals = []
    if flow_log.exists():
        for line in flow_log.read_text(encoding="utf-8").splitlines():
            try:
                signals.append(json.loads(line))
            except ValueError:
                continue
    rows = []
    for event in signals:
        sid = event.get("source", {}).get("session_id", "")
        at = parse_timestamp(event.get("timestamp", ""))
        path = next(projects_dir.glob(f"*/{sid}.jsonl"), None) if sid else None
        if path is None or at is None:
            continue
        records = []
        for line in path.open(encoding="utf-8", errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            ts = parse_timestamp(r.get("timestamp", "")) if isinstance(r, dict) else None
            if ts is not None and ts >= at - 1 and not r.get("isSidechain"):
                records.append((ts, r))
        records.sort(key=lambda x: x[0])
        tools = failed = 0
        last, stopped = at, False
        for ts, r in records:
            new_prompt = starts_model_turn(r) and not any(b.get("type") == "tool_result" for b in _blocks(r))
            if ts > at and new_prompt:
                break
            if is_interruption(r) and ts >= at - 1:
                stopped = True
                last = ts
                break
            if ts <= at:
                continue
            last = ts
            tools += sum(b.get("type") == "tool_use" for b in _blocks(r)) if r.get("type") == "assistant" else 0
            failed += sum(b.get("type") == "tool_result" and bool(b.get("is_error")) for b in _blocks(r))
        rows.append(
            {
                "reason": (event.get("signal") or {}).get("reasons", ["?"])[0],
                "tools_after": tools,
                "failed_after": failed,
                "seconds_after": last - at,
                "stopped_by_user": stopped,
            }
        )
    by_reason: dict[str, int] = defaultdict(int)
    for row in rows:
        by_reason[row["reason"]] += 1
    return {
        "signals": len(signals),
        "found_in_logs": len(rows),
        "by_reason": dict(by_reason),
        "stopped_by_user": sum(r["stopped_by_user"] for r in rows),
        "median_tools_after": _quantile([r["tools_after"] for r in rows], 0.5),
        "median_failed_after": _quantile([r["failed_after"] for r in rows], 0.5),
        "median_seconds_after": round(_quantile([r["seconds_after"] for r in rows], 0.5), 1),
    }
