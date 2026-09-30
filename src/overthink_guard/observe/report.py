from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from overthink_guard.observe.claude_code import parse_timestamp, starts_model_turn, thinking_tokens


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
