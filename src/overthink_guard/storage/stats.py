from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def default_stats_path() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "overthink-guard" / "shadow.jsonl"


def _tier_summary(records: list[dict], key: str) -> dict:
    rows = [r for r in records if r.get(key)]
    thinking = sum(r["thinking_tokens"] for r in rows)
    stopped = [r for r in rows if r[key]["stop_at"] is not None]
    saved = sum(r["thinking_tokens"] - r[key]["stop_at"] for r in stopped)
    return {
        "requests": len(rows),
        "would_stop": len(stopped),
        "saved_tokens": saved,
        "saved_ratio": round(saved / thinking, 4) if thinking else 0.0,
        "mismatch": sum(1 for r in stopped if r[key]["match"] is False),
        "unknown": sum(1 for r in stopped if r[key]["match"] is None),
    }


class ShadowStats:
    """Per-request Shadow outcomes as JSONL: counts and booleans only, never prompt, thinking or answer text."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._records: list[dict] = []
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    self._records.append(json.loads(line))
                except ValueError:
                    continue

    @property
    def path(self) -> Path | None:
        return self._path

    def add(self, record: dict) -> None:
        self._records.append(record)
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def summary(self) -> dict:
        probed = [r for r in self._records if r.get("tier2")]
        probe_s = sum(r["tier2"]["probe_seconds"] for r in probed)
        elapsed = sum(r["elapsed_seconds"] for r in probed)
        return {
            "requests": len(self._records),
            "thinking_tokens": sum(r["thinking_tokens"] for r in self._records),
            "tier0": _tier_summary(self._records, "tier0"),
            "tier2": {
                **_tier_summary(self._records, "tier2"),
                "overhead_ratio": round(probe_s / elapsed, 4) if elapsed else 0.0,
            },
        }
