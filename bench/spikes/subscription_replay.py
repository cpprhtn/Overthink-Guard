"""V1: replays real Claude Code session logs through the real observer with a simulated clock.

Ground truth is an independent reading of each log, following the spec in docs/validation/subscription.md: a wait
starts at a user prompt or at the last outstanding tool result (compaction neither starts nor ends one), and is labeled
by the record that ends it. An alert is a true positive when its wait ended with model output or with the user
interrupting the model mid-turn; otherwise (API error, new prompt, end of log) it is a false alarm.

Usage: python bench/spikes/subscription_replay.py [--after 60] [--projects-dir ~/.claude/projects]
"""

import argparse
import json
import tempfile
from pathlib import Path

from overthink_guard.observe import ClaudeCodeObserver, default_projects_dir
from overthink_guard.observe.claude_code import parse_timestamp, thinking_tokens

NOISE = {"attachment", "queue-operation", "file-history-snapshot", "last-prompt", "ai-title", "atis-latch"}
LOCAL = ("<local-command", "<command-name>", "<command-message>", "<bash-")
WORKING = ("output", "interrupted")


def main_chain(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and not r.get("isSidechain") and parse_timestamp(r.get("timestamp", "")):
                records.append(r)
    records.sort(key=lambda r: parse_timestamp(r["timestamp"]))
    return records


def _blocks(r: dict) -> list[dict]:
    c = (r.get("message") or {}).get("content")
    return [b for b in c if isinstance(b, dict)] if isinstance(c, list) else []


def _texts(r: dict) -> list[str]:
    c = (r.get("message") or {}).get("content")
    return [c] if isinstance(c, str) else [b.get("text", "") for b in _blocks(r) if b.get("type") == "text"]


def waits(records: list[dict]) -> list[dict]:
    """Every interval in which the model had what it needed and owed a response, labeled by how it ended."""
    out: list[dict] = []
    start: float | None = None
    pending: set[str] = set()

    def close(r: dict | None, label: str) -> None:
        nonlocal start
        if start is not None:
            out.append({"start": start, "end": parse_timestamp(r["timestamp"]) if r else float("inf"), "label": label,
                        "thinking": thinking_tokens(r) if r and label == "output" else None})
        start = None

    for r in records:
        kind, ts = r.get("type"), parse_timestamp(r["timestamp"])
        if kind in NOISE:
            continue
        if kind == "assistant":
            close(r, "output")
            pending |= {b["id"] for b in _blocks(r) if b.get("type") == "tool_use" and "id" in b}
        elif kind == "system" and r.get("subtype") == "api_error":
            close(r, "api_error")
            pending.clear()
        elif kind != "user" or r.get("isMeta") or r.get("isCompactSummary"):
            continue  # compaction is Claude Code's own work: a wait under way continues, none starts
        elif any(t.lstrip().startswith("[Request interrupted") for t in _texts(r)):
            close(r, "interrupted")
            pending.clear()
        else:
            results = {b.get("tool_use_id") for b in _blocks(r) if b.get("type") == "tool_result"}
            texts = [t for t in _texts(r) if not t.lstrip().startswith(LOCAL)]
            if results:
                pending -= results
                if not pending:
                    close(r, "new_prompt")
                    start = ts
            elif texts:
                close(r, "new_prompt")
                pending.clear()
                start = ts
    close(None, "end_of_log")
    return out


def replay_alerts(records: list[dict], after: float, wait_list: list[dict]) -> list[float]:
    """Alert times from the real observer, fed the log record by record with the clock stepped to each due time."""
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "proj" / "s.jsonl"
        log.parent.mkdir()
        log.write_text("")
        clock = {"now": parse_timestamp(records[0]["timestamp"]) - 1}
        alerts: list[float] = []
        observer = ClaudeCodeObserver(Path(tmp), after, lambda e: alerts.append(clock["now"]), clock=lambda: clock["now"])
        observer.poll()
        dues = sorted(w["start"] + after for w in wait_list)
        with log.open("a") as fh:
            for r in records:
                ts = parse_timestamp(r["timestamp"])
                while dues and dues[0] < ts:
                    clock["now"] = dues.pop(0)
                    observer.poll()
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                clock["now"] = ts
                observer.poll()
            for due in dues:
                clock["now"] = max(clock["now"], due)
                observer.poll()
    return alerts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--after", type=float, default=60)
    parser.add_argument("--projects-dir", type=Path, default=default_projects_dir())
    parser.add_argument("--strict", action="store_true", help="count only model output as working (not interrupts)")
    args = parser.parse_args()
    working = ("output",) if args.strict else WORKING
    tp = fp = relevant = n_waits = sessions = 0
    fp_labels: list[str] = []
    tp_thinking: list[int] = []
    all_thinking: list[int] = []
    for path in sorted(args.projects_dir.glob("*/*.jsonl")):
        records = main_chain(path)
        if not records:
            continue
        sessions += 1
        wait_list = waits(records)
        n_waits += len(wait_list)
        all_thinking += [w["thinking"] for w in wait_list if w["thinking"] is not None]
        relevant += sum(w["end"] - w["start"] >= args.after and w["label"] in working for w in wait_list)
        for t in replay_alerts(records, args.after, wait_list):
            w = next((w for w in wait_list if w["start"] <= t <= w["end"]), None)
            if w and w["label"] in working:
                tp += 1
                if w["thinking"] is not None:
                    tp_thinking.append(w["thinking"])
            else:
                fp += 1
                fp_labels.append(w["label"] if w else "outside any wait")

    def q(xs: list[int], p: float) -> int | None:
        return sorted(xs)[min(len(xs) - 1, int(len(xs) * p))] if xs else None

    print(f"threshold {args.after:g}s: {sessions} sessions, {n_waits} waits")
    print(f"  alerts {tp + fp}: model working {tp}, false alarm {fp} -> precision {tp / max(1, tp + fp):.1%}")
    print(f"  waits >= threshold ending in model work (independent): {relevant} -> recall {tp / max(1, relevant):.1%}")
    print(f"  false alarm causes: {sorted((x, fp_labels.count(x)) for x in set(fp_labels))}")
    if tp_thinking:
        print(f"  thinking tokens: alerted median {q(tp_thinking, .5)}, p90 {q(tp_thinking, .9)}, "
              f"zero {sum(t == 0 for t in tp_thinking)}/{len(tp_thinking)}; all turns median {q(all_thinking, .5)}, "
              f"p90 {q(all_thinking, .9)}, p99 {q(all_thinking, .99)}")


if __name__ == "__main__":
    main()
