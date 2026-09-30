"""A: do live action-flow signals in Claude Code sessions precede the user stopping the work? (offline, read-only)

Pre-registered in docs/validation/action-flow.md. Reads ~/.claude/projects session logs of human-driven sessions only,
never prints message text. Usage: python bench/spikes/action_flow.py [--projects-dir ~/.claude/projects]
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from overthink_guard.observe import default_projects_dir
from overthink_guard.observe.claude_code import parse_timestamp

HUMAN_PROJECTS = ("LiteDeck", "Overthink-Guard", "memo")
EXPERIMENTS = ("scratchpad", "otg-")
LOCAL = ("<local-command", "<command-name>", "<command-message>", "<bash-", "[Request interrupted")
CORRECTION = re.compile(r"^\s*(아니|아냐|그게 아니|잠깐|그만|멈춰|no\b|stop\b|wait\b|don'?t\b)", re.IGNORECASE)
EDITS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
SIGNALS = ["same_fail", "repeat_cmd", "reread", "errors3", "err_streak3", "edit_spread", "edit_churn"]
BASELINES = ["tools40", "time10"]


def blocks(r: dict) -> list[dict]:
    c = (r.get("message") or {}).get("content")
    return [b for b in c if isinstance(b, dict)] if isinstance(c, list) else []


def texts(r: dict) -> list[str]:
    c = (r.get("message") or {}).get("content")
    return [c] if isinstance(c, str) else [b.get("text", "") for b in blocks(r) if b.get("type") == "text"]


def is_prompt(r: dict) -> bool:
    if r.get("type") != "user" or r.get("isMeta") or r.get("isCompactSummary"):
        return False
    if any(b.get("type") == "tool_result" for b in blocks(r)):
        return False
    ts = [t for t in texts(r) if t.strip()]
    return bool(ts) and not all(t.lstrip().startswith(LOCAL) for t in ts)


def key_of(block: dict) -> tuple[str, str]:
    inp = block.get("input") or {}
    name = block.get("name", "")
    if name == "Bash":
        return name, " ".join(str(inp.get("command", "")).split())
    if "file_path" in inp:
        return name, str(inp["file_path"])
    return name, json.dumps(inp, sort_keys=True)[:500]


def turns_of(path: Path) -> list[dict]:
    records = []
    for line in path.open(encoding="utf-8", errors="replace"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and not r.get("isSidechain") and parse_timestamp(r.get("timestamp", "")):
            records.append(r)
    records.sort(key=lambda r: parse_timestamp(r["timestamp"]))
    turns, current = [], None
    for r in records:
        if is_prompt(r):
            if current:
                current["next_prompt"] = " ".join(texts(r))
                turns.append(current)
            current = {"start": parse_timestamp(r["timestamp"]), "records": [], "next_prompt": None}
        elif current is not None:
            current["records"].append(r)
    if current:
        turns.append(current)
    return turns


def evaluate(turn: dict) -> dict:
    """First time each signal fires, and when (if ever) the user stopped the work."""
    fired: dict[str, float] = {}
    uses: dict[str, tuple[str, str]] = {}
    runs, fails, reads, edits = Counter(), Counter(), Counter(), Counter()
    errors = streak = tools = 0
    stopped = None
    start = turn["start"]

    def fire(name: str, ts: float) -> None:
        fired.setdefault(name, ts)

    for r in turn["records"]:
        ts = parse_timestamp(r["timestamp"])
        if ts - start >= 600:
            fire("time10", start + 600)
        if r.get("type") == "user" and any(t.lstrip().startswith("[Request interrupted") for t in texts(r)):
            stopped = stopped or ts
        if r.get("type") == "assistant":
            for b in blocks(r):
                if b.get("type") != "tool_use":
                    continue
                tools += 1
                name, key = key_of(b)
                uses[b.get("id", "")] = (name, key)
                if name == "Bash":
                    runs[key] += 1
                    if runs[key] >= 3:
                        fire("repeat_cmd", ts)
                if name == "Read":
                    reads[key] += 1
                    if reads[key] >= 3:
                        fire("reread", ts)
                if name in EDITS:
                    edits[key] += 1
                    if len(edits) >= 8:
                        fire("edit_spread", ts)
                    if edits[key] >= 6:
                        fire("edit_churn", ts)
                if tools >= 40:
                    fire("tools40", ts)
        if r.get("type") == "user":
            for b in blocks(r):
                if b.get("type") != "tool_result":
                    continue
                if b.get("is_error"):
                    errors += 1
                    streak += 1
                    use = uses.get(b.get("tool_use_id", ""))
                    if use:
                        fails[use] += 1
                        if fails[use] >= 2:
                            fire("same_fail", ts)
                    if errors >= 3:
                        fire("errors3", ts)
                    if streak >= 3:
                        fire("err_streak3", ts)
                else:
                    streak = 0
    corrected = bool(turn["next_prompt"] and CORRECTION.search(turn["next_prompt"]))
    return {"fired": fired, "stopped": stopped, "corrected": corrected}


def table(results: list[dict], label: str, positive) -> None:
    n = len(results)
    pos = sum(positive(r) for r in results)
    base = pos / n if n else 0
    print(f"\n[{label}] turns {n}, positive {pos} ({base:.1%})")
    print(f"  {'signal':12} {'fired':>5} {'hit':>4} {'precision':>9} {'recall':>6} {'lift':>5}")
    for name in SIGNALS + BASELINES:
        fired = hit = 0
        for r in results:
            t = r["fired"].get(name)
            if t is None:
                continue
            if positive(r):
                cutoff = r["stopped"]
                if cutoff is None or t < cutoff:
                    fired += 1
                    hit += 1
            else:
                fired += 1
        precision = hit / fired if fired else 0
        print(
            f"  {name:12} {fired:5d} {hit:4d} {precision:9.1%} {hit / pos if pos else 0:6.1%} "
            f"{precision / base if base else 0:5.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects-dir", type=Path, default=default_projects_dir())
    args = parser.parse_args()
    results = []
    for path in sorted(args.projects_dir.glob("*/*.jsonl")):
        project = path.parent.name
        if not any(p in project for p in HUMAN_PROJECTS) or any(x in project for x in EXPERIMENTS):
            continue
        results += [evaluate(t) for t in turns_of(path)]
    table(results, "stopped by the user (Esc or rejected tool)", lambda r: r["stopped"] is not None)
    table(
        results,
        "secondary: stopped or corrected in the next prompt",
        lambda r: r["stopped"] is not None or r["corrected"],
    )


if __name__ == "__main__":
    main()
