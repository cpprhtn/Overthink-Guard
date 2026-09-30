"""V6: what stopping a headless Claude Code turn early costs, on problems with known answers.

For each problem: a full `claude -p` run, and a run interrupted with SIGINT at --stop-at seconds followed (only if the
turn really was interrupted) by one `--resume` follow-up asking for the best answer now. Problems: the 12 of the
held-out OpenR1 set (bench/data/math_d_qwen3_2026-09-30.json) on which qwen3 thought longest.

Uses your Claude subscription (about 36 Sonnet calls). Usage:
python bench/spikes/subscription_early_stop.py --cache <dir with openr1_math_*.json> [--stop-at 20] [--out results.json]
"""

import argparse
import json
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shadow_validate import held_out  # noqa: E402

DATA = Path(__file__).parents[1] / "data" / "math_d_qwen3_2026-09-30.json"
SUFFIX = "\n\nDo not use any tools. Reply with only the final number."
FOLLOW_UP = "Stop analysing. Give your best answer now from the reasoning so far. Reply with only the final number."
FLAGS = ["--model", "sonnet", "--effort", "high", "--output-format", "stream-json", "--verbose"]


def claude(args: list[str], cwd: Path, sigint_after: float | None = None) -> dict:
    start = time.time()
    proc = subprocess.Popen(["claude", "-p", *args, *FLAGS], cwd=cwd, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, text=True)
    if sigint_after is not None:
        threading.Timer(sigint_after, lambda: proc.poll() is None and proc.send_signal(signal.SIGINT)).start()
    events = [json.loads(line) for line in proc.stdout if line.startswith("{")]
    proc.wait()
    result = next((e for e in events if e.get("type") == "result"), {})
    usage = result.get("usage") or {}
    return {
        "seconds": round(time.time() - start, 1),
        "session_id": next((e["session_id"] for e in events if e.get("session_id")), None),
        "end": result.get("terminal_reason") or result.get("subtype"),
        "answer": _last_int(result.get("result") or ""),
        "output_tokens": usage.get("output_tokens", 0),
        "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens", 0),
    }


def _last_int(text: str) -> str | None:
    found = re.findall(r"-?\d[\d,]*", text)
    return found[-1].replace(",", "") if found else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--stop-at", type=float, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    problems = held_out(args.cache, 12, 100, 30)
    recorded = json.loads(DATA.read_text())["problems"]
    assert [gold for _, gold in problems] == [p["gold"] for p in recorded], "problem set does not match the data file"
    hardest = sorted(range(len(recorded)), key=lambda i: -recorded[i]["thinking_tokens"])[:12]
    rows = []
    for i in sorted(hardest):
        question, gold = problems[i]
        with tempfile.TemporaryDirectory() as tmp:
            full = claude([question + SUFFIX], Path(tmp))
        with tempfile.TemporaryDirectory() as tmp:
            early = claude([question + SUFFIX], Path(tmp), sigint_after=args.stop_at)
            follow = None
            if early["end"] == "aborted_streaming" and early["session_id"]:
                follow = claude(["--resume", early["session_id"], FOLLOW_UP], Path(tmp))
        final = follow["answer"] if follow else early["answer"]
        row = {"index": i, "gold": gold, "full": full, "early": early, "follow_up": follow,
               "full_correct": full["answer"] == gold, "early_correct": final == gold}
        rows.append(row)
        print(f"{i:2d} gold {gold:>8} | full {full['answer']!s:>8} {full['seconds']:5.1f}s {full['thinking_tokens']:5d}t "
              f"| early {early['end']} -> {final!s:>8} "
              f"{early['seconds'] + (follow['seconds'] if follow else 0):5.1f}s", flush=True)
    stopped = [r for r in rows if r["follow_up"]]
    print(f"full correct {sum(r['full_correct'] for r in rows)}/{len(rows)}; "
          f"interrupted {len(stopped)}: full correct {sum(r['full_correct'] for r in stopped)}, "
          f"early+follow-up correct {sum(r['early_correct'] for r in stopped)}; "
          f"time full {sum(r['full']['seconds'] for r in stopped):.0f}s vs early "
          f"{sum(r['early']['seconds'] + r['follow_up']['seconds'] for r in stopped):.0f}s; "
          f"output tokens full {sum(r['full']['output_tokens'] for r in stopped)} vs early+follow-up "
          f"{sum(r['early']['output_tokens'] + r['follow_up']['output_tokens'] for r in stopped)}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
