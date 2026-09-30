"""Runs a headless `claude -p` turn and registers its session so stop_on_suggest.py may interrupt it.

Usage: python run_claude.py "prompt" [--follow-up "Answer now from what you have."] [extra claude flags...]
If the turn is interrupted, the follow-up is sent once with --resume. The registry lives in $OTG_RECIPE_DIR.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REGISTRY = Path(os.environ.get("OTG_RECIPE_DIR") or Path(tempfile.gettempdir()) / "otg-claude-headless")


def run_turn(args: list[str]) -> tuple[dict | None, str | None]:
    cmd = ["claude", "-p", *args, "--output-format", "stream-json", "--verbose"]
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
    entry, result, session_id = None, None, None
    try:
        for line in proc.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if entry is None and event.get("session_id"):
                # Only sessions this script started are ever interrupted.
                session_id = event["session_id"]
                entry = REGISTRY / f"{session_id}.json"
                entry.write_text(json.dumps({"pid": proc.pid, "interrupted": False}))
            if "is_error" in event:
                result = event
    finally:
        proc.wait()
        if entry is not None:
            entry.unlink(missing_ok=True)
    return result, session_id


def main() -> int:
    argv = sys.argv[1:]
    follow_up = None
    if "--follow-up" in argv:
        i = argv.index("--follow-up")
        follow_up = argv[i + 1]
        del argv[i : i + 2]
    prompt, extra = argv[0], argv[1:]
    REGISTRY.mkdir(parents=True, exist_ok=True)
    result, session_id = run_turn([prompt, *extra])
    interrupted = result is not None and result.get("terminal_reason") == "aborted_streaming"
    if interrupted and follow_up and session_id:
        print("[interrupted: sending the follow-up once]", file=sys.stderr)
        result, _ = run_turn(["--resume", session_id, follow_up, *extra])
    if result:
        print(result.get("result") or "")
        print(f"[{result.get('terminal_reason') or result.get('subtype')}]", file=sys.stderr)
    return 0 if result and not result.get("is_error") else 1


if __name__ == "__main__":
    sys.exit(main())
