"""Hook for `otg claude-code watch --hook`: sends SIGINT once to a headless turn that run_claude.py started.

Never touches interactive sessions or sessions it did not start. SIGINT ends the current turn cleanly with a partial
result (Claude Code's documented behavior); the process is never killed.
"""

import json
import os
import signal
import sys

from run_claude import REGISTRY


def main() -> int:
    event = json.load(sys.stdin)
    if event.get("event") != "stop_suggested" or event["source"].get("run_mode") == "interactive":
        return 0
    entry = REGISTRY / f"{event['source'].get('session_id')}.json"
    if not entry.exists():
        return 0
    state = json.loads(entry.read_text())
    if state.get("interrupted"):
        return 0  # at most once per session, so an automatic loop cannot form
    state["interrupted"] = True
    entry.write_text(json.dumps(state))
    if sys.platform == "win32":
        return 0  # SIGINT cannot be sent to another console process on Windows; notify-only there
    os.kill(state["pid"], signal.SIGINT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
