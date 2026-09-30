"""Claude Code hook for action-flow study B: counts tool failures per session and, in nudge mode, adds a note to the
model's context when the pre-registered signals fire (docs/validation/action-flow.md). Never stops the turn.

Usage (from --settings): python3 action_flow_hook.py <log|nudge> <state_dir>
"""

import fcntl
import json
import sys
import time
from pathlib import Path

NUDGE = (
    "[Overthink Guard, a monitor the user runs on this session] {reason}. Retrying the same thing rarely helps. "
    "Before the next attempt, state in one or two sentences what the error tells you about the cause, then change "
    "approach. If it truly cannot be done in this environment, say so."
)
MAX_NUDGES = 2


def main() -> None:
    mode, state_dir = sys.argv[1], Path(sys.argv[2])
    event = json.load(sys.stdin)
    path = state_dir / f"{event.get('session_id', 'unknown')}.json"
    # Parallel tool calls run their hooks concurrently; serialise the read-modify-write of the state file.
    with open(state_dir / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        output = update(mode, event, path)
    if output:
        print(json.dumps(output))


def update(mode: str, event: dict, path: Path) -> dict | None:
    state = (
        json.loads(path.read_text())
        if path.exists()
        else {"tools": 0, "errors": 0, "fails": {}, "fires": [], "nudges": 0}
    )
    state["tools"] += 1
    reason = None
    if event.get("hook_event_name") == "PostToolUseFailure":
        state["errors"] += 1
        inp = event.get("tool_input") or {}
        key = f"{event.get('tool_name')}|{' '.join(str(inp.get('command', json.dumps(inp, sort_keys=True))).split())}"
        state["fails"][key] = state["fails"].get(key, 0) + 1
        if state["fails"][key] >= 2:
            reason = "The same command has now failed twice"
        elif state["errors"] >= 3:
            reason = f"{state['errors']} tool calls have failed in this task"
    output = None
    if reason:
        state["fires"].append(
            {"time": time.time(), "errors": state["errors"], "tools": state["tools"], "reason": reason}
        )
        if mode == "nudge" and state["nudges"] < MAX_NUDGES:
            state["nudges"] += 1
            output = {
                "hookSpecificOutput": {
                    "hookEventName": event["hook_event_name"],
                    "additionalContext": NUDGE.format(reason=reason),
                }
            }
    path.write_text(json.dumps(state))
    return output


if __name__ == "__main__":
    main()
