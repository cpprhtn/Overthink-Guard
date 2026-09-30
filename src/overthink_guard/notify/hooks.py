from __future__ import annotations

import json
import os
import shlex
import subprocess


def run_hook(command: str, event: dict, timeout_s: float = 10) -> int | None:
    """Runs the user's hook command with the event JSON on stdin, without a shell; None if it failed to run."""
    if not command.strip():
        return None
    argv = shlex.split(command, posix=os.name != "nt")
    try:
        done = subprocess.run(
            argv, input=json.dumps(event), text=True, timeout=timeout_s, capture_output=True, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.returncode
