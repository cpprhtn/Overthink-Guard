from __future__ import annotations

import shutil
import subprocess
import sys

_APPLESCRIPT = [
    "on run argv",
    "display notification (item 2 of argv) with title (item 1 of argv)",
    "end run",
]


def notification_command(title: str, message: str, platform: str = sys.platform) -> list[str] | None:
    """Argument list for the platform's notifier; text is passed as arguments, never through a shell."""
    if platform == "darwin" and shutil.which("osascript"):
        script = [arg for line in _APPLESCRIPT for arg in ("-e", line)]
        return ["osascript", *script, title, message]
    if platform.startswith("linux") and shutil.which("notify-send"):
        return ["notify-send", title, message]
    return None


def notify(title: str, message: str) -> bool:
    """Shows a desktop notification; returns False when none is available (the caller still prints)."""
    command = notification_command(title, message)
    if command is None:
        return False
    try:
        subprocess.run(command, check=False, timeout=5, capture_output=True)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return True
