import json
import sys
from datetime import datetime, timezone

import pytest

from overthink_guard.notify import notification_command, run_hook
from overthink_guard.observe import ClaudeCodeObserver, session_states, starts_model_turn, usage_report

T0 = 1_790_000_000.0


def ts(seconds: float) -> str:
    return datetime.fromtimestamp(T0 + seconds, timezone.utc).isoformat().replace("+00:00", "Z")


def user(seconds, content="fix the bug", **extra):
    return {"type": "user", "timestamp": ts(seconds), "sessionId": "s1", "message": {"content": content}, **extra}


def tool_result(seconds):
    return user(seconds, [{"type": "tool_result", "content": "ok"}])


def assistant(seconds, msg_id="m1", thinking=100, output=150, effort="high"):
    usage = {"output_tokens": output, "output_tokens_details": {"thinking_tokens": thinking}}
    return {
        "type": "assistant",
        "timestamp": ts(seconds),
        "sessionId": "s1",
        "effort": effort,
        "message": {"id": msg_id, "content": [{"type": "text", "text": "hi"}], "usage": usage},
    }


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now


@pytest.fixture
def log(tmp_path):
    path = tmp_path / "-Users-me-proj" / "s1.jsonl"
    path.parent.mkdir()
    path.write_text("")

    def append(*records, partial=""):
        with path.open("a") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
            fh.write(partial)

    return tmp_path, append


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (user(0), True),
        (tool_result(0), True),
        (
            user(0, [{"type": "text", "text": "<ide_opened_file>x</ide_opened_file>"}, {"type": "text", "text": "hi"}]),
            True,
        ),
        (user(0, "<command-name>/model</command-name>"), False),
        (user(0, "<local-command-stdout>ok</local-command-stdout>"), False),
        (user(0, [{"type": "text", "text": "[Request interrupted by user]"}]), False),
        (user(0, "<bash-input>ls</bash-input>"), False),
        (user(0, isMeta=True), False),
        (user(0, isSidechain=True), False),
        (assistant(0), False),
    ],
)
def test_which_user_records_start_a_model_turn(record, expected):
    assert starts_model_turn(record) is expected


def test_alerts_once_per_silent_turn_and_resets_on_output(log):
    root, append = log
    clock, alerts = Clock(), []
    observer = ClaudeCodeObserver(root, after_seconds=60, on_alert=alerts.append, clock=clock)
    observer.poll()
    append(user(0))
    clock.now = T0 + 30
    observer.poll()
    assert alerts == []
    clock.now = T0 + 61
    observer.poll()
    observer.poll()
    assert len(alerts) == 1
    event = alerts[0]
    assert event["event"] == "stop_suggested" and event["source"]["tool"] == "claude_code"
    assert event["signal"]["reasons"] == ["long_silence"] and event["stats"]["thinking_elapsed_s"] == 61
    assert "fix the bug" not in json.dumps(event)

    append(assistant(62), tool_result(63))
    clock.now = T0 + 63 + 59
    observer.poll()
    assert len(alerts) == 1
    clock.now = T0 + 63 + 61
    observer.poll()
    assert len(alerts) == 2


def test_no_alert_for_local_commands_or_turns_already_silent_at_startup(log):
    root, append = log
    clock, alerts = Clock(), []
    append(user(0))
    clock.now = T0 + 300
    observer = ClaudeCodeObserver(root, after_seconds=60, on_alert=alerts.append, clock=clock)
    observer.poll()
    append(assistant(301), user(302, "<command-name>/clear</command-name>"))
    clock.now = T0 + 500
    observer.poll()
    assert alerts == []


def test_partial_lines_are_completed_on_the_next_poll(log):
    root, append = log
    clock, alerts = Clock(), []
    observer = ClaudeCodeObserver(root, after_seconds=10, on_alert=alerts.append, clock=clock)
    observer.poll()
    line = json.dumps(user(0))
    append(partial=line[:20])
    clock.now = T0 + 5
    observer.poll()
    append(partial=line[20:] + "\n")
    clock.now = T0 + 11
    observer.poll()
    assert len(alerts) == 1


def test_report_dedupes_block_records_and_splits_by_effort(log):
    root, append = log
    append(
        user(0),
        assistant(4, "m1", thinking=300, output=400),
        assistant(4.5, "m1", thinking=300, output=400),
        tool_result(5),
        assistant(70, "m2", thinking=900, output=1000, effort="xhigh"),
    )
    report = usage_report(root)
    assert (report["responses"], report["thinking_tokens"], report["output_tokens"]) == (2, 1200, 1400)
    assert report["thinking_share"] == round(1200 / 1400, 3)
    assert set(report["by_effort"]) == {"high", "xhigh"}
    assert report["silence_seconds"]["over_60s"] == 1


def test_desktop_notification_passes_text_as_arguments(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    cmd = notification_command('say "hi"; rm -rf ~', "body", platform="darwin")
    assert cmd[0] == "osascript" and cmd[-2:] == ['say "hi"; rm -rf ~', "body"]
    assert notification_command("t", "b", platform="linux") == ["notify-send", "t", "b"]
    assert notification_command("t", "b", platform="win32") is None


def test_hook_receives_the_event_on_stdin(tmp_path):
    out = tmp_path / "event.json"
    script = tmp_path / "hook.py"
    script.write_text(f"import sys; open({str(out)!r}, 'w').write(sys.stdin.read())")
    assert run_hook(f'"{sys.executable}" "{script}"', {"schema_version": 1, "event": "stop_suggested"}) == 0
    assert json.loads(out.read_text())["event"] == "stop_suggested"
    assert run_hook("", {}) is None
    assert run_hook("definitely-not-a-command-xyz", {}) is None


def write_state(sessions_dir, pid, status, kind="interactive", session_id="s1"):
    sessions_dir.mkdir(exist_ok=True)
    (sessions_dir / f"{pid}.json").write_text(
        json.dumps({"pid": pid, "sessionId": session_id, "kind": kind, "status": status})
    )
    (sessions_dir / f"{pid}.deadbeef.key").write_text("not json and never read")


@pytest.mark.parametrize(
    ("status", "alive", "alerts", "run_mode"),
    [("busy", True, 1, "interactive"), ("idle", True, 0, None), ("busy", False, 0, None), (None, True, 1, "headless")],
)
def test_interactive_session_state_gates_alerts(log, tmp_path, status, alive, alerts, run_mode):
    root, append = log
    sessions = tmp_path / "sessions"
    write_state(sessions, 4242, status)
    clock, got = Clock(), []
    observer = ClaudeCodeObserver(root, 60, got.append, clock=clock, sessions_dir=sessions, pid_alive=lambda pid: alive)
    observer.poll()
    append(user(0))
    clock.now = T0 + 61
    observer.poll()
    assert len(got) == alerts
    if got:
        assert got[0]["source"]["run_mode"] == run_mode


def test_sessions_without_a_state_file_are_treated_as_headless(log, tmp_path):
    root, append = log
    sessions = tmp_path / "sessions"
    write_state(sessions, 4242, "idle", session_id="someone-else")
    clock, got = Clock(), []
    observer = ClaudeCodeObserver(root, 60, got.append, clock=clock, sessions_dir=sessions, pid_alive=lambda pid: True)
    observer.poll()
    append(user(0))
    clock.now = T0 + 61
    observer.poll()
    assert [e["source"]["run_mode"] for e in got] == ["unknown"]


def test_a_state_file_that_disappears_means_the_run_ended(log, tmp_path):
    root, append = log
    sessions = tmp_path / "sessions"
    write_state(sessions, 4242, None)
    clock, got = Clock(), []
    observer = ClaudeCodeObserver(root, 60, got.append, clock=clock, sessions_dir=sessions, pid_alive=lambda pid: True)
    observer.poll()
    append(tool_result(0))
    clock.now = T0 + 10
    observer.poll()
    (sessions / "4242.json").unlink()
    clock.now = T0 + 61
    observer.poll()
    assert got == []


def test_session_states_reads_only_json_metadata(tmp_path):
    write_state(tmp_path, 1, "busy")
    (tmp_path / "2.json").write_text("{broken")
    assert session_states(tmp_path, pid_alive=lambda pid: True) == {"s1": {"status": "busy", "alive": True}}
