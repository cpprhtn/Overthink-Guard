from pathlib import Path

import pytest

from overthink_guard.cli import build_parser, resolve_settings
from overthink_guard.config import ConfigError, Settings, load_settings

FULL = """
server:
  host: 127.0.0.1
  port: 9000
local:
  backend_url: http://gpu-box:11434
  auto_stop:
    converge_k: 4
    repetition_threshold: 1
  signals:
    active_probe: true
    probe_interval_tokens: 256
    probe_converge_k: 5
privacy:
  stats_file: ~/otg/shadow.jsonl
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_default_file_means_defaults(tmp_path):
    assert load_settings(tmp_path / "absent.yaml") == Settings()


def test_full_file_is_applied(tmp_path):
    s = load_settings(write(tmp_path, FULL))
    assert (s.host, s.port, s.backend_url) == ("127.0.0.1", 9000, "http://gpu-box:11434")
    assert (s.judge.converge_k, s.judge.repetition_threshold, s.judge.min_thinking_tokens) == (4, 1, 300)
    assert (s.probe, s.probe_interval, s.probe_converge_k) == (True, 256, 5)
    assert s.stats_file == Path("~/otg/shadow.jsonl").expanduser()


def test_null_stats_file_keeps_stats_in_memory(tmp_path):
    assert load_settings(write(tmp_path, "privacy:\n  stats_file: null\n")).stats_file is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("server:\n  prot: 1\n", "server.prot: unknown setting"),
        ("hooks:\n  on_stop_suggested: x\n", "hooks: unknown setting"),
        ("server:\n  port: '8484'\n", "server.port: invalid value"),
        ("server:\n  port: true\n", "server.port: invalid value"),
        ("local:\n  signals:\n    active_probe: 1\n", "local.signals.active_probe: invalid value"),
        ("- just\n- a list\n", "config: expected a mapping"),
    ],
)
def test_bad_settings_are_rejected_not_ignored(tmp_path, text, message):
    with pytest.raises(ConfigError, match=message):
        load_settings(write(tmp_path, text))


def test_invalid_judge_value_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="converge_k"):
        load_settings(write(tmp_path, "local:\n  auto_stop:\n    converge_k: 1\n"))


def start_args(*argv: str):
    return build_parser().parse_args(["start", *argv])


def test_flags_override_the_file(tmp_path):
    path = write(tmp_path, FULL)
    s = resolve_settings(start_args("--config", str(path), "--port", "7000", "--no-probe", "--probe-k", "6"))
    assert (s.port, s.probe, s.probe_converge_k, s.probe_interval) == (7000, False, 6, 256)


def test_no_stats_flag_wins(tmp_path):
    s = resolve_settings(start_args("--config", str(write(tmp_path, FULL)), "--no-stats"))
    assert s.stats_file is None


def test_explicit_missing_config_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        resolve_settings(start_args("--config", str(tmp_path / "nope.yaml")))
