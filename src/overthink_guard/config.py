from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from overthink_guard.analysis import JudgeConfig
from overthink_guard.analysis.prober import DEFAULT_PROBE_K, DEFAULT_PROBE_MIN_TOKENS
from overthink_guard.storage import default_stats_path


def default_config_path() -> Path:
    if sys.platform == "win32":
        return (
            Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "overthink-guard" / "config.yaml"
        )
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "overthink-guard" / "config.yaml"


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8484
    backend_url: str = "http://localhost:11434"
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    mode: str = "shadow"
    probe: bool = False
    probe_interval: int = 400
    probe_min_tokens: int = DEFAULT_PROBE_MIN_TOKENS
    probe_converge_k: int = DEFAULT_PROBE_K
    stats_file: Path | None = field(default_factory=default_stats_path)


_SCHEMA: dict = {
    "server": {"host": str, "port": int},
    "local": {
        "backend_url": str,
        "mode": str,
        "auto_stop": {
            "converge_k": int,
            "min_thinking_tokens": int,
            "revision_cooldown_tokens": int,
            "repetition_threshold": (int, float),
        },
        "signals": {
            "active_probe": bool,
            "probe_interval_tokens": int,
            "probe_min_tokens": int,
            "probe_converge_k": int,
        },
    },
    "privacy": {"stats_file": (str, type(None))},
}


MODES = ("shadow", "auto")


class ConfigError(ValueError):
    pass


def _check(data: object, schema: dict, where: str) -> None:
    if not isinstance(data, dict):
        raise ConfigError(f"{where or 'config'}: expected a mapping")
    for key, value in data.items():
        path = f"{where}.{key}" if where else key
        if key not in schema:
            raise ConfigError(f"{path}: unknown setting (supported: {', '.join(schema)})")
        expected = schema[key]
        if isinstance(expected, dict):
            _check(value, expected, path)
        elif not isinstance(value, expected) or (isinstance(value, bool) and expected is not bool):
            raise ConfigError(f"{path}: invalid value {value!r}")


def load_settings(path: Path | None) -> Settings:
    """Settings from a YAML file; a missing default file just means defaults."""
    settings = Settings()
    if path is None or not path.exists():
        return settings
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _check(data, _SCHEMA, "")

    server = data.get("server", {})
    local = data.get("local", {})
    signals = local.get("signals", {})
    settings.host = server.get("host", settings.host)
    settings.port = server.get("port", settings.port)
    settings.backend_url = local.get("backend_url", settings.backend_url)
    settings.mode = local.get("mode", settings.mode)
    if settings.mode not in MODES:
        raise ConfigError(f"local.mode: must be one of {', '.join(MODES)}")
    settings.judge = replace(settings.judge, **local.get("auto_stop", {}))
    settings.probe = signals.get("active_probe", settings.probe)
    settings.probe_interval = signals.get("probe_interval_tokens", settings.probe_interval)
    settings.probe_min_tokens = signals.get("probe_min_tokens", settings.probe_min_tokens)
    settings.probe_converge_k = signals.get("probe_converge_k", settings.probe_converge_k)
    privacy = data.get("privacy", {})
    if "stats_file" in privacy:
        value = privacy["stats_file"]
        settings.stats_file = None if value is None else Path(value).expanduser()
    return settings
