from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from functools import cache
from importlib import resources

import yaml


@dataclass(frozen=True)
class RawFormat:
    system: str
    user: str
    assistant: str


@dataclass(frozen=True)
class Template:
    id: str
    match: tuple[str, ...]
    think_start: str
    think_end: str
    starts_in_thinking: bool
    prefill_supported: bool
    raw: RawFormat | None
    stop_thinking_prefix: str
    stop_injection_text: str
    probe_answer_prefix: str
    probe_max_tokens: int
    resume_content_prefix: str
    extract_boxed: bool
    max_answer_chars: int
    answer_patterns: tuple[re.Pattern[str], ...]
    revision_patterns: tuple[re.Pattern[str], ...]

    @property
    def can_intervene(self) -> bool:
        """Answer now and probing need either a prefill-capable chat template or a verified raw prompt format."""
        return self.prefill_supported or self.raw is not None


def _build(raw: dict) -> Template:
    thinking = raw["thinking"]
    if not raw["probe"]["answer_prefix"].endswith("\\boxed{"):
        raise ValueError(f"template {raw['id']}: probe.answer_prefix must end with an open \\boxed{{")
    return Template(
        id=raw["id"],
        match=tuple(raw.get("match", [])),
        think_start=thinking["start"],
        think_end=thinking["end"],
        starts_in_thinking=bool(thinking.get("starts_in_thinking", False)),
        prefill_supported=bool(raw["prefill_supported"]),
        raw=RawFormat(**raw["raw_prompt"]) if raw.get("raw_prompt") else None,
        stop_thinking_prefix=raw["stop_injection"].get("thinking_prefix", ""),
        stop_injection_text=raw["stop_injection"]["text"],
        probe_answer_prefix=raw["probe"]["answer_prefix"],
        probe_max_tokens=int(raw["probe"]["max_tokens"]),
        resume_content_prefix=raw["resume"]["content_prefix"],
        extract_boxed=bool(raw.get("extract_boxed", False)),
        max_answer_chars=int(raw.get("max_answer_chars", 40)),
        answer_patterns=tuple(re.compile(p) for p in raw.get("tentative_answer_patterns", [])),
        revision_patterns=tuple(re.compile(p) for p in raw.get("revision_patterns", [])),
    )


@cache
def _load_all() -> dict[str, Template]:
    raws: dict[str, dict] = {}
    for entry in resources.files("overthink_guard.templates").joinpath("models").iterdir():
        if entry.name.endswith(".yaml"):
            raw = yaml.safe_load(entry.read_text(encoding="utf-8"))
            raws[raw["id"]] = raw

    def resolve(raw: dict) -> dict:
        parent_id = raw.get("extends")
        if not parent_id:
            return raw
        return {**resolve(raws[parent_id]), **{k: v for k, v in raw.items() if k != "extends"}}

    return {tid: _build(resolve(raw)) for tid, raw in raws.items()}


def get_template(template_id: str) -> Template:
    templates = _load_all()
    if template_id not in templates:
        available = ", ".join(sorted(templates))
        raise KeyError(f"unknown template {template_id!r}; available: {available}")
    return templates[template_id]


def template_for_model(model: str) -> Template:
    name = model.lower()
    for template in _load_all().values():
        if any(fnmatch.fnmatch(name, pattern) for pattern in template.match):
            return template
    return get_template("generic")


def template_ids() -> list[str]:
    return sorted(_load_all())
