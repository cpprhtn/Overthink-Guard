"""Replay public DeepSeek-R1 math traces (open-r1/OpenR1-Math-220k) through the Tier 0 judge.

Usage: python bench/r1_replay.py [--rows 500] [--cache .bench-cache]
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from dataclasses import replace
from pathlib import Path

from overthink_guard.analysis import JudgeConfig, replay
from overthink_guard.analysis.signals import estimate_tokens, extract_boxed
from overthink_guard.templates import get_template

ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=open-r1/OpenR1-Math-220k"
    "&config=default&split=train&offset={offset}&length=100"
)
LOOSE_NUMERIC = re.compile(
    r"(?i)\b(?:so|thus|therefore|hence|total|we get|gives|equals)\b[^.\n]*?(?P<ans>-?\d+(?:\.\d+)?)\s*(?:\.(?:\s|$)|\n|$)"
)


def load_rows(total: int, cache: Path) -> list[dict]:
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset in range(0, total, 100):
        path = cache / f"openr1_math_{offset}.json"
        if not path.exists():
            with urllib.request.urlopen(ROWS_URL.format(offset=offset), timeout=60) as resp:
                path.write_bytes(resp.read())
        rows.extend(r["row"] for r in json.loads(path.read_text(encoding="utf-8"))["rows"])
    return rows


def traces_of(rows: list[dict]) -> list[tuple[str, bool]]:
    return [
        (gen, bool(ok))
        for row in rows
        for gen, done, ok in zip(row["generations"], row["is_reasoning_complete"], row["correctness_math_verify"])
        if done and "</think>" in gen
    ]


def first_mention_bound(rows: list[dict]) -> tuple[int, float, list[float]]:
    """Where a distinctive numeric final answer first appears anywhere in thinking (extractor-independent)."""
    positions = []
    for row in rows:
        problem_numbers = set(re.findall(r"\d+(?:\.\d+)?", row["problem"]))
        for gen, done, ok in zip(row["generations"], row["is_reasoning_complete"], row["correctness_math_verify"]):
            if not (done and ok and "</think>" in gen):
                continue
            think, answer = gen.split("</think>", 1)
            boxed = extract_boxed(answer)
            value = boxed[-1][1].strip() if boxed else ""
            if len(value) < 2 or not re.fullmatch(r"-?\d+(?:\.\d+)?", value) or value in problem_numbers:
                continue
            match = re.search(rf"(?<![\d.]){re.escape(value)}(?!\d)", think)
            if match:
                positions.append(estimate_tokens(think[: match.start()]) / estimate_tokens(think))
    positions.sort()
    return len(positions), sum(1 - p for p in positions) / len(positions), positions


def evaluate(traces, template, config) -> dict:
    think = saved = stopped = agree = disagree = risk = 0
    for gen, correct in traces:
        report = replay(gen, template, config)
        think += report.thinking_tokens
        if report.decision:
            stopped += 1
            saved += report.saved_tokens
            if report.answer_match:
                agree += 1
            elif report.answer_match is False:
                disagree += 1
                risk += correct
    n = len(traces)
    return {"stopped": stopped / n, "saved": saved / think, "agree": agree, "disagree": disagree, "risk": risk / n}


def extractor_bound(traces, template) -> float:
    """Savings if we stopped exactly when the extractor first saw the eventual final answer (hindsight)."""
    total = saved = 0
    for gen, _ in traces:
        report = replay(gen, template, JudgeConfig())
        if report.final_answer is None:
            continue
        total += report.thinking_tokens
        hit = next((s for s in report.segments if s.answer == report.final_answer), None)
        saved += report.thinking_tokens - hit.end_tokens if hit else 0
    return saved / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--cache", type=Path, default=Path(".bench-cache"))
    args = parser.parse_args()

    rows = load_rows(args.rows, args.cache)
    traces = traces_of(rows)
    print(f"problems {len(rows)}, complete traces {len(traces)}, final correct {sum(c for _, c in traces)}")

    count, bound, positions = first_mention_bound(rows)
    quartiles = ", ".join(f"p{q}={positions[q * (count - 1) // 100]:.0%}" for q in (25, 50, 75))
    print(f"first mention of final value in thinking ({count} traces): {quartiles}; loose bound {bound:.1%}")

    generic = get_template("generic")
    loose = replace(generic, answer_patterns=generic.answer_patterns + (LOOSE_NUMERIC,))
    print(f"extractor hindsight bound: generic {extractor_bound(traces, generic):.1%}")

    base = JudgeConfig()
    runs = [
        ("generic", generic, "default", base),
        ("generic", generic, "k=2", replace(base, converge_k=2)),
        ("generic", generic, "k=2 rep off", replace(base, converge_k=2, repetition_threshold=1.0)),
        ("loose", loose, "default", base),
        ("loose", loose, "k=3 rep off", replace(base, repetition_threshold=1.0)),
        ("loose", loose, "k=4 rep off", replace(base, converge_k=4, repetition_threshold=1.0)),
    ]
    print(f"\n{'extractor':9} {'config':12} {'stopped':>8} {'saved':>7} {'agree':>6} {'disagree':>9} {'risk':>6}")
    for tname, template, cname, config in runs:
        r = evaluate(traces, template, config)
        print(f"{tname:9} {cname:12} {r['stopped']:8.0%} {r['saved']:7.1%} {r['agree']:6} {r['disagree']:9} {r['risk']:6.1%}")


if __name__ == "__main__":
    main()
