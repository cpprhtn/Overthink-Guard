"""Confusion matrix of the probe stop rule on every recorded math probe trail in bench/data (offline).

Per run, with the stopped answer taken to be the recorded probe answer at the stop point (a proxy: Auto generates a
fresh answer there) and runs that hit the 12k cap counted as producing no answer:
  TP        stopped a run that finished correctly, and the stopped answer is also correct (tokens saved)
  FP        stopped a run that finished correctly, and the stopped answer is wrong (a correct answer lost)
  TP_wrong  stopped a run that finished wrong; stopped answer correct (gain) or not (no loss)
  rescue    stopped a runaway run (hit the cap) with a correct answer; runaway_wrong: with a wrong one
  TN        left a completed run alone;  FN_runaway: left a runaway run alone
"eligible" counts the correct completed runs long enough for the rule to fire at all: only those can become FP.

Usage: python bench/spikes/probe_rule_matrix.py [--k 4] [--min-tokens 6000] [--sweep]
"""

import argparse
import json
from pathlib import Path

from overthink_guard.analysis.prober import (
    DEFAULT_OPEN_BUDGET_TOKENS,
    DEFAULT_PROBE_K,
    DEFAULT_PROBE_MIN_TOKENS,
    Probe,
    ProbeTracker,
)
from overthink_guard.analysis.signals import normalize_answer

DATA = Path(__file__).parents[1] / "data"
GROUPS = {
    "tuning (samples 1-4)": [
        "shadow_live_2026-09-29.json",
        "shadow_validate_qwen3_2026-09-29.json",
        "shadow_third_qwen3_2026-09-30.json",
        "shadow_r1_2026-09-30.json",
    ],
    "pre-registered C": ["shadow_fifth_qwen3_2026-09-30.json", "shadow_sixth_r1_2026-09-30.json"],
    "pre-registered D": ["math_d_qwen3_2026-09-30.json", "math_d_r1_2026-09-30.json"],
}
CELLS = ["TP", "FP", "TP_wrong", "rescue", "runaway_wrong", "TN", "FN_runaway"]


def same(a: str | None, b: str) -> bool:
    return a is not None and normalize_answer(a) == normalize_answer(b)


def load(name: str | Path) -> list[dict]:
    path = Path(name) if Path(name).exists() else DATA / name
    data = json.loads(path.read_text(encoding="utf-8"))
    name = path.name
    runs = []
    for i, p in enumerate(data["problems"]):
        truncated = bool(p.get("truncated"))
        if i in data.get("ungradable", []) and not truncated:
            continue
        probes = [(q[0], q[1], q[2] if len(q) > 2 else True) for q in p["probes"]]
        correct = not truncated and (same(p["final"], p["gold"]) if "final" in p else bool(p["final_correct"]))
        runs.append(
            {
                "id": f"{name.split('_2026')[0]}#{i}",
                "gold": p["gold"],
                "correct": correct,
                "truncated": truncated,
                "tokens": p["thinking_tokens"],
                "probes": probes,
            }
        )
    return runs


def plain_budget(run: dict, tokens: int) -> tuple[int, str | None] | None:
    """Stop at the first probe at or past `tokens`, whatever it says (no convergence check)."""
    return next(((at, answer) for at, answer, _ in run["probes"] if at >= tokens), None)


def stop_point(run: dict, k: int, min_tokens: int) -> tuple[int, str | None] | None:
    tracker = ProbeTracker(k, DEFAULT_OPEN_BUDGET_TOKENS)
    for at, answer, grounded in run["probes"]:
        if at >= min_tokens and tracker.add(Probe(at, answer, 0.0, grounded)):
            return at, answer
    return None


def matrix(runs: list[dict], k: int, min_tokens: int, rule=None) -> dict:
    """rule(run) -> (stop tokens, answer) or None; defaults to the probe rule with k and min_tokens."""
    rule = rule or (lambda run: stop_point(run, k, min_tokens))
    cells = dict.fromkeys(CELLS, 0)
    saved = {"completed": 0, "runaway": 0}
    fps, stops = [], {}
    for run in runs:
        stop = rule(run)
        stops[run["id"]] = stop
        if stop is None:
            cells["FN_runaway" if run["truncated"] else "TN"] += 1
            continue
        at, answer = stop
        ok = same(answer, run["gold"])
        if run["truncated"]:
            cells["rescue" if ok else "runaway_wrong"] += 1
        elif run["correct"]:
            cells["TP" if ok else "FP"] += 1
            if not ok:
                fps.append(f"{run['id']} stopped at {at} with {answer}, gold {run['gold']}")
        else:
            cells["TP_wrong"] += 1
        saved["runaway" if run["truncated"] else "completed"] += run["tokens"] - at
    completed = [r for r in runs if not r["truncated"]]
    eligible = sum(r["correct"] and r["tokens"] >= min_tokens + (k - 1) * 400 for r in completed)
    total = sum(r["tokens"] for r in runs)
    return {
        "n": len(runs),
        "cells": cells,
        "eligible": eligible,
        "fps": fps,
        "stops": stops,
        "saved_all": sum(saved.values()) / max(1, total),
        "saved_completed": saved["completed"] / max(1, sum(r["tokens"] for r in completed)),
    }


def show(label: str, m: dict) -> None:
    cells = " ".join(f"{c}={m['cells'][c]}" for c in CELLS if m["cells"][c])
    bound = f"{3 / m['eligible']:.0%}" if m["eligible"] else "n/a"
    print(
        f"{label:22} n={m['n']:3} {cells} | FP {m['cells']['FP']}/{m['eligible']} eligible "
        f"(95% upper ~{bound}) | saved {m['saved_all']:.0%} overall, {m['saved_completed']:.0%} on completed runs"
    )
    for fp in m["fps"]:
        print(f"    FP: {fp}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=DEFAULT_PROBE_K)
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_PROBE_MIN_TOKENS)
    parser.add_argument("--sweep", action="store_true", help="also sweep k and min tokens over all data (in-sample)")
    parser.add_argument("--files", nargs="+", type=Path, help="analyse these files instead of the recorded samples")
    parser.add_argument(
        "--overnight", action="store_true", help="also judge B1-B3 (docs/validation/local-overnight.md)"
    )
    args = parser.parse_args()
    if args.files:
        runs = [r for f in args.files for r in load(f)]
        show("files", matrix(runs, args.k, args.min_tokens))
        if args.overnight:
            overnight(runs, args.k, args.min_tokens)
        return
    groups = {g: [r for f in files for r in load(f)] for g, files in GROUPS.items()}
    groups["held-out (C + D)"] = groups["pre-registered C"] + groups["pre-registered D"]
    groups["all"] = [r for g in GROUPS for r in groups[g]]
    print(f"rule: k={args.k}, probes from {args.min_tokens} thinking tokens")
    for label, runs in groups.items():
        show(label, matrix(runs, args.k, args.min_tokens))
    if args.sweep:
        print("sweep over all data (in-sample: do not pick a default from this)")
        for k in (2, 3, 4, 5):
            for m in (3000, 4000, 5000, 6000, 7000, 8000):
                show(f"k={k} min={m}", matrix(groups["all"], k, m))


def overnight(runs: list[dict], k: int, min_tokens: int) -> None:
    """The pre-registered judgements B1-B3 (docs/validation/local-overnight.md)."""
    fire = min_tokens + (k - 1) * 400
    d = matrix(runs, k, min_tokens)
    fp, eligible = d["cells"]["FP"], d["eligible"]
    verdict = ("PASS" if fp <= 1 else "FAIL") if eligible >= 100 else "not enough eligible runs to judge"
    print(f"B1 rule D: FP {fp} of {eligible} eligible correct runs -> {verdict}")
    plain = matrix(runs, k, min_tokens, rule=lambda run: plain_budget(run, fire))
    keep = d["cells"]["FP"] < plain["cells"]["FP"]
    print(
        f"B2 plain {fire}-token budget: FP {plain['cells']['FP']} vs rule D {fp}, saved {plain['saved_all']:.0%} vs "
        f"{d['saved_all']:.0%} -> {'keep convergence' if keep else 'convergence adds nothing measurable here'}"
    )
    show(f"plain budget {fire}", plain)
    long_runs = [r for r in runs if r["tokens"] > 12000 and d["stops"][r["id"]] is not None]
    hidden = [r for r in long_runs if r["correct"] and not same(d["stops"][r["id"]][1], r["gold"])]
    rate = len(hidden) / len(long_runs) if long_runs else 0.0
    b3 = ("PASS" if rate <= 0.05 else "FAIL") if len(long_runs) >= 20 else "not enough runs to judge"
    print(
        f"B3 runs past the old 12k cap that rule D stopped: {len(long_runs)}; finished correct but stopped wrong: "
        f"{len(hidden)} ({rate:.0%}) -> {b3}"
    )


if __name__ == "__main__":
    main()
