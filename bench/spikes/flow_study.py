"""Can the flow of the thinking tell wasteful re-checking from checking that is about to fix the answer?

Pre-registered in docs/validation/flow-signals.md. Offline only: recorded thinking text and probe answers.
    python bench/spikes/flow_study.py            # pick rules on the development runs only
    python bench/spikes/flow_study.py --test     # the one-time evaluation on held-out (and overnight) runs
"""

import argparse
import itertools
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from probe_rule_matrix import DATA, load, matrix, same, stop_point

from overthink_guard.analysis.prober import DEFAULT_PROBE_K, DEFAULT_PROBE_MIN_TOKENS
from overthink_guard.analysis.signals import normalize_answer

DEV = ["shadow_third_qwen3_2026-09-30.json", "shadow_r1_2026-09-30.json"]
TEST = [
    "shadow_fifth_qwen3_2026-09-30.json",
    "shadow_sixth_r1_2026-09-30.json",
    "math_d_qwen3_2026-09-30.json",
    "math_d_r1_2026-09-30.json",
]
WINDOW = 800
NUMBER = re.compile(r"\d+(?:\.\d+)?")
WORD = re.compile(r"\w+")
DOUBT = re.compile(
    r"\b(wait|hmm|mistake|wrong|incorrect|actually|oops|contradict\w*|re-?examine|but no)\b", re.IGNORECASE
)
VERIFY = re.compile(r"\b(check\w*|verif\w*|double-check\w*|confirm\w*|sanity)\b", re.IGNORECASE)
CUE = re.compile(r"(answer is|therefore|thus|hence|\bso\b|=|boxed)", re.IGNORECASE)
NEW_WORK = re.compile(
    r"\b(let'?s (?:find|compute|calculate|consider|try|solve|set up|denote)|consider the case|"
    r"another (?:approach|way|case|method)|alternatively)\b",
    re.IGNORECASE,
)
GRID_V2 = {"s": (1, 2, 3), "m": (0, 400, 800, 1200), "d": (0, 1, 2), "w": (0, 1, 2)}
GRID = {
    "s": (2, 3),
    "n": (0, 0.1, 0.2, 0.3, 1),
    "d": (0, 1, 2, math.inf),
    "r": (0, 0.2, 0.4),
    "t0": (400, 1200, 2000),
}


def ngrams(words: list[str], n: int = 6) -> list[tuple[str, ...]]:
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def features(run: dict) -> list[dict]:
    """One row per probe: the flow of the thinking up to that probe."""
    segments = run["segments"]
    rows, streak, last = [], 0, None
    for at, answer, grounded in run["probes"]:
        key = normalize_answer(answer) if answer is not None and grounded else None
        streak = streak + 1 if key is not None and key == last else (1 if key is not None else 0)
        last = key
        window = " ".join(t for s, e, t in segments if at - WINDOW < e <= at)
        before = " ".join(t for s, e, t in segments if e <= at - WINDOW)
        seen = set(NUMBER.findall(before))
        numbers = NUMBER.findall(window)
        earlier = set(ngrams(WORD.findall(before.lower())))
        grams = ngrams(WORD.findall(window.lower()))
        rows.append(
            {
                "at": at,
                "answer": answer,
                "streak": streak,
                "novelty": sum(x not in seen for x in numbers) / len(numbers) if numbers else 0.0,
                "repeat": sum(g in earlier for g in grams) / len(grams) if grams else 0.0,
                "doubt": len(DOUBT.findall(window)),
                "verify": len(VERIFY.findall(window)),
                "restated": answer is not None and answer in window,
            }
        )
    return rows


def first_conclusion(text: str, ends: list[tuple[int, int]], answer: str) -> tuple[int, int] | None:
    """(token, char) where the thinking first states `answer` as a conclusion, or None."""
    pattern = re.compile(r"(?<![0-9A-Za-z.])" + re.escape(answer) + r"(?![0-9A-Za-z]|\.\d)")
    for match in pattern.finditer(text):
        if CUE.search(text[max(0, match.start() - 60) : match.start()]):
            token = next(tok for char_end, tok in ends if char_end >= match.end())
            return token, match.end()
    return None


def features_v2(run: dict) -> list[dict]:
    """Per probe: has the model itself concluded the probed answer, and what has it done since?"""
    text, ends = "", []
    for _, end, piece in run["segments"]:
        text += piece
        ends.append((len(text), end))
    char_at = lambda t: max((c for c, tok in ends if tok <= t), default=0)  # noqa: E731
    cache: dict[str, tuple[int, int] | None] = {}
    rows = []
    for f in run["flow"]:
        answer = f["answer"]
        if answer is None or not answer.strip():
            rows.append({**f, "concluded": False})
            continue
        if answer not in cache:
            cache[answer] = first_conclusion(text, ends, answer.strip())
        found = cache[answer]
        if found is None or found[0] > f["at"]:
            rows.append({**f, "concluded": False})
            continue
        after = text[found[1] : char_at(f["at"])]
        rows.append(
            {
                **f,
                "concluded": True,
                "since": f["at"] - found[0],
                "new_work": len(NEW_WORK.findall(after)),
                "doubt_since": len(DOUBT.findall(after)),
            }
        )
    return rows


def v2_rule(p: dict):
    def rule(run: dict):
        for f in run["flow2"]:
            if (
                f["concluded"]
                and f["streak"] >= p["s"]
                and f["since"] >= p["m"]
                and f["doubt_since"] <= p["d"]
                and f["new_work"] <= p["w"]
            ):
                return f["at"], f["answer"]
        return None

    return rule


def flow_rule(p: dict):
    def rule(run: dict):
        for f in run["flow"]:
            if (
                f["at"] >= p["t0"]
                and f["streak"] >= p["s"]
                and f["novelty"] <= p["n"]
                and f["doubt"] <= p["d"]
                and f["repeat"] >= p["r"]
            ):
                return f["at"], f["answer"]
        return None

    return rule


def with_rule_d(rule):
    """The flow rule, with rule D kept as the runaway backstop: whichever stops first."""

    def combined(run: dict):
        stops = [x for x in (rule(run), stop_point(run, DEFAULT_PROBE_K, DEFAULT_PROBE_MIN_TOKENS)) if x]
        return min(stops) if stops else None

    return combined


def oracle_capture(runs: list[dict], rule) -> float:
    """Share of the thinking done after the answer had settled on gold (correct runs) that the rule cuts."""
    total = cut = 0
    for run in runs:
        if not run["correct"]:
            continue
        last_bad = max((at for at, a, _ in run["probes"] if not same(a, run["gold"])), default=0)
        settled = next((at for at, _, _ in run["probes"] if at > last_bad), None)
        if settled is None:
            continue
        total += run["tokens"] - settled
        stop = rule(run)
        if stop and same(stop[1], run["gold"]):
            cut += run["tokens"] - stop[0]
    return cut / total if total else 0.0


def load_runs(files: list) -> list[dict]:
    runs = []
    for f in files:
        raw = json.loads((Path(f) if Path(f).exists() else DATA / f).read_text(encoding="utf-8"))["problems"]
        by_index = {}
        for i, p in enumerate(raw):
            by_index[i] = p.get("segments") or []
        for run in load(f):
            run["segments"] = by_index[int(run["id"].rsplit("#", 1)[1])]
            run["flow"] = features(run)
            run["flow2"] = features_v2(run)
            runs.append(run)
    return runs


def summary(label: str, runs: list[dict], rule) -> dict:
    m = matrix(runs, DEFAULT_PROBE_K, DEFAULT_PROBE_MIN_TOKENS, rule=rule)
    c = m["cells"]
    correct = sum(r["correct"] for r in runs)
    print(
        f"  {label:34} FP {c['FP']}/{correct} correct runs | TP {c['TP']} | saved on completed runs "
        f"{m['saved_completed']:.0%}, overall {m['saved_all']:.0%} | runaway missed {c['FN_runaway']} | "
        f"oracle captured {oracle_capture(runs, rule):.0%}"
    )
    for fp in m["fps"]:
        print(f"      FP: {fp}")
    return m


def pick(runs: list[dict], grid: dict, make=None, conservative=None) -> tuple[dict | None, dict | None]:
    make = make or flow_rule
    conservative = conservative or (lambda p: (p["s"], -p["n"], -p["d"], p["r"], p["t0"]))
    best, best_m = None, None
    for values in itertools.product(*grid.values()):
        p = dict(zip(grid, values, strict=True))
        m = matrix(runs, DEFAULT_PROBE_K, DEFAULT_PROBE_MIN_TOKENS, rule=with_rule_d(make(p)))
        if m["cells"]["FP"]:
            continue
        # most savings on completed runs; ties go to the more conservative rule
        key = (round(m["saved_completed"], 4), *conservative(p))
        if best is None or key > best[0]:
            best, best_m = (key, p), m
    return (best[1], best_m) if best else (None, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="the one-time held-out evaluation")
    parser.add_argument("--flow", type=json.loads, help="rule chosen on dev, as JSON (required with --test)")
    parser.add_argument("--baseline", type=json.loads, help="streak-only rule chosen on dev, as JSON")
    parser.add_argument("--overnight", nargs="*", type=Path, default=[])
    parser.add_argument("--v2", action="store_true", help="second study: pick on all 210 runs, test on overnight")
    parser.add_argument("--v2-rule", type=json.loads, help="rule chosen for the second study (with --test)")
    args = parser.parse_args()
    rule_d = lambda run: stop_point(run, DEFAULT_PROBE_K, DEFAULT_PROBE_MIN_TOKENS)  # noqa: E731

    v2_conservative = lambda p: (p["s"], p["m"], -p["d"], -p["w"])  # noqa: E731
    if args.v2 and not args.test:
        dev = load_runs(DEV + TEST)
        print(f"second study, development runs: {len(dev)}")
        summary("rule D", dev, rule_d)
        concluded = sum(any(f["concluded"] for f in r["flow2"]) for r in dev if r["correct"])
        print(f"  correct runs where the probed answer is ever stated as a conclusion: {concluded}")
        chosen, _ = pick(dev, GRID_V2, make=v2_rule, conservative=v2_conservative)
        print(f"chosen rule: {chosen}")
        if chosen:
            summary("conclusion rule + rule D", dev, with_rule_d(v2_rule(chosen)))
        return
    if args.v2:
        runs = load_runs(args.overnight)
        print(f"second study, overnight runs: {len(runs)}")
        d = summary("rule D", runs, rule_d)
        f = summary("conclusion rule + rule D", runs, with_rule_d(v2_rule(args.v2_rule)))
        c, n_correct = f["cells"], sum(r["correct"] for r in runs)
        if n_correct < 30:
            print(f"  only {n_correct} correct completed runs: numbers only, no verdict")
            return
        print(
            f"  F1 FP<=1: {'PASS' if c['FP'] <= 1 else 'FAIL'} | F2 saved on completed >=20%: "
            f"{'PASS' if f['saved_completed'] >= 0.20 else 'FAIL'} | F3 runaway missed <= rule D: "
            f"{'PASS' if c['FN_runaway'] <= d['cells']['FN_runaway'] else 'FAIL'}"
        )
        return

    if not args.test:
        dev = load_runs(DEV)
        print(f"development runs: {len(dev)}")
        summary("rule D", dev, rule_d)
        flow, _ = pick(dev, GRID)
        base, _ = pick(dev, {**GRID, "n": (1,), "d": (math.inf,), "r": (0,)})
        print(f"chosen flow rule: {flow}")
        if flow:
            summary("flow rule + rule D", dev, with_rule_d(flow_rule(flow)))
        print(f"chosen streak-only baseline: {base}")
        if base:
            summary("streak-only + rule D", dev, with_rule_d(flow_rule(base)))
        return

    sets = {"held-out (test 1)": load_runs(TEST)}
    if args.overnight:
        sets["overnight (test 2)"] = load_runs(args.overnight)
    for label, runs in sets.items():
        print(f"{label}: {len(runs)} runs")
        d = summary("rule D", runs, rule_d)
        f = summary("flow rule + rule D", runs, with_rule_d(flow_rule(args.flow)))
        if args.baseline:
            summary("streak-only + rule D", runs, with_rule_d(flow_rule(args.baseline)))
        c = f["cells"]
        print(
            f"  F1 FP<=1: {'PASS' if c['FP'] <= 1 else 'FAIL'} | F2 saved on completed >=20%: "
            f"{'PASS' if f['saved_completed'] >= 0.20 else 'FAIL'} | F3 runaway missed <= rule D: "
            f"{'PASS' if c['FN_runaway'] <= d['cells']['FN_runaway'] else 'FAIL'}"
        )


if __name__ == "__main__":
    main()
