"""Replays recorded probe trails from several samples under answer-only and answer+redundancy stop rules.

Answers are compared by value when both sides are plain arithmetic (e.g. 2013^2 == 4052169).
Usage: python bench/spikes/probe_rule_sim_all.py bench/data/shadow_*.json
"""

import ast
import json
import operator
import sys

from overthink_guard.analysis.signals import novelty

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.USub: operator.neg}


def value(answer):
    if answer is None:
        return None
    try:
        tree = ast.parse(answer.replace("^", "**").replace("\\times", "*").replace("\\cdot", "*"), mode="eval")
    except SyntaxError:
        return answer

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 64:
                raise ValueError
            return OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.operand))
        raise ValueError

    try:
        return round(float(ev(tree.body)), 9)
    except (ValueError, ZeroDivisionError, OverflowError):
        return answer


def same(a, b):
    return a is not None and value(a) == value(b)


def text_between(segments, a, b):
    return "".join(t for s, e, t in segments if s >= a and e <= b)


def stop(problem, k, min_tokens, tau):
    trail = [(t, a) for t, a in problem["probes"] if t >= min_tokens]
    for j in range(k - 1, len(trail)):
        window = trail[j - k + 1 : j + 1]
        if window[0][1] is None or not all(same(a, window[0][1]) for _, a in window):
            continue
        if tau is None:
            return trail[j]
        segments = problem.get("segments")
        if not segments:
            return None
        span = text_between(segments, window[0][0], window[-1][0])
        if span and novelty(span, text_between(segments, 0, window[0][0])) <= tau:
            return trail[j]
    return None


samples = []
for path in sys.argv[1:]:
    data = json.loads(open(path, encoding="utf-8").read())
    gradable = [p for i, p in enumerate(data["problems"]) if i not in data["ungradable"]]
    for p in gradable:
        p["final_correct"] = same(p.get("final"), p["gold"]) if "final" in p else p["final_correct"]
    samples.append((path.rsplit("/", 1)[-1], gradable))
    print(f"{path}: gradable {len(gradable)}, correct with full thinking {sum(p['final_correct'] for p in gradable)}")

rules = [(k, m, None) for k in (3, 4, 5, 6) for m in (0, 2000, 3000)]
rules += [(k, m, tau) for k in (3, 4) for m in (0, 3000) for tau in (0.6, 0.5)]
print(f"\n{'rule':32}" + "".join(f"{n[:22]:>26}" for n, _ in samples))
for k, m, tau in rules:
    row = f"k={k} from {m}" + ("" if tau is None else f" nov<={tau}")
    row = f"{row:32}"
    for _, gradable in samples:
        saved = total = lost = gained = 0
        for p in gradable:
            total += p["thinking_tokens"]
            s = stop(p, k, m, tau)
            if s:
                saved += p["thinking_tokens"] - s[0]
                right = same(s[1], p["gold"])
                lost += p["final_correct"] and not right
                gained += (not p["final_correct"]) and right
        cell = "n/a" if tau is not None and not any(p.get("segments") for p in gradable) else f"{saved / total:3.0%} -{lost}/+{gained}"
        row += f"{cell:>26}"
    print(row)
