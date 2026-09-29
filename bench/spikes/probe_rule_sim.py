"""Replays recorded probe trails under different stop rules: python bench/spikes/probe_rule_sim.py [data.json]"""

import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent.parent / "data" / "shadow_live_2026-09-29.json")
data = json.loads(path.read_text(encoding="utf-8"))
gradable = [p for i, p in enumerate(data["problems"]) if i not in data["ungradable"]]
baseline = sum(p["final_correct"] for p in gradable)
print(f"gradable {len(gradable)}, correct with full thinking {baseline}")
print(f"{'rule':26} fired  saved  correct  lost  gained")
for k in (3, 4, 5, 6):
    for min_tokens in (0, 1200, 2000, 3000):
        fired = saved = total = correct = lost = gained = 0
        for p in gradable:
            total += p["thinking_tokens"]
            trail = [(t, a) for t, a in p["probes"] if t >= min_tokens]
            stop = next(
                (
                    trail[j]
                    for j in range(k - 1, len(trail))
                    if trail[j - k + 1][1] is not None and len({a for _, a in trail[j - k + 1 : j + 1]}) == 1
                ),
                None,
            )
            if stop is None:
                correct += p["final_correct"]
                continue
            fired += 1
            saved += p["thinking_tokens"] - stop[0]
            right = stop[1] == p["gold"]
            correct += right
            lost += p["final_correct"] and not right
            gained += not p["final_correct"] and right
        print(
            f"k={k} probes from {min_tokens:>4} tok  {fired:5}  {saved / total:5.0%}  {correct:4} ({correct - baseline:+d})"
            f"  {lost:4}  {gained:6}"
        )
