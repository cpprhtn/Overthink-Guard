"""Collects probe trails for held-out OpenR1 problems through a running `otg start --probe` proxy.

Usage: python bench/spikes/shadow_validate.py --cache .bench-cache --out bench/data/shadow_validate.json
Output matches bench/data/shadow_live_*.json, so probe_rule_sim.py can replay it.
"""

import argparse
import glob
import json
import re
from pathlib import Path

import httpx
from openai import OpenAI

from overthink_guard.analysis.signals import extract_boxed, normalize_answer

MULTIPLE_CHOICE = re.compile(r"\(A\)|\bA[.)]\s|\(a\)|options|choices", re.I)


def held_out(cache: Path, skip: int, start: int, count: int) -> list[tuple[str, str]]:
    rows = []
    files = sorted(glob.glob(str(cache / "openr1_math_*.json")), key=lambda p: int(re.search(r"_(\d+)\.json", p)[1]))
    for f in files:
        rows += [r["row"] for r in json.loads(Path(f).read_text(encoding="utf-8"))["rows"]]
    short = [(r["problem"], r["answer"].strip()) for r in rows if re.fullmatch(r"-?\d+", r["answer"].strip())]
    short = [p for p in short if len(p[0]) < 260]
    return [p for p in short[skip:] if not MULTIPLE_CHOICE.search(p[0])][start : start + count]


def session_state(proxy: str, session: str) -> tuple[dict, list, list]:
    summary, probes, segments = {}, [], []
    try:
        with httpx.stream("GET", f"{proxy}/otg/api/events", timeout=1.5) as r:
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("session") == session and event["type"] == "probe":
                    probes.append([event["at_tokens"], event["answer"], event.get("grounded", True)])
                elif event.get("session") == session and event["type"] == "segment":
                    segments.append([event["start_tokens"], event["end_tokens"], event["text"]])
                elif event["type"] == "session" and event["id"] == session:
                    summary = event
    except httpx.ReadTimeout:
        pass
    return summary, probes, sorted(segments)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", default="http://127.0.0.1:8485")
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--cache", type=Path, default=Path(".bench-cache"))
    parser.add_argument("--skip", type=int, default=12, help="integer-answer problems already used for tuning")
    parser.add_argument("--start", type=int, default=0, help="offset into the held-out list (0-29 used on 2026-09-29)")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=12000, help="caps runaway thinking loops")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    client = OpenAI(base_url=f"{args.proxy}/v1", api_key="unused")
    data = {"note": f"{args.model}, held-out OpenR1 problems {args.start}-{args.start + args.count - 1} after the first {args.skip}", "problems": [], "ungradable": []}
    for i, (question, gold) in enumerate(held_out(args.cache, args.skip, args.start, args.count)):
        content, session = "", None
        messages = [{"role": "user", "content": question}]
        stream = client.chat.completions.create(
            model=args.model, stream=True, messages=messages, max_tokens=args.max_tokens
        )
        finish = None
        for chunk in stream:
            session = session or chunk.id.rsplit("-", 1)[-1]
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            if chunk.choices and chunk.choices[0].finish_reason:
                finish = chunk.choices[0].finish_reason
        boxed = extract_boxed(content)
        final = normalize_answer(boxed[-1][1]) if boxed and finish != "length" else None
        summary, probes, segments = session_state(args.proxy, session)
        shadow = summary.get("shadow") or {}
        data["problems"].append(
            {
                "gold": gold,
                "final": final,
                "final_correct": final == gold,
                "truncated": finish == "length",
                "thinking_tokens": shadow.get("thinking_tokens", summary.get("thinking_tokens")),
                "tier0_stop": (shadow.get("tier0") or {}).get("stop_at"),
                "probes": probes,
                "segments": segments,
            }
        )
        if finish == "length":
            data["ungradable"].append(i)
        args.out.write_text(json.dumps(data, indent=1), encoding="utf-8")
        print(f"{i:2d} gold={gold:>7} final={final!s:>8} ok={final == gold!s:5} "
              f"think={data['problems'][-1]['thinking_tokens']} probes={len(probes)} finish={finish}", flush=True)


if __name__ == "__main__":
    main()
