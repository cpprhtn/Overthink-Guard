from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from overthink_guard import __version__
from overthink_guard.analysis import JudgeConfig, ReplayReport, replay
from overthink_guard.templates import get_template, template_ids


def _report_dict(path: str, report: ReplayReport) -> dict:
    decision = report.decision
    return {
        "file": path,
        "thinking_tokens_est": report.thinking_tokens,
        "segments": len(report.segments),
        "phases": dict(Counter(s.phase for s in report.segments)),
        "tentative_answers": [
            {"segment": s.index, "at_tokens": s.end_tokens, "answer": s.answer}
            for s in report.segments
            if s.answer is not None
        ],
        "stop": None
        if decision is None
        else {
            "segment": decision.segment_index,
            "at_tokens": decision.at_tokens,
            "answer": decision.answer,
            "reasons": list(decision.reasons),
            "span_novelty": round(decision.span_novelty, 3),
        },
        "saved_tokens_est": report.saved_tokens,
        "final_answer": report.final_answer,
        "answer_match": report.answer_match,
    }


def _print_report(item: dict) -> None:
    print(f"== {item['file']}")
    print(f"  thinking   ~{item['thinking_tokens_est']:,} tokens (estimated), {item['segments']} segments")
    print("  phases     " + ", ".join(f"{k} {v}" for k, v in sorted(item["phases"].items())))
    answers = item["tentative_answers"]
    if answers:
        print("  answers    " + " -> ".join(f"{a['answer']}@{a['at_tokens']}" for a in answers))
    stop = item["stop"]
    if stop is None:
        print("  stop       none (conditions never all held)")
        return
    pct = item["saved_tokens_est"] / item["thinking_tokens_est"] * 100 if item["thinking_tokens_est"] else 0
    match = {True: "match", False: "MISMATCH", None: "unknown"}[item["answer_match"]]
    print(f"  stop       segment {stop['segment']} @ ~{stop['at_tokens']:,} tokens, span novelty {stop['span_novelty']}")
    print(f"  saved      ~{item['saved_tokens_est']:,} tokens ({pct:.0f}%)")
    print(f"  answer     at stop {stop['answer']!r} vs final {item['final_answer']!r}: {match}")


def _summary(items: list[dict]) -> dict:
    thinking = sum(i["thinking_tokens_est"] for i in items)
    saved = sum(i["saved_tokens_est"] for i in items)
    return {
        "files": len(items),
        "stopped": sum(1 for i in items if i["stop"] is not None),
        "thinking_tokens_est": thinking,
        "saved_tokens_est": saved,
        "saved_ratio": round(saved / thinking, 4) if thinking else 0.0,
        "answer_match": dict(Counter({True: "match", False: "mismatch", None: "unknown"}[i["answer_match"]] for i in items if i["stop"])),
    }


def _analyze(args: argparse.Namespace) -> int:
    template = get_template(args.template)
    config = JudgeConfig(
        converge_k=args.converge_k,
        min_thinking_tokens=args.min_thinking_tokens,
        revision_cooldown_tokens=args.revision_cooldown_tokens,
        repetition_threshold=args.repetition_threshold,
    )
    items = []
    for name in args.files:
        text = sys.stdin.read() if name == "-" else Path(name).read_text(encoding="utf-8")
        if args.thinking_only:
            text += template.think_end
        items.append(_report_dict(name, replay(text, template, config)))

    if args.json:
        json.dump({"results": items, "summary": _summary(items)}, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0
    for item in items:
        _print_report(item)
    if len(items) > 1:
        s = _summary(items)
        print(f"== summary: {s['stopped']}/{s['files']} stopped, ~{s['saved_tokens_est']:,} of ~{s['thinking_tokens_est']:,} thinking tokens saved ({s['saved_ratio']:.0%}), answers {s['answer_match']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="otg", description="Overthink Guard")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    defaults = JudgeConfig()
    analyze = sub.add_parser("analyze", help="replay recorded completions offline and report where auto-stop would fire")
    analyze.add_argument("files", nargs="+", help="completion text files ('-' for stdin)")
    analyze.add_argument("--template", default="generic", choices=template_ids())
    analyze.add_argument("--thinking-only", action="store_true", help="files contain only thinking text, no tags")
    analyze.add_argument("--converge-k", type=int, default=defaults.converge_k)
    analyze.add_argument("--min-thinking-tokens", type=int, default=defaults.min_thinking_tokens)
    analyze.add_argument("--revision-cooldown-tokens", type=int, default=defaults.revision_cooldown_tokens)
    analyze.add_argument("--repetition-threshold", type=float, default=defaults.repetition_threshold)
    analyze.add_argument("--json", action="store_true", help="machine-readable output")
    analyze.set_defaults(func=_analyze)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
