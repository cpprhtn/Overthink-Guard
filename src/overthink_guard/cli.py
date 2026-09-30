from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from overthink_guard import __version__
from overthink_guard.analysis import JudgeConfig, ReplayReport, replay
from overthink_guard.analysis.prober import DEFAULT_OPEN_BUDGET_TOKENS, DEFAULT_PROBE_K, DEFAULT_PROBE_MIN_TOKENS
from overthink_guard.config import MODES, ConfigError, Settings, default_config_path, load_settings
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
    print(
        f"  stop       segment {stop['segment']} @ ~{stop['at_tokens']:,} tokens, span novelty {stop['span_novelty']}"
    )
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
        "answer_match": dict(
            Counter({True: "match", False: "mismatch", None: "unknown"}[i["answer_match"]] for i in items if i["stop"])
        ),
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
        print(
            f"== summary: {s['stopped']}/{s['files']} stopped, "
            f"~{s['saved_tokens_est']:,} of ~{s['thinking_tokens_est']:,} thinking tokens saved "
            f"({s['saved_ratio']:.0%}), answers {s['answer_match']}"
        )
    return 0


def resolve_settings(args: argparse.Namespace) -> Settings:
    """Defaults, then the config file, then any flags given explicitly."""
    if args.config is not None and not args.config.exists():
        raise ConfigError(f"config file not found: {args.config}")
    settings = load_settings(args.config or default_config_path())
    for flag, attr in [
        ("host", "host"),
        ("port", "port"),
        ("backend_url", "backend_url"),
        ("mode", "mode"),
        ("probe", "probe"),
        ("probe_interval", "probe_interval"),
        ("probe_min_tokens", "probe_min_tokens"),
        ("probe_k", "probe_converge_k"),
        ("open_budget_tokens", "open_budget_tokens"),
        ("stats_file", "stats_file"),
    ]:
        if getattr(args, flag) is not None:
            setattr(settings, attr, getattr(args, flag))
    if args.no_stats:
        settings.stats_file = None
    return settings


def _start(args: argparse.Namespace) -> int:
    import httpx
    import uvicorn

    from overthink_guard.proxy import create_app
    from overthink_guard.storage import ShadowStats

    try:
        s = resolve_settings(args)
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    try:
        version = httpx.get(f"{s.backend_url.rstrip('/')}/api/version", timeout=3).json()["version"]
        print(f"✓ Ollama {version} at {s.backend_url}")
    except (httpx.HTTPError, ValueError, KeyError):
        print(f"! Ollama not reachable at {s.backend_url} yet; requests will fail until it is up")
    base = f"http://{s.host}:{s.port}"
    print(f"→ proxy: {base}/v1   (set this as your client's base URL)")
    print(f"→ UI:    {base}/ui")
    if s.mode == "auto":
        print(
            f"→ mode:  auto (runaway guard: once thinking passes {s.probe_min_tokens} tokens and "
            f"{s.probe_converge_k} probes agree, it answers from the thinking so far)"
        )
    else:
        print("→ mode:  shadow (nothing is stopped unless you press 'Answer now'; would-stop points are recorded)")
    if s.probe or s.mode == "auto":
        print(
            f"→ probe: every {s.probe_interval} thinking tokens after the first {s.probe_min_tokens}, "
            f"k={s.probe_converge_k} (Tier 2, adds short pauses)"
        )
    print(f"→ stats: {s.stats_file or 'memory only'} (counts only, no text)", flush=True)
    app = create_app(
        s.backend_url,
        host=s.host,
        port=s.port,
        stats=ShadowStats(s.stats_file),
        judge_config=s.judge,
        probe=s.probe,
        auto=s.mode == "auto",
        probe_interval=s.probe_interval,
        probe_min_tokens=s.probe_min_tokens,
        probe_converge_k=s.probe_converge_k,
        open_budget_tokens=s.open_budget_tokens,
    )
    uvicorn.run(app, host=s.host, port=s.port, log_level="warning")
    return 0


def _claude_watch(args: argparse.Namespace) -> int:
    import time

    from overthink_guard.notify import notify, run_hook
    from overthink_guard.observe import ClaudeCodeObserver

    def on_alert(event: dict) -> None:
        seconds = event["stats"]["thinking_elapsed_s"]
        message = f"Claude has been working for {seconds}s with no output yet ({event['source']['project']})"
        print(
            f"[{time.strftime('%H:%M:%S')}] {message}. Long turns are usually productive; "
            "stopping one early often costs the right answer (docs/validation/subscription.md, V7).",
            flush=True,
        )
        if args.desktop:
            notify("Overthink Guard", message + ".")
        if args.hook:
            run_hook(args.hook, event)

    def on_flow(event: dict) -> None:
        # Record only by default: whether these signals mark wasted work is still being measured
        # (docs/validation/action-flow.md). The log holds counts, reasons and ids, never commands or text.
        if args.flow_log:
            args.flow_log.parent.mkdir(parents=True, exist_ok=True)
            with args.flow_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")
        if args.flow_alerts:
            failed = event["stats"]["failed_tool_calls_in_turn"]
            message = f"{failed} tool calls have failed in this turn ({event['source']['project']})"
            print(f"[{time.strftime('%H:%M:%S')}] {message}.", flush=True)
            if args.desktop:
                notify("Overthink Guard", message + ".")
            if args.hook:
                run_hook(args.hook, event)

    if args.flow_log is None and not args.no_flow_log:
        from overthink_guard.observe import default_flow_log_path

        args.flow_log = default_flow_log_path()
    if args.no_flow_log:
        args.flow_log = None
    observer = ClaudeCodeObserver(
        args.projects_dir, args.after, on_alert, sessions_dir=args.projects_dir.parent / "sessions", on_flow=on_flow
    )
    print(
        f"Watching {args.projects_dir} (read-only; no credentials, no requests). "
        f"Alert after {args.after:g}s of silence. "
        f"Repeated tool failures: {'alert and ' if args.flow_alerts else ''}"
        f"{'record to ' + str(args.flow_log) if args.flow_log else 'not recorded'}.",
        flush=True,
    )
    try:
        while True:
            observer.poll()
            time.sleep(1)
    except KeyboardInterrupt:
        return 0


def _claude_report(args: argparse.Namespace) -> int:
    from overthink_guard.observe import default_flow_log_path, flow_review, usage_report

    report = usage_report(args.projects_dir, days=args.days)
    report["flow"] = flow_review(args.projects_dir, args.flow_log or default_flow_log_path())
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        print()
        return 0
    span = f"last {args.days:g} days" if args.days else "all local sessions"
    print(f"Claude Code reasoning usage ({span}): {report['responses']:,} responses")
    print(
        f"  thinking {report['thinking_tokens']:,} of {report['output_tokens']:,} output tokens "
        f"({report['thinking_share']:.0%})"
    )
    for effort, row in report["by_effort"].items():
        print(
            f"  effort {effort:8} {row['responses']:6,} responses, thinking median {row['median_thinking']:,.0f} "
            f"/ p90 {row['p90_thinking']:,.0f} tokens"
        )
    s = report["silence_seconds"]
    print(f"  wait before first output: median {s['p50']}s, p90 {s['p90']}s, over 60s in {s['over_60s']} turns")
    if report["thinking_share"] >= 0.5:
        print(
            "  → Thinking is most of your output. A lower effort (/effort or --effort) would cut it; how much that "
            "costs in answer quality has not been measured yet."
        )
    f = report["flow"]
    if f["signals"]:
        print(
            f"  repeated tool failures: {f['signals']} signals {f['by_reason']}; afterwards the user stopped the turn "
            f"{f['stopped_by_user']} times; median after the signal: {f['median_tools_after']:.0f} more tool calls, "
            f"{f['median_failed_after']:.0f} more failures, {f['median_seconds_after']:.0f}s"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="otg", description="Overthink Guard")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    defaults = JudgeConfig()
    analyze = sub.add_parser(
        "analyze", help="replay recorded completions offline and report where auto-stop would fire"
    )
    analyze.add_argument("files", nargs="+", help="completion text files ('-' for stdin)")
    analyze.add_argument("--template", default="generic", choices=template_ids())
    analyze.add_argument("--thinking-only", action="store_true", help="files contain only thinking text, no tags")
    analyze.add_argument("--converge-k", type=int, default=defaults.converge_k)
    analyze.add_argument("--min-thinking-tokens", type=int, default=defaults.min_thinking_tokens)
    analyze.add_argument("--revision-cooldown-tokens", type=int, default=defaults.revision_cooldown_tokens)
    analyze.add_argument("--repetition-threshold", type=float, default=defaults.repetition_threshold)
    analyze.add_argument("--json", action="store_true", help="machine-readable output")
    analyze.set_defaults(func=_analyze)

    start = sub.add_parser(
        "start", help="run the local proxy and UI in front of Ollama (flags override the config file)"
    )
    start.add_argument("--config", type=Path, help=f"YAML settings (default: {default_config_path()})")
    start.add_argument("--host", help="default 127.0.0.1")
    start.add_argument("--port", type=int, help="default 8484")
    start.add_argument("--backend-url", help="default http://localhost:11434")
    start.add_argument(
        "--mode", choices=MODES, help="shadow (default) records only; auto also stops runaway thinking (probing on)"
    )
    start.add_argument(
        "--probe", action=argparse.BooleanOptionalAction, help="opt-in Tier 2 active probing (off by default, D12)"
    )
    start.add_argument("--probe-interval", type=int, help="thinking tokens between probes (default 400)")
    start.add_argument(
        "--probe-min-tokens",
        type=int,
        help=f"no probes before this many thinking tokens (default {DEFAULT_PROBE_MIN_TOKENS})",
    )
    start.add_argument(
        "--probe-k",
        type=int,
        help=f"identical probe answers in a row to count as converged (default {DEFAULT_PROBE_K})",
    )
    start.add_argument(
        "--open-budget-tokens",
        type=int,
        help=f"thinking budget for open-ended prompts with no short answer (default {DEFAULT_OPEN_BUDGET_TOKENS})",
    )
    start.add_argument("--stats-file", type=Path, help="where Shadow statistics are appended (JSONL)")
    start.add_argument("--no-stats", action="store_true", help="keep Shadow statistics in memory only")
    start.set_defaults(func=_start)

    claude = sub.add_parser(
        "claude-code", help="observe Claude Code (subscription) sessions: read-only, no credentials, no proxy"
    )
    claude_sub = claude.add_subparsers(dest="claude_command", required=True)
    watch = claude_sub.add_parser("watch", help="notify when a turn has gone unusually long without output")
    watch.add_argument("--after", type=float, default=60, help="seconds of silence before notifying (default 60)")
    watch.add_argument("--hook", default="", help="command to run with the event JSON on stdin")
    watch.add_argument("--no-desktop", dest="desktop", action="store_false", help="print only")
    watch.add_argument("--flow-log", type=Path, help="where repeated-failure signals are recorded (JSONL, counts only)")
    watch.add_argument("--no-flow-log", action="store_true", help="do not record repeated-failure signals")
    watch.add_argument("--flow-alerts", action="store_true", help="also alert on repeated tool failures in a turn")
    watch.set_defaults(func=_claude_watch)
    report = claude_sub.add_parser("report", help="how much output went to thinking, per effort level")
    report.add_argument("--days", type=float, help="only the last N days")
    report.add_argument("--json", action="store_true")
    report.add_argument("--flow-log", type=Path, help="repeated-failure log to review (default: the watch default)")
    report.set_defaults(func=_claude_report)
    for p in (watch, report):
        p.add_argument("--projects-dir", type=Path, default=None, help="default ~/.claude/projects")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "claude_command", None) and args.projects_dir is None:
        from overthink_guard.observe import default_projects_dir

        args.projects_dir = default_projects_dir()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
