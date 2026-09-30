"""V2-V4b: live headless `claude -p` runs watched by the real `otg claude-code watch` CLI.

Each run gets its own working directory (so its own project log). Alerts are captured through --hook. Whether a run
should have been alerted comes from its session log, read with the V1 ground truth (subscription_replay.waits): a wait
of at least --after seconds that ended in model output. V3 compares `otg claude-code report` on that run's project
with the run's own stream-json result.usage. V4b samples the watcher's CPU time and RSS.

Uses your Claude subscription (about 15 short Sonnet runs). Usage: python bench/spikes/subscription_live.py [--after 20]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from subscription_replay import WORKING, main_chain, waits

from overthink_guard.observe import default_projects_dir, usage_report

LONG = [
    "Without using any tools: how many ordered pairs of integers (a, b) with 1 <= a, b <= 100 satisfy that a^2 + b^2 "
    "is divisible by 13? Give only the number.",
    "Without using any tools: how many integers n with 1 <= n <= 1000 make n^2 + n + 41 composite? "
    "Give only the number.",
    "Without using any tools: find the number of subsets of {1, 2, ..., 15} whose elements sum to a multiple of 5. "
    "Give only the number.",
]
SHORT = "What is 17 * 23? Reply with only the number."
TOOL = "Use the Bash tool to run `echo hello` and tell me what it printed."
HOOK = (
    "import json, sys, time\nwith open(sys.argv[1], 'a') as f:\n"
    "    f.write(json.dumps({'received': time.time(), **json.load(sys.stdin)}) + '\\n')\n"
)


def run(prompt: str, cwd: Path, extra: list[str], sigint_after: float | None = None) -> dict:
    cwd.mkdir(parents=True)
    cmd = ["claude", "-p", prompt, "--model", "sonnet", "--output-format", "stream-json", "--verbose", *extra]
    start = time.time()
    proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
    if sigint_after is not None:
        threading.Timer(sigint_after, lambda: proc.poll() is None and proc.send_signal(signal.SIGINT)).start()
    events = [json.loads(line) for line in proc.stdout if line.startswith("{")]
    proc.wait()
    result = next((e for e in events if e.get("type") == "result"), {})
    session_id = next((e["session_id"] for e in events if e.get("session_id")), None)
    return {"start": start, "end": time.time(), "session_id": session_id, "result": result,
            "responses": len({e["message"]["id"] for e in events if e.get("type") == "assistant"})}


def expected_alerts(session_id: str, after: float) -> tuple[int, float]:
    path = next(default_projects_dir().glob(f"*/{session_id}.jsonl"), None)
    if path is None:
        return 0, 0.0
    ws = waits(main_chain(path))
    longest = max((min(w["end"], time.time()) - w["start"] for w in ws), default=0.0)
    return sum(w["end"] - w["start"] >= after and w["label"] in WORKING for w in ws), longest


def report_check(r: dict) -> dict:
    path = next(default_projects_dir().glob(f"*/{r['session_id']}.jsonl"))
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / path.parent.name).symlink_to(path.parent)
        rep = usage_report(Path(tmp))
    usage = r["result"].get("usage") or {}
    truth = (r["responses"], (usage.get("output_tokens_details") or {}).get("thinking_tokens", 0),
             usage.get("output_tokens", 0))
    got = (rep["responses"], rep["thinking_tokens"], rep["output_tokens"])
    return {"truth": truth, "report": got, "match": truth == got}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--after", type=float, default=20)
    parser.add_argument("--settle", type=float, default=30, help="seconds to keep watching after each scenario")
    parser.add_argument("--scenarios", default="abcde", help="subset to run, e.g. 'ae'")
    parser.add_argument("--long", type=int, nargs="*", help="indexes into LONG for (a) and (e) (default: all / 0 1)")
    args = parser.parse_args()
    base = Path(tempfile.mkdtemp(prefix="otg-live-"))
    hook, alerts_file = base / "hook.py", base / "alerts.jsonl"
    hook.write_text(HOOK)
    alerts_file.touch()
    watcher = subprocess.Popen(
        [sys.executable, "-m", "overthink_guard.cli", "claude-code", "watch", "--after", str(args.after),
         "--no-desktop", "--hook", f"{sys.executable} {hook} {alerts_file}"],
        stdout=subprocess.DEVNULL,
    )
    rss: list[int] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(2):
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(watcher.pid)], capture_output=True, text=True).stdout
            if out.strip():
                rss.append(int(out))

    threading.Thread(target=sample, daemon=True).start()
    time.sleep(3)
    t0, cpu0 = time.time(), _cpu_seconds(watcher.pid)

    runs: list[tuple[str, dict]] = []
    effort_high = ["--effort", "high"]
    long_a = [LONG[i] for i in args.long] if args.long else LONG
    long_e = [LONG[i] for i in args.long] if args.long else LONG[:2]
    if "a" in args.scenarios:
        for i, prompt in enumerate(long_a):
            runs.append(("a_long", run(prompt, base / f"a{i}", effort_high)))
    if "b" in args.scenarios:
        for i in range(3):
            runs.append(("b_short", run(SHORT, base / f"b{i}", ["--effort", "low"])))
    if "c" in args.scenarios:
        for i in range(2):
            runs.append(("c_max_turns", run(TOOL, base / f"c{i}", ["--max-turns", "1"])))
    if "d" in args.scenarios:
        for i in range(2):
            runs.append(("d_sigint", run(LONG[i], base / f"d{i}", effort_high, sigint_after=8)))
        time.sleep(args.settle)
    if "e" in args.scenarios:
        pair: list[dict] = []
        threads = [threading.Thread(target=lambda i=i, q=q: pair.append(run(q, base / f"e{i}", effort_high)))
                   for i, q in enumerate(long_e[:2] if len(long_e) > 1 else long_e * 2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        runs += [("e_concurrent", r) for r in pair]
    time.sleep(args.settle)

    wall, cpu = time.time() - t0, _cpu_seconds(watcher.pid) - cpu0
    stop.set()
    watcher.send_signal(signal.SIGINT)
    watcher.wait(timeout=10)
    alerts = [json.loads(line) for line in alerts_file.read_text().splitlines()]

    print(f"threshold {args.after:g}s, workdir {base}")
    for scenario, r in runs:
        sid = r["session_id"]
        mine = [a for a in alerts if a["source"]["session_id"] == sid]
        want, longest = expected_alerts(sid, args.after) if sid else (0, 0.0)
        lag = [a["stats"]["thinking_elapsed_s"] - args.after for a in mine]
        rc = report_check(r) if sid else {}
        print(f"  {scenario:13} {sid and sid[:8]} longest wait {longest:5.1f}s  expected {want} got {len(mine)} "
              f"lag {lag} run_mode {[a['source']['run_mode'] for a in mine]} "
              f"end {r['result'].get('terminal_reason') or r['result'].get('subtype')} "
              f"V3 {rc.get('truth')} vs {rc.get('report')} {'OK' if rc.get('match') else 'DIFF'}")
    stray = [a for a in alerts if a["source"]["session_id"] not in {r["session_id"] for _, r in runs}]
    print(f"  alerts for other sessions (not ours): {len(stray)}")
    print(f"V4b watcher over {wall:.0f}s: CPU {100 * cpu / wall:.2f}%, RSS max {max(rss) / 1024:.1f} MB")


def _cpu_seconds(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "cputime=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    parts = [float(p) for p in out.replace("-", ":").split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


if __name__ == "__main__":
    os.environ.pop("CLAUDECODE", None)
    main()
