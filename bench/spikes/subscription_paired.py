"""V7: does stopping a Claude Code turn at the 60 s alert lose correct answers? (docs/validation/subscription.md)

Problems come from the local OpenR1 cache, hardest first (by mean length of the R1 solutions it ships). Each candidate
is run to completion once (screening); it is eligible when that run went 60 s or more without output. Eligible problems
get 3 more full runs and 3 runs under the recipe policy: the real ClaudeCodeObserver alerts at 60 s of silence, the
turn gets SIGINT, and if it really was interrupted one `--resume` follow-up asks for the best answer now.

Uses your Claude subscription heavily (up to ~120 Sonnet runs, some several minutes long). Results are appended to
--out after every run, and a rerun with the same --out resumes where it stopped.

Usage: python bench/spikes/subscription_paired.py --cache <dir with openr1_math_*.json> --out v7.json [--analyze]
"""

import argparse
import glob
import json
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from subscription_replay import main_chain, waits

from overthink_guard.observe import ClaudeCodeObserver, default_projects_dir, default_sessions_dir

AFTER = 60
FULL_LIMIT = 1200
SUFFIX = "\n\nDo not use any tools. Reply with only the final number."
FOLLOW_UP = "Stop analysing. Give your best answer now from the reasoning so far. Reply with only the final number."
FLAGS = ["--model", "sonnet", "--effort", "high", "--output-format", "stream-json", "--verbose"]
MULTIPLE_CHOICE = re.compile(r"\(A\)|\bA[.)]\s|\(a\)|options|choices", re.I)


def pool(cache: Path) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(cache / "openr1_math_*.json"))):
        rows += [r["row"] for r in json.loads(Path(f).read_text(encoding="utf-8"))["rows"]]
    keep = [
        r
        for r in rows
        if re.fullmatch(r"-?\d+", r["answer"].strip())
        and r.get("question_type") == "math-word-problem"
        and not MULTIPLE_CHOICE.search(r["problem"])
        and len(r["problem"]) < 400
        and (r.get("correctness_count") or 0) >= 1
    ]
    length = {r["uuid"]: sum(map(len, r["generations"])) / max(1, len(r["generations"])) for r in keep}
    return sorted(keep, key=lambda r: (-length[r["uuid"]], r["uuid"]))


class Stopper:
    """The recipe policy: SIGINT a registered session once when the real observer alerts."""

    def __init__(self) -> None:
        self.procs: dict[str, subprocess.Popen] = {}
        self.stopped: set[str] = set()
        self.lock = threading.Lock()
        self.observer = ClaudeCodeObserver(
            default_projects_dir(), AFTER, self._alert, sessions_dir=default_sessions_dir()
        )
        threading.Thread(target=self._loop, daemon=True).start()

    def _alert(self, event: dict) -> None:
        sid = event["source"]["session_id"]
        with self.lock:
            proc = self.procs.get(sid)
            if proc and sid not in self.stopped and proc.poll() is None:
                self.stopped.add(sid)
                proc.send_signal(signal.SIGINT)

    def _loop(self) -> None:
        while True:
            self.observer.poll()
            time.sleep(1)


def claude(args: list[str], cwd: Path, stopper: Stopper | None = None) -> dict:
    start = time.time()
    proc = subprocess.Popen(
        ["claude", "-p", *args, *FLAGS], cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True
    )
    killer = threading.Timer(FULL_LIMIT, lambda: proc.poll() is None and proc.kill())
    killer.start()
    events, sid = [], None
    for line in proc.stdout:
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        events.append(event)
        if sid is None and event.get("session_id"):
            sid = event["session_id"]
            if stopper:
                with stopper.lock:
                    stopper.procs[sid] = proc
    proc.wait()
    killer.cancel()
    result = next((e for e in events if e.get("type") == "result"), {})
    usage = result.get("usage") or {}
    found = re.findall(r"-?\d[\d,]*", result.get("result") or "")
    return {
        "seconds": round(time.time() - start, 1),
        "session_id": sid,
        "end": result.get("terminal_reason") or result.get("subtype") or "killed",
        "answer": found[-1].replace(",", "") if found else None,
        "output_tokens": usage.get("output_tokens", 0),
        "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens", 0),
        "longest_silence": longest_silence(sid, time.time() - start),
    }


def longest_silence(sid: str | None, fallback: float) -> float:
    path = next(default_projects_dir().glob(f"*/{sid}.jsonl"), None) if sid else None
    if path is None:
        return round(fallback, 1)
    ws = waits(main_chain(path))
    return round(max((min(w["end"], w["start"] + fallback) - w["start"] for w in ws), default=0.0), 1)


def run_arm(problem: dict, arm: str, stopper: Stopper) -> dict:
    prompt = problem["problem"] + SUFFIX
    with tempfile.TemporaryDirectory() as tmp:
        if arm == "full":
            return {"arm": arm, "turn": claude([prompt], Path(tmp)), "follow_up": None}
        turn = claude([prompt], Path(tmp), stopper)
        follow = None
        if turn["session_id"] in stopper.stopped and turn["end"] == "aborted_streaming":
            follow = claude(["--resume", turn["session_id"], FOLLOW_UP], Path(tmp))
        return {"arm": arm, "turn": turn, "follow_up": follow}


def final_answer(run: dict) -> str | None:
    return (run["follow_up"] or run["turn"])["answer"]


def analyze(records: list[dict]) -> None:
    by_problem: dict[str, dict] = {}
    for r in records:
        p = by_problem.setdefault(r["uuid"], {"gold": r["gold"], "screen": None, "full": [], "stop": []})
        if r["role"] == "screen":
            p["screen"] = r
        else:
            p[r["arm"]].append(r)
    eligible = {u: p for u, p in by_problem.items() if p["screen"] and p["screen"]["turn"]["longest_silence"] >= AFTER}
    print(f"screened {len(by_problem)}, eligible {len(eligible)}")
    full_ok = stop_ok = fp = gained = 0
    matrix = {(a, b): 0 for a in (True, False) for b in (True, False)}
    for u, p in eligible.items():
        f = sum(final_answer(r) == p["gold"] for r in p["full"])
        s = sum(final_answer(r) == p["gold"] for r in p["stop"])
        full_ok, stop_ok = full_ok + f, stop_ok + s
        majority = f >= 2
        fp += majority and s <= 1
        gained += f <= 1 and s >= 2
        for r in p["stop"]:
            if r["follow_up"]:
                matrix[(majority, final_answer(r) == p["gold"])] += 1
        interrupted = sum(bool(r["follow_up"]) for r in p["stop"])
        print(
            f"  {u[:8]} gold {p['gold']:>8} full {f}/{len(p['full'])} stop {s}/{len(p['stop'])} "
            f"(interrupted {interrupted}) full answers {[final_answer(r) for r in p['full']]} "
            f"stop answers {[final_answer(r) for r in p['stop']]}"
        )
    n_full = sum(len(p["full"]) for p in eligible.values())
    n_stop = sum(len(p["stop"]) for p in eligible.values())
    print(f"Q1 correct: full {full_ok}/{n_full}, stop-at-60s {stop_ok}/{n_stop}; FP problems {fp}, gained {gained}")
    print("   interrupted turns (full majority right?, stopped answer right?): " + str(matrix))
    t_full = [r["turn"]["seconds"] for p in eligible.values() for r in p["full"]]
    t_stop = [
        r["turn"]["seconds"] + (r["follow_up"] or {}).get("seconds", 0) for p in eligible.values() for r in p["stop"]
    ]
    if t_full and t_stop:
        print(f"   mean seconds: full {sum(t_full) / len(t_full):.0f}, stop {sum(t_stop) / len(t_stop):.0f}")
    buckets: dict[str, list[bool]] = {"<60": [], "60-300": [], ">=300": []}
    for p in by_problem.values():
        for r in ([p["screen"]] if p["screen"] else []) + p["full"]:
            s = r["turn"]["longest_silence"]
            key = "<60" if s < AFTER else "60-300" if s < 300 else ">=300"
            buckets[key].append(final_answer(r) == p["gold"])
    print("Q2 full-run accuracy by silence: " + ", ".join(f"{k}: {sum(v)}/{len(v)}" for k, v in buckets.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eligible", type=int, default=10)
    parser.add_argument("--max-screen", type=int, default=60)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.out.read_text().splitlines()] if args.out.exists() else []
    if args.analyze:
        analyze(records)
        return
    lock = threading.Lock()

    def save(record: dict) -> None:
        with lock:
            records.append(record)
            with args.out.open("a") as fh:
                fh.write(json.dumps(record) + "\n")

    stopper = Stopper()
    candidates = pool(args.cache)[: args.max_screen]
    screened = {r["uuid"]: r for r in records if r["role"] == "screen"}
    with ThreadPoolExecutor(args.workers) as ex:
        todo = [p for p in candidates if p["uuid"] not in screened]
        i = 0
        while sum(r["turn"]["longest_silence"] >= AFTER for r in screened.values()) < args.eligible and i < len(todo):
            batch, i = todo[i : i + args.workers], i + args.workers
            for p, run in zip(batch, ex.map(lambda p: run_arm(p, "full", stopper), batch), strict=True):
                rec = {"uuid": p["uuid"], "gold": p["answer"].strip(), "role": "screen", **run}
                save(rec)
                screened[p["uuid"]] = rec
                print(f"screen {p['uuid'][:8]} silence {run['turn']['longest_silence']}s", flush=True)
        eligible = [
            p for p in candidates if p["uuid"] in screened and screened[p["uuid"]]["turn"]["longest_silence"] >= AFTER
        ]
        eligible = eligible[: args.eligible]
        done = {(r["uuid"], r["arm"], r["rep"]) for r in records if r["role"] == "test"}
        jobs = [
            (p, arm, rep)
            for p in eligible
            for rep in range(3)
            for arm in ("full", "stop")
            if (p["uuid"], arm, rep) not in done
        ]

        def job(item: tuple[dict, str, int]) -> None:
            p, arm, rep = item
            run = run_arm(p, arm, stopper)
            save({"uuid": p["uuid"], "gold": p["answer"].strip(), "role": "test", "rep": rep, **run})
            print(
                f"{arm:4} {p['uuid'][:8]} #{rep} -> {final_answer(run)} (gold {p['answer'].strip()}) "
                f"{run['turn']['end']} {run['turn']['seconds']}s",
                flush=True,
            )

        list(ex.map(job, jobs))
    analyze(records)


if __name__ == "__main__":
    main()
