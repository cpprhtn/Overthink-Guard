"""Action-flow study B: headless Claude Code on coding tasks, hook logging only (control) vs nudging (treatment).

Pre-registered in docs/validation/action-flow.md. Uses your Claude subscription (48 Sonnet runs, up to 10 min each).
Results are appended to --out after every run; rerunning with the same --out resumes.
Usage: python bench/spikes/action_flow_live.py --out b.jsonl [--analyze]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from action_flow_tasks import TASKS

HOOK = Path(__file__).with_name("action_flow_hook.py")
LIMIT = 600
ALLOWED = ["Read", "Edit", "Write", "Glob", "Grep", "Bash"]
HARD: dict[str, dict] = {}  # filled from --hard <manifest.json> (SWE-bench-style tasks, docs/validation/action-flow.md)
HARD_LIMIT = 900
DENIED = ["Bash(pip:*)", "Bash(pip3:*)", "Bash(python3 -m pip:*)", "Bash(brew:*)", "Bash(curl:*)", "Bash(sudo:*)"]


def load_hard(manifest: Path) -> None:
    root = manifest.parent
    for t in json.loads(manifest.read_text()):
        HARD[t["task_id"]] = {**t, "root": root, "venv": root / "venv"}


def hard_env(spec: dict, work: Path) -> dict:
    env = {
        **os.environ,
        "PATH": f"{spec['venv'] / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "VIRTUAL_ENV": str(spec["venv"]),
    }
    if spec["pythonpath"]:
        env["PYTHONPATH"] = str(work / spec["pythonpath"])
    return env


def grade_hard(spec: dict, work: Path, grade: Path) -> tuple[bool, bool]:
    shutil.copytree(work, grade)
    for rel in spec["hidden_files"]:
        target = grade / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(spec["root"] / "tasks" / spec["task_id"] / "hidden" / rel, target)
    results = []
    for key in ("grade_command", "regression_command"):
        command = spec[key].replace("{venv}", str(spec["venv"]))
        try:
            r = subprocess.run(
                command, shell=True, cwd=grade, env=hard_env(spec, grade), capture_output=True, timeout=600
            )
            results.append(r.returncode == 0)
        except subprocess.TimeoutExpired:
            results.append(False)
    return results[0], results[1]


def run(task: str, arm: str, rep: int) -> dict:
    hard = task in HARD
    spec = HARD[task] if hard else TASKS[task]
    with tempfile.TemporaryDirectory() as tmp:
        work, state = Path(tmp) / "work", Path(tmp) / "state"
        state.mkdir()
        if hard:
            shutil.copytree(spec["root"] / "tasks" / task / "workspace", work)
        else:
            work.mkdir()
            for name, content in spec["files"].items():
                (work / name).write_text(content)
        command = f"{sys.executable} {HOOK} {'nudge' if arm == 'nudge' else 'log'} {state}"
        hooks = [{"matcher": "*", "hooks": [{"type": "command", "command": command, "timeout": 30}]}]
        settings = json.dumps({"hooks": {"PostToolUse": hooks, "PostToolUseFailure": hooks}})
        cmd = ["claude", "-p", spec["prompt"], "--model", "sonnet", "--output-format", "stream-json", "--verbose",
               "--settings", settings, "--allowedTools", *ALLOWED, "--disallowedTools", *DENIED]  # fmt: skip
        start = time.time()
        proc = subprocess.Popen(
            cmd,
            cwd=work,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            text=True,
            env=hard_env(spec, work) if hard else None,
        )
        timer = threading.Timer(HARD_LIMIT if hard else LIMIT, lambda: proc.poll() is None and proc.kill())
        timer.start()
        events = [json.loads(line) for line in proc.stdout if line.startswith("{")]
        proc.wait()
        timer.cancel()
        end = time.time()
        result = next((e for e in events if e.get("type") == "result"), {})
        tools = sum(
            1
            for e in events
            if e.get("type") == "assistant"
            for b in e["message"].get("content") or []
            if b.get("type") == "tool_use"
        )
        failed = sum(
            1
            for e in events
            if e.get("type") == "user"
            for b in (e.get("message") or {}).get("content") or []
            if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("is_error")
        )
        states = [json.loads(p.read_text()) for p in state.glob("*.json")]
        fires = sorted((f for s in states for f in s["fires"]), key=lambda f: f["time"])
        grade = Path(tmp) / "grade"
        regression = None
        if hard:
            target_ok, regression = grade_hard(spec, work, grade)
            passed = target_ok and regression
        else:
            shutil.copytree(work, grade)
            (grade / "hidden_test.py").write_text(spec["hidden"])
            try:
                check = [sys.executable, "hidden_test.py"]
                passed = subprocess.run(check, cwd=grade, capture_output=True, timeout=120).returncode == 0
            except subprocess.TimeoutExpired:
                passed = False
        first = fires[0] if fires else None
        return {
            "task": task,
            "arm": arm,
            "rep": rep,
            "passed": passed,
            "regression_ok": regression,
            "seconds": round(end - start, 1),
            "end": result.get("terminal_reason") or result.get("subtype") or "killed",
            "tools": tools,
            "failed": failed,
            "output_tokens": (result.get("usage") or {}).get("output_tokens", 0),
            "fired": first is not None,
            "fire_reason": first and first["reason"],
            "failed_after_fire": failed - first["errors"] if first else None,
            "seconds_after_fire": round(end - first["time"], 1) if first else None,
            "nudges": sum(s.get("nudges", 0) for s in states),
            "result_text": (result.get("result") or "")[:1500],
        }


def analyze(records: list[dict]) -> None:
    arms = ("control", "nudge")
    print(f"{'task':15} " + " ".join(f"{a:>8}" for a in arms))
    fp_tasks, totals = [], dict.fromkeys(arms, 0)
    for task in dict.fromkeys(r["task"] for r in records):
        passed = {a: sum(r["passed"] for r in records if r["task"] == task and r["arm"] == a) for a in arms}
        runs = {a: sum(1 for r in records if r["task"] == task and r["arm"] == a) for a in arms}
        for a in arms:
            totals[a] += passed[a]
        if passed["control"] >= 2 and passed["nudge"] <= 1:
            fp_tasks.append(task)
        print(f"{task:15} " + " ".join(f"{passed[a]}/{runs[a]:<6}" for a in arms))
    print(f"total passed: control {totals['control']}, nudge {totals['nudge']}; FP tasks {fp_tasks}")
    b1 = not fp_tasks and totals["nudge"] >= totals["control"] - 1
    print(f"B1 correct answers kept: {'PASS' if b1 else 'FAIL'}")
    for a in arms:
        rs = [r for r in records if r["arm"] == a]
        mean = lambda k, rows=rs: sum(r[k] for r in rows) / len(rows) if rows else 0  # noqa: E731
        print(f"  {a:8} runs {len(rs)}: tools {mean('tools'):.1f}, failed {mean('failed'):.1f}, "
              f"seconds {mean('seconds'):.0f}, output tokens {mean('output_tokens'):.0f}, fired {sum(r['fired'] for r in rs)}")  # fmt: skip
    fired = {a: [r for r in records if r["arm"] == a and r["fired"]] for a in arms}
    if min(len(v) for v in fired.values()) < 6:
        print(f"B2: fewer than 6 fired runs in an arm ({ {a: len(v) for a, v in fired.items()} }): numbers only")
    ratios = {}
    for key in ("failed_after_fire", "seconds_after_fire"):
        means = {a: sum(r[key] for r in v) / len(v) if v else 0 for a, v in fired.items()}
        ratios[key] = means["nudge"] / means["control"] if means["control"] else None
        print(f"  after the signal, mean {key}: control {means['control']:.1f}, nudge {means['nudge']:.1f}")
    if min(len(v) for v in fired.values()) >= 6:
        ok = all(v is not None and v <= 0.8 for v in ratios.values())
        print(f"B2 less waste after the signal: {'PASS' if ok else 'FAIL'} ({ratios})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--hard", type=Path, help="manifest.json of SWE-bench-style tasks to run instead")
    args = parser.parse_args()
    if args.hard:
        load_hard(args.hard)
        if args.tasks == list(TASKS):
            args.tasks = list(HARD)
    records = [json.loads(line) for line in args.out.read_text().splitlines()] if args.out.exists() else []
    if not args.analyze:
        done = {(r["task"], r["arm"], r["rep"]) for r in records}
        jobs = [
            (t, a, i)
            for i in range(args.reps)
            for t in args.tasks
            for a in ("control", "nudge")
            if (t, a, i) not in done
        ]
        lock = threading.Lock()

        def job(item: tuple[str, str, int]) -> None:
            r = run(*item)
            with lock:
                records.append(r)
                with args.out.open("a") as fh:
                    fh.write(json.dumps(r) + "\n")
            print(f"{r['task']:15} {r['arm']:7} #{r['rep']} passed={r['passed']} {r['seconds']}s tools={r['tools']} "
                  f"failed={r['failed']} fired={r['fire_reason']} nudges={r['nudges']}", flush=True)  # fmt: skip

        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(job, jobs))
    analyze(records)


if __name__ == "__main__":
    main()
