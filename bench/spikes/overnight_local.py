"""Overnight local run for docs/validation/local-overnight.md: probe trails on fresh, hard-first OpenR1 problems.

Starts `otg start --probe` in Shadow mode (nothing is stopped; probes are recorded every 400 thinking tokens from 400),
then alternates the models problem by problem until --hours runs out or the problems do. Each run is saved as soon as
it finishes, and rerunning with the same --out-dir resumes. Analyse with:
    python bench/spikes/probe_rule_matrix.py --files bench/data/overnight_*.json --overnight

Needs `ollama serve` running with the models pulled. Usage: python bench/spikes/overnight_local.py [--hours 8]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from shadow_validate import collect, held_out

ROOT = Path(__file__).parents[2]
FIRST_FRESH = 160  # held-out problems 0-159 were used by earlier samples (bench/data notes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--models", nargs="+", default=["qwen3:1.7b", "deepseek-r1:1.5b"])
    parser.add_argument("--max-tokens", type=int, default=24000, help="twice the old 12k cap, to see hidden losses")
    parser.add_argument("--cache", type=Path, default=ROOT / ".bench-cache")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "bench" / "data")
    parser.add_argument("--backend-url", default="http://localhost:11434")
    parser.add_argument("--port", type=int, default=8487)
    args = parser.parse_args()

    tags = httpx.get(f"{args.backend_url}/api/tags", timeout=5).json()
    missing = set(args.models) - {m["name"] for m in tags["models"]}
    if missing:
        sys.exit(f"pull these models first: {', '.join(sorted(missing))}")
    problems = held_out(args.cache, 12, FIRST_FRESH, 100000, hard_first=True)
    if len(problems) < 100:
        sys.exit(f"only {len(problems)} fresh problems in {args.cache}; run bench/r1_replay.py once to fill the cache")

    stamp = time.strftime("%Y-%m-%d")
    outs, data = {}, {}
    for model in args.models:
        tag = model.split(":")[0].replace("deepseek-r1", "r1")
        existing = sorted(args.out_dir.glob(f"overnight_{tag}_*.json"))
        outs[model] = existing[-1] if existing else args.out_dir / f"overnight_{tag}_{stamp}.json"
        data[model] = (
            json.loads(outs[model].read_text())
            if outs[model].exists()
            else {
                "note": f"{model}, fresh held-out OpenR1 problems from {FIRST_FRESH}, hardest first by mean R1 "
                f"solution length; Shadow probes every 400 tokens from 400; max_tokens {args.max_tokens}",
                "problems": [],
                "ungradable": [],
            }
        )

    proxy = f"http://127.0.0.1:{args.port}"
    server = subprocess.Popen(
        [sys.executable, "-m", "overthink_guard.cli", "start", "--port", str(args.port), "--backend-url",
         args.backend_url, "--probe", "--probe-min-tokens", "400", "--probe-interval", "400", "--no-stats"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        for _ in range(50):
            try:
                httpx.get(f"{proxy}/otg/api/stats", timeout=1)
                break
            except httpx.HTTPError:
                time.sleep(0.2)
        deadline = time.time() + args.hours * 3600
        while time.time() < deadline:
            model = min(args.models, key=lambda m: len(data[m]["problems"]))
            i = len(data[model]["problems"])
            if i >= len(problems):
                break
            question, gold = problems[i]
            print(f"[{time.strftime('%H:%M:%S')}] {model} #{i}", end=" ", flush=True)
            collect(proxy, model, question, gold, args.max_tokens, data[model], outs[model])
    finally:
        server.terminate()
        server.wait(timeout=10)
    print("done:", {m: len(d["problems"]) for m, d in data.items()}, "->", [str(p) for p in outs.values()])


if __name__ == "__main__":
    main()
