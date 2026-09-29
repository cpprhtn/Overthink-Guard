import json, re, time, httpx

URL = "http://localhost:11434"
M = "qwen3:1.7b"
PROBLEMS = [
    ("How many positive divisors does 3600 have?", "45"),
    ("What is the sum of the first 50 positive odd integers?", "2500"),
    ("A train travels 180 km in 2.5 hours. What is its average speed in km/h?", "72"),
    ("In how many ways can 5 people be seated in a row?", "120"),
    ("What is the remainder when 2^10 is divided by 7?", "2"),
    ("If 3x + 7 = 25, what is x?", "6"),
    ("What is the least common multiple of 12 and 18?", "36"),
    ("How many prime numbers are less than 30?", "10"),
]
SUFFIX = "\n\nI have enough to answer now."


def full(q):
    thinking = ""
    with httpx.stream(
        "POST",
        URL + "/api/chat",
        json={"model": M, "think": True, "stream": True, "messages": [{"role": "user", "content": q}]},
        timeout=900,
    ) as r:
        for line in r.iter_lines():
            d = json.loads(line)
            thinking += d["message"].get("thinking", "")
            if d.get("done"):
                return thinking, d["eval_count"]


def probe(q, prefix):
    t0 = time.time()
    d = httpx.post(
        URL + "/api/chat",
        json={
            "model": M,
            "think": True,
            "stream": False,
            "options": {"num_predict": 12},
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "thinking": "\n" + prefix + SUFFIX, "content": "The final answer is $\\boxed{"},
            ],
        },
        timeout=900,
    ).json()
    m = re.match(r"\s*([^}]*)\}", d["message"]["content"])
    return (
        (m.group(1).strip() if m else d["message"]["content"][:12]),
        time.time() - t0,
        d["prompt_eval_cached_count"],
        d["prompt_eval_count"],
    )


total_think = total_saved = 0
results = []
for q, gold in PROBLEMS:
    thinking, n_tokens = full(q)
    answers = []
    probe_secs = 0
    stop_frac = None
    for step in range(1, 10):
        cut = thinking[: int(len(thinking) * step / 10)]
        cut = cut[: cut.rfind("\n") + 1] or cut
        a, secs, cached, prompt = probe(q, cut.rstrip())
        answers.append(a)
        probe_secs += secs
        if stop_frac is None and len(answers) >= 2 and answers[-1] == answers[-2]:
            stop_frac = step / 10
            stop_ans = a
    ok = stop_frac is not None and stop_ans == gold
    first_gold = next((i + 1 for i, a in enumerate(answers) if a == gold), None)
    results.append((q[:34], gold, answers, stop_frac, ok))
    total_think += n_tokens
    total_saved += (1 - stop_frac) * n_tokens if stop_frac else 0
    print(
        f"{q[:34]:34s} gold={gold:5s} probes@10..90%={answers} | first gold @{first_gold and first_gold * 10}% | "
        f"stop(k=2)@{stop_frac} ok={ok} | 9 probes took {probe_secs:.1f}s (last cached {cached}/{prompt})",
        flush=True,
    )
fired = [r for r in results if r[3]]
print(
    f"\nk=2 agreement rule: fired {len(fired)}/8, correct {sum(r[4] for r in fired)}/{len(fired)}, "
    f"saved {total_saved / total_think:.0%} of generated tokens (probe cost excluded)"
)
