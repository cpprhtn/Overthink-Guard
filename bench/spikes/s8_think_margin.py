import json, time, httpx
URL = "http://localhost:11434"; M = "qwen3:1.7b"; END = "</think>"
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
TAUS = (1.0, 2.0, 3.0, 5.0)
SUFFIX = "\n\nI have enough to answer now."

def full_run(q):
    toks = []  # (thinking_text_so_far_len, token, margin_at_this_position)
    thinking = content = ""; t0 = time.time()
    with httpx.stream("POST", URL + "/api/chat", json={"model": M, "think": True, "stream": True, "logprobs": True, "top_logprobs": 20,
            "messages": [{"role": "user", "content": q}]}, timeout=900) as r:
        for line in r.iter_lines():
            d = json.loads(line)
            if d.get("done"): fin = d; break
            thinking_before = thinking
            thinking += d["message"].get("thinking", ""); content += d["message"].get("content", "")
            for lp in d.get("logprobs") or []:
                top = lp["top_logprobs"]
                end_lp = next((t["logprob"] for t in top if t["token"] == END), None)
                margin = top[0]["logprob"] - end_lp if end_lp is not None else float("inf")
                toks.append((len(thinking_before), lp["token"], margin, d["message"].get("content", "") == ""))
    return thinking, content, toks, fin, time.time() - t0

def inject(q, thinking_prefix):
    content = ""
    with httpx.stream("POST", URL + "/api/chat", json={"model": M, "think": True, "stream": True, "messages": [
            {"role": "user", "content": q}, {"role": "assistant", "thinking": "\n" + thinking_prefix + SUFFIX, "content": ""}]}, timeout=900) as r:
        for line in r.iter_lines():
            d = json.loads(line); content += d["message"].get("content", "")
            if d.get("done"): return content, d

rows = []
for q, ans in PROBLEMS:
    thinking, content, toks, fin, secs = full_run(q)
    think_toks = [t for t in toks if t[3]]
    n_think = len(think_toks)
    full_ok = ans in content[-300:]
    res = {"q": q[:40], "think_tokens": n_think, "full_ok": full_ok, "secs": round(secs)}
    # margin evaluated at the token right after a sentence boundary inside thinking
    for tau in TAUS:
        hit = None
        for i in range(1, n_think):
            prev_tok = think_toks[i - 1][1]
            if prev_tok.rstrip(" ").endswith((".", "\n", "?", "!")) and think_toks[i][2] <= tau:
                hit = i; break
        if hit is None:
            res[tau] = None; continue
        cut_chars = think_toks[hit][0]
        ans_text, d = inject(q, thinking[:cut_chars].rstrip())
        res[tau] = {"at": hit / n_think, "ok": ans in ans_text[-300:], "cached": d["prompt_eval_cached_count"], "prompt": d["prompt_eval_count"]}
    rows.append(res)
    print(json.dumps(res, default=str), flush=True)

print("\nsummary")
print(f"full thinking correct: {sum(r['full_ok'] for r in rows)}/{len(rows)}")
for tau in TAUS:
    fired = [r for r in rows if r[tau]]
    saved = sum((1 - r[tau]["at"]) * r["think_tokens"] for r in fired) / sum(r["think_tokens"] for r in rows)
    print(f"tau={tau}: fired {len(fired)}/{len(rows)}, correct after inject {sum(r[tau]['ok'] for r in fired)}/{len(fired)}, "
          f"lost vs full {sum(r['full_ok'] and not r[tau]['ok'] for r in fired)}, saved {saved:.0%} of thinking tokens")
