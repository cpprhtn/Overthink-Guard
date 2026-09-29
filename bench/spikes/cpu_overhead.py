import json, re, sys, time, httpx
URL = "http://localhost:11434"; M = "qwen3:1.7b"
Q = "How many positive divisors does 3600 have?"
USER = [{"role": "user", "content": Q}]
CPU = {"num_gpu": 0}
LIMIT, EVERY = 2000, 400
SUFFIX = "\n\nI have enough to answer now."

def run_stream(body, until, count_from):
    n = count_from; text = ""; done = False
    with httpx.stream("POST", URL + "/api/chat", json=body, timeout=None) as r:
        for line in r.iter_lines():
            d = json.loads(line); m = d["message"]
            piece = m.get("thinking") or m.get("content") or ""
            if piece:
                n += 1; text += piece
                if n >= until: break
            if d.get("done"): done = True; break
    return n, text, done

def probe(thinking):
    t0 = time.time()
    d = httpx.post(URL + "/api/chat", timeout=None, json={"model": M, "think": True, "stream": False,
        "options": {**CPU, "num_predict": 16, "temperature": 0},
        "messages": USER + [{"role": "assistant", "thinking": "\n" + thinking.rstrip() + SUFFIX, "content": "The final answer is $\\boxed{"}]}).json()
    return time.time() - t0, d["prompt_eval_cached_count"], d["prompt_eval_count"], d["message"]["content"][:10]

# warm the CPU-placed model
httpx.post(URL + "/api/chat", timeout=None, json={"model": M, "stream": False, "options": {**CPU, "num_predict": 1}, "messages": [{"role": "user", "content": "hi"}]})

t0 = time.time(); n, _, _ = run_stream({"model": M, "stream": True, "options": CPU, "messages": USER}, LIMIT, 0)
base = time.time() - t0
print(f"baseline CPU: {n} tokens in {base:.1f}s ({n/base:.1f} tok/s)")

t0 = time.time(); thinking = ""; n = 0; probe_time = 0; first = True
while n < LIMIT:
    target = min(n + EVERY, LIMIT)
    if first:
        body = {"model": M, "stream": True, "options": CPU, "messages": USER}
    else:
        body = {"model": M, "stream": True, "options": CPU, "messages": USER + [{"role": "assistant", "content": "<think>\n" + thinking}]}
    n, piece, done = run_stream(body, target, n); thinking += piece; first = False
    if done or "</think>" in piece:
        break
    if n < LIMIT:
        dt, cached, prompt, ans = probe(thinking); probe_time += dt
        print(f"  probe@{n}: {dt:.2f}s cached {cached}/{prompt} answer {ans!r}")
total = time.time() - t0
print(f"with probes: {n} tokens in {total:.1f}s; probes {probe_time:.1f}s; overhead vs baseline {(total-base)/base:+.1%}")
