import json, re, httpx
URL = "http://localhost:11434"; M = "deepseek-r1:1.5b"
def render(q): return f"<｜User｜>{q}<｜Assistant｜>"
def stream_gen(prompt, cap, n=None):
    body = {"model": M, "raw": True, "prompt": prompt, "stream": True}
    if n: body["options"] = {"num_predict": n}
    th = rs = ""; k = 0; first = None
    with httpx.stream("POST", URL + "/api/generate", json=body, timeout=None) as r:
        for line in r.iter_lines():
            d = json.loads(line)
            if first is None: first = {k: v for k, v in d.items() if k in ("thinking", "response")}
            th += d.get("thinking", ""); rs += d.get("response", ""); k += 1
            if d.get("done") or k >= cap: break
    return th, rs, first
def gen(prompt, n):
    return httpx.post(URL + "/api/generate", json={"model": M, "raw": True, "prompt": prompt, "stream": False, "options": {"num_predict": n}}, timeout=None).json()

for q, gold in [("What is the least common multiple of 12 and 18?", "36"),
                ("How many positive divisors does 3600 have?", "45"),
                ("What is the remainder when 2^10 is divided by 7?", "2")]:
    base = render(q)
    th, rs, first = stream_gen(base, cap=400)
    print(f"\nQ={q[:40]!r}: first chunk {first}; thinking {len(th)} chars, response {len(rs)} chars; thinking head {th[:40]!r}")
    for label, lead in (("'<think>\\n'+thinking", "<think>\n"), ("thinking only", "")):
        ans = gen(base + lead + th + "\n\nI have enough to answer now.\n</think>\n\n", 400)
        a = ans.get("response", "") + ans.get("thinking", "")
        print(f"  answer now [{label}]: cached {ans['prompt_eval_cached_count']}/{ans['prompt_eval_count']}, contains {gold}: {gold in a}, head {a[:60]!r}")
    pr = gen(base + "<think>\n" + th + "\n\nI have enough to answer now.\n</think>\n\nThe final answer is $\\boxed{", 16)
    m = re.match(r"\s*([^}]*)\}", pr.get("response", ""))
    print(f"  probe: cached {pr['prompt_eval_cached_count']}/{pr['prompt_eval_count']}, answer {m.group(1) if m else pr.get('response','')[:12]!r}, thinking field {pr.get('thinking','')[:20]!r}")
    th2, rs2, _ = stream_gen(base + "<think>\n" + th, cap=60)
    print(f"  resume: thinking {th2[:70]!r} | response {rs2[:40]!r}")
