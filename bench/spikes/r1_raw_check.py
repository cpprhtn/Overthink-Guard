import json, httpx
URL = "http://localhost:11434"; M = "deepseek-r1:1.5b"
Q = "What is the least common multiple of 12 and 18?"
thinking = ""
with httpx.stream("POST", URL + "/api/chat", json={"model": M, "stream": True, "messages": [{"role": "user", "content": Q}]}, timeout=None) as r:
    for line in r.iter_lines():
        thinking += json.loads(line)["message"].get("thinking", "")
        if len(thinking) >= 400: break
for name, prompt in [
    ("answer", f"<｜begin▁of▁sentence｜><｜User｜>{Q}<｜Assistant｜><think>\n{thinking}\n\nI have enough to answer now.\n</think>\n\n"),
    ("resume", f"<｜begin▁of▁sentence｜><｜User｜>{Q}<｜Assistant｜><think>\n{thinking}"),
]:
    out = ""; fin = None
    with httpx.stream("POST", URL + "/api/generate", json={"model": M, "raw": True, "prompt": prompt, "stream": True, "options": {"num_predict": 200}}, timeout=None) as r:
        for line in r.iter_lines():
            d = json.loads(line); out += d.get("response", "")
            if d.get("done"): fin = d
    print(f"{name}: cached {fin['prompt_eval_cached_count']}/{fin['prompt_eval_count']}; says 36: {'36' in out}; head {out[:110]!r}")
