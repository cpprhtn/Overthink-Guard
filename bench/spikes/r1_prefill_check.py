import json, time, httpx
URL = "http://localhost:11434"; M = "deepseek-r1:1.5b"
USER = [{"role": "user", "content": "What is the least common multiple of 12 and 18?"}]
def stream(body, cap=None):
    th = ct = ""; fin = None
    with httpx.stream("POST", URL + "/api/chat", json=body, timeout=None) as r:
        if r.status_code != 200:
            r.read(); return None, r.status_code, r.text[:200]
        for line in r.iter_lines():
            d = json.loads(line); th += d["message"].get("thinking", ""); ct += d["message"].get("content", "")
            if d.get("done"): fin = d; break
            if cap and len(th) + len(ct) >= cap: break
    return (th, ct, fin), 200, ""
print("1) plain stream (no think param)")
(th, ct, fin), st, err = stream({"model": M, "stream": True, "messages": USER}, cap=800)
print(f"   thinking-field chars {len(th)}, content chars {len(ct)}; content starts {ct[:80]!r}")
partial = th or ct.split("</think>")[0].replace("<think>", "")
print("2) answer prefill via thinking field, think=true")
res, st, err = stream({"model": M, "stream": True, "think": True, "messages": USER + [{"role": "assistant", "content": "", "thinking": "\n" + partial + "\n\nI have enough to answer now."}]})
if res:
    th2, ct2, fin2 = res
    print(f"   new thinking {len(th2)} chars, content {len(ct2)} chars, cached {fin2 and fin2.get('prompt_eval_cached_count')}/{fin2 and fin2.get('prompt_eval_count')}; content head {ct2[:100]!r}; thinking head {th2[:80]!r}")
else:
    print(f"   HTTP {st}: {err}")
print("3) resume via content '<think>\\n' + thinking")
res, st, err = stream({"model": M, "stream": True, "messages": USER + [{"role": "assistant", "content": "<think>\n" + partial}]}, cap=600)
if res:
    th3, ct3, fin3 = res
    print(f"   thinking {len(th3)} chars, content {len(ct3)} chars; head {(th3 or ct3)[:120]!r}")
else:
    print(f"   HTTP {st}: {err}")
