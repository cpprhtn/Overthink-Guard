import json, time, httpx
from openai import OpenAI
P = "http://127.0.0.1:8487"
for _ in range(40):
    try: httpx.get(P + "/ui", timeout=1); break
    except httpx.HTTPError: time.sleep(0.5)
c = OpenAI(base_url=P + "/v1", api_key="x")
def run(q, gold, stop_after=None, sid_holder=[0]):
    reasoning = content = ""; otg = None; n = 0; t_stop = t_first = None; sid = None
    for ch in c.chat.completions.create(model="deepseek-r1:1.5b", stream=True, messages=[{"role": "user", "content": q}]):
        sid = sid or ch.id.rsplit("-", 1)[-1]
        otg = (ch.model_extra or {}).get("otg", otg)
        if not ch.choices: continue
        d = ch.choices[0].delta
        r = (d.model_extra or {}).get("reasoning") or ""
        reasoning += r; n += bool(r)
        if d.content:
            content += d.content
            t_first = t_first or time.time()
        if stop_after and n == stop_after and not t_stop:
            t_stop = time.time()
            print("   stop ->", httpx.post(f"{P}/otg/api/sessions/{sid}/stop", json={}).json())
    lag = f"{t_first - t_stop:.2f}s" if t_stop and t_first else "-"
    print(f"   reasoning {len(reasoning)} chars, content has {gold}: {gold in content}, stop->answer {lag}, otg {otg}")
    print(f"   content head {content[:90]!r}")
    return sid
print("1) LCM, probing on, no stop")
sid = run("What is the least common multiple of 12 and 18?", "36")
with httpx.stream("GET", P + "/otg/api/events", timeout=2) as r:
    try:
        for line in r.iter_lines():
            if line.startswith("data: "):
                e = json.loads(line[6:])
                if e["type"] == "probe": print("   probe", e["at_tokens"], e["answer"], f"{e['seconds']}s")
                if e["type"] == "session" and e["id"] == sid: print("   can_intervene", e["can_intervene"], "probe_skipped", e["probe_skipped"])
    except httpx.ReadTimeout: pass
print("2) divisors of 3600, Answer now after 150 reasoning chunks")
run("How many positive divisors does 3600 have?", "45", stop_after=150)
