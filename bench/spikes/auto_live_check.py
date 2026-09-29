import json, sys, time, glob, re, httpx
from openai import OpenAI
def wait(port):
    for _ in range(40):
        try: httpx.get(f"http://127.0.0.1:{port}/ui", timeout=1); return
        except httpx.HTTPError: time.sleep(0.5)
def run(port, model, q):
    c = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="x")
    t0 = time.time(); reasoning = content = ""; otg = None
    for ch in c.chat.completions.create(model=model, stream=True, messages=[{"role": "user", "content": q}]):
        otg = (ch.model_extra or {}).get("otg", otg)
        if not ch.choices: continue
        d = ch.choices[0].delta
        reasoning += (d.model_extra or {}).get("reasoning") or ""
        content += d.content or ""
    return time.time() - t0, len(reasoning), content, otg
wait(8488); wait(8487)
dt, rl, content, otg = run(8488, "qwen3:1.7b", "How many positive divisors does 3600 have?")
print(f"1) wiring (qwen3, min 800, k=2): {dt:.0f}s, reasoning {rl} chars, otg {otg}, says 45: {'45' in content}")

rows = []
for f in sorted(glob.glob(sys.argv[1] + "/cache/openr1_math_*.json"), key=lambda p: int(re.search(r"_(\d+)\.json", p)[1])):
    rows += [r["row"] for r in json.load(open(f))["rows"]]
mc = re.compile(r"\(A\)|\bA[.)]\s|\(a\)|options|choices", re.I)
short = [(r["problem"], r["answer"].strip()) for r in rows if re.fullmatch(r"-?\d+", r["answer"].strip()) and len(r["problem"]) < 260]
held = [p for p in short[12:] if not mc.search(p[0])]
q, gold = held[0]   # deepseek-r1 hit the 12k cap on this one in the 4th sample
dt, rl, content, otg = run(8487, "deepseek-r1:1.5b", q)
print(f"2) runaway guard (deepseek-r1, defaults): {dt:.0f}s, reasoning {rl} chars, otg {otg}, gold {gold} in answer tail: {gold in content[-300:]}")
print("   answer tail:", repr(content[-160:]))
