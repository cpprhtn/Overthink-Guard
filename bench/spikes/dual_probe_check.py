import json, re, glob, sys, httpx
from overthink_guard.analysis import read_probe_answer
from overthink_guard.backends import requests_for, to_native_chat
from overthink_guard.templates import template_for_model
OLLAMA = "http://localhost:11434"
src = open("bench/spikes/probe_rule_sim_all.py").read().split("samples = []")[0]
ns = {}; exec(src, ns)

def plain_probe(model, question, thinking):
    native = to_native_chat({"model": model, "stream": True, "messages": [{"role": "user", "content": question}]})
    t = template_for_model(model)
    req = requests_for(native, t)
    call = req.probe(thinking)
    body = json.loads(json.dumps(call.body))
    if "messages" in body:
        body["messages"][-1]["content"] = "Final answer (one line): "
    else:
        body["prompt"] = body["prompt"][: -len(t.probe_answer_prefix)] + "Final answer (one line): "
    body["options"]["num_predict"] = 24
    r = httpx.post(OLLAMA + call.path, json=body, timeout=None).json()
    return (r.get("message") or {}).get("content") or r.get("response", "")

def boxed_probe(model, question, thinking):
    native = to_native_chat({"model": model, "stream": True, "messages": [{"role": "user", "content": question}]})
    call = requests_for(native, template_for_model(model)).probe(thinking)
    r = httpx.post(OLLAMA + call.path, json=call.body, timeout=None).json()
    return read_probe_answer((r.get("message") or {}).get("content") or r.get("response", ""))

def agrees(boxed, plain):
    if not boxed: return False
    text = re.sub(r"\s+", "", plain.lower().replace("\\boxed{", "").replace("$", "").replace("*", ""))
    return boxed in text[: max(len(boxed) * 3, 30)]

mode = sys.argv[1]
if mode == "math":
    rows = []
    for f in sorted(glob.glob(sys.argv[2] + "/cache/openr1_math_*.json"), key=lambda p: int(re.search(r"_(\d+)\.json", p)[1])):
        rows += [r["row"] for r in json.load(open(f))["rows"]]
    mc = re.compile(r"\(A\)|\bA[.)]\s|\(a\)|options|choices", re.I)
    short = [(r["problem"], r["answer"].strip()) for r in rows if re.fullmatch(r"-?\d+", r["answer"].strip()) and len(r["problem"]) < 260]
    held = [p for p in short[12:] if not mc.search(p[0])]
    total = ok = 0
    for f, start, model in (("bench/data/shadow_fifth_qwen3_2026-09-30.json", 70, "qwen3:1.7b"), ("bench/data/shadow_sixth_r1_2026-09-30.json", 40, "deepseek-r1:1.5b")):
        d = json.load(open(f))
        for i, p in enumerate(d["problems"]):
            s = ns["stop"](p, 4, 6000, None)
            if not s: continue
            q = held[start + i][0]
            thinking = "".join(t for a, e, t in p["segments"] if e <= s[0])
            b = boxed_probe(model, q, thinking); pl = plain_probe(model, q, thinking)
            total += 1; ok += agrees(b, pl)
            print(f"  {model[:5]} #{i:2} boxed={b!r:12} plain={pl[:60]!r} agree={agrees(b, pl)}", flush=True)
    print(f"MATH agreement: {ok}/{total}")
else:
    qs = ["Design a URL shortener that handles 10k writes per second. Outline the components and the data model.",
          "How would you structure the database schema for a multi-tenant SaaS invoicing system?",
          "Design a rate limiter for a public API that runs on multiple servers.",
          "Plan logging and alerting for a small team running 20 services.",
          "월 100만 명이 쓰는 쿠폰 발급 시스템에서 같은 사용자에게 쿠폰이 중복 발급되지 않도록 하는 설계를 제안해줘."]
    total = ok = 0
    for model in ("qwen3:1.7b", "deepseek-r1:1.5b"):
        for q in qs:
            native = to_native_chat({"model": model, "stream": True, "messages": [{"role": "user", "content": q}]})
            call = requests_for(native, template_for_model(model)).original()
            thinking = ""; n = 0
            with httpx.stream("POST", OLLAMA + call.path, json=call.body, timeout=None) as r:
                for line in r.iter_lines():
                    c = json.loads(line); t = (c.get("message") or {}).get("thinking") or c.get("thinking") or ""
                    thinking += t; n += bool(t)
                    if n >= 700 or c.get("done"): break
            b = boxed_probe(model, q, thinking); pl = plain_probe(model, q, thinking)
            total += 1; ok += agrees(b, pl)
            print(f"  {model[:5]} boxed={b!r:12} plain={pl[:70]!r} agree={agrees(b, pl)}", flush=True)
    print(f"OPEN agreement: {ok}/{total}")
