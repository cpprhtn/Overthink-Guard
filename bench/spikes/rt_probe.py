import asyncio, json, re, time, httpx
URL = "http://localhost:11434"; M = "qwen3:1.7b"
Q = "How many positive divisors does 3600 have?"
USER = [{"role": "user", "content": Q}]
EVERY = 400
SUFFIX = "\n\nI have enough to answer now."

async def probe(client, thinking):
    t0 = time.time()
    r = await client.post(URL + "/api/chat", json={"model": M, "think": True, "stream": False, "options": {"num_predict": 12},
        "messages": USER + [{"role": "assistant", "thinking": "\n" + thinking.rstrip() + SUFFIX, "content": "The final answer is $\\boxed{"}]})
    d = r.json(); m = re.match(r"\s*([^}]*)\}", d["message"]["content"])
    return (m.group(1) if m else d["message"]["content"][:10]), time.time() - t0, d["prompt_eval_cached_count"], d["prompt_eval_count"]

async def baseline(client):
    t0 = time.time(); n = 0
    async with client.stream("POST", URL + "/api/chat", json={"model": M, "stream": True, "messages": USER}) as r:
        async for line in r.aiter_lines():
            d = json.loads(line)
            if d["message"].get("thinking"): n += 1
            if d["message"].get("content"): break
    dt = time.time() - t0
    print(f"[baseline] thinking {n} tokens in {dt:.1f}s -> {n/dt:.0f} tok/s")
    return n / dt

async def parallel(client):
    t0 = time.time(); n = 0; thinking = ""; tasks = []
    async with client.stream("POST", URL + "/api/chat", json={"model": M, "stream": True, "messages": USER}) as r:
        async for line in r.aiter_lines():
            d = json.loads(line); th = d["message"].get("thinking", "")
            if th:
                n += 1; thinking += th
                if n % EVERY == 0:
                    tasks.append((n, asyncio.create_task(probe(client, thinking))))
            if d["message"].get("content"): break
    dt = time.time() - t0
    print(f"[P parallel] thinking {n} tokens in {dt:.1f}s -> {n/dt:.0f} tok/s while probing")
    for at, t in tasks:
        a, lat, cached, prompt = await t
        print(f"   probe@{at}: answer={a!r} latency {lat:.2f}s cached {cached}/{prompt}")

async def pause_resume(client):
    t0 = time.time(); thinking = ""; n = 0; probes = []; resumed_fields = set(); probe_time = 0
    first = True
    while True:
        if first:
            body = {"model": M, "stream": True, "messages": USER}
        else:
            body = {"model": M, "stream": True, "messages": USER + [{"role": "assistant", "content": "<think>\n" + thinking}]}
        finished = False; paused = False; tail = ""
        async with client.stream("POST", URL + "/api/chat", json=body) as r:
            async for line in r.aiter_lines():
                d = json.loads(line); msg = d["message"]
                if d.get("done"):
                    if not first: resume_stats = (d.get("prompt_eval_cached_count"), d.get("prompt_eval_count"))
                    finished = True; break
                th = msg.get("thinking", ""); ct = msg.get("content", "")
                if not first:
                    if th: resumed_fields.add("thinking")
                    if ct: resumed_fields.add("content")
                    # after resume Ollama may route continued thinking to content; stop at the closing tag
                    tail += ct
                    if "</think>" in tail:
                        thinking += tail.split("</think>")[0]; finished = True; break
                    th = th or ct
                if first and ct: finished = True; break
                if th:
                    thinking += th; n += 1
                    if n % EVERY == 0:
                        paused = True; break
        if paused:
            p = await probe(client, thinking); probes.append((n,) + p); probe_time += p[1]
            first = False
            continue
        break
    dt = time.time() - t0
    print(f"[S pause/resume] thinking ~{n} tokens in {dt:.1f}s, {len(probes)} probes took {probe_time:.1f}s; resumed output routed to: {resumed_fields}")
    for at, a, lat, cached, prompt in probes:
        print(f"   probe@{at}: answer={a!r} latency {lat:.2f}s cached {cached}/{prompt}")

async def main():
    async with httpx.AsyncClient(timeout=None) as client:
        await baseline(client)
        await parallel(client)
        await pause_resume(client)
asyncio.run(main())
