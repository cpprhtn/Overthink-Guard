import json, httpx

URL = "http://localhost:11434"
M = "qwen3:1.7b"
Q = "How many positive divisors does 3600 have?"


def gen_and_abort(chars):
    thinking = ""
    with httpx.stream(
        "POST",
        URL + "/api/chat",
        json={"model": M, "think": True, "stream": True, "messages": [{"role": "user", "content": Q}]},
        timeout=600,
    ) as r:
        for line in r.iter_lines():
            thinking += json.loads(line)["message"].get("thinking", "")
            if len(thinking) >= chars:
                break
    return thinking


def chat_prefill(thinking):
    with httpx.stream(
        "POST",
        URL + "/api/chat",
        json={
            "model": M,
            "think": True,
            "stream": True,
            "messages": [{"role": "user", "content": Q}, {"role": "assistant", "thinking": thinking, "content": ""}],
        },
        timeout=600,
    ) as r:
        for line in r.iter_lines():
            d = json.loads(line)
            if d.get("done"):
                return d


def raw_prefill(prompt):
    with httpx.stream(
        "POST", URL + "/api/generate", json={"model": M, "raw": True, "stream": True, "prompt": prompt}, timeout=600
    ) as r:
        for line in r.iter_lines():
            d = json.loads(line)
            if d.get("done"):
                return d


def show(name, d):
    print(
        f"  {name:48s} prompt {d['prompt_eval_count']:5d} cached {d['prompt_eval_cached_count']:5d} ({d['prompt_eval_duration'] / 1e6:.0f}ms)"
    )


suffix = "\n\nI have enough to answer now."
for name, make in [
    ("chat prefill, thinking as received", lambda t: ("chat", t + suffix)),
    ("chat prefill, '\\n' + thinking", lambda t: ("chat", "\n" + t + suffix)),
    (
        "raw: '<think>\\n' + thinking",
        lambda t: (
            "raw",
            f"<|im_start|>user\n{Q}<|im_end|>\n<|im_start|>assistant\n<think>\n{t}{suffix}\n</think>\n\n",
        ),
    ),
]:
    t = gen_and_abort(2000)
    kind, arg = make(t)
    show(name, chat_prefill(arg) if kind == "chat" else raw_prefill(arg))
