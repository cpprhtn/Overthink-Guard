"""Runs design-style prompts through a `otg start --probe --probe-min-tokens 0` proxy and, where the runaway guard
would have stopped, generates the answer it would have produced from the same run's thinking.

Usage: python bench/spikes/design_eval.py --model qwen3:1.7b --out design_qwen3.json
"""

import argparse
import json
from pathlib import Path

import httpx
from openai import OpenAI

from overthink_guard.analysis.signals import extract_boxed, normalize_answer
from overthink_guard.backends import requests_for, to_native_chat
from overthink_guard.templates import template_for_model

BOX = " End your answer with only the choice in \\boxed{}."
PROMPTS = [
    # open-ended system and data design
    ("open", "Design a URL shortener that handles 10k writes per second. Outline the components and the data model."),
    ("open", "How would you structure the database schema for a multi-tenant SaaS invoicing system?"),
    ("open", "Propose a caching strategy for a read-heavy product catalog API whose prices change several times a day."),
    ("open", "Design a rate limiter for a public API that runs on multiple servers."),
    ("open", "How would you move the user service out of a monolith into its own service without downtime?"),
    ("open", "Design an offline-first sync protocol for a mobile note-taking app, including conflict resolution."),
    ("open", "Plan logging and alerting for a small team running 20 services."),
    ("open", "Design retries and idempotency for a consumer of payment webhooks."),
    # design decisions with a discrete choice
    ("choice", "A chat app needs per-conversation message ordering and 100k concurrent users. Which primary store fits "
     "best: PostgreSQL, Cassandra, or Redis?" + BOX),
    ("choice", "A team of 3 is building an internal CRUD admin tool used by 20 people. Monolith or microservices?" + BOX),
    ("choice", "You need to run a nightly job that must not run twice concurrently across 5 servers. Use a database "
     "advisory lock, a cron entry on every server, or a message queue with 5 consumers?" + BOX),
    ("choice", "For storing user passwords, which is appropriate: SHA-256, bcrypt, or AES-256?" + BOX),
    ("choice", "A mobile app must show a live-updating stock price every second. WebSocket, long polling, or a REST "
     "call every second?" + BOX),
    ("choice", "An e-commerce order table has 500M rows and queries always filter by customer_id and created_at. "
     "Index on (created_at, customer_id) or (customer_id, created_at)?" + BOX),
    ("choice", "To share configuration secrets between 30 services, use environment variables baked into images, a "
     "secrets manager, or a shared Git repo?" + BOX),
    ("choice", "A search box should query the backend as the user types. Debounce, throttle, or query on every "
     "keystroke?" + BOX),
    # code reasoning with a short answer
    ("code", "Which line has the bug?\n1 def average(xs):\n2     total = 0\n3     for x in xs:\n4         total += x\n"
     "5     return total / len(xs) + 1\nEnd your answer with the line number in \\boxed{}."),
    ("code", "Which line has the bug?\n1 def is_palindrome(s):\n2     s = s.lower()\n3     for i in range(len(s)):\n"
     "4         if s[i] != s[len(s) - i]:\n5             return False\n6     return True\n"
     "End your answer with the line number in \\boxed{}."),
    ("code", "What does this print?\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(len(x))\n"
     "End your answer with the number in \\boxed{}."),
    ("code", "Which line has the bug?\n1 def find_max(xs):\n2     best = 0\n3     for x in xs:\n4         if x > best:\n"
     "5             best = x\n6     return best\n(xs may contain only negative numbers.) "
     "End your answer with the line number in \\boxed{}."),
    # Korean design prompts
    ("open", "월 100만 명이 쓰는 쿠폰 발급 시스템에서 같은 사용자에게 쿠폰이 중복 발급되지 않도록 하는 설계를 제안해줘."),
    ("open", "사내 게시판 서비스의 검색 기능을 설계해줘. 게시글은 약 50만 건이고 한국어 검색이 중요해."),
    ("choice", "하루 주문 1천 건인 쇼핑몰의 재고 차감을 구현하려고 해. 비관적 락, 낙관적 락, 분산 락(Redis) 중 무엇이 "
     "적합할까? 답의 마지막에 선택지 하나만 \\boxed{} 안에 써줘."),
    ("open", "레거시 PHP 서비스를 Kotlin 백엔드로 점진적으로 옮기는 전략을 세워줘."),
]
GOLD = {16: "5", 17: "4", 18: "4", 19: "2"}


def boxed(text: str) -> str | None:
    found = extract_boxed(text)
    return normalize_answer(found[-1][1]) if found else None


def rule_c(probes: list, k: int = 4, min_tokens: int = 6000):
    trail = [(t, a) for t, a in probes if t >= min_tokens]
    for j in range(k - 1, len(trail)):
        window = [a for _, a in trail[j - k + 1 : j + 1]]
        if window[0] is not None and len(set(window)) == 1:
            return trail[j]
    return None


def session_state(proxy: str, session: str) -> tuple[dict, list, list]:
    summary, probes, segments = {}, [], []
    try:
        with httpx.stream("GET", f"{proxy}/otg/api/events", timeout=1.5) as r:
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("session") == session and event["type"] == "probe":
                    probes.append([event["at_tokens"], event["answer"]])
                elif event.get("session") == session and event["type"] == "segment":
                    segments.append([event["start_tokens"], event["end_tokens"], event["text"]])
                elif event["type"] == "session" and event["id"] == session:
                    summary = event
    except httpx.ReadTimeout:
        pass
    return summary, probes, sorted(segments)


def answer_at(ollama: str, model: str, question: str, thinking: str, generated: int) -> str:
    """The answer Auto would have produced: the same injection Answer now uses, from the same run's thinking."""
    native = to_native_chat({"model": model, "stream": True, "messages": [{"role": "user", "content": question}]})
    call = requests_for(native, template_for_model(model)).answer(thinking, generated)
    text = ""
    with httpx.stream("POST", ollama + call.path, json=call.body, timeout=None) as r:
        for line in r.iter_lines():
            chunk = json.loads(line)
            text += (chunk.get("message") or {}).get("content", "") or chunk.get("response", "")
            if chunk.get("done"):
                break
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", default="http://127.0.0.1:8485")
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    client = OpenAI(base_url=f"{args.proxy}/v1", api_key="unused")
    results = []
    for i, (category, prompt) in enumerate(PROMPTS):
        content, session, finish = "", None, None
        messages = [{"role": "user", "content": prompt}]
        for chunk in client.chat.completions.create(
            model=args.model, stream=True, messages=messages, max_tokens=args.max_tokens
        ):
            session = session or chunk.id.rsplit("-", 1)[-1]
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            if chunk.choices and chunk.choices[0].finish_reason:
                finish = chunk.choices[0].finish_reason
        summary, probes, segments = session_state(args.proxy, session)
        thinking_tokens = summary.get("thinking_tokens")
        stop = rule_c(probes)
        auto_answer = None
        if stop:
            thinking = "".join(t for s, e, t in segments if e <= stop[0])
            auto_answer = answer_at(args.ollama, args.model, prompt, thinking, stop[0])
        row = {
            "i": i, "category": category, "prompt": prompt, "finish": finish, "thinking_tokens": thinking_tokens,
            "final": content, "final_boxed": boxed(content), "gold": GOLD.get(i), "probes": probes,
            "auto_stop": stop, "auto_answer": auto_answer, "auto_boxed": boxed(auto_answer) if auto_answer else None,
        }
        results.append(row)
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        parsed = sum(a is not None for _, a in probes)
        print(f"{i:2d} {category:6} think={thinking_tokens} finish={finish} probes={len(probes)} parsed={parsed} "
              f"auto_stop={stop} final_boxed={row['final_boxed']} auto_boxed={row['auto_boxed']}", flush=True)


if __name__ == "__main__":
    main()
