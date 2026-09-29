import json

import httpx
import pytest
from starlette.testclient import TestClient

from overthink_guard.backends import to_native_chat
from overthink_guard.proxy import create_app

BASE = "http://127.0.0.1:8484"
QUESTION = "What is 17*24?"
THINKING = ["Okay, ", "17 times 24. ", "So the answer is 408.\n\n", "Wait, let me check ", "again. ", "17*24 = 408."]
# After a resume Ollama streams the continuation as content, closing tag included (spike: resume shape).
RESUMED = ["More checking. ", "Still 408.", "</think>", "\n\n", "408"]


def ndjson(*chunks: dict) -> bytes:
    return b"".join(json.dumps(c).encode() + b"\n" for c in chunks)


def thinking_chunk(text: str) -> dict:
    return {"message": {"role": "assistant", "content": "", "thinking": text}, "done": False}


def content_chunk(text: str) -> dict:
    return {"message": {"role": "assistant", "content": text}, "done": False}


DONE = {
    "message": {"role": "assistant", "content": ""},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 20,
    "prompt_eval_cached_count": 0,
    "eval_count": 9,
}


class FakeOllama:
    """Upstream double: records requests; /api/chat thinks, then answers unless told to stop."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.app = None
        self.stop_after: int | None = None
        self.native_status = 200
        self.probe_status = 200
        self.resume_status = 200

    def bodies(self, path: str) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.url.path == path]

    async def handler(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        if request.url.path == "/api/generate":
            return self.generate(json.loads(request.content))
        if request.url.path != "/api/chat":
            echo = b'{"echo": ' + (request.content or b"null") + b"}"
            return httpx.Response(
                201, headers={"x-upstream": "yes", "content-type": "application/json"}, stream=httpx.ByteStream(echo)
            )
        if self.native_status != 200:
            return httpx.Response(self.native_status, stream=httpx.ByteStream(b'{"error": "think not supported"}'))
        body = json.loads(request.content)
        if body.get("stream") is False:
            probe = {"message": {"role": "assistant", "content": "408}$."}, "done": True, "prompt_eval_count": 50}
            return httpx.Response(self.probe_status, json=probe if self.probe_status == 200 else {"error": "x"})
        if not body.get("think") and body["messages"][-1]["role"] == "assistant":
            if self.resume_status != 200:
                return httpx.Response(self.resume_status, stream=httpx.ByteStream(b'{"error": "x"}'))
            return httpx.Response(200, content=ndjson(*(content_chunk(t) for t in RESUMED), DONE))
        if body.get("think") and body["messages"][-1]["role"] == "assistant":
            answer = ndjson(
                content_chunk("The answer is "),
                content_chunk("408."),
                {**DONE, "prompt_eval_count": 60, "prompt_eval_cached_count": 55, "eval_count": 4},
            )
            return httpx.Response(200, content=answer)

        async def stream():
            for i, text in enumerate(THINKING):
                if self.stop_after is not None and i == self.stop_after:
                    self.app.state.hub.request_stop("1")
                yield ndjson(thinking_chunk(text))
            yield ndjson(content_chunk("408"), DONE)

        return httpx.Response(200, content=stream())


def generate_chunk(thinking: str = "", response: str = "") -> dict:
    return {"thinking": thinking, "response": response, "done": False}


def _generate(self, body: dict) -> httpx.Response:
    """Raw /api/generate: original stream, answer after a closed think block, probe, or resume."""
    prompt = body["prompt"]
    done = {**DONE, "response": ""}
    del done["message"]
    if body.get("stream") is False:
        return httpx.Response(200, json={"response": "408}$.", "done": True, "prompt_eval_count": 50})
    if prompt.endswith("</think>\n\n"):
        answer = ndjson(generate_chunk(response="The answer is "), generate_chunk(response="408."), done)
        return httpx.Response(200, content=answer)
    if "<think>" in prompt:
        return httpx.Response(200, content=ndjson(*(generate_chunk(response=t) for t in RESUMED), done))

    async def stream():
        for i, text in enumerate(THINKING):
            if self.stop_after is not None and i == self.stop_after:
                self.app.state.hub.request_stop("1")
            yield ndjson(generate_chunk(thinking=text))
        yield ndjson(generate_chunk(response="408"), done)

    return httpx.Response(200, content=stream())


FakeOllama.generate = _generate


@pytest.fixture
def fake():
    return FakeOllama()


@pytest.fixture
def app_kwargs():
    return {}


@pytest.fixture
def client(fake, app_kwargs):
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    app = create_app("http://ollama.test", client=upstream, **app_kwargs)
    fake.app = app
    with TestClient(app, base_url=BASE) as test_client:
        yield test_client


def sse_events(text: str) -> list:
    return [
        json.loads(line[6:]) if line != "data: [DONE]" else "[DONE]"
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def chat(client: TestClient, **extra) -> httpx.Response:
    body = {"model": "qwen3:1.7b", "messages": [{"role": "user", "content": QUESTION}], "stream": True, **extra}
    return client.post("/v1/chat/completions", json=body)


def test_streams_reasoning_then_content_and_marks_no_intervention(client, fake):
    resp = chat(client, stream_options={"include_usage": True})
    events = sse_events(resp.text)
    deltas = [e["choices"][0]["delta"] for e in events[:-3]]
    assert "".join(d.get("reasoning", "") for d in deltas) == "".join(THINKING)
    assert "".join(d.get("content", "") for d in deltas) == "408"
    assert events[-3]["choices"][0]["finish_reason"] == "stop"
    assert events[-3]["otg"] == {"session": "1", "intervened": False}
    assert events[-2]["usage"] == {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29}
    assert events[-1] == "[DONE]"
    assert resp.headers["x-otg-session"] == "1"
    assert "think" not in fake.bodies("/api/chat")[0]
    assert fake.app.state.hub.get("1").status == "done"


def test_answer_now_aborts_thinking_and_splices_prefilled_answer(client, fake):
    fake.stop_after = 3
    events = sse_events(chat(client, max_tokens=100).text)
    deltas = [e["choices"][0]["delta"] for e in events[:-2]]
    received = "".join(d.get("reasoning", "") for d in deltas)
    assert received == "".join(THINKING[:4])
    assert "".join(d.get("content", "") for d in deltas) == "The answer is 408."

    prefill = fake.bodies("/api/chat")[1]
    assert prefill["think"] is True
    assert prefill["messages"][-1] == {
        "role": "assistant",
        "content": "",
        "thinking": "\n" + received.rstrip() + "\n\nI have enough to answer now.",
    }
    assert prefill["options"]["num_predict"] == 100 - 4
    otg = events[-2]["otg"]
    assert otg["intervened"] is True
    assert otg["stopped_after_tokens"] == 4
    assert otg["answer_prompt_cached_tokens"] == 55
    assert fake.app.state.hub.get("1").intervened


@pytest.mark.parametrize(
    "extra",
    [
        {"tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]},
        {"stream": False},
        {"response_format": {"type": "json_object"}},
    ],
)
def test_requests_we_do_not_understand_pass_through_byte_for_byte(client, fake, extra):
    body = json.dumps(
        {"model": "qwen3:1.7b", "messages": [{"role": "user", "content": "hi"}], "stream": True, **extra},
        separators=(",", ":"),
    ).encode()
    resp = client.post("/v1/chat/completions?x=1", content=body, headers={"content-type": "application/json"})
    upstream = fake.requests[-1]
    assert upstream.url.path == "/v1/chat/completions" and upstream.url.query == b"x=1"
    assert upstream.content == body
    assert resp.status_code == 201
    assert resp.headers["x-upstream"] == "yes"
    assert resp.content == b'{"echo": ' + body + b"}"


def test_other_paths_pass_through(client, fake):
    assert client.get("/api/tags").status_code == 201
    assert client.get("/").status_code == 201
    assert [r.url.path for r in fake.requests] == ["/api/tags", "/"]


def test_backend_rejecting_native_request_falls_back_to_passthrough(client, fake):
    fake.native_status = 400
    resp = chat(client)
    assert resp.status_code == 201
    assert [r.url.path for r in fake.requests] == ["/api/chat", "/v1/chat/completions"]


@pytest.mark.parametrize(
    ("headers", "status"),
    [
        ({"host": "evil.example:8484"}, 403),
        ({"origin": "https://evil.example"}, 403),
        ({"origin": "null"}, 403),
        ({"origin": "http://localhost:3000"}, 201),
        ({"host": "localhost:8484"}, 201),
    ],
)
def test_only_local_hosts_and_origins_are_served(client, headers, status):
    assert client.get("/api/tags", headers=headers).status_code == status


def test_stop_unknown_session_is_rejected(client):
    assert client.post("/otg/api/sessions/nope/stop").status_code == 409


def test_ui_is_served_without_external_assets(client):
    page = client.get("/ui").text
    assert "Overthink Guard" in page
    assert "http://" not in page and "https://" not in page


@pytest.mark.parametrize(
    "body",
    [
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "q2"},
            ]
        },
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]},
        {"messages": [{"role": "user", "content": "q", "name": "bob"}]},
        {"messages": [{"role": "user", "content": "q"}], "n": 2},
        {"messages": [{"role": "user", "content": "q"}], "stream_options": {"other": 1}},
        {"messages": [{"role": "user", "content": "q"}], "logprobs": True},
    ],
)
def test_only_single_turn_text_is_intervened(body):
    assert to_native_chat({"model": "m", "stream": True, **body}) is None


def test_openai_options_map_to_ollama_options():
    native = to_native_chat(
        {
            "model": "m",
            "stream": True,
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
            "temperature": 0.2,
            "max_tokens": 50,
            "stop": "END",
            "seed": 1,
        }
    )
    assert native["options"] == {"temperature": 0.2, "num_predict": 50, "stop": ["END"], "seed": 1}
    assert native["messages"][0] == {"role": "system", "content": "s"}


PROBE_EVERY_3 = [{"probe": True, "probe_interval": 3, "probe_min_tokens": 0}]


@pytest.mark.parametrize("app_kwargs", PROBE_EVERY_3)
def test_probing_pauses_probes_and_resumes_transparently(client, fake):
    events = sse_events(chat(client, max_tokens=100).text)
    deltas = [e["choices"][0]["delta"] for e in events if isinstance(e, dict) and e["choices"]]
    assert "".join(d.get("reasoning", "") for d in deltas) == "".join(THINKING[:3]) + "More checking. Still 408."
    assert "".join(d.get("content", "") for d in deltas) == "408"

    _, probe, resume = fake.bodies("/api/chat")
    assert probe["stream"] is False and probe["think"] is True
    assert probe["messages"][-1]["content"].endswith("\\boxed{")
    assert probe["options"] == {"num_predict": 16, "temperature": 0}
    assert resume["messages"][-1] == {"role": "assistant", "content": "<think>\n" + "".join(THINKING[:3])}
    assert "think" not in resume
    assert resume["options"]["num_predict"] == 100 - 3

    session = fake.app.state.hub.get("1")
    assert [(p.at_tokens, p.answer) for p in session.probes.probes] == [(3, "408")]
    assert session.judge.thinking_tokens == 5
    assert events[-2]["otg"]["probes"] == 1
    assert session.shadow["tier2"]["probes"] == 1
    assert session.shadow["perturbed"] is True
    assert fake.app.state.hub.stats.summary()["tier2"]["requests"] == 1


@pytest.mark.parametrize("app_kwargs", PROBE_EVERY_3)
def test_failed_probe_is_recorded_without_answer_and_thinking_continues(client, fake):
    fake.probe_status = 500
    events = sse_events(chat(client).text)
    deltas = [e["choices"][0]["delta"] for e in events if isinstance(e, dict) and e["choices"]]
    assert "".join(d.get("content", "") for d in deltas) == "408"
    assert "error" not in events[-2]["otg"]
    assert [(p.at_tokens, p.answer) for p in fake.app.state.hub.get("1").probes.probes] == [(3, None)]


@pytest.mark.parametrize("app_kwargs", PROBE_EVERY_3)
def test_failed_resume_ends_the_stream_with_an_error_and_no_shadow(client, fake):
    fake.resume_status = 503
    events = sse_events(chat(client).text)
    deltas = [e["choices"][0]["delta"] for e in events if isinstance(e, dict) and e["choices"]]
    assert "".join(d.get("reasoning", "") for d in deltas) == "".join(THINKING[:3])
    assert events[-2]["otg"]["error"] == "resume request failed with HTTP 503"
    assert events[-1] == "[DONE]"
    session = fake.app.state.hub.get("1")
    assert session.status == "error"
    assert session.shadow is None


@pytest.mark.parametrize("app_kwargs", PROBE_EVERY_3)
def test_client_seed_disables_probing(client, fake):
    chat(client, seed=7)
    assert len(fake.bodies("/api/chat")) == 1
    session = fake.app.state.hub.get("1")
    assert session.probes is None
    assert session.probe_skipped == "seed"


def test_completed_request_records_shadow_counts_but_no_text(client, fake):
    chat(client)
    record = fake.app.state.hub.get("1").shadow
    assert set(record) == {"ts", "model", "thinking_tokens", "elapsed_seconds", "tier0", "tier2", "perturbed"}
    assert record["thinking_tokens"] == len(THINKING)
    assert record["tier2"] is None
    assert record["perturbed"] is False
    assert QUESTION not in json.dumps(record) and "408" not in json.dumps(record)
    assert client.get("/otg/api/stats").json()["requests"] == 1


def test_answer_now_is_not_recorded_as_shadow(client, fake):
    fake.stop_after = 3
    chat(client)
    assert fake.app.state.hub.get("1").shadow is None
    assert fake.app.state.hub.stats.summary()["requests"] == 0


@pytest.mark.parametrize("app_kwargs", [{"probe": True, "probe_interval": 3, "probe_min_tokens": 5}])
def test_no_probes_before_the_minimum_thinking_length(client, fake):
    chat(client)
    session = fake.app.state.hub.get("1")
    assert [p.at_tokens for p in session.probes.probes] == [5]
    resume = fake.bodies("/api/chat")[-1]
    assert resume["messages"][-1]["content"] == "<think>\n" + "".join(THINKING[:5])


@pytest.mark.parametrize("app_kwargs", PROBE_EVERY_3)
def test_models_without_a_verified_template_are_only_observed(client, fake):
    fake.stop_after = 3
    events = sse_events(chat(client, model="llama3.1:8b").text)
    deltas = [e["choices"][0]["delta"] for e in events if isinstance(e, dict) and e["choices"]]
    assert "".join(d.get("reasoning", "") for d in deltas) == "".join(THINKING)
    assert "".join(d.get("content", "") for d in deltas) == "408"
    assert len(fake.bodies("/api/chat")) == 1
    session = fake.app.state.hub.get("1")
    assert (session.probes, session.probe_skipped, session.intervened) == (None, "template", False)
    assert fake.app.state.hub.summary(session)["can_intervene"] is False
    assert session.shadow is not None


R1 = "deepseek-r1:1.5b"
R1_PROMPT = "<｜User｜>" + QUESTION + "<｜Assistant｜>"  # noqa: RUF001 - DeepSeek's special tokens use fullwidth bars


def test_raw_mode_answer_now_renders_the_prompt_and_closes_the_think_block(client, fake):
    fake.stop_after = 3
    events = sse_events(chat(client, model=R1).text)
    deltas = [e["choices"][0]["delta"] for e in events[:-2]]
    received = "".join(d.get("reasoning", "") for d in deltas)
    assert received == "".join(THINKING[:4])
    assert "".join(d.get("content", "") for d in deltas) == "The answer is 408."
    original, answer = fake.bodies("/api/generate")
    assert (original["prompt"], original["raw"], original["stream"]) == (R1_PROMPT, True, True)
    expected = R1_PROMPT + "<think>\n" + received.rstrip() + "\n\nI have enough to answer now.\n</think>\n\n"
    assert answer["prompt"] == expected
    assert events[-2]["otg"]["intervened"] is True
    assert fake.bodies("/api/chat") == []


@pytest.mark.parametrize("app_kwargs", PROBE_EVERY_3)
def test_raw_mode_probes_and_resumes_with_the_open_think_block(client, fake):
    events = sse_events(
        chat(
            client,
            model=R1,
            messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": QUESTION}],
        ).text
    )
    deltas = [e["choices"][0]["delta"] for e in events if isinstance(e, dict) and e["choices"]]
    assert "".join(d.get("reasoning", "") for d in deltas) == "".join(THINKING[:3]) + "More checking. Still 408."
    assert "".join(d.get("content", "") for d in deltas) == "408"
    original, probe, resume = fake.bodies("/api/generate")
    prompt = "Be brief." + R1_PROMPT
    assert original["prompt"] == prompt
    assert probe["stream"] is False and probe["prompt"].endswith("</think>\n\nThe final answer is $\\boxed{")
    assert resume["prompt"] == prompt + "<think>\n" + "".join(THINKING[:3])
    session = fake.app.state.hub.get("1")
    assert [(p.at_tokens, p.answer) for p in session.probes.probes] == [(3, "408")]


AUTO_FAST = [{"auto": True, "probe_interval": 1, "probe_min_tokens": 0, "probe_converge_k": 2}]


@pytest.mark.parametrize("app_kwargs", AUTO_FAST)
def test_auto_mode_answers_once_probe_answers_agree(client, fake):
    events = sse_events(chat(client).text)
    deltas = [e["choices"][0]["delta"] for e in events if isinstance(e, dict) and e["choices"]]
    assert "".join(d.get("reasoning", "") for d in deltas) == THINKING[0] + RESUMED[0]
    assert "".join(d.get("content", "") for d in deltas) == "The answer is 408."
    otg = events[-2]["otg"]
    assert (otg["intervened"], otg["auto"], otg["probes"]) == (True, True, 2)
    session = fake.app.state.hub.get("1")
    assert session.auto_stopped and session.shadow is None
    stats = fake.app.state.hub.stats.summary()
    assert (stats["auto_stops"], stats["requests"]) == (1, 0)


@pytest.mark.parametrize("app_kwargs", AUTO_FAST)
def test_auto_mode_leaves_unverified_models_alone(client, fake):
    events = sse_events(chat(client, model="llama3.1:8b").text)
    deltas = [e["choices"][0]["delta"] for e in events if isinstance(e, dict) and e["choices"]]
    assert "".join(d.get("content", "") for d in deltas) == "408"
    session = fake.app.state.hub.get("1")
    assert (session.auto_stopped, session.probe_skipped) == (False, "template")
