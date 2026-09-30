# Overthink Guard

Watch a local reasoning model think, and stop it when its thinking runs away, without costing correct answers.

Overthink Guard is a small proxy that sits between your OpenAI-compatible client and [Ollama](https://ollama.com). It shows the model's thinking live, records where it *could* have stopped (Shadow mode), and gives you an **Answer now** button that ends the thinking and gets the answer in a fraction of a second. No GPU needed for the proxy itself; it runs on macOS, Linux and Windows.

> **Status: early.** Ollama is the only backend. Answer now and probing are enabled only for model families where they have been verified to work (currently Qwen3 and DeepSeek-R1); other models are observed but never interrupted. By default nothing is stopped automatically: Shadow mode only records, and you decide when to press Answer now. Opt into `--mode auto` to have runaway thinking cut automatically.

## Quick start

```bash
ollama serve                 # if it is not already running
ollama pull qwen3:1.7b

uv tool install overthink-guard     # or: pipx install overthink-guard
otg start
```

Then point your client's base URL at `http://127.0.0.1:8484/v1` and open the UI at `http://127.0.0.1:8484/ui`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8484/v1", api_key="unused")
stream = client.chat.completions.create(
    model="qwen3:1.7b",
    messages=[{"role": "user", "content": "How many positive divisors does 3600 have?"}],
    stream=True,
)
for chunk in stream:
    ...  # thinking arrives in delta.reasoning, the answer in delta.content
```

To run from a checkout instead: `uv tool install .`

## What it does to your requests

| Request | Handling |
| --- | --- |
| Streamed, single-turn, plain-text chat (`/v1/chat/completions`) | Relayed through Ollama's native API so the thinking can be observed and cut. The response looks like Ollama's own OpenAI-compatible stream. |
| Anything else: tool calls, images, multi-turn, `stream: false`, other endpoints, unknown fields | Passed through to Ollama **byte for byte**, untouched. |

**Answer now.** Pressing it in the UI (or `POST /otg/api/sessions/{id}/stop`) closes the connection to Ollama. The thinking received so far is then prefilled back to the model, which answers immediately; the prompt lines up with Ollama's KV cache, so the answer usually starts within 0.1 s. The answer is spliced into the same response, and the final chunk carries an `otg` field saying whether the request was intervened.

**Shadow mode** (always on). Every request that thinks to completion is recorded: where Overthink Guard *would* have stopped, and whether the answer at that point matched the final one. The UI shows the running totals.

**Auto mode** (opt-in, `--mode auto`): a runaway guard. Once thinking passes 6,000 tokens, if the model's probed answer is the same 4 times in a row, Overthink Guard does an automatic Answer now. A probed answer only counts when a second, one-line probe states the same answer. For open-ended questions (designs, explanations), which have no short answer, Auto instead stops thinking that runs past a 10,000-token budget. It leaves normal-length thinking alone, so most overthinking inside normal-length runs is not cut. In pre-registered tests on 130 fresh math problems it cost no correct answers. Only 15 correct runs were long enough to be at risk, though, so this bounds the loss below about 20% of those runs (95% confidence), not at zero. It stopped every runaway run; 13 of 35 then gave the correct answer where the run had hit our 12,000-token test cap without one. The final chunk's `otg.auto` is `true` when this happens.

**Active probing** (opt-in, `--probe`; always on in Auto mode). Once thinking passes 6,000 tokens, the generation pauses briefly every 400 tokens. The model is asked for its current answer in a few tokens, and then the thinking resumes where it left off. Probing is off by default because it adds short pauses; the default detector reads only the thinking text.

## Measured so far

These results come from two small models (qwen3:1.7b, deepseek-r1:1.5b), mostly math questions, and small samples, so treat them as early evidence. Details and scripts are in [`docs/spikes/`](docs/spikes/). What matters most is how many correct answers a stop costs, counted over the correct runs that were long enough to be stopped at all; `python bench/spikes/probe_rule_matrix.py` recomputes that for every recorded run.

| | Result |
| --- | --- |
| Answer now, Ollama + `qwen3:1.7b` | First answer token ~0.09 s after pressing; 327 of 337 prompt tokens served from cache; 809 generated tokens instead of 3,193 |
| Default text-only detector, 986 public DeepSeek-R1 traces | It would have stopped 59 traces, and in 6 of them (10%) the answer at the stop point differed from the final answer. It saves only ~0.2% of thinking, or ~2% with looser settings; models rarely state their answer before the end |
| Opt-in probing, `qwen3:1.7b`, 17 problems used to choose the rule | `k=3` from the start would have saved 76% of thinking tokens, but lost 2 answers by locking onto an early guess. `k=4` lost none on this sample. |
| Same, 20 held-out problems | `k=4` from the start lost 2 answers here (57% saved): the model can hold a wrong answer for 2,000+ tokens before correcting it. Ignoring probes before 3,000 thinking tokens (the new default) lost none on either sample and saved 40% overall. That rule was also chosen on these samples, so it still needs validating. |
| Same, 34 more held-out problems (rule fixed beforehand) | Ignoring probes before 3,000 tokens and then `k=4` saved 32%, but lost 2 answers and gained 1 (net −1 of 34). The misses were late corrections, where a wrong answer was held for 5,000+ tokens, which no answer-stability rule can foresee. Probing is not yet accurate enough to stop automatically. |
| Runaway guard, rule fixed beforehand, 51 fresh completed runs + 19 runaway runs (qwen3:1.7b and deepseek-r1:1.5b) | Probing only after 6,000 tokens lost no correct answers, but only 7 correct runs were long enough to be at risk. It stopped all 19 runaway runs, which had hit our 12,000-token cap without answering, and 8 of them gave the correct answer. It saved 21–23% of thinking tokens under that cap, short of the 25% target set beforehand, and only 9% on the runs that finished normally. This is the current default. |
| Same guard with the one-line cross-check, rule fixed beforehand: 48 new design prompts (open designs, design decisions, code bugs, some in Korean) and 60 new math problems | No completed math answer was lost (8 correct runs were long enough to be at risk). None of the 46 completed design answers was cut, but all of them finished under 6,000 thinking tokens, so the rule could not have fired: this shows that normal design answers are left alone, not that stopping a long design answer is safe. It stopped all 18 runaway runs; the design ones gave a correct choice and a correct bug explanation where waiting produced nothing. |
| Probe cost | ~4.7% of wall time on Apple Silicon GPU, ~5% CPU-only; each probe hits the KV cache |

## Claude Code (subscription)

Claude Code does not show its thinking while it happens. Headless stream-json carries empty thinking deltas, and the session log gets each thinking block only after it ends. So Overthink Guard cannot judge Claude's reasoning live. What it can see is how long a turn has gone without any output, and how much of your usage went to thinking.

```bash
otg claude-code watch            # desktop alert when a turn has gone 60s without output
otg claude-code watch --after 90 --hook "your-command"   # also pass a stop_suggested event (JSON on stdin)
otg claude-code report --days 7  # thinking tokens per effort level, share of output spent on thinking
```

- It reads `~/.claude/projects/*/*.jsonl` (only record types, timestamps, effort and token counts; message text is neither kept nor sent) and the session status files in `~/.claude/sessions/*.json`. It never opens credential or key files, never proxies requests, and never interrupts a session.
- Alerts are advice based on time alone. On the author's machine, 0.8% of turns were silent for over 60 s. In a pre-registered check ([`docs/validation/subscription.md`](docs/validation/subscription.md)), replaying 66 real sessions gave 96.5% precision and 100% recall at 60 s. That measures whether Claude was really still working when alerted, not whether stopping it then would have been right. In live headless runs, every turn that went past the threshold was alerted within 1 s of it (one ran for 593 s and 64k thinking tokens), and short, aborted and interrupted runs got no alert.
- To stop long **headless** runs automatically, you can wire the alert to Claude Code's documented SIGINT yourself: see [`docs/recipes/claude-code-headless.md`](docs/recipes/claude-code-headless.md). Don't, at 60 s: in a pre-registered test (V7), Sonnet turns that went past 60 s were right 8 of 9 times when left alone and 4 of 9 times when stopped at the alert, because the interrupted thinking is lost. On these problems long turns were needed thinking, not overthinking. That is why stopping is not built in, and why the alert only tells you a turn is taking long.
- `report` counts only turns that finished: tokens spent before an interrupt are not in the logs.
- Claude Code's local file formats are internal and may change between releases. Tested with Claude Code 2.1.233.

## Configuration

Flags override the config file, and the config file overrides the defaults. The file lives at `~/.config/overthink-guard/config.yaml` (`%APPDATA%\overthink-guard\config.yaml` on Windows) or wherever `--config` points. Unknown keys are rejected, not ignored.

```yaml
server:
  host: 127.0.0.1
  port: 8484
local:
  backend_url: http://localhost:11434
  mode: shadow                # shadow | auto
  auto_stop:                  # text-only detector (shown as "Looks safe to stop" in the UI)
    converge_k: 3
    min_thinking_tokens: 300
    revision_cooldown_tokens: 200
    repetition_threshold: 0.5
  signals:
    active_probe: false       # same as --probe
    probe_interval_tokens: 400
    probe_min_tokens: 6000    # no probes before this much thinking
    probe_converge_k: 4
    open_budget_tokens: 10000 # open-ended prompts: stop thinking past this
privacy:
  stats_file: null            # null keeps Shadow statistics in memory only
```

`otg analyze trace.txt` replays a saved completion offline and reports where the text-only detector would have stopped.

## Privacy and security

- Listens on `127.0.0.1` only. Requests with an unexpected `Host` header (DNS rebinding) or a non-local `Origin` (a web page in your browser) are rejected with 403.
- Shadow statistics are counts and true/false flags only; no prompt, thinking or answer text is stored. By default they go to `~/Library/Application Support/overthink-guard/shadow.jsonl` on macOS, `~/.local/share/overthink-guard/` on Linux, and `%LOCALAPPDATA%\overthink-guard\` on Windows.
- The live UI shows thinking text from memory; nothing is written to disk.
- No telemetry.
- Overthink Guard does not read, use or proxy subscription credentials for Claude Code, Codex or any other tool. For Claude Code it only reads local session logs and status files, and it keeps no message text.

## Known limitations

- Ollama only. llama.cpp server and LM Studio are planned.
- Answer now and probing need a verified way to continue the model's own thinking. Qwen3 uses Ollama's chat template (prefilled assistant turn). DeepSeek-R1's Ollama template cannot do that, so its prompt is rendered by hand and sent raw. Models that are not verified yet are observed only.
- Intervention covers streamed single-turn text chat; everything else passes through.
- No CORS headers, so browser apps that call the proxy directly from a page will not work. Server-side clients are fine.
- When a request is intervened or probed, `usage.prompt_tokens` is an estimate, because Ollama does not report it for a cut stream.
- Open-ended prompts have no short answer to probe, so they fall back to a plain thinking budget. That budget has not yet been seen stopping a runaway design question in testing.

## Prior work

The ideas here build on published research: [Dynasor](https://github.com/hao-ai-lab/Dynasor) (probe-in-the-middle), [DEER](https://arxiv.org/abs/2504.15895), [PUMA](https://github.com/giovanni-vaccarino/PUMA), [ThinkBrake](https://arxiv.org/abs/2510.00546) (end-of-thinking token margin), and s1's budget forcing.

## License

MIT
