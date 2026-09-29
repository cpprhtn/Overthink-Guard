# Overthink Guard

See when a local reasoning model has already reached its answer, and cut the rest of the thinking.

Overthink Guard is a small proxy that sits between your OpenAI-compatible client and [Ollama](https://ollama.com). It shows the model's thinking live, records where it *could* have stopped (Shadow mode), and gives you an **Answer now** button that ends the thinking and gets the answer in a fraction of a second. No GPU needed for the proxy itself; it runs on macOS, Linux and Windows.

> **Status: early (0.2.0.dev0).** Ollama is the only backend. Answer now and probing are enabled only for model families where they have been verified to work (currently Qwen3); other models are observed but never interrupted. Nothing is stopped automatically yet: Shadow mode only records, and you decide when to press Answer now.

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

Until 0.2.0 is published, the PyPI package is a name placeholder, so install from a checkout instead: `uv tool install .`

## What it does to your requests

| Request | Handling |
| --- | --- |
| Streamed, single-turn, plain-text chat (`/v1/chat/completions`) | Relayed through Ollama's native API so the thinking can be observed and cut. The response looks like Ollama's own OpenAI-compatible stream. |
| Anything else: tool calls, images, multi-turn, `stream: false`, other endpoints, unknown fields | Passed through to Ollama **byte for byte**, untouched. |

**Answer now.** Pressing it in the UI (or `POST /otg/api/sessions/{id}/stop`) closes the connection to Ollama. The thinking received so far is then prefilled back to the model, which answers immediately; the prompt lines up with Ollama's KV cache, so the answer usually starts within 0.1 s. The answer is spliced into the same response, and the final chunk carries an `otg` field saying whether the request was intervened.

**Shadow mode** (always on). Every request that thinks to completion is recorded: where Overthink Guard *would* have stopped, and whether the answer at that point matched the final one. The UI shows the running totals.

**Active probing** (opt-in, `--probe`). After the first 3,000 thinking tokens, the generation pauses briefly every 400 tokens. The model is asked for its current answer in a few tokens, and then the thinking resumes where it left off. Probing is off by default because it adds short pauses; the default detector reads only the thinking text.

## Measured so far

These results come from one small model, math questions only, and small samples, so treat them as early evidence. Details and scripts are in [`docs/spikes/`](docs/spikes/).

| | Result |
| --- | --- |
| Answer now, Ollama + `qwen3:1.7b` | First answer token ~0.09 s after pressing; 327 of 337 prompt tokens served from cache; 809 generated tokens instead of 3,193 |
| Default text-only detector, 986 public DeepSeek-R1 traces | Safe (the answer at the stop point differed from the final answer in 6 of 986 traces) but saves only ~0.2% of thinking, or ~2% with looser settings; models rarely state their answer before the end |
| Opt-in probing, `qwen3:1.7b`, 17 problems used to choose the rule | `k=3` from the start would have saved 76% of thinking tokens, but lost 2 answers by locking onto an early guess. `k=4` lost none on this sample. |
| Same, 20 held-out problems | `k=4` from the start lost 2 answers here (57% saved): the model can hold a wrong answer for 2,000+ tokens before correcting it. Ignoring probes before 3,000 thinking tokens (the new default) lost none on either sample and saved 40% overall. That rule was also chosen on these samples, so it still needs validating. |
| Probe cost | ~4.7% of wall time on Apple Silicon GPU, ~5% CPU-only; each probe hits the KV cache |

## Configuration

Flags override the config file, and the config file overrides the defaults. The file lives at `~/.config/overthink-guard/config.yaml` (`%APPDATA%\overthink-guard\config.yaml` on Windows) or wherever `--config` points. Unknown keys are rejected, not ignored.

```yaml
server:
  host: 127.0.0.1
  port: 8484
local:
  backend_url: http://localhost:11434
  auto_stop:                  # text-only detector (shown as "Looks safe to stop" in the UI)
    converge_k: 3
    min_thinking_tokens: 300
    revision_cooldown_tokens: 200
    repetition_threshold: 0.5
  signals:
    active_probe: false       # same as --probe
    probe_interval_tokens: 400
    probe_min_tokens: 3000    # no probes before this much thinking
    probe_converge_k: 4
privacy:
  stats_file: null            # null keeps Shadow statistics in memory only
```

`otg analyze trace.txt` replays a saved completion offline and reports where the text-only detector would have stopped.

## Privacy and security

- Listens on `127.0.0.1` only. Requests with an unexpected `Host` header (DNS rebinding) or a non-local `Origin` (a web page in your browser) are rejected with 403.
- Shadow statistics are counts and true/false flags only; no prompt, thinking or answer text is stored. By default they go to `~/Library/Application Support/overthink-guard/shadow.jsonl` on macOS, `~/.local/share/overthink-guard/` on Linux, and `%LOCALAPPDATA%\overthink-guard\` on Windows.
- The live UI shows thinking text from memory; nothing is written to disk.
- No telemetry.
- Overthink Guard does not read, use or proxy subscription credentials for Claude Code, Codex or any other tool.

## Known limitations

- Ollama only. llama.cpp server and LM Studio are planned.
- Answer now and probing rely on the model's chat template accepting a prefilled assistant turn. This is verified for Qwen3. It does not work for Ollama's DeepSeek-R1 template, so those models, and any model not yet verified, are observed only.
- Intervention covers streamed single-turn text chat; everything else passes through.
- No CORS headers, so browser apps that call the proxy directly from a page will not work. Server-side clients are fine.
- When a request is intervened or probed, `usage.prompt_tokens` is an estimate, because Ollama does not report it for a cut stream.
- Probing assumes a short answer that fits in `\boxed{}`. It does not converge on open-ended or multiple-choice questions.

## Prior work

The ideas here build on published research: [Dynasor](https://github.com/hao-ai-lab/Dynasor) (probe-in-the-middle), [DEER](https://arxiv.org/abs/2504.15895), [PUMA](https://github.com/giovanni-vaccarino/PUMA), [ThinkBrake](https://arxiv.org/abs/2510.00546) (end-of-thinking token margin), and s1's budget forcing.

## License

MIT
