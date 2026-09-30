# Recipe: stop a long headless Claude Code turn yourself

Overthink Guard only **observes** Claude Code and sends alerts. It never reads your credentials, never proxies or sends requests on your behalf, and never interrupts a session itself (concept doc D5, D7). This recipe shows how *you* can wire its alert to Claude Code's documented way of ending a headless turn, for runs your own script starts.

- **Scope: headless `claude -p` runs started by `run_claude.py` only.** Interactive sessions are never touched; press Esc there yourself. The hook ignores any session it did not start and any event with `run_mode: interactive`.
- **Tested with**: Claude Code 2.1.233 on macOS, 2026-09-30, `--model sonnet`. Claude Code's session files and signal handling can change between versions; re-check after upgrading.
- **Your responsibility**: you are running your own subscription's official CLI. Check that automating it this way fits Anthropic's current terms for your plan, and remember that subscription limits assume personal use.
- **No loops**: each session is interrupted at most once, and the follow-up prompt is sent at most once.

## How it works

1. `run_claude.py` runs `claude -p ... --output-format stream-json`. It records the session id and process id in a local registry and removes the entry when the run ends.
2. `otg claude-code watch --hook ".../stop_on_suggest.py"` reads Claude Code's local session logs. When a turn has produced no output for `--after` seconds, it passes a `stop_suggested` event (schema v1) to the hook on stdin.
3. `stop_on_suggest.py` finds the session in the registry and sends it **SIGINT**, which Claude Code documents as ending the current turn cleanly. It never kills the process.
4. If you passed `--follow-up`, `run_claude.py` resumes the session once with that prompt, for example "Give your best answer now from the reasoning so far."

```bash
# terminal 1
otg claude-code watch --after 60 --hook "python /path/to/docs/recipes/claude_code_headless/stop_on_suggest.py"

# terminal 2
python docs/recipes/claude_code_headless/run_claude.py "your task" \
  --follow-up "Stop analysing. Give your best answer now from the reasoning so far." \
  --max-turns 10
```

On Windows SIGINT cannot be sent to another console process, so the hook only records the event there.

## What we saw when testing it

With a deliberately aggressive `--after 5` and a number-theory question:

| Step | Result |
| --- | --- |
| Alert | 6 s after the prompt, while the model was still thinking |
| SIGINT | The turn ended at once (`terminal_reason: aborted_streaming`) with an empty result, because the thinking was interrupted |
| Follow-up (once) | An answer 13 s after the start |
| Correctness | **Wrong**: it missed one of the four solutions (x = 32257), which a full-length run had found |

Stopping early saves time, but it can cost accuracy. Claude Code does not expose the thinking while it happens (see `docs/spikes/s4-claude-code.md`), so this alert is based on time alone, not on whether the model has already found its answer. Use a generous `--after` (the default is 60 s; on this machine 0.8% of turns go past it). Treat the alert as advice, not a verdict.
