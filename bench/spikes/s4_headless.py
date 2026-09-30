import json, subprocess, sys, time
from collections import Counter
model, out = sys.argv[1], sys.argv[2]
prompt = sys.argv[3] if len(sys.argv) > 3 else "How many positive divisors does 3600 have? Think it through carefully, then answer briefly."
cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "stream-json", "--include-partial-messages",
       "--verbose", "--effort", "high", "--max-turns", "1"]
t0 = time.time()
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
events = []
for line in proc.stdout:
    try: events.append((round(time.time() - t0, 2), json.loads(line)))
    except ValueError: pass
proc.wait()
json.dump(events, open(out, "w"))
kinds = Counter()
for t, e in events:
    k = e.get("type")
    if k == "stream_event":
        ev = e.get("event", {}); k = f"stream:{ev.get('type')}"
        if ev.get("type") == "content_block_delta": k += ":" + ev.get("delta", {}).get("type", "?")
        if ev.get("type") == "content_block_start": k += ":" + ev.get("content_block", {}).get("type", "?")
    kinds[k] += 1
res = next((e for _, e in events if "is_error" in e), {})
print("model", model, "exit", proc.returncode, "is_error", res.get("is_error"), "| elapsed", events[-1][0] if events else None)
print("  result:", str(res.get("result"))[:150])
for k, v in kinds.most_common(): print(f"  {v:4}  {k}")
th = [(t, e["event"]["delta"].get("thinking", "")) for t, e in events if e.get("type") == "stream_event" and e.get("event", {}).get("type") == "content_block_delta" and e["event"]["delta"].get("type") == "thinking_delta"]
if th:
    print(f"  thinking deltas: {len(th)}, first {th[0][0]}s, last {th[-1][0]}s, chars {sum(len(x) for _, x in th)}, first sizes {[len(x) for _, x in th[:8]]}")
text = [t for t, e in events if e.get("type") == "stream_event" and e.get("event", {}).get("type") == "content_block_delta" and e["event"]["delta"].get("type") == "text_delta"]
if text: print(f"  text deltas {text[0]}s .. {text[-1]}s")
u = res.get("usage", {})
print("  usage thinking_tokens:", (u.get("output_tokens_details") or {}).get("thinking_tokens"), "output_tokens:", u.get("output_tokens"))
