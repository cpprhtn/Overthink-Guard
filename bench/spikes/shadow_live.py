import glob, json, re, sys, time
from openai import OpenAI
from overthink_guard.analysis.signals import extract_boxed, normalize_answer
SP = sys.argv[1]
EASY = [
    ("How many positive divisors does 3600 have?", "45"),
    ("What is the sum of the first 50 positive odd integers?", "2500"),
    ("A train travels 180 km in 2.5 hours. What is its average speed in km/h?", "72"),
    ("In how many ways can 5 people be seated in a row?", "120"),
    ("What is the remainder when 2^10 is divided by 7?", "2"),
    ("If 3x + 7 = 25, what is x?", "6"),
    ("What is the least common multiple of 12 and 18?", "36"),
    ("How many prime numbers are less than 30?", "10"),
]
r1 = []
for f in sorted(glob.glob(f"{SP}/cache/openr1_math_*.json")):
    for r in json.load(open(f))["rows"]:
        row = r["row"]
        if re.fullmatch(r"-?\d+", row["answer"].strip()) and len(row["problem"]) < 260:
            r1.append((row["problem"], row["answer"].strip()))
problems = EASY + r1[:12]
client = OpenAI(base_url="http://127.0.0.1:8485/v1", api_key="x")
rows = []
for i, (q, gold) in enumerate(problems):
    t0 = time.time(); content = ""
    for ch in client.chat.completions.create(model="qwen3:1.7b", stream=True, messages=[{"role": "user", "content": q}]):
        if ch.choices and ch.choices[0].delta.content: content += ch.choices[0].delta.content
    boxed = extract_boxed(content)
    final = normalize_answer(boxed[-1][1]) if boxed else None
    rec = json.loads(open(f"{SP}/shadow.jsonl").read().splitlines()[-1])
    correct = final == gold
    t2 = rec["tier2"]
    rows.append({"correct": correct, **rec})
    print(f"{i:2d} {'OK ' if correct else 'BAD'} gold={gold:>6} final={final!s:>8} think={rec['thinking_tokens']:5d} "
          f"t0_stop={rec['tier0']['stop_at']!s:>5} t2_stop={t2['stop_at']!s:>5} t2_match={t2['match']!s:>5} "
          f"probes={t2['probes']:2d} ({t2['probe_seconds']:.1f}s of {rec['elapsed_seconds']:.0f}s)", flush=True)
json.dump(rows, open(f"{SP}/shadow_live_rows.json", "w"))
