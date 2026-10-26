#!/usr/bin/env python3
# needle-set-01.py <label>: 12 long-needle questions for A/B of a change, all against ONE cold read per size (fresh salt per run,
# then the questions reuse it from the cache): 4500 records (74k tokens) x 4 gates, 6000 (100k) x 4, 6900 (116k) x 4.
import json, sys, time, urllib.request
TAG = (sys.argv[1] if len(sys.argv) > 1 else "x") + time.strftime("%H%M%S")
recs = lambda n: "\n".join(f"Record {i}: gate {i} uses code K{(i * 7919) % 9973}." for i in range(1, n + 1))
ok = 0; miss = []
for n, gates in [(4500, [4001, 1234, 77, 3333]), (6000, [5555, 5553, 2468, 999]), (6900, [6789, 3456, 6001, 150])]:
    for g in gates:
        body = {"model": "glm", "messages": [{"role": "user", "content": recs(n) + f"\n\nWhat code does gate {g} use? Answer with the code only, for example K123."}],
                "max_tokens": 64, "temperature": 0, "cache_salt": f"ns-{TAG}-{n}"}
        r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900).read())
        c = r["choices"][0]["message"].get("content") or ""; want = f"K{(g * 7919) % 9973}"
        if want in c: ok += 1
        else: miss.append(f"{n}/{g} got {c.strip()[:12]!r}")
print(f"{ok}/12 right" + ("; missed: " + ", ".join(miss) if miss else ""))
